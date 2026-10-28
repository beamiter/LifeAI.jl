# Bounded-memory split-K attention for Flint. Each block handles one query,
# one query head and a key partition. Thirty-two keys are evaluated together,
# with eight lanes contributing to each dot product. Online softmax avoids a
# context-by-query score array; the learned attention sink is added only once
# when merging partitions.
const _FLINT_LONG_ATTN_MAX_SPLITS = 128
const _FLINT_LONG_ATTN_MAX_QUERIES = 64
const _FLINT_LONG_ATTN_SCRATCH_BYTES = 256 * 1024 * 1024
const _FLINT_LONG_ATTN_MASK = typemax(UInt32)

function _flint_long_attention_plan(d, heads, queries, keys, window)
    d > 0 && heads > 0 && queries > 0 ||
        throw(ArgumentError("Flint attention dimensions must be positive"))
    keys >= 0 && window >= 0 || throw(ArgumentError("Flint attention lengths must be nonnegative"))
    effective_keys = window > 0 ? min(keys, window) : keys
    bytes_per_partition = (d+2) * heads * sizeof(Float32)
    max_splits = min(_FLINT_LONG_ATTN_MAX_SPLITS,
        max(1, _FLINT_LONG_ATTN_SCRATCH_BYTES ÷ bytes_per_partition))
    splits = min(max_splits, max(1, cld(effective_keys, 128)))
    tile_keys = max(128, cld(effective_keys, splits))
    splits = max(1, cld(effective_keys, tile_keys))
    bytes_per_query = bytes_per_partition * splits
    bytes_per_query <= _FLINT_LONG_ATTN_SCRATCH_BYTES ||
        throw(ArgumentError("Flint attention head dimensions exceed the scratch budget"))
    query_batch = min(queries, _FLINT_LONG_ATTN_MAX_QUERIES,
        max(1, _FLINT_LONG_ATTN_SCRATCH_BYTES ÷ bytes_per_query))
    return (splits=splits, tile_keys=tile_keys, query_batch=query_batch,
        scratch_bytes=bytes_per_query*query_batch,
        partial_shared_bytes=(2d+67)*sizeof(Float32),
        merge_shared_bytes=(splits+1)*sizeof(Float32))
end

@inline function _flint_long_warp_sum(value)
    value += CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 16)
    value += CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 8)
    value += CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 4)
    value += CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 2)
    return value + CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 1)
end

@inline function _flint_long_warp_max(value)
    value = max(value, CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 16))
    value = max(value, CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 8))
    value = max(value, CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 4))
    value = max(value, CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 2))
    return max(value, CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK, value, 1))
end

function _flint_long_attention_partials_kernel!(partials, query, key, value,
        first_key_position, query_position, window, query_offset, tile_keys, scale)
    thread = CUDA.threadIdx().x
    head = CUDA.blockIdx().x
    local_query = CUDA.blockIdx().y
    partition = CUDA.blockIdx().z
    query_index = query_offset + local_query
    d = Int32(size(query,1))
    heads_per_kv = Int32(size(query,2) ÷ size(key,2))
    kvhead = (head-Int32(1)) ÷ heads_per_kv + Int32(1)
    position = query_position + query_index-Int32(1)
    upper = min(Int32(size(key,3)), position-first_key_position+Int32(1))
    lower = window > 0 ? max(Int32(1),position-window-first_key_position+Int32(2)) : Int32(1)
    first_key = lower + (partition-Int32(1))*tile_keys
    last_key = min(lower + partition*tile_keys-Int32(1), upper)
    if first_key > last_key
        row = thread
        while row <= d
            @inbounds partials[row,head,local_query,partition] = 0f0
            row += CUDA.blockDim().x
        end
        if thread == 1
            @inbounds begin
                partials[d+1,head,local_query,partition] = -Inf32
                partials[d+2,head,local_query,partition] = 0f0
            end
        end
        return
    end

    shared = CUDA.CuDynamicSharedArray(Float32, 2d+67)
    row = thread
    while row <= d
        @inbounds begin
            shared[row] = query[row,head,query_index]
            shared[d+row] = 0f0
        end
        row += CUDA.blockDim().x
    end
    state = 2d+65
    if thread == 1
        @inbounds begin
            shared[state] = -Inf32
            shared[state+1] = 0f0
            shared[state+2] = 0f0
        end
    end
    CUDA.sync_threads()

    lane = (thread-Int32(1)) % Int32(8)
    key_lane = (thread-Int32(1)) ÷ Int32(8)
    start = first_key
    parity = Int32(0)
    while start <= last_key
        key_index = start+key_lane
        dot = 0f0
        if key_index <= last_key
            row = lane+Int32(1)
            while row <= d
                @inbounds dot = muladd(shared[row],key[row,kvhead,key_index],dot)
                row += Int32(8)
            end
        end
        dot += CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK,dot,4,8)
        dot += CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK,dot,2,8)
        dot += CUDA.shfl_down_sync(_FLINT_LONG_ATTN_MASK,dot,1,8)
        score_base = 2d+parity*Int32(32)
        if lane == 0
            @inbounds shared[score_base+key_lane+1] = key_index <= last_key ? dot*scale : -Inf32
        end
        CUDA.sync_threads()
        if thread <= 32
            @inbounds score = shared[score_base+thread]
            maximum_score = _flint_long_warp_max(score)
            maximum_score = CUDA.shfl_sync(_FLINT_LONG_ATTN_MASK,maximum_score,1)
            @inbounds old_maximum = shared[state]
            maximum_score = max(old_maximum,maximum_score)
            old_scale = old_maximum == -Inf32 ? 0f0 : exp(old_maximum-maximum_score)
            probability = score == -Inf32 ? 0f0 : exp(score-maximum_score)
            @inbounds shared[score_base+thread] = probability
            denominator = _flint_long_warp_sum(probability)
            if thread == 1
                @inbounds begin
                    shared[state] = maximum_score
                    shared[state+1] = shared[state+1]*old_scale+denominator
                    shared[state+2] = old_scale
                end
            end
        end
        CUDA.sync_threads()
        @inbounds old_scale = shared[state+2]
        row = thread
        while row <= d
            weighted_value = 0f0
            for offset in Int32(0):Int32(31)
                key_index = start+offset
                if key_index <= last_key
                    @inbounds weighted_value = muladd(shared[score_base+offset+1],
                        value[row,kvhead,key_index],weighted_value)
                end
            end
            @inbounds shared[d+row] = shared[d+row]*old_scale+weighted_value
            row += CUDA.blockDim().x
        end
        # The next iteration writes the other score buffer. Its first barrier
        # also waits for all current value accumulations before changing state.
        parity = Int32(1)-parity
        start += Int32(32)
    end
    row = thread
    while row <= d
        @inbounds partials[row,head,local_query,partition] = shared[d+row]
        row += CUDA.blockDim().x
    end
    if thread == 1
        @inbounds begin
            partials[d+1,head,local_query,partition] = shared[state]
            partials[d+2,head,local_query,partition] = shared[state+1]
        end
    end
    return
end

function _flint_long_attention_merge_kernel!(output, partials, sinks, query_offset, splits)
    thread = CUDA.threadIdx().x
    head = CUDA.blockIdx().x
    local_query = CUDA.blockIdx().y
    query_index = query_offset+local_query
    d = Int32(size(output,1))
    shared = CUDA.CuDynamicSharedArray(Float32,splits+1)
    if thread <= 32
        @inbounds sink = sinks[head]
        maximum_score = thread == 1 ? sink : -Inf32
        partition = thread
        while partition <= splits
            @inbounds maximum_score = max(maximum_score,partials[d+1,head,local_query,partition])
            partition += Int32(32)
        end
        maximum_score = _flint_long_warp_max(maximum_score)
        maximum_score = CUDA.shfl_sync(_FLINT_LONG_ATTN_MASK,maximum_score,1)
        denominator = thread == 1 ? exp(sink-maximum_score) : 0f0
        partition = thread
        while partition <= splits
            @inbounds weight = exp(partials[d+1,head,local_query,partition]-maximum_score)
            @inbounds begin
                shared[partition] = weight
                denominator += partials[d+2,head,local_query,partition]*weight
            end
            partition += Int32(32)
        end
        denominator = _flint_long_warp_sum(denominator)
        thread == 1 && (@inbounds shared[splits+1] = inv(denominator))
    end
    CUDA.sync_threads()
    @inbounds inverse_denominator = shared[splits+1]
    row = thread
    while row <= d
        total = 0f0
        for partition in Int32(1):splits
            @inbounds total = muladd(partials[row,head,local_query,partition],shared[partition],total)
        end
        @inbounds output[row,head,query_index] = total*inverse_denominator
        row += CUDA.blockDim().x
    end
    return
end

_flint_long_array_context(array::CUDA.CuArray) = CUDA.context(array)
_flint_long_array_context(array::SubArray) = _flint_long_array_context(parent(array))

function LifeAI._flint_attention_long(query::CUDA.CuArray{Float32,3},
        key::CUDA.StridedCuArray{Float32,3},value::CUDA.StridedCuArray{Float32,3},
        sinks::CUDA.CuVector{Float32},first_key_position::Integer,query_position::Integer,window::Integer)
    d,heads,queries = size(query)
    d > 0 && heads > 0 && queries > 0 || throw(ArgumentError("Flint attention requires nonempty queries"))
    size(key) == size(value) && size(key,1) == d ||
        throw(DimensionMismatch("Flint attention key/value dimensions differ"))
    size(key,2) > 0 && heads % size(key,2) == 0 ||
        throw(DimensionMismatch("Flint query heads must be divisible by KV heads"))
    length(sinks) == heads || throw(DimensionMismatch("Flint attention sink count differs"))
    window >= 0 || throw(ArgumentError("Flint attention window must be nonnegative"))
    all(n -> typemin(Int32) <= n <= typemax(Int32),
        (first_key_position,query_position,query_position+queries,window,size(key,3))) ||
        throw(ArgumentError("Flint attention positions exceed supported Int32 indexing"))
    context = CUDA.context()
    CUDA.context(query) == _flint_long_array_context(key) == _flint_long_array_context(value) ==
        CUDA.context(sinks) == context || throw(ArgumentError("Flint attention arrays must share the active CUDA context"))
    if queries >= 16
        return _flint_attention_gemm(query,key,value,sinks,first_key_position,query_position,window)
    end
    return _flint_attention_splitk(query,key,value,sinks,first_key_position,query_position,window)
end

function _flint_attention_splitk(query,key,value,sinks,first_key_position,query_position,window)
    d,heads,queries = size(query)
    plan = _flint_long_attention_plan(d,heads,queries,size(key,3),window)
    plan.partial_shared_bytes <= 48*1024 ||
        throw(ArgumentError("Flint attention head dimension exceeds shared-memory capacity"))
    output = similar(query)
    partials = CUDA.CuArray{Float32}(undef,d+2,heads,plan.query_batch,plan.splits)
    scale = inv(sqrt(Float32(d)))
    GC.@preserve query key value sinks output partials begin
        for offset in 0:plan.query_batch:queries-1
            count = min(plan.query_batch,queries-offset)
            CUDA.@cuda threads=256 blocks=(heads,count,plan.splits) shmem=plan.partial_shared_bytes _flint_long_attention_partials_kernel!(
                partials,query,key,value,Int32(first_key_position),Int32(query_position),Int32(window),
                Int32(offset),Int32(plan.tile_keys),scale)
            CUDA.@cuda threads=256 blocks=(heads,count) shmem=plan.merge_shared_bytes _flint_long_attention_merge_kernel!(
                output,partials,sinks,Int32(offset),Int32(plan.splits))
        end
    end
    return output
end

# Long prefill uses FP32 batched GEMMs and online softmax over key tiles.
# The score tile is overwritten with probabilities; KV packing, score storage,
# and query microbatches together stay within one GiB of temporary storage.
import NNlib: batched_mul!
const _FLINT_GEMM_ATTN_SCRATCH_BYTES = 1024*1024*1024
function _flint_gemm_attention_plan(d,heads,kvheads,queries,keys)
    key_tile=min(max(1,keys),8192)
    fixed_bytes=2*d*kvheads*key_tile*sizeof(Float32)
    per_query=(key_tile*heads+2*d*heads+4*heads)*sizeof(Float32)
    query_batch=min(queries,512,max(1,(_FLINT_GEMM_ATTN_SCRATCH_BYTES-fixed_bytes)÷per_query))
    scratch_bytes=fixed_bytes+per_query*query_batch
    scratch_bytes<=_FLINT_GEMM_ATTN_SCRATCH_BYTES || throw(ArgumentError("Flint GEMM attention dimensions exceed scratch capacity"))
    return (;key_tile,query_batch,scratch_bytes)
end

function _flint_gemm_pack_kv_kernel!(packed_key,packed_value,key,value,offset,count)
    tx,ty=CUDA.threadIdx().x,CUDA.threadIdx().y
    row=(CUDA.blockIdx().x-Int32(1))*Int32(32)+tx
    token=(CUDA.blockIdx().y-Int32(1))*Int32(32)+ty
    head=CUDA.blockIdx().z
    tile=CUDA.CuStaticSharedArray(Float32,(33,32))
    for step in Int32(0):Int32(8):Int32(24)
        if row<=size(key,1) && token+step<=count
            @inbounds begin
                tile[tx,ty+step]=key[row,head,offset+token+step]
                packed_value[row,token+step,head]=value[row,head,offset+token+step]
            end
        end
    end
    CUDA.sync_threads()
    out_token=(CUDA.blockIdx().y-Int32(1))*Int32(32)+tx
    out_row=(CUDA.blockIdx().x-Int32(1))*Int32(32)+ty
    for step in Int32(0):Int32(8):Int32(24)
        if out_token<=count && out_row+step<=size(key,1)
            @inbounds packed_key[out_token,out_row+step,head]=tile[ty+step,tx]
        end
    end
    return
end

function _flint_gemm_state_kernel!(state,sinks)
    index=(CUDA.blockIdx().x-Int32(1))*CUDA.blockDim().x+CUDA.threadIdx().x
    if index<=size(state,2)*size(state,3)
        head=(index-Int32(1))%Int32(size(state,2))+Int32(1)
        token=(index-Int32(1))÷Int32(size(state,2))+Int32(1)
        @inbounds begin
            state[1,head,token]=sinks[head]
            state[2,head,token]=1f0
            state[3,head,token]=0f0
            state[4,head,token]=0f0
        end
    end
    return
end

function _flint_gemm_softmax_kernel!(scores,state,group,first_key_position,query_position,window,scale)
    thread=CUDA.threadIdx().x
    column=CUDA.blockIdx().x
    kvhead=CUDA.blockIdx().y
    head=(column-Int32(1))%group+group*(kvhead-Int32(1))+Int32(1)
    query=(column-Int32(1))÷group+Int32(1)
    query_absolute=query_position+query-Int32(1)
    warp=(thread-Int32(1))÷Int32(32)+Int32(1)
    lane=(thread-Int32(1))%Int32(32)+Int32(1)
    shared=CUDA.CuStaticSharedArray(Float32,19)
    maximum_score=-Inf32
    row=thread
    while row<=size(scores,1)
        key_absolute=first_key_position+row-Int32(1)
        allowed=key_absolute<=query_absolute && (window==0 || key_absolute>query_absolute-window)
        @inbounds score=allowed ? scores[row,column,kvhead]*scale : -Inf32
        @inbounds scores[row,column,kvhead]=score
        maximum_score=max(maximum_score,score)
        row+=CUDA.blockDim().x
    end
    maximum_score=_flint_long_warp_max(maximum_score)
    lane==1 && (@inbounds shared[warp]=maximum_score)
    CUDA.sync_threads()
    if thread<=32
        @inbounds maximum_score=thread<=8 ? shared[thread] : -Inf32
        maximum_score=_flint_long_warp_max(maximum_score)
        if thread==1
            @inbounds begin
                old_maximum=state[1,head,query]
                shared[19]=maximum_score
                shared[17]=max(old_maximum,maximum_score)
                shared[18]=exp(old_maximum-shared[17])
            end
        end
    end
    CUDA.sync_threads()
    @inbounds maximum_score=shared[19]
    denominator=0f0
    row=thread
    while row<=size(scores,1)
        @inbounds score=scores[row,column,kvhead]
        probability=score == -Inf32 ? 0f0 : exp(score-maximum_score)
        @inbounds scores[row,column,kvhead]=probability
        denominator+=probability
        row+=CUDA.blockDim().x
    end
    denominator=_flint_long_warp_sum(denominator)
    lane==1 && (@inbounds shared[8+warp]=denominator)
    CUDA.sync_threads()
    if thread<=32
        @inbounds denominator=thread<=8 ? shared[8+thread] : 0f0
        denominator=_flint_long_warp_sum(denominator)
        if thread==1
            @inbounds begin
                tile_scale=shared[19] == -Inf32 ? 0f0 : exp(shared[19]-shared[17])
                state[1,head,query]=shared[17]
                state[2,head,query]=state[2,head,query]*shared[18]+denominator*tile_scale
                state[3,head,query]=shared[18]
                state[4,head,query]=tile_scale
            end
        end
    end
    return
end

function _flint_gemm_accumulate_kernel!(output,context,state,group,query_offset,last_tile)
    index=(CUDA.blockIdx().x-Int32(1))*CUDA.blockDim().x+CUDA.threadIdx().x
    stride=CUDA.blockDim().x*CUDA.gridDim().x
    d=Int32(size(context,1))
    while index<=length(context)
        row=(index-Int32(1))%d+Int32(1)
        column=(index-Int32(1))÷d%Int32(size(context,2))+Int32(1)
        kvhead=(index-Int32(1))÷(d*Int32(size(context,2)))+Int32(1)
        head=(column-Int32(1))%group+group*(kvhead-Int32(1))+Int32(1)
        query=(column-Int32(1))÷group+Int32(1)
        @inbounds begin
            accumulated=muladd(context[row,column,kvhead],state[4,head,query],
                output[row,head,query_offset+query]*state[3,head,query])
            output[row,head,query_offset+query]=last_tile ? accumulated/state[2,head,query] : accumulated
        end
        index+=stride
    end
    return
end

function _flint_attention_gemm(query,key,value,sinks,first_key_position,query_position,window)
    d,heads,queries=size(query)
    kvheads,keys=size(key,2),size(key,3)
    group=heads÷kvheads
    plan=_flint_gemm_attention_plan(d,heads,kvheads,queries,keys)
    output=CUDA.zeros(Float32,d,heads,queries)
    keys==0 && return output
    packed_key=CUDA.CuArray{Float32}(undef,plan.key_tile,d,kvheads)
    packed_value=CUDA.CuArray{Float32}(undef,d,plan.key_tile,kvheads)
    scores=CUDA.CuArray{Float32}(undef,plan.key_tile,group*plan.query_batch,kvheads)
    context=CUDA.CuArray{Float32}(undef,d,group*plan.query_batch,kvheads)
    state=CUDA.CuArray{Float32}(undef,4,heads,plan.query_batch)
    scale=inv(sqrt(Float32(d)))
    for query_offset in 0:plan.query_batch:queries-1
        query_count=min(plan.query_batch,queries-query_offset)
        query_view=view(query,:,:,query_offset+1:query_offset+query_count)
        packed_query=reshape(permutedims(reshape(query_view,d,group,kvheads,query_count),(1,2,4,3)),d,group*query_count,kvheads)
        query_state=view(state,:,:,1:query_count)
        tile_context=view(context,:,1:group*query_count,:)
        CUDA.@cuda threads=256 blocks=cld(heads*query_count,256) _flint_gemm_state_kernel!(query_state,sinks)
        # Causal queries cannot read future chunks; local queries also cannot
        # read before their earliest window. This bounds sliding work to W+Q.
        first_key=window>0 ? max(0,query_position+query_offset-window-first_key_position+1) : 0
        last_key=min(keys,query_position+query_offset+query_count-first_key_position)
        first_key=min(keys,max(0,first_key))
        if last_key<=first_key
            continue
        end
        for key_offset in first_key:plan.key_tile:last_key-1
            key_count=min(plan.key_tile,last_key-key_offset)
            tile_key=view(packed_key,1:key_count,:,:)
            tile_value=view(packed_value,:,1:key_count,:)
            tile_scores=view(scores,1:key_count,1:group*query_count,:)
            CUDA.@cuda threads=(32,8) blocks=(cld(d,32),cld(key_count,32),kvheads) _flint_gemm_pack_kv_kernel!(packed_key,packed_value,key,value,Int32(key_offset),Int32(key_count))
            batched_mul!(tile_scores,tile_key,packed_query)
            CUDA.@cuda threads=256 blocks=(group*query_count,kvheads) _flint_gemm_softmax_kernel!(tile_scores,query_state,Int32(group),Int32(first_key_position+key_offset),Int32(query_position+query_offset),Int32(window),scale)
            batched_mul!(tile_context,tile_value,tile_scores)
            last_tile=key_offset+key_count==last_key
            CUDA.@cuda threads=256 blocks=min(cld(length(tile_context),256),65535) _flint_gemm_accumulate_kernel!(output,tile_context,query_state,Int32(group),Int32(query_offset),last_tile)
        end
    end
    return output
end
