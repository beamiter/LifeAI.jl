"""Transport interface for one-node Ripple tensor parallelism."""
abstract type AbstractRippleTPBackend end

"""A CPU rendezvous backend for numerical tests of actual tensor partitioning."""
mutable struct RippleTPReferenceBackend <: AbstractRippleTPBackend
    world_size::Int
    condition::Threads.Condition
    generation::Int
    arrived::Int
    operation::Symbol
    shape::Tuple
    inputs::Vector{Any}
    outputs::Vector{Any}
    failure::Any
end
function RippleTPReferenceBackend(world_size::Integer)
    n=_hf_strict_host_int(world_size,"Ripple TP world size")
    n>0 || throw(ArgumentError("Ripple TP world size must be positive"))
    RippleTPReferenceBackend(n,Threads.Condition(),0,0,:none,(),Any[nothing for _ in 1:n],Any[nothing for _ in 1:n],nothing)
end

_ripple_tp_world(b::RippleTPReferenceBackend)=b.world_size
_ripple_tp_with_rank(f,b::RippleTPReferenceBackend,rank)=f()
_ripple_tp_finish_rank(b::RippleTPReferenceBackend,rank)=nothing
_ripple_tp_root!(b::RippleTPReferenceBackend)=nothing
_ripple_tp_upload(b::RippleTPReferenceBackend,rank,raw,dtype,shape,target)=
    _decode_safetensors_values(raw,dtype,collect(shape);target_dtype=target)

function _ripple_tp_abort!(b::RippleTPReferenceBackend,error)
    lock(b.condition) do
        b.failure=error
        notify(b.condition;all=true)
    end
end
function _ripple_tp_reference_collective!(b,rank,input,output,operation)
    lock(b.condition) do
        b.failure===nothing || throw(b.failure)
        generation=b.generation
        if b.arrived==0
            b.operation=operation;b.shape=size(input)
        else
            b.operation==operation && b.shape==size(input) || throw(ArgumentError("Ripple TP collective mismatch"))
        end
        b.inputs[rank+1]=input;b.outputs[rank+1]=output;b.arrived+=1
        if b.arrived==b.world_size
            combined=if operation===:sum
                value=copy(b.inputs[1])
                for item in b.inputs[2:end];value .+= item;end
                value
            else
                vcat(b.inputs...)
            end
            for destination in b.outputs;copyto!(destination,combined);end
            b.arrived=0;b.generation+=1
            notify(b.condition;all=true)
        else
            while b.generation==generation && b.failure===nothing;wait(b.condition);end
            b.failure===nothing || throw(b.failure)
        end
    end
    return output
end
_ripple_tp_allreduce!(b::RippleTPReferenceBackend,rank,value)=
    _ripple_tp_reference_collective!(b,rank,value,value,:sum)
function _ripple_tp_allgather(b::RippleTPReferenceBackend,rank,value)
    output=similar(value,size(value,1)*b.world_size,size(value,2))
    return _ripple_tp_reference_collective!(b,rank,value,output,:gather)
end
function _ripple_tp_run_ranks(f,b::RippleTPReferenceBackend)
    results=Vector{Any}(undef,b.world_size)
    @sync for rank in 0:b.world_size-1
        @async try
            results[rank+1]=f(rank)
        catch error
            _ripple_tp_abort!(b,error)
            rethrow()
        end
    end
    return results
end

"""Create a CUDA/NCCL backend after loading `CUDA` and `NCCL`. Devices must be unique."""
function ripple_nccl_backend end
function close_ripple_tp_backend! end

"""
    prepare_ripple_tp_projections!(bundle; mode=:bf16_gemv)

Choose `:reference` (original BF16-to-FP32 GEMM), `:bf16_gemv` (direct BF16
single-token projections with Float32 accumulation), or `:fp32` (resident
FP32 projections). Only call while idle. Conversions back to BF16 must be
lossless; the source checkpoint values are never quantized.
"""
function prepare_ripple_tp_projections! end
_ripple_projection_array(weight)=weight
function _ripple_bf16_projection end

_ripple_tp_before_forward!(b::AbstractRippleTPBackend)=nothing
_ripple_tp_before_forward!(b::AbstractRippleTPBackend,tokens::Int)=_ripple_tp_before_forward!(b)
close_ripple_tp_backend!(b::RippleTPReferenceBackend)=nothing

function ripple_tensor_parallel_plan(c::RippleConfig,world_size::Integer)
    n=_hf_strict_host_int(world_size,"Ripple TP world size")
    n>0 || throw(ArgumentError("Ripple TP world size must be positive"))
    for (name,value) in ((:num_attention_heads,c.num_attention_heads),(:num_key_value_heads,c.num_key_value_heads),
            (:intermediate_size,c.intermediate_size),(:vocab_size,c.vocab_size))
        value%n==0 || throw(ArgumentError("Ripple TP $name must be divisible by world size"))
    end
    return (world_size=n,query_heads_per_rank=c.num_attention_heads÷n,
        kv_heads_per_rank=c.num_key_value_heads÷n,ffn_width_per_rank=c.intermediate_size÷n,
        vocabulary_per_rank=c.vocab_size÷n,embedding=:replicated,attention_output=:sum,
        feedforward_output=:sum,logits=:vocabulary_allgather)
end

"""One-based tensor slice. Rows are output-parallel; columns are reduction-parallel."""
function ripple_tensor_partition(c::RippleConfig,name::AbstractString,rank::Integer,world_size::Integer)
    plan=ripple_tensor_parallel_plan(c,world_size)
    shapes=ripple_expected_tensor_shapes(c)
    haskey(shapes,name) || throw(KeyError(name))
    return _ripple_tp_partition(name,shapes[name],rank,plan)
end
function _ripple_tp_partition(name,shape,rank,plan)
    rank=_hf_strict_host_int(rank,"Ripple TP rank")
    0<=rank<plan.world_size || throw(ArgumentError("Ripple TP rank out of range"))
    axis=if name=="lm_head.weight" || occursin(r"\.(q_proj|k_proj|v_proj|g_proj|gate_proj|up_proj)\.weight$",name)
        1
    elseif occursin(r"\.(o_proj|down_proj)\.weight$",name)
        2
    else
        0
    end
    if axis==0
        return (axis=0,range=1:shape[1],shape=shape)
    end
    shape[axis]%plan.world_size==0 || throw(ArgumentError("Ripple TP tensor dimension is not divisible"))
    width=shape[axis]÷plan.world_size
    local_shape=ntuple(i->i==axis ? width : shape[i],length(shape))
    return (axis=axis,range=rank*width+1:(rank+1)*width,shape=local_shape)
end

function _ripple_tp_slice_raw(raw,source_shape,partition,element_bytes)
    partition.axis==0 && return raw
    rows,columns=source_shape
    if partition.axis==1
        byte_start=(first(partition.range)-1)*columns*element_bytes+1
        byte_stop=last(partition.range)*columns*element_bytes
        return view(raw,byte_start:byte_stop)
    end
    width=length(partition.range)*element_bytes
    result=Vector{UInt8}(undef,rows*width)
    offset=(first(partition.range)-1)*element_bytes
    for row in 0:rows-1
        copyto!(result,row*width+1,raw,row*columns*element_bytes+offset+1,width)
    end
    return result
end

struct RippleTPRank
    parameters::Dict{String,Any}
    vocabulary_rows::UnitRange{Int}
end
struct RippleTensorParallelModel{B<:AbstractRippleTPBackend}
    model::RippleConfig
    local_model::RippleConfig
    ranks::Vector{RippleTPRank}
    backend::B
    source::String
end
mutable struct RippleTPKVCache
    model::RippleConfig
    ranks::Vector{RippleKVCache}
    tokens::Int
end
RippleTPKVCache(bundle::RippleTensorParallelModel)=RippleTPKVCache(bundle.model,
    [RippleKVCache(bundle.local_model) for _ in bundle.ranks],0)
function reset_ripple_tp_cache!(cache::RippleTPKVCache)
    foreach(reset_ripple_cache!,cache.ranks);cache.tokens=0;return cache
end

"""
    load_hf_ripple_tensor_parallel(path; backend, dtype=BFloat16, on_progress=nothing)

Load the full unquantized checkpoint with output-row Q/K/V/gate/FFN shards,
input-column O/down shards and vocabulary-row output shards. Embeddings and
norms are replicated. Each source tensor is read once, and GPU layout conversion
is performed on its owning device. `backend` is required explicitly.
"""
function load_hf_ripple_tensor_parallel(path::AbstractString;backend::AbstractRippleTPBackend,
        dtype::Type=BFloat16,on_progress=nothing)
    dtype in (Float32,BFloat16) || throw(ArgumentError("invalid Ripple TP dtype"))
    c=load_hf_ripple_config(path);n=_ripple_tp_world(backend);plan=ripple_tensor_parallel_plan(c,n)
    options=Dict{Symbol,Any}(key=>getfield(c,key) for key in fieldnames(RippleConfig))
    merge!(options,Dict(:num_attention_heads=>plan.query_heads_per_rank,:num_key_value_heads=>plan.kv_heads_per_rank,
        :intermediate_size=>plan.ffn_width_per_rank))
    local_model=RippleConfig(;options...)
    ranks=[RippleTPRank(Dict{String,Any}(),rank*plan.vocabulary_per_rank+1:(rank+1)*plan.vocabulary_per_rank) for rank in 0:n-1]
    reader=open_safetensors_reader(path);shapes=verify_ripple_checkpoint(reader,c)
    for (index,name) in enumerate(sort!(collect(keys(shapes))))
        location=reader.locations[name]
        element_bytes=location.dtype=="BF16" ? 2 : location.dtype=="F32" ? 4 : throw(ArgumentError("unsupported Ripple TP source dtype"))
        raw=open(location.path,"r") do io
            seek(io,location.data_base+location.data_start)
            read(io,location.data_stop-location.data_start)
        end
        length(raw)==location.data_stop-location.data_start || throw(ArgumentError("truncated Ripple TP source tensor"))
        partitions=[_ripple_tp_partition(name,shapes[name],rank,plan) for rank in 0:n-1]
        target=length(shapes[name])==1 ? Float32 : dtype
        _ripple_tp_run_ranks(backend) do rank
            _ripple_tp_with_rank(backend,rank) do
                part=partitions[rank+1]
                data=_ripple_tp_slice_raw(raw,shapes[name],part,element_bytes)
                ranks[rank+1].parameters[name]=_ripple_tp_upload(backend,rank,data,location.dtype,part.shape,target)
            end
        end
        on_progress===nothing || on_progress(name,index,length(shapes))
        index%20==0 && GC.gc(false)
    end
    _ripple_tp_root!(backend)
    return RippleTensorParallelModel(c,local_model,ranks,backend,abspath(path))
end

# Complete local CUDA work before entering a collective phase. Allocation and
# CUDA memory-pool reclamation may synchronize streams, so they must not run
# between different GPUs' collective launches in the same process.
function _ripple_tp_compute_ranks(f,b)
    return _ripple_tp_run_ranks(b) do rank
        _ripple_tp_with_rank(b,rank) do
            result=f(rank)
            _ripple_tp_finish_rank(b,rank)
            result
        end
    end
end
function _ripple_tp_allreduce_batch!(b,values)
    _ripple_tp_compute_ranks(b) do rank
        _ripple_tp_allreduce!(b,rank,values[rank+1])
    end
    return values
end
function _ripple_tp_allgather_batch(b,values)
    return _ripple_tp_compute_ranks(b) do rank
        _ripple_tp_allgather(b,rank,values[rank+1])
    end
end

@inline function _ripple_tp_stage(callback,layer,stage,started)
    callback===nothing && return UInt64(0)
    stopped=time_ns();callback(layer,stage,Float64(stopped-started)/1e9)
    return time_ns()
end

function hf_ripple_tp_forward(bundle::RippleTensorParallelModel,token_ids::AbstractVector{<:Integer};
        cache::Union{Nothing,RippleTPKVCache}=nothing,last_only::Bool=false,
        output_logits::Bool=true,cache_capacity::Integer=0,
        schedule::Symbol=:merged,on_stage=nothing,prefill_query_batch::Integer=512,prefill_gc_layers::Integer=8)
    c=bundle.model;b=bundle.backend;n=_ripple_tp_world(b)
    !isempty(token_ids) && !(eltype(token_ids)<:Bool) || throw(ArgumentError("invalid Ripple TP input IDs"))
    ids=Int.(collect(token_ids))
    all(id->1<=id<=c.vocab_size,ids) || throw(ArgumentError("Ripple TP token ID out of range"))
    cache===nothing || (cache.model===c && length(cache.ranks)==n) || throw(ArgumentError("Ripple TP cache belongs to another model"))
    position=cache===nothing ? 0 : cache.tokens
    length(ids)<=c.max_position_embeddings-position || throw(ArgumentError("Ripple TP context limit exceeded"))
    !(cache_capacity isa Bool) && 0<=cache_capacity<=c.max_position_embeddings || throw(ArgumentError("invalid Ripple TP cache reservation"))
    schedule in (:staged,:merged) || throw(ArgumentError("invalid Ripple TP schedule"))
    !(prefill_query_batch isa Bool) && prefill_query_batch>0 || throw(ArgumentError("invalid prefill query batch"))
    !(prefill_gc_layers isa Bool) && prefill_gc_layers>0 || throw(ArgumentError("invalid prefill GC layer interval"))
    merged=schedule===:merged
    clock=on_stage===nothing ? UInt64(0) : time_ns()
    _ripple_tp_before_forward!(b,length(ids))
    clock=_ripple_tp_stage(on_stage,0,:gc,clock)
    states=_ripple_tp_compute_ranks(b) do rank
        parameters=bundle.ranks[rank+1].parameters
        x=permutedims(Float32.(parameters["model.embed_tokens.weight"][ids,:]))
        c.embed_scale===nothing || (x=x.*c.embed_scale)
        (x=x,seed=x,updates=Any[],pending=nothing)
    end
    clock=_ripple_tp_stage(on_stage,0,:embedding,clock)
    for layer in 1:c.num_hidden_layers
        prefix="model.layers.$(layer-1)."
        attention=_ripple_tp_compute_ranks(b) do rank
            parameters=bundle.ranks[rank+1].parameters;state=states[rank+1]
            x=state.x
            if merged && layer>1
                previous="model.layers.$(layer-2).post_feedforward_layernorm.weight"
                x=x.+_flint_norm(state.pending,parameters[previous],c.rms_norm_eps)
            end
            normalized=_flint_norm(x,parameters[prefix*"input_layernorm.weight"],c.rms_norm_eps)
            past=cache===nothing ? nothing : cache.ranks[rank+1].layers[layer]
            value,updated=_ripple_attention(bundle.local_model,parameters,prefix*"self_attn.",normalized,
                state.seed,layer,past,position;cache_capacity=Int(cache_capacity),reuse_constants=merged,prefill_query_batch=Int(prefill_query_batch))
            (;x,value,updated)
        end
        clock=_ripple_tp_stage(on_stage,layer,:attention,clock)
        _ripple_tp_allreduce_batch!(b,[item.value for item in attention])
        clock=_ripple_tp_stage(on_stage,layer,:attention_reduce,clock)
        feedforward=_ripple_tp_compute_ranks(b) do rank
            parameters=bundle.ranks[rank+1].parameters;state=states[rank+1]
            x=attention[rank+1].x.+_flint_norm(attention[rank+1].value,parameters[prefix*"post_attention_layernorm.weight"],c.rms_norm_eps)
            normalized=_flint_norm(x,parameters[prefix*"pre_feedforward_layernorm.weight"],c.rms_norm_eps)
            value=_ripple_mlp(bundle.local_model,parameters,prefix*"mlp.",normalized;fuse=merged)
            push!(state.updates,attention[rank+1].updated)
            (x=x,value=value,seed=state.seed,updates=state.updates,pending=value)
        end
        clock=_ripple_tp_stage(on_stage,layer,:feedforward,clock)
        _ripple_tp_allreduce_batch!(b,[item.value for item in feedforward])
        clock=_ripple_tp_stage(on_stage,layer,:feedforward_reduce,clock)
        states=if merged
            feedforward
        else
            _ripple_tp_compute_ranks(b) do rank
                parameters=bundle.ranks[rank+1].parameters;state=states[rank+1]
                x=feedforward[rank+1].x.+_flint_norm(feedforward[rank+1].value,
                    parameters[prefix*"post_feedforward_layernorm.weight"],c.rms_norm_eps)
                (x=x,seed=state.seed,updates=state.updates,pending=nothing)
            end
        end
        clock=_ripple_tp_stage(on_stage,layer,:residual,clock)
        if length(ids)>=512 && layer%prefill_gc_layers==0
            # Completed phases are safe collection boundaries. Clear persistent
            # workers' last arguments/results and collect old large GPU arrays.
            _ripple_tp_compute_ranks(rank->nothing,b)
            GC.gc(true)
            clock=_ripple_tp_stage(on_stage,layer,:prefill_gc,clock)
        end
    end
    logits=if output_logits
        local_logits=_ripple_tp_compute_ranks(b) do rank
            parameters=bundle.ranks[rank+1].parameters;x=states[rank+1].x
            if merged
                final_norm_name="model.layers.$(c.num_hidden_layers-1).post_feedforward_layernorm.weight"
                x=x.+_flint_norm(states[rank+1].pending,parameters[final_norm_name],c.rms_norm_eps)
            end
            last_only && (x=x[:,end:end])
            x=_flint_norm(x,parameters["model.norm.weight"],c.rms_norm_eps)
            head=c.tie_word_embeddings ? parameters["model.embed_tokens.weight"][bundle.ranks[rank+1].vocabulary_rows,:] : parameters["lm_head.weight"]
            _ripple_linear(head,x)
        end
        clock=_ripple_tp_stage(on_stage,c.num_hidden_layers,:head,clock)
        result=_ripple_tp_allgather_batch(b,local_logits)[1]
        clock=_ripple_tp_stage(on_stage,c.num_hidden_layers,:gather,clock)
        result
    else
        nothing
    end
    _ripple_tp_root!(b)
    if cache!==nothing
        for rank in 1:n
            cache.ranks[rank].layers=states[rank].updates
            cache.ranks[rank].tokens=position+length(ids)
        end
        cache.tokens=position+length(ids)
    end
    return logits
end

function generate_ripple_tp_tokens(bundle,prompt::AbstractVector{<:Integer};max_new_tokens::Integer=128,
        temperature::Real=0f0,top_k=nothing,top_p=1f0,rng=default_rng(),
        eos_ids=bundle.model.eos_token_id.+1,prefill_chunk_size::Integer=256,on_token=nothing,on_prefill=nothing,
        schedule::Symbol=:merged,prefix_cache=nothing,prefill_query_batch::Integer=512,prefill_gc_layers::Integer=8)
    !(max_new_tokens isa Bool) && max_new_tokens>=0 || throw(ArgumentError("invalid Ripple TP max_new_tokens"))
    !(prefill_chunk_size isa Bool) && prefill_chunk_size>0 || throw(ArgumentError("invalid Ripple TP chunk size"))
    !isempty(prompt) && !(eltype(prompt)<:Bool) && all(id->1<=id<=bundle.model.vocab_size,prompt) || throw(ArgumentError("invalid Ripple TP prompt"))
    max_new_tokens<=bundle.model.max_position_embeddings-length(prompt) || throw(ArgumentError("Ripple TP generation exceeds context limit"))
    generated=Int[];max_new_tokens==0 && return generated
    capacity=length(prompt)+Int(max_new_tokens)
    policy=(schedule,Int(prefill_chunk_size),Int(prefill_query_batch),Int(prefill_gc_layers))
    try
    cache,reused,logits=prefix_cache===nothing ? (RippleTPKVCache(bundle),0,nothing) :
        _ripple_tp_prefix_start!(prefix_cache,bundle,prompt,capacity,policy)
    if reused>0 && on_prefill!==nothing && on_prefill(reused,length(prompt))===false
        reset_ripple_tp_prefix!(prefix_cache);return generated
    end
    first=reused+1
    while first<=length(prompt)
        last=min(length(prompt),first+prefill_chunk_size-1,(fld(first-1,4096)+1)*4096)
        logits=hf_ripple_tp_forward(bundle,prompt[first:last];cache,last_only=true,cache_capacity=capacity,
            output_logits=last==length(prompt),schedule,prefill_query_batch,prefill_gc_layers)
        if on_prefill!==nothing && on_prefill(last,length(prompt))===false
            prefix_cache===nothing || reset_ripple_tp_prefix!(prefix_cache)
            return generated
        end
        first=last+1
    end
    prefix_cache===nothing || _ripple_tp_prefix_checkpoint!(prefix_cache,bundle,cache,prompt,logits,policy)
    for step in 1:max_new_tokens
        token=_flint_sample_token(logits,rng;temperature,top_k,top_p);push!(generated,token)
        if on_token!==nothing && on_token(token)===false
            prefix_cache===nothing || reset_ripple_tp_prefix!(prefix_cache)
            break
        end
        (token in eos_ids || step==max_new_tokens) && break
        logits=hf_ripple_tp_forward(bundle,[token];cache,last_only=true,cache_capacity=capacity,schedule,prefill_query_batch,prefill_gc_layers)
    end
    return generated
    catch
        prefix_cache===nothing || reset_ripple_tp_prefix!(prefix_cache)
        rethrow()
    end
end

function generate_ripple_tp(bundle,tokenizer,prompt;tools=nothing,clear_thinking::Bool=false,kwargs...)
    text=prompt isa AbstractString ? String(prompt) : apply_ripple_chat_template(tokenizer,prompt;tools,clear_thinking)
    ids=generate_ripple_tp_tokens(bundle,encode(tokenizer,text);kwargs...)
    return (text=decode(tokenizer,ids;skip_special_tokens=true,errors=:replace),token_ids=ids)
end
