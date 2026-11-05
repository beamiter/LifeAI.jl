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

function hf_ripple_tp_forward(bundle::RippleTensorParallelModel,token_ids::AbstractVector{<:Integer};
        cache::Union{Nothing,RippleTPKVCache}=nothing,last_only::Bool=false,
        output_logits::Bool=true,cache_capacity::Integer=0)
    c=bundle.model;b=bundle.backend;n=_ripple_tp_world(b)
    !isempty(token_ids) && !(eltype(token_ids)<:Bool) || throw(ArgumentError("invalid Ripple TP input IDs"))
    ids=Int.(collect(token_ids))
    all(id->1<=id<=c.vocab_size,ids) || throw(ArgumentError("Ripple TP token ID out of range"))
    cache===nothing || (cache.model===c && length(cache.ranks)==n) || throw(ArgumentError("Ripple TP cache belongs to another model"))
    position=cache===nothing ? 0 : cache.tokens
    length(ids)<=c.max_position_embeddings-position || throw(ArgumentError("Ripple TP context limit exceeded"))
    !(cache_capacity isa Bool) && 0<=cache_capacity<=c.max_position_embeddings || throw(ArgumentError("invalid Ripple TP cache reservation"))
    results=_ripple_tp_run_ranks(b) do rank
        _ripple_tp_with_rank(b,rank) do
            parameters=bundle.ranks[rank+1].parameters
            x=permutedims(Float32.(parameters["model.embed_tokens.weight"][ids,:]))
            c.embed_scale===nothing || (x=x.*c.embed_scale)
            seed=x;updates=Any[]
            for layer in 1:c.num_hidden_layers
                prefix="model.layers.$(layer-1)."
                normalized=_flint_norm(x,parameters[prefix*"input_layernorm.weight"],c.rms_norm_eps)
                past=cache===nothing ? nothing : cache.ranks[rank+1].layers[layer]
                attention,updated=_ripple_attention(bundle.local_model,parameters,prefix*"self_attn.",normalized,
                    seed,layer,past,position;cache_capacity=Int(cache_capacity))
                _ripple_tp_allreduce!(b,rank,attention)
                x=x.+_flint_norm(attention,parameters[prefix*"post_attention_layernorm.weight"],c.rms_norm_eps)
                normalized=_flint_norm(x,parameters[prefix*"pre_feedforward_layernorm.weight"],c.rms_norm_eps)
                feedforward=_ripple_mlp(bundle.local_model,parameters,prefix*"mlp.",normalized)
                _ripple_tp_allreduce!(b,rank,feedforward)
                x=x.+_flint_norm(feedforward,parameters[prefix*"post_feedforward_layernorm.weight"],c.rms_norm_eps)
                push!(updates,updated)
            end
            logits=if output_logits
                last_only && (x=x[:,end:end])
                x=_flint_norm(x,parameters["model.norm.weight"],c.rms_norm_eps)
                head=c.tie_word_embeddings ? parameters["model.embed_tokens.weight"][bundle.ranks[rank+1].vocabulary_rows,:] : parameters["lm_head.weight"]
                local_logits=_ripple_linear(head,x)
                _ripple_tp_allgather(b,rank,local_logits)
            else
                nothing
            end
            _ripple_tp_finish_rank(b,rank)
            return (;logits,updates)
        end
    end
    _ripple_tp_root!(b)
    if cache!==nothing
        for rank in 1:n
            cache.ranks[rank].layers=results[rank].updates
            cache.ranks[rank].tokens=position+length(ids)
        end
        cache.tokens=position+length(ids)
    end
    return results[1].logits
end

function generate_ripple_tp_tokens(bundle,prompt::AbstractVector{<:Integer};max_new_tokens::Integer=128,
        temperature::Real=0f0,top_k=nothing,top_p=1f0,rng=default_rng(),
        eos_ids=bundle.model.eos_token_id.+1,prefill_chunk_size::Integer=256,on_token=nothing,on_prefill=nothing)
    !(max_new_tokens isa Bool) && max_new_tokens>=0 || throw(ArgumentError("invalid Ripple TP max_new_tokens"))
    !(prefill_chunk_size isa Bool) && prefill_chunk_size>0 || throw(ArgumentError("invalid Ripple TP chunk size"))
    !isempty(prompt) && !(eltype(prompt)<:Bool) && all(id->1<=id<=bundle.model.vocab_size,prompt) || throw(ArgumentError("invalid Ripple TP prompt"))
    max_new_tokens<=bundle.model.max_position_embeddings-length(prompt) || throw(ArgumentError("Ripple TP generation exceeds context limit"))
    generated=Int[];max_new_tokens==0 && return generated
    cache=RippleTPKVCache(bundle);logits=nothing;capacity=length(prompt)+Int(max_new_tokens)
    for first in 1:Int(min(prefill_chunk_size,length(prompt))):length(prompt)
        last=min(length(prompt),first+prefill_chunk_size-1)
        logits=hf_ripple_tp_forward(bundle,prompt[first:last];cache,last_only=true,cache_capacity=capacity,
            output_logits=last==length(prompt))
        on_prefill===nothing || on_prefill(last,length(prompt))!==false || return generated
    end
    for step in 1:max_new_tokens
        token=_flint_sample_token(logits,rng;temperature,top_k,top_p);push!(generated,token)
        on_token===nothing || on_token(token)!==false || break
        (token in eos_ids || step==max_new_tokens) && break
        logits=hf_ripple_tp_forward(bundle,[token];cache,last_only=true,cache_capacity=capacity)
    end
    return generated
end

function generate_ripple_tp(bundle,tokenizer,prompt;tools=nothing,clear_thinking::Bool=false,kwargs...)
    text=prompt isa AbstractString ? String(prompt) : apply_ripple_chat_template(tokenizer,prompt;tools,clear_thinking)
    ids=generate_ripple_tp_tokens(bundle,encode(tokenizer,text);kwargs...)
    return (text=decode(tokenizer,ids;skip_special_tokens=true,errors=:replace),token_ids=ids)
end
