"""One serialized, exact-token prompt checkpoint. Generated suffixes are never reused.

Full-attention storage keeps the immutable prompt prefix. Sliding-window keys
and values are copied at the prompt boundary so later decode can safely wrap
its ring without corrupting the checkpoint. Reset when changing runtime policy.
"""
mutable struct RippleTPPrefixCache
    model::RippleConfig
    max_tokens::Int
    ids::Vector{Int}
    cache::Union{Nothing,RippleTPKVCache}
    checkpoint::Any
    logits::Any
    signature::Any
    hits::Int
    misses::Int
    reused_tokens::Int
    prefilled_tokens::Int
    needs_full_gc::Bool
end
function RippleTPPrefixCache(bundle::RippleTensorParallelModel;max_tokens::Integer=bundle.model.max_position_embeddings)
    !(max_tokens isa Bool) && 0<max_tokens<=bundle.model.max_position_embeddings || throw(ArgumentError("invalid prefix cache token limit"))
    RippleTPPrefixCache(bundle.model,Int(max_tokens),Int[],nothing,nothing,nothing,nothing,0,0,0,0,false)
end
function reset_ripple_tp_prefix!(store::RippleTPPrefixCache)
    store.needs_full_gc |= max(length(store.ids),store.prefilled_tokens)>=32768
    empty!(store.ids);store.cache=nothing;store.checkpoint=nothing;store.logits=nothing;store.signature=nothing
    store.reused_tokens=0;store.prefilled_tokens=0
    return store
end
ripple_tp_prefix_stats(store::RippleTPPrefixCache)=(cached_tokens=length(store.ids),hits=store.hits,
    misses=store.misses,reused_tokens=store.reused_tokens,prefilled_tokens=store.prefilled_tokens)

function _ripple_tp_prefix_signature(bundle,schedule)
    weight=bundle.ranks[1].parameters["model.layers.0.self_attn.q_proj.weight"]
    return (schedule,typeof(weight),objectid(_ripple_projection_array(weight)))
end
_ripple_tp_reserve_prefix_layer(past,capacity)=past

function _ripple_tp_release_prefix!(backend,large)
    # Persistent rank tasks otherwise retain their previous closure/results.
    _ripple_tp_compute_ranks(rank->nothing,backend)
    GC.gc(large)
    return
end
function _ripple_tp_prefix_start!(store::RippleTPPrefixCache,bundle,prompt,capacity,policy)
    store.model===bundle.model || throw(ArgumentError("prefix cache belongs to another model"))
    signature=_ripple_tp_prefix_signature(bundle,policy)
    reused=length(store.ids)
    match=store.cache!==nothing && signature==store.signature && reused>0 &&
        reused<=length(prompt) && length(prompt)<=store.max_tokens &&
        all(i->prompt[i]==store.ids[i],eachindex(store.ids))
    if match
        for rank in eachindex(store.cache.ranks)
            store.cache.ranks[rank].layers=copy(store.checkpoint[rank])
            store.cache.ranks[rank].tokens=reused
        end
        store.cache.tokens=reused
        # Release the old snapshot before reserving. Retarget full checkpoints
        # to any grown storage before decode, including exact-prompt repeats.
        store.checkpoint=nothing
        _ripple_tp_compute_ranks(rank->nothing,bundle.backend)
        _ripple_tp_compute_ranks(bundle.backend) do rank
            layers=store.cache.ranks[rank+1].layers
            for layer in eachindex(layers)
                if bundle.model.layer_types[layer]=="full_attention"
                    layers[layer]=_ripple_tp_reserve_prefix_layer(layers[layer],capacity)
                end
            end
        end
        reused>=32768 && GC.gc(true)
        store.hits+=1;store.reused_tokens=reused
        store.prefilled_tokens=length(prompt)-reused
        return store.cache,reused,store.logits
    end
    large=store.needs_full_gc || reused>=32768
    reset_ripple_tp_prefix!(store)
    _ripple_tp_release_prefix!(bundle.backend,large)
    store.needs_full_gc=false
    store.misses+=1;store.prefilled_tokens=length(prompt)
    return RippleTPKVCache(bundle),0,nothing
end
function _ripple_tp_prefix_checkpoint!(store::RippleTPPrefixCache,bundle,cache,prompt,logits,policy)
    length(prompt)<=store.max_tokens || return reset_ripple_tp_prefix!(store)
    checkpoint=_ripple_tp_compute_ranks(bundle.backend) do rank
        [if bundle.model.layer_types[layer]=="sliding_attention"
            past=cache.ranks[rank+1].layers[layer]
            (key=copy(past.key),value=copy(past.value),first_position=past.first_position)
        else
            cache.ranks[rank+1].layers[layer]
        end for layer in 1:bundle.model.num_hidden_layers]
    end
    store.ids=Int.(prompt);store.cache=cache;store.checkpoint=checkpoint
    # Host logits avoid a cross-request CUDA-stream dependency on exact repeats.
    store.logits=Array(logits);store.signature=_ripple_tp_prefix_signature(bundle,policy)
    return store
end
