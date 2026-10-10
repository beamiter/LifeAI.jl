using MLDataDevices: get_device

"""Record cache storage after proving that it cannot overlap earlier tensors."""
_qwen3_vl_storage_parent(storage::SubArray) = _qwen3_vl_storage_parent(parent(storage))
_qwen3_vl_storage_parent(storage::AbstractArray) = storage

function _qwen3_vl_ranges_overlap(a::Integer, b::Integer)
    return a == b
end

function _qwen3_vl_ranges_overlap(a::Integer, b::AbstractUnitRange)
    return first(b) <= a <= last(b)
end

function _qwen3_vl_ranges_overlap(a::AbstractUnitRange, b::Integer)
    return first(a) <= b <= last(a)
end

function _qwen3_vl_ranges_overlap(a::AbstractUnitRange, b::AbstractUnitRange)
    return max(first(a), first(b)) <= min(last(a), last(b))
end

_qwen3_vl_indices_overlap(::Tuple{}, ::Tuple{}) = true

function _qwen3_vl_indices_overlap(
    left::Tuple{Any,Vararg{Any}},
    right::Tuple{Any,Vararg{Any}},
)
    return _qwen3_vl_ranges_overlap(left[1], right[1]) &&
        _qwen3_vl_indices_overlap(Base.tail(left), Base.tail(right))
end

function _qwen3_vl_storage_overlaps(left::SubArray, right::SubArray)
    _qwen3_vl_storage_parent(left) === _qwen3_vl_storage_parent(right) ||
        return false
    return _qwen3_vl_indices_overlap(left.indices, right.indices)
end

function _qwen3_vl_storage_overlaps(left::AbstractArray, right::AbstractArray)
    parent_left = _qwen3_vl_storage_parent(left)
    parent_right = _qwen3_vl_storage_parent(right)
    parent_left === parent_right || return false
    return true
end

function _record_qwen3_vl_cache_storage!(
    storages::Vector{T},
    storage::T,
    message::AbstractString,
) where {T<:AbstractArray}
    for existing in storages
        _qwen3_vl_storage_overlaps(storage, existing) &&
            throw(ArgumentError(message))
    end
    push!(storages, storage)
    return nothing
end

"""Validate Qwen3-VL dynamic layer storage without reading or copying tensors."""
function _validate_qwen3_vl_dynamic_cache_layers(
    layers,
    position::Int,
    batch_size::Int,
)
    layers isa Tuple || throw(ArgumentError(
        "Qwen3-VL dynamic cache layers must be a tuple",
    ))
    isempty(layers) && return nothing

    expected_shape = nothing
    expected_dtype = nothing
    expected_device = nothing
    seen_storage = nothing
    for (layer_index, layer) in enumerate(layers)
        layer isa LayerKVCache || throw(ArgumentError(
            "Qwen3-VL dynamic cache layer $layer_index must be LayerKVCache storage",
        ))
        keys = layer.keys
        values = layer.values
        (keys === nothing) == (values === nothing) || throw(ArgumentError(
            "Qwen3-VL dynamic cache layer $layer_index must contain both keys and values",
        ))
        if position == 0
            keys === nothing || throw(DimensionMismatch(
                "empty Qwen3-VL dynamic cache contains layer storage",
            ))
            continue
        end
        keys === nothing && throw(DimensionMismatch(
            "populated Qwen3-VL dynamic cache is missing layer $layer_index storage",
        ))
        keys isa AbstractArray && values isa AbstractArray || throw(ArgumentError(
            "Qwen3-VL dynamic cache keys and values must be arrays",
        ))
        ndims(keys) == 4 && ndims(values) == 4 || throw(DimensionMismatch(
            "Qwen3-VL dynamic cache keys and values must be four-dimensional",
        ))
        size(keys) == size(values) || throw(DimensionMismatch(
            "Qwen3-VL dynamic cache key and value shapes must match",
        ))
        all(dimension -> dimension > 0, size(keys)) || throw(ArgumentError(
            "Qwen3-VL dynamic cache dimensions must be positive",
        ))
        size(keys, 3) == position || throw(DimensionMismatch(
            "Qwen3-VL dynamic cache token dimension must match position",
        ))
        size(keys, 4) == batch_size || throw(DimensionMismatch(
            "Qwen3-VL dynamic cache batch dimension must match batch_size",
        ))
        dtype = eltype(keys)
        dtype in (Float32, BFloat16) || throw(ArgumentError(
            "Qwen3-VL dynamic cache storage must contain Float32 or BFloat16 values",
        ))
        eltype(values) == dtype || throw(ArgumentError(
            "Qwen3-VL dynamic cache key and value dtypes must match",
        ))
        device = get_device(keys)
        get_device(values) == device || throw(ArgumentError(
            "Qwen3-VL dynamic cache keys and values must use the same device",
        ))
        if expected_shape === nothing
            expected_shape = size(keys)
            expected_dtype = dtype
            expected_device = device
        else
            size(keys) == expected_shape || throw(DimensionMismatch(
                "Qwen3-VL dynamic cache layer shapes must match",
            ))
            dtype == expected_dtype || throw(ArgumentError(
                "Qwen3-VL dynamic cache layer dtypes must match",
            ))
            device == expected_device || throw(ArgumentError(
                "Qwen3-VL dynamic cache layer devices must match",
            ))
        end
        if seen_storage === nothing
            seen_storage = Vector{typeof(keys)}()
        end
        for storage in (keys, values)
            _record_qwen3_vl_cache_storage!(
                seen_storage,
                storage,
                "Qwen3-VL dynamic cache layers must use non-overlapping storage",
            )
        end
    end
    return nothing
end

"""
    Qwen3VLKVCache(layers, position, rope_delta, batch_size)

Per-request dynamic KV cache for the Qwen3-VL text decoder.  Every layer uses
`LayerKVCache`, whose key and value tensors have layout
`(head_dim, num_kv_heads, cached_tokens, batch)`.  Keys are stored after
QK-Norm and multimodal RoPE; values are stored directly after projection.  GQA
expansion is deliberately confined to attention and is never persisted.

`position` is the physical number of tokens already processed.  `rope_delta`
is the request-local offset produced by the multimodal prompt, so the next
decode token uses the zero-based T/H/W coordinate
`position + rope_delta`.
"""
struct Qwen3VLKVCache{C}
    layers::C
    position::Int
    rope_delta::Int
    batch_size::Int

    function Qwen3VLKVCache(layers, position, rope_delta, batch_size)
        resolved_position = _strict_host_int(
            position,
            "Qwen3-VL KV cache position",
        )
        resolved_rope_delta = _strict_host_int(
            rope_delta,
            "Qwen3-VL KV cache rope_delta",
        )
        resolved_batch = _strict_host_int(
            batch_size,
            "Qwen3-VL KV cache batch_size",
        )
        resolved_position >= 0 || throw(ArgumentError(
            "Qwen3-VL KV cache position must be non-negative",
        ))
        resolved_batch == 1 || throw(ArgumentError(
            "Qwen3-VL dynamic generation currently supports batch size one",
        ))
        resolved_position == 0 && resolved_rope_delta != 0 &&
            throw(ArgumentError(
                "an empty Qwen3-VL KV cache must have rope_delta == 0",
            ))
        _validate_qwen3_vl_dynamic_cache_layers(
            layers,
            resolved_position,
            resolved_batch,
        )
        return new{typeof(layers)}(
            layers,
            resolved_position,
            resolved_rope_delta,
            resolved_batch,
        )
    end
end

Base.length(cache::Qwen3VLKVCache) = cache.position
Base.isempty(cache::Qwen3VLKVCache) = cache.position == 0

function _qwen3_vl_cache_spec(parameters)
    hasproperty(parameters, :spec) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain a decoder spec",
    ))
    hasproperty(parameters, :blocks) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain decoder blocks",
    ))
    hasproperty(parameters, :embedding) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain token embeddings",
    ))
    hasproperty(parameters, :final_norm) || throw(ArgumentError(
        "Qwen3-VL text parameters must contain a final norm",
    ))
    return parameters.spec
end

struct _Qwen3VLGenerationPromptValidated end
const _QWEN3_VL_GENERATION_PROMPT_VALIDATED =
    _Qwen3VLGenerationPromptValidated()

"""Sealed proof that one generation prompt passed every non-cache check."""
struct _Qwen3VLGenerationPromptContract{S}
    sources::S
    sequence_length::Int
    batch_size::Int
    rope_delta::Int

    function _Qwen3VLGenerationPromptContract(
        ::_Qwen3VLGenerationPromptValidated,
        sources::S,
        sequence_length::Int,
        batch_size::Int,
        rope_delta::Int,
    ) where {S}
        return new{S}(sources, sequence_length, batch_size, rope_delta)
    end
end

function _qwen3_vl_seal_generation_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    vision_features,
    vision_sources,
    rope_delta::Int,
    checkpoint_token_ids,
)
    checkpoint_present = checkpoint_token_ids !== nothing
    blocks = parameters.blocks
    sources = (;
        parameters,
        spec=parameters.spec,
        embedding=parameters.embedding,
        blocks,
        block_sources=Tuple(blocks),
        final_norm=parameters.final_norm,
        checkpoint_present,
        checkpoint=checkpoint_present ?
            checkpoint_token_ids.checkpoint : nothing,
        checkpoint_image_token_id=checkpoint_present ?
            checkpoint_token_ids.image_token_id : nothing,
        checkpoint_video_token_id=checkpoint_present ?
            checkpoint_token_ids.video_token_id : nothing,
        tokens,
        rope_layout,
        position_ids=rope_layout.position_ids,
        rope_deltas=rope_layout.rope_deltas,
        visual_mask=rope_layout.visual_mask,
        attention_mask=rope_layout.attention_mask,
        vision_features,
        vision_sources...,
    )
    sequence_length, batch_size = size(tokens)
    return _Qwen3VLGenerationPromptContract(
        _QWEN3_VL_GENERATION_PROMPT_VALIDATED,
        sources,
        sequence_length,
        batch_size,
        rope_delta,
    )
end

function _qwen3_vl_validate_generation_prompt_sources(
    contract::_Qwen3VLGenerationPromptContract,
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    vision_features,
)
    sources = contract.sources
    unchanged = sources.parameters === parameters &&
        sources.spec === parameters.spec &&
        sources.embedding === parameters.embedding &&
        sources.blocks === parameters.blocks &&
        sources.final_norm === parameters.final_norm &&
        sources.tokens === tokens &&
        sources.rope_layout === rope_layout &&
        sources.position_ids === rope_layout.position_ids &&
        sources.rope_deltas === rope_layout.rope_deltas &&
        sources.visual_mask === rope_layout.visual_mask &&
        sources.attention_mask === rope_layout.attention_mask &&
        sources.vision_features === vision_features &&
        sources.checkpoint_present == hasproperty(parameters, :checkpoint)
    if unchanged
        blocks = parameters.blocks
        unchanged = length(blocks) == length(sources.block_sources) &&
            all(
                index -> sources.block_sources[index] === blocks[index],
                eachindex(sources.block_sources),
            )
    end
    if unchanged && sources.checkpoint_present
        checkpoint = parameters.checkpoint
        unchanged = sources.checkpoint === checkpoint &&
            hasproperty(checkpoint, :image_token_id) &&
            hasproperty(checkpoint, :video_token_id)
        if unchanged
            unchanged = sources.checkpoint_image_token_id == _strict_host_int(
                checkpoint.image_token_id,
                "Qwen3-VL checkpoint image_token_id",
            ) && sources.checkpoint_video_token_id == _strict_host_int(
                checkpoint.video_token_id,
                "Qwen3-VL checkpoint video_token_id",
            )
        end
    end
    if unchanged && vision_features !== nothing
        deepstack = vision_features.deepstack
        unchanged = sources.visual_embeddings ===
            vision_features.visual_embeddings &&
            sources.deepstack === deepstack &&
            length(deepstack) == length(sources.deepstack_features) &&
            all(
                index -> sources.deepstack_features[index] === deepstack[index],
                eachindex(sources.deepstack_features),
            )
    end
    unchanged || throw(ArgumentError(
        "Qwen3-VL prevalidated generation prompt sources changed",
    ))
    size(tokens) == (contract.sequence_length, contract.batch_size) || throw(
        DimensionMismatch(
            "Qwen3-VL prevalidated generation token shape changed",
        ),
    )
    return sources.spec
end

"""
    init_qwen3_vl_kv_cache(text_parameters; batch_size=1)

Create an empty dynamic cache for one Qwen3-VL generation request.  Layer
storage is adopted lazily from the first cached prefill, preserving the text
parameters' dtype and CPU/CUDA residency.  Chapter 45 supports batch size one.
"""
function init_qwen3_vl_kv_cache(text_parameters; batch_size::Integer=1)
    batch = _strict_host_int(batch_size, "batch_size")
    batch == 1 || throw(ArgumentError(
        "Qwen3-VL dynamic generation currently supports batch size one",
    ))
    spec = _qwen3_vl_cache_spec(text_parameters)
    length(text_parameters.blocks) == spec.num_hidden_layers ||
        throw(DimensionMismatch(
            "Qwen3-VL decoder parameter layer count is invalid",
        ))
    layers = ntuple(_ -> LayerKVCache(), spec.num_hidden_layers)
    return Qwen3VLKVCache(layers, 0, 0, batch)
end

function _validate_qwen3_vl_kv_cache(parameters, cache::Qwen3VLKVCache)
    spec = _qwen3_vl_cache_spec(parameters)
    cache.batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL dynamic generation currently supports batch size one",
    ))
    length(cache.layers) == spec.num_hidden_layers || throw(DimensionMismatch(
        "Qwen3-VL KV cache layer count does not match the decoder",
    ))
    0 <= cache.position <= spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL KV cache position is outside the decoder context",
    ))
    cache.position == 0 && cache.rope_delta != 0 && throw(ArgumentError(
        "an empty Qwen3-VL KV cache must have rope_delta == 0",
    ))
    _validate_qwen3_vl_dynamic_cache_layers(
        cache.layers,
        cache.position,
        cache.batch_size,
    )

    expected_shape = (
        spec.head_dim,
        spec.num_key_value_heads,
        cache.position,
        cache.batch_size,
    )
    parameter_dtype = eltype(parameters.embedding)
    parameter_dtype in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL dynamic cache supports Float32 or BFloat16 parameters",
    ))
    parameter_device = get_device(parameters.embedding)
    for (layer, layer_cache) in enumerate(cache.layers)
        (layer_cache.keys === nothing) == (layer_cache.values === nothing) ||
            throw(ArgumentError(
                "Qwen3-VL layer $layer cache must contain both keys and values",
            ))
        if cache.position == 0
            layer_cache.keys === nothing || throw(DimensionMismatch(
                "empty Qwen3-VL cache contains layer storage",
            ))
            continue
        end
        layer_cache.keys === nothing && throw(DimensionMismatch(
            "populated Qwen3-VL cache is missing layer $layer storage",
        ))
        size(layer_cache.keys) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL layer $layer key cache has the wrong shape",
        ))
        size(layer_cache.values) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL layer $layer value cache has the wrong shape",
        ))
        eltype(layer_cache.keys) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL cached keys do not match the text parameter dtype",
        ))
        eltype(layer_cache.values) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL cached values do not match the text parameter dtype",
        ))
        get_device(layer_cache.keys) == parameter_device || throw(ArgumentError(
            "Qwen3-VL cached keys do not match the text parameter device",
        ))
        get_device(layer_cache.values) == parameter_device || throw(ArgumentError(
            "Qwen3-VL cached values do not match the text parameter device",
        ))
    end
    return nothing
end

"""
    Qwen3VLStaticLayerKVCache(keys, values)

Fixed-capacity key/value storage for one Qwen3-VL decoder layer. Both tensors
use `(head_dim, num_kv_heads, capacity, batch)` layout. Only the prefix ending
at `Qwen3VLStaticKVCache.position` is logically valid.
"""
struct Qwen3VLStaticLayerKVCache{K,V}
    keys::K
    values::V

    function Qwen3VLStaticLayerKVCache(
        keys::AbstractArray,
        values::AbstractArray,
    )
        ndims(keys) == 4 && ndims(values) == 4 || throw(DimensionMismatch(
            "Qwen3-VL static layer keys and values must be four-dimensional",
        ))
        all(
            dimension -> axes(keys, dimension) ==
                Base.OneTo(size(keys, dimension)),
            1:4,
        ) || throw(ArgumentError(
            "Qwen3-VL static layer keys must use one-based axes",
        ))
        all(
            dimension -> axes(values, dimension) ==
                Base.OneTo(size(values, dimension)),
            1:4,
        ) || throw(ArgumentError(
            "Qwen3-VL static layer values must use one-based axes",
        ))
        size(keys) == size(values) || throw(DimensionMismatch(
            "Qwen3-VL static layer key and value shapes must match",
        ))
        all(dimension -> dimension > 0, size(keys)) || throw(ArgumentError(
            "Qwen3-VL static layer dimensions must be positive",
        ))
        key_type = eltype(keys)
        key_type in (Float32, BFloat16) || throw(ArgumentError(
            "Qwen3-VL static layer storage must contain Float32 or BFloat16 values",
        ))
        eltype(values) == key_type || throw(ArgumentError(
            "Qwen3-VL static layer key and value dtypes must match",
        ))
        get_device(keys) == get_device(values) || throw(ArgumentError(
            "Qwen3-VL static layer keys and values must use the same device",
        ))
        Base.mightalias(keys, values) && throw(ArgumentError(
            "Qwen3-VL static layer keys and values must use non-overlapping storage",
        ))
        return new{typeof(keys),typeof(values)}(keys, values)
    end
end

function _validate_qwen3_vl_static_cache_layers(
    layers,
    batch_size::Int,
    capacity::Int,
    reused=nothing,
)
    layers isa Tuple || throw(ArgumentError(
        "Qwen3-VL static cache layers must be a tuple",
    ))
    isempty(layers) && throw(ArgumentError(
        "Qwen3-VL static cache layers must not be empty",
    ))
    first_layer = first(layers)
    first_layer isa Qwen3VLStaticLayerKVCache || throw(ArgumentError(
        "Qwen3-VL static cache layers must contain static layer storage",
    ))
    expected_shape = (
        size(first_layer.keys, 1),
        size(first_layer.keys, 2),
        capacity,
        batch_size,
    )
    expected_dtype = eltype(first_layer.keys)
    expected_device = get_device(first_layer.keys)
    seen_storage = if reused isa Vector{typeof(first_layer.keys)}
        empty!(reused)
        reused
    else
        Vector{typeof(first_layer.keys)}()
    end
    for (layer_index, layer) in enumerate(layers)
        layer isa Qwen3VLStaticLayerKVCache || throw(ArgumentError(
            "Qwen3-VL static cache layer $layer_index is not static layer storage",
        ))
        size(layer.keys) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL static cache layer $layer_index has the wrong shape",
        ))
        size(layer.values) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL static cache layer $layer_index has the wrong shape",
        ))
        eltype(layer.keys) == expected_dtype &&
            eltype(layer.values) == expected_dtype || throw(ArgumentError(
            "Qwen3-VL static cache layer dtypes must match",
        ))
        get_device(layer.keys) == expected_device &&
            get_device(layer.values) == expected_device || throw(ArgumentError(
            "Qwen3-VL static cache layer devices must match",
        ))
        for storage in (layer.keys, layer.values)
            _record_qwen3_vl_cache_storage!(
                seen_storage,
                storage,
                "Qwen3-VL static cache layers must use non-overlapping storage",
            )
        end
    end
    return nothing
end

"""
    Qwen3VLStaticKVCache(layers, position, rope_delta, batch_size, capacity)

Mutable request state backed by fixed-capacity per-layer K/V tensors.
`position` remains the physical number of processed tokens and `rope_delta`
remains the request-local multimodal RoPE offset. Prefill, decode, and reset
mutate this request object in place without replacing its layer storage.
"""
mutable struct Qwen3VLStaticKVCache{C}
    layers::C
    position::Int
    rope_delta::Int
    batch_size::Int
    capacity::Int
    attention_scratch::Any
    mlp_activation_scratch::Any
    qk_rope_scratch::Any
    qk_norm_scratch::Any
    hidden_norm_scratch::Any
    gate_up_scratch::Any
    hidden_residual_scratch::Any
    tied_logits_scratch::Any
    qkv_scratch::Any
    token_embedding_scratch::Any
    mrope_scratch::Any
    seen_storage::Any
    decode_pack::Any
    decode_workspace::Any

    function Qwen3VLStaticKVCache(
        layers,
        position,
        rope_delta,
        batch_size,
        capacity,
    )
        resolved_position = _strict_host_int(
            position,
            "Qwen3-VL static cache position",
        )
        resolved_rope_delta = _strict_host_int(
            rope_delta,
            "Qwen3-VL static cache rope_delta",
        )
        resolved_batch = _strict_host_int(
            batch_size,
            "Qwen3-VL static cache batch_size",
        )
        resolved_capacity = _strict_host_int(
            capacity,
            "Qwen3-VL static cache capacity",
        )
        resolved_capacity > 0 || throw(ArgumentError(
            "Qwen3-VL static cache capacity must be positive",
        ))
        resolved_position >= 0 || throw(ArgumentError(
            "Qwen3-VL static cache position must be non-negative",
        ))
        resolved_position <= resolved_capacity || throw(ArgumentError(
            "Qwen3-VL static cache position must not exceed capacity",
        ))
        resolved_batch == 1 || throw(ArgumentError(
            "Qwen3-VL static generation currently supports batch size one",
        ))
        resolved_position == 0 && resolved_rope_delta != 0 &&
            throw(ArgumentError(
                "an empty Qwen3-VL static cache must have rope_delta == 0",
            ))
        _validate_qwen3_vl_static_cache_layers(
            layers,
            resolved_batch,
            resolved_capacity,
        )
        return new{typeof(layers)}(
            layers,
            resolved_position,
            resolved_rope_delta,
            resolved_batch,
            resolved_capacity,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
            nothing,
        )
    end
end

Base.length(cache::Qwen3VLStaticKVCache) = cache.position
Base.isempty(cache::Qwen3VLStaticKVCache) = cache.position == 0

function _qwen3_vl_seen_storage!(
    cache::Qwen3VLStaticKVCache,
    sample::T,
    count::Int,
) where {T<:AbstractArray}
    seen = cache.seen_storage
    if seen isa Vector{T}
        empty!(seen)
        return seen
    end
    seen = Vector{T}()
    sizehint!(seen, count)
    cache.seen_storage = seen
    return seen
end

function _qwen3_vl_static_cache_dimensions(spec)
    num_layers = _strict_host_int(
        spec.num_hidden_layers,
        "Qwen3-VL static cache num_hidden_layers",
    )
    head_dim = _strict_host_int(
        spec.head_dim,
        "Qwen3-VL static cache head_dim",
    )
    num_kv_heads = _strict_host_int(
        spec.num_key_value_heads,
        "Qwen3-VL static cache num_key_value_heads",
    )
    max_positions = _strict_host_int(
        spec.max_position_embeddings,
        "Qwen3-VL static cache max_position_embeddings",
    )
    num_layers > 0 || throw(ArgumentError(
        "Qwen3-VL static cache num_hidden_layers must be positive",
    ))
    head_dim > 0 || throw(ArgumentError(
        "Qwen3-VL static cache head_dim must be positive",
    ))
    num_kv_heads > 0 || throw(ArgumentError(
        "Qwen3-VL static cache num_key_value_heads must be positive",
    ))
    max_positions > 0 || throw(ArgumentError(
        "Qwen3-VL static cache max_position_embeddings must be positive",
    ))
    return (; num_layers, head_dim, num_kv_heads, max_positions)
end

function _qwen3_vl_static_cache_layout(
    dimensions,
    capacity::Int,
    batch_size::Int,
    dtype::Type,
)
    per_tensor_elements = BigInt(dimensions.head_dim) *
        dimensions.num_kv_heads * capacity * batch_size
    per_tensor_elements <= typemax(Int) || throw(ArgumentError(
        "Qwen3-VL static cache tensor element count exceeds the host integer range",
    ))

    total_elements = 2 * BigInt(dimensions.num_layers) * per_tensor_elements
    total_elements <= typemax(Int) || throw(ArgumentError(
        "Qwen3-VL static cache total element count exceeds the host integer range",
    ))

    total_bytes = total_elements * sizeof(dtype)
    total_bytes <= typemax(Int) || throw(ArgumentError(
        "Qwen3-VL static cache byte count exceeds the host integer range",
    ))
    return (;
        shape=(
            dimensions.head_dim,
            dimensions.num_kv_heads,
            capacity,
            batch_size,
        ),
        per_tensor_elements=Int(per_tensor_elements),
        total_elements=Int(total_elements),
        total_bytes=Int(total_bytes),
    )
end

"""
    init_qwen3_vl_static_kv_cache(text_parameters; capacity, batch_size=1)

Allocate bounded Qwen3-VL K/V storage on the same device and with the same
element type as the text embedding. `capacity` is required deliberately: the
official context limit is large enough that silently allocating the maximum
would consume excessive host or accelerator memory.
"""
function init_qwen3_vl_static_kv_cache(
    text_parameters;
    capacity::Integer,
    batch_size::Integer=1,
)
    requested_capacity = _strict_host_int(capacity, "capacity")
    batch = _strict_host_int(batch_size, "batch_size")
    batch == 1 || throw(ArgumentError(
        "Qwen3-VL static generation currently supports batch size one",
    ))
    spec = _qwen3_vl_cache_spec(text_parameters)
    dimensions = _qwen3_vl_static_cache_dimensions(spec)
    length(text_parameters.blocks) == dimensions.num_layers ||
        throw(DimensionMismatch(
            "Qwen3-VL decoder parameter layer count is invalid",
        ))
    0 < requested_capacity <= dimensions.max_positions ||
        throw(ArgumentError(
            "Qwen3-VL static cache capacity must be in 1:max_position_embeddings",
        ))
    dtype = eltype(text_parameters.embedding)
    dtype in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL static cache supports Float32 or BFloat16 parameters",
    ))
    layout = _qwen3_vl_static_cache_layout(
        dimensions,
        requested_capacity,
        batch,
        dtype,
    )

    layers = ntuple(dimensions.num_layers) do _
        keys = similar(text_parameters.embedding, dtype, layout.shape)
        values = similar(text_parameters.embedding, dtype, layout.shape)
        fill!(keys, zero(dtype))
        fill!(values, zero(dtype))
        Qwen3VLStaticLayerKVCache(keys, values)
    end
    cache = Qwen3VLStaticKVCache(layers, 0, 0, batch, requested_capacity)
    # The blocks tuple is copied once here. The first decode used to copy it
    # again while installing the steady-step pack.
    cache.decode_pack = _Qwen3VLDecodePack(
        text_parameters.blocks,
        text_parameters.embedding,
        text_parameters.final_norm,
        text_parameters.spec,
    )
    _validate_qwen3_vl_static_kv_cache(text_parameters, cache)
    return cache
end

function _validate_qwen3_vl_static_kv_cache(
    parameters,
    cache::Qwen3VLStaticKVCache,
)
    spec = _qwen3_vl_cache_spec(parameters)
    dimensions = _qwen3_vl_static_cache_dimensions(spec)
    cache.batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL static generation currently supports batch size one",
    ))
    length(cache.layers) == dimensions.num_layers || throw(DimensionMismatch(
        "Qwen3-VL static KV cache layer count does not match the decoder",
    ))
    0 < cache.capacity <= dimensions.max_positions || throw(ArgumentError(
        "Qwen3-VL static cache capacity is outside the decoder context",
    ))
    0 <= cache.position <= cache.capacity || throw(ArgumentError(
        "Qwen3-VL static cache position is outside its capacity",
    ))
    cache.position == 0 && cache.rope_delta != 0 && throw(ArgumentError(
        "an empty Qwen3-VL static cache must have rope_delta == 0",
    ))
    _validate_qwen3_vl_static_cache_layers(
        cache.layers,
        cache.batch_size,
        cache.capacity,
        _qwen3_vl_seen_storage!(
            cache,
            first(cache.layers).keys,
            2 * length(cache.layers),
        ),
    )

    expected_shape = (
        dimensions.head_dim,
        dimensions.num_kv_heads,
        cache.capacity,
        cache.batch_size,
    )
    parameter_dtype = eltype(parameters.embedding)
    parameter_dtype in (Float32, BFloat16) || throw(ArgumentError(
        "Qwen3-VL static cache supports Float32 or BFloat16 parameters",
    ))
    parameter_device = get_device(parameters.embedding)
    for (layer, layer_cache) in enumerate(cache.layers)
        size(layer_cache.keys) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL static layer $layer key cache has the wrong shape",
        ))
        size(layer_cache.values) == expected_shape || throw(DimensionMismatch(
            "Qwen3-VL static layer $layer value cache has the wrong shape",
        ))
        eltype(layer_cache.keys) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL static cached keys do not match the text parameter dtype",
        ))
        eltype(layer_cache.values) == parameter_dtype || throw(ArgumentError(
            "Qwen3-VL static cached values do not match the text parameter dtype",
        ))
        get_device(layer_cache.keys) == parameter_device || throw(ArgumentError(
            "Qwen3-VL static cached keys do not match the text parameter device",
        ))
        get_device(layer_cache.values) == parameter_device || throw(ArgumentError(
            "Qwen3-VL static cached values do not match the text parameter device",
        ))
    end
    return nothing
end

"""
    reset_qwen3_vl_static_kv_cache!(cache; clear=false)

Reset a bounded cache for another request while retaining all allocations.
With `clear=false`, stale tail bytes remain physically present but are never
visible to attention because `position` returns to zero. `clear=true` also
zeroes every K/V buffer.
"""
function reset_qwen3_vl_static_kv_cache!(
    cache::Qwen3VLStaticKVCache;
    clear::Bool=false,
)
    _validate_qwen3_vl_static_cache_layers(
        cache.layers,
        cache.batch_size,
        cache.capacity,
    )
    if clear
        for layer_cache in cache.layers
            fill!(layer_cache.keys, zero(eltype(layer_cache.keys)))
            fill!(layer_cache.values, zero(eltype(layer_cache.values)))
        end
    end
    cache.position = 0
    cache.rope_delta = 0
    return cache
end

@inline function _qwen3_vl_profile_stage(
    thunk,
    runner,
    stage::Symbol,
    layer_index::Int,
)
    runner === nothing && return thunk()
    return runner(stage, layer_index, thunk)
end

macro _qwen3_vl_stage(runner, stage, layer, body)
    quote
        let _runner = $(esc(runner))
            if _runner === nothing
                $(esc(body))
            else
                _qwen3_vl_profile_stage(
                    () -> $(esc(body)),
                    _runner,
                    $(esc(stage)),
                    $(esc(layer)),
                )
            end
        end
    end
end

function _qwen3_vl_cache_block_core(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    mask,
    store,
    profile_runner=nothing,
    layer_index::Int=0;
    attention_scratch=nothing,
    mlp_scratch=nothing,
    rope_scratch=nothing,
    norm_scratch=nothing,
    hidden_norm_scratch=nothing,
    gate_up_scratch=nothing,
    hidden_residual_scratch=nothing,
    qkv_scratch=nothing,
)
    head_dim = spec.head_dim
    sequence_length, batch_size = size(x, 2), size(x, 3)
    normed = @_qwen3_vl_stage profile_runner :pre_attention_norm layer_index begin
        if hidden_norm_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            size(x) == size(hidden_norm_scratch.output)
            _qwen3_vl_decode_hidden_norm!(
                hidden_norm_scratch,
                x,
                block.norm1,
                spec.rms_norm_eps,
            )
        else
            _qwen3_vl_text_rmsnorm(x, block.norm1, spec.rms_norm_eps)
        end
    end
    queries, keys, values = @_qwen3_vl_stage profile_runner :qkv_projection layer_index begin
        queries, keys, values = if qkv_scratch !== nothing &&
            eltype(normed) === BFloat16 &&
            size(normed) == (qkv_scratch.hidden_size, 1, 1)
            _qwen3_vl_decode_qkv!(
                qkv_scratch,
                block.q_weight,
                block.k_weight,
                block.v_weight,
                normed,
            )
        else
            queries = reshape(
                _bf16a_linear(block.q_weight, normed),
                head_dim, spec.num_attention_heads, sequence_length, batch_size,
            )
            keys = reshape(
                _bf16a_linear(block.k_weight, normed),
                head_dim, spec.num_key_value_heads, sequence_length, batch_size,
            )
            values = reshape(
                _bf16a_linear(block.v_weight, normed),
                head_dim, spec.num_key_value_heads, sequence_length, batch_size,
            )
            (queries, keys, values)
        end
    end
    queries, keys = @_qwen3_vl_stage profile_runner :qk_norm layer_index begin
        if norm_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            size(queries, 3) == 1 &&
            size(queries) == size(norm_scratch.query_out) &&
            size(keys) == size(norm_scratch.key_out)
            _qwen3_vl_decode_qk_norm!(
                norm_scratch,
                queries,
                keys,
                block.q_norm,
                block.k_norm,
                spec.rms_norm_eps,
            )
        else
            queries = _qwen3_vl_text_rmsnorm(
                queries,
                block.q_norm,
                spec.rms_norm_eps,
            )
            keys = _qwen3_vl_text_rmsnorm(
                keys,
                block.k_norm,
                spec.rms_norm_eps,
            )
            (queries, keys)
        end
    end
    queries, keys = @_qwen3_vl_stage profile_runner :qk_rope layer_index begin
        if rope_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            size(queries, 3) == 1 &&
            size(queries) == size(rope_scratch.query_out) &&
            size(keys) == size(rope_scratch.key_out)
            _qwen3_vl_decode_qk_rope!(
                rope_scratch,
                queries,
                keys,
                cos_values,
                sin_values,
            )
        else
            (
                _qwen3_vl_text_apply_rope(queries, cos_values, sin_values),
                _qwen3_vl_text_apply_rope(keys, cos_values, sin_values),
            )
        end
    end

    attention_keys, attention_values, updated_layer_cache =
        @_qwen3_vl_stage profile_runner :kv_write layer_index begin
            store(keys, values)
        end
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    context = @_qwen3_vl_stage profile_runner :attention layer_index begin
        if attention_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            mask === nothing &&
            size(queries, 3) == 1
            _qwen3_vl_decode_attention!(
                attention_scratch,
                queries,
                attention_keys,
                attention_values;
                scaling,
            )
        elseif eltype(x) === BFloat16
            _bf16a_attention(
                queries,
                attention_keys,
                attention_values;
                scaling,
                mask,
            )
        else
            _qwen3_vl_text_f32_attention(
                queries,
                attention_keys,
                attention_values,
                mask;
                scaling,
            )
        end
    end
    x = @_qwen3_vl_stage profile_runner :attention_output_projection_residual layer_index begin
        attention = if hidden_residual_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            size(x) == size(hidden_residual_scratch.output)
            projected = reshape(
                context,
                head_dim * spec.num_attention_heads,
                sequence_length,
                batch_size,
            )
            if size(projected) == (
                hidden_residual_scratch.hidden_size,
                1,
                1,
            )
                _qwen3_vl_decode_hidden_residual!(
                    hidden_residual_scratch,
                    block.o_weight,
                    projected,
                    x,
                )
            else
                _qwen3_vl_text_residual(x, _bf16a_linear(block.o_weight, projected))
            end
        else
            _qwen3_vl_text_residual(
                x,
                _bf16a_linear(
                    block.o_weight,
                    reshape(
                        context,
                        head_dim * spec.num_attention_heads,
                        sequence_length,
                        batch_size,
                    ),
                ),
            )
        end
    end
    normed = @_qwen3_vl_stage profile_runner :post_attention_norm layer_index begin
        if hidden_norm_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            size(x) == size(hidden_norm_scratch.output)
            _qwen3_vl_decode_hidden_norm!(
                hidden_norm_scratch,
                x,
                block.norm2,
                spec.rms_norm_eps,
            )
        else
            _qwen3_vl_text_rmsnorm(x, block.norm2, spec.rms_norm_eps)
        end
    end
    gate, up = @_qwen3_vl_stage profile_runner :mlp_gate_up_projection layer_index begin
        if gate_up_scratch !== nothing &&
            eltype(normed) === BFloat16 &&
            size(normed) == (gate_up_scratch.hidden_size, 1, 1)
            _qwen3_vl_decode_gate_up!(
                gate_up_scratch,
                block.gate_weight,
                block.up_weight,
                normed,
            )
        else
            (
                _bf16a_linear(block.gate_weight, normed),
                _bf16a_linear(block.up_weight, normed),
            )
        end
    end
    hidden = @_qwen3_vl_stage profile_runner :mlp_activation layer_index begin
        if mlp_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            size(gate) == size(mlp_scratch.product)
            _qwen3_vl_mlp_activation!(mlp_scratch, gate, up)
        elseif eltype(x) === BFloat16
            gate_f32 = _bf16a_f32(gate)
            activated = BFloat16.(gate_f32 ./ (1.0f0 .+ exp.(.-gate_f32)))
            BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
        else
            activated = gate ./ (1.0f0 .+ exp.(.-gate))
            activated .* up
        end
    end
    x = @_qwen3_vl_stage profile_runner :mlp_down_projection_residual layer_index begin
        if hidden_residual_scratch !== nothing &&
            eltype(hidden) === BFloat16 &&
            eltype(x) === BFloat16 &&
            size(x) == size(hidden_residual_scratch.output) &&
            size(hidden) == (size(block.down_weight, 2), 1, 1)
            _qwen3_vl_decode_hidden_residual!(
                hidden_residual_scratch,
                block.down_weight,
                hidden,
                x,
            )
        else
            mlp = _bf16a_linear(block.down_weight, hidden)
            _qwen3_vl_text_residual(x, mlp)
        end
    end
    return x, updated_layer_cache
end

function _qwen3_vl_cache_block(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    mask,
    layer_cache::LayerKVCache,
)
    store = function (keys, values)
        updated = _append_kv(layer_cache, keys, values)
        return updated.keys, updated.values, updated
    end
    return _qwen3_vl_cache_block_core(
        spec,
        block,
        x,
        cos_values,
        sin_values,
        mask,
        store,
    )
end

function _qwen3_vl_cached_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    cache::Qwen3VLKVCache,
    logits_to_keep::Int,
)
    spec = _qwen3_vl_cache_spec(parameters)
    sequence_length, batch_size = size(tokens)
    _validate_qwen3_vl_kv_cache(parameters, cache)
    isempty(cache) || throw(ArgumentError(
        "Qwen3-VL cached prefill requires an empty cache",
    ))
    batch_size == cache.batch_size || throw(DimensionMismatch(
        "Qwen3-VL prompt batch size does not match the cache",
    ))
    0 < sequence_length <= spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL prompt length is outside the decoder context",
    ))
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL input_ids contain an out-of-vocabulary id",
    ))
    size(rope_layout.visual_mask) == size(tokens) || throw(DimensionMismatch(
        "Qwen3-VL visual mask does not match input_ids",
    ))
    _qwen3_vl_prompt_visual_token_contract(
        parameters,
        tokens,
        rope_layout.visual_mask,
        rope_layout.attention_mask,
    )
    rope_deltas = _qwen3_vl_prompt_rope_deltas(
        spec,
        rope_layout,
        sequence_length,
        batch_size,
    )
    all(rope_layout.attention_mask) || throw(ArgumentError(
        "Qwen3-VL cached generation currently requires an all-ones attention mask",
    ))
    rope_delta = only(rope_deltas)
    0 <= logits_to_keep <= sequence_length || throw(ArgumentError(
        "logits_to_keep must be between zero and the prefill length",
    ))
    return spec, rope_delta
end

"""
    hf_qwen3_vl_text_prefill_cached(parameters, input_ids, rope_layout;
                                    vision_features=nothing, cache,
                                    logits_to_keep=1,
                                    capture_input_embeddings=true,
                                    capture_final_hidden=true)

Run one batch-1, all-ones-mask Qwen3-VL prompt and populate a dynamic KV
cache.  Main visual embeddings replace image-token embeddings before layer 0,
and the three DeepStack tensors are added after decoder layers 0, 1, and 2.

Returns `(prefill_result, updated_cache)`.  The result uses the same
`Qwen3VLTextPrefill` container as cache-free prefill; cached prefill does not
capture intermediate layers, so its capture dictionaries are empty.
The two `capture_*` keywords can suppress prompt-sized diagnostic tensors;
when final hidden capture is disabled, RMSNorm is applied only to positions
needed by `logits_to_keep`.
"""
function hf_qwen3_vl_text_prefill_cached(
    parameters,
    input_ids,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    cache::Qwen3VLKVCache,
    logits_to_keep::Integer=1,
    capture_input_embeddings::Bool=true,
    capture_final_hidden::Bool=true,
)
    options = _qwen3_vl_prefill_options(logits_to_keep)
    tokens = _qwen3_vl_token_matrix(input_ids)
    spec, rope_delta = _qwen3_vl_cached_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        cache,
        options.logits_to_keep,
    )

    if vision_features === nothing
        any(rope_layout.visual_mask) && throw(ArgumentError(
            "Qwen3-VL visual placeholders require vision features",
        ))
    else
        length(vision_features.deepstack) == 3 || throw(DimensionMismatch(
            "Qwen3-VL cached prefill requires exactly three DeepStack features",
        ))
        _validate_qwen3_vl_text_feature_geometry(
            parameters,
            vision_features,
            count(rope_layout.visual_mask),
        )
        _validate_qwen3_vl_text_feature_residency(
            parameters,
            vision_features,
        )
    end
    return _qwen3_vl_text_prefill_cached_compute(
        parameters,
        tokens,
        rope_layout,
        vision_features,
        cache,
        spec,
        rope_delta,
        options.logits_to_keep,
        capture_input_embeddings,
        capture_final_hidden,
    )
end

function _qwen3_vl_prevalidated_dynamic_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    vision_features,
    cache::Qwen3VLKVCache,
    contract::_Qwen3VLGenerationPromptContract,
)
    spec = _qwen3_vl_cache_spec(parameters)
    _validate_qwen3_vl_kv_cache(parameters, cache)
    isempty(cache) || throw(ArgumentError(
        "Qwen3-VL cached prefill requires an empty cache",
    ))
    contract.batch_size == cache.batch_size || throw(DimensionMismatch(
        "Qwen3-VL prompt batch size does not match the cache",
    ))
    0 < contract.sequence_length <= spec.max_position_embeddings || throw(
        ArgumentError("Qwen3-VL prompt length is outside the decoder context"),
    )
    sealed_spec = _qwen3_vl_validate_generation_prompt_sources(
        contract,
        parameters,
        tokens,
        rope_layout,
        vision_features,
    )
    spec === sealed_spec || throw(ArgumentError(
        "Qwen3-VL prevalidated generation decoder spec changed",
    ))
    return spec, contract.rope_delta
end

function _qwen3_vl_prevalidated_vision_features(
    contract::_Qwen3VLGenerationPromptContract,
)
    sources = contract.sources
    sources.vision_features === nothing && return nothing
    return (;
        visual_embeddings=sources.visual_embeddings,
        deepstack=sources.deepstack_features,
    )
end

function _qwen3_vl_prevalidated_text_parameters(
    contract::_Qwen3VLGenerationPromptContract,
)
    sources = contract.sources
    return (;
        spec=sources.spec,
        embedding=sources.embedding,
        blocks=sources.block_sources,
        final_norm=sources.final_norm,
    )
end

function _qwen3_vl_text_prefill_cached_prevalidated(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    cache::Qwen3VLKVCache,
    contract::_Qwen3VLGenerationPromptContract,
    capture_input_embeddings::Bool=true,
    capture_final_hidden::Bool=true,
)
    spec, rope_delta = _qwen3_vl_prevalidated_dynamic_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        vision_features,
        cache,
        contract,
    )
    sealed_parameters = _qwen3_vl_prevalidated_text_parameters(contract)
    sealed_vision_features = _qwen3_vl_prevalidated_vision_features(contract)
    return _qwen3_vl_text_prefill_cached_compute(
        sealed_parameters,
        tokens,
        rope_layout,
        sealed_vision_features,
        cache,
        spec,
        rope_delta,
        1,
        capture_input_embeddings,
        capture_final_hidden,
    )
end

function _qwen3_vl_text_prefill_cached_compute(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    vision_features,
    cache::Qwen3VLKVCache,
    spec::Qwen3VLTextSpec,
    rope_delta::Int,
    logits_to_keep::Int,
    capture_input_embeddings::Bool,
    capture_final_hidden::Bool,
)
    sequence_length, batch_size = size(tokens)
    x = reshape(
        gather(parameters.embedding, tokens),
        spec.hidden_size,
        sequence_length,
        batch_size,
    )
    if vision_features !== nothing
        x = _qwen3_vl_replace_visual_embeddings(
            x,
            vision_features.visual_embeddings,
            rope_layout.visual_mask,
        )
    end
    input_embeddings = capture_input_embeddings ? x : nothing
    cos_values, sin_values = _qwen3_vl_text_mrope(
        spec,
        parameters.embedding,
        rope_layout.position_ids,
    )
    mask = _qwen3_vl_causal_mask(
        parameters.embedding,
        Bool.(rope_layout.attention_mask),
    )

    layer_caches = Vector{Any}(undef, spec.num_hidden_layers)
    for julia_layer in 1:spec.num_hidden_layers
        layer = julia_layer - 1
        x, layer_caches[julia_layer] = _qwen3_vl_cache_block(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            mask,
            cache.layers[julia_layer],
        )
        if vision_features !== nothing && layer < 3
            x = _qwen3_vl_add_deepstack(
                x,
                vision_features.deepstack[layer + 1],
                rope_layout.visual_mask,
            )
        end
    end

    final_hidden, projection = _qwen3_vl_prefill_projection(
        parameters,
        x,
        sequence_length,
        logits_to_keep,
        capture_final_hidden,
    )
    logits = _qwen3_vl_project_tied(parameters.embedding, projection)
    updated_cache = Qwen3VLKVCache(
        Tuple(layer_caches),
        sequence_length,
        rope_delta,
        batch_size,
    )
    _validate_qwen3_vl_kv_cache(parameters, updated_cache)
    result = Qwen3VLTextPrefill(
        input_embeddings,
        Dict{Int,Any}(),
        Dict{Int,Any}(),
        final_hidden,
        logits,
        rope_layout,
    )
    return result, updated_cache
end

function _qwen3_vl_prefill_projection(
    parameters,
    hidden,
    sequence_length::Int,
    logits_to_keep::Int,
    capture_final_hidden::Bool,
)
    if capture_final_hidden
        final_hidden = _qwen3_vl_text_rmsnorm(
            hidden,
            parameters.final_norm,
            parameters.spec.rms_norm_eps,
        )
        projection = logits_to_keep == 0 ? final_hidden :
            final_hidden[:, (sequence_length - logits_to_keep + 1):end, :]
        return final_hidden, projection
    end

    projection_input = logits_to_keep == 0 ? hidden : view(
        hidden,
        :,
        (sequence_length - logits_to_keep + 1):sequence_length,
        :,
    )
    projection = _qwen3_vl_text_rmsnorm(
        projection_input,
        parameters.final_norm,
        parameters.spec.rms_norm_eps,
    )
    return nothing, projection
end

function _qwen3_vl_decode_token_matrix(token, batch_size::Int)
    tokens = if token isa Integer && !(token isa Bool)
        fill(_strict_host_int(token, "Qwen3-VL decode token"), 1, 1)
    else
        _qwen3_vl_token_matrix(token)
    end
    size(tokens) == (1, batch_size) || throw(DimensionMismatch(
        "Qwen3-VL decode_step expects exactly one token per batch item",
    ))
    return tokens
end

function _qwen3_vl_decode_coordinate(
    position::Int,
    rope_delta::Int,
    limit::Int,
)
    coordinate = try
        Base.Checked.checked_add(position, rope_delta)
    catch error
        error isa OverflowError || rethrow()
        throw(ArgumentError(
            "Qwen3-VL decode mRoPE coordinate exceeds the host integer range",
        ))
    end
    0 <= coordinate < limit || throw(ArgumentError(
        "Qwen3-VL decode mRoPE coordinate is outside the decoder context",
    ))
    return coordinate
end

"""
    hf_qwen3_vl_text_decode_step(parameters, token, cache)

Append one token to a populated Qwen3-VL dynamic cache and return
`(logits, updated_cache)`.  The logits shape is `(vocab_size, 1, 1)`.  Decode
uses the request-local coordinate `cache.position + cache.rope_delta` on all
three mRoPE axes and never reruns vision replacement or DeepStack injection.
"""
function hf_qwen3_vl_text_decode_step(
    parameters,
    token,
    cache::Qwen3VLKVCache,
)
    spec = _qwen3_vl_cache_spec(parameters)
    _validate_qwen3_vl_kv_cache(parameters, cache)
    isempty(cache) && throw(ArgumentError(
        "call hf_qwen3_vl_text_prefill_cached before decode_step",
    ))
    cache.position < spec.max_position_embeddings || throw(ArgumentError(
        "Qwen3-VL KV cache has reached the decoder context limit",
    ))
    coordinate = _qwen3_vl_decode_coordinate(
        cache.position,
        cache.rope_delta,
        spec.max_position_embeddings,
    )
    tokens = _qwen3_vl_decode_token_matrix(token, cache.batch_size)
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL decode token is outside the vocabulary",
    ))

    x = reshape(
        gather(parameters.embedding, tokens),
        spec.hidden_size,
        1,
        cache.batch_size,
    )
    position_ids = fill(coordinate, 3, 1, cache.batch_size)
    cos_values, sin_values = _qwen3_vl_text_mrope(
        spec,
        parameters.embedding,
        position_ids,
    )
    layer_caches = Vector{Any}(undef, spec.num_hidden_layers)
    for julia_layer in 1:spec.num_hidden_layers
        x, layer_caches[julia_layer] = _qwen3_vl_cache_block(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            nothing,
            cache.layers[julia_layer],
        )
    end
    final_hidden = _qwen3_vl_text_rmsnorm(
        x,
        parameters.final_norm,
        spec.rms_norm_eps,
    )
    logits = _qwen3_vl_project_tied(parameters.embedding, final_hidden)
    updated_cache = Qwen3VLKVCache(
        Tuple(layer_caches),
        cache.position + 1,
        cache.rope_delta,
        cache.batch_size,
    )
    _validate_qwen3_vl_kv_cache(parameters, updated_cache)
    return logits, updated_cache
end

function _qwen3_vl_attention_projected(context, hidden_size::Int)
    return reshape(context, hidden_size, size(context, 3), size(context, 4))
end

function _qwen3_vl_static_decode_block!(
    spec,
    block,
    x::AbstractArray{BFloat16,3},
    cos_values,
    sin_values,
    layer_cache::Qwen3VLStaticLayerKVCache,
    write_start::Int,
    attention_scratch::Qwen3VLDecodeAttentionScratch,
    mlp_scratch::Qwen3VLMlpActivationScratch,
    rope_scratch::Qwen3VLQKRopeScratch,
    norm_scratch::Qwen3VLQKNormScratch,
    hidden_norm_scratch::Qwen3VLHiddenNormScratch,
    gate_up_scratch::Qwen3VLGateUpScratch,
    hidden_residual_scratch::Qwen3VLHiddenResidualScratch,
    qkv_scratch::Qwen3VLQKVScratch,
)
    size(x, 2) == 1 || throw(ArgumentError(
        "Qwen3-VL static decode block requires one token",
    ))
    head_dim = spec.head_dim
    write_stop = write_start
    1 <= write_start <= size(layer_cache.keys, 3) || throw(ArgumentError(
        "Qwen3-VL static cache write would exceed its capacity",
    ))
    normed = _qwen3_vl_decode_hidden_norm!(
        hidden_norm_scratch, x, block.norm1, spec.rms_norm_eps,
    )
    queries, keys, values = _qwen3_vl_decode_qkv!(
        qkv_scratch, block.q_weight, block.k_weight, block.v_weight, normed,
    )
    queries, keys = _qwen3_vl_decode_qk_norm!(
        norm_scratch, queries, keys, block.q_norm, block.k_norm, spec.rms_norm_eps,
    )
    queries, keys = _qwen3_vl_decode_qk_rope!(
        rope_scratch, queries, keys, cos_values, sin_values,
    )
    if !_qwen3_vl_device_write_kv!(
        layer_cache.keys, layer_cache.values, keys, values, write_start,
    )
        copyto!(view(layer_cache.keys, :, :, write_start:write_stop, :), keys)
        copyto!(view(layer_cache.values, :, :, write_start:write_stop, :), values)
    end
    attention_keys = view(layer_cache.keys, :, :, 1:write_stop, :)
    attention_values = view(layer_cache.values, :, :, 1:write_stop, :)
    scaling = 1.0f0 / sqrt(Float32(head_dim))
    context = _qwen3_vl_decode_attention!(
        attention_scratch, queries, attention_keys, attention_values; scaling,
    )
    projected = _qwen3_vl_attention_projected(
        context, head_dim * spec.num_attention_heads,
    )
    x = _qwen3_vl_decode_hidden_residual!(
        hidden_residual_scratch, block.o_weight, projected, x,
    )
    normed = _qwen3_vl_decode_hidden_norm!(
        hidden_norm_scratch, x, block.norm2, spec.rms_norm_eps,
    )
    gate, up = _qwen3_vl_decode_gate_up!(
        gate_up_scratch, block.gate_weight, block.up_weight, normed,
    )
    hidden = _qwen3_vl_mlp_activation!(mlp_scratch, gate, up)
    x = _qwen3_vl_decode_hidden_residual!(
        hidden_residual_scratch, block.down_weight, hidden, x,
    )
    return x, layer_cache
end

mutable struct _Qwen3VLDecodePack{B,E,F,S}
    blocks::B
    embedding::E
    final_norm::F
    spec::S
end

mutable struct _Qwen3VLDecodeReady{
    P,TE,MR,AT,ML,RO,QK,HI,GU,RE,TI,QV,
}
    pack::P
    token_embedding::TE
    mrope::MR
    attention::AT
    mlp::ML
    rope::RO
    qk_norm::QK
    hidden::HI
    gate_up::GU
    residual::RE
    tied::TI
    qkv::QV
end

function _qwen3_vl_after_impl!(parameters, token, cache)
    return nothing
end

function _qwen3_vl_load_scratch(
    cache::Qwen3VLStaticKVCache,
    field::Symbol,
    ready_field::Symbol,
    ::Type{T},
) where {T}
    scratch = getfield(cache, field)
    scratch isa T && return scratch
    ready = cache.decode_pack
    ready isa _Qwen3VLDecodeReady || return scratch
    installed = getfield(ready, ready_field)
    return installed isa T ? installed : scratch
end

function _qwen3_vl_decode_pack(cache::Qwen3VLStaticKVCache, parameters)
    pack = cache.decode_pack
    if pack isa _Qwen3VLDecodePack && pack.embedding === parameters.embedding
        return pack
    end
    pack = _Qwen3VLDecodePack(
        parameters.blocks,
        parameters.embedding,
        parameters.final_norm,
        parameters.spec,
    )
    cache.decode_pack = pack
    return pack
end

function _qwen3_vl_install_ready!(
    cache::Qwen3VLStaticKVCache,
    pack::_Qwen3VLDecodePack{B,E,F,S},
    token_embedding_scratch::Qwen3VLTokenEmbeddingScratch,
    mrope_scratch::Qwen3VLMropeScratch,
    attention_scratch::Qwen3VLDecodeAttentionScratch,
    mlp_scratch::Qwen3VLMlpActivationScratch,
    rope_scratch::Qwen3VLQKRopeScratch,
    norm_scratch::Qwen3VLQKNormScratch,
    hidden_norm_scratch::Qwen3VLHiddenNormScratch,
    gate_up_scratch::Qwen3VLGateUpScratch,
    hidden_residual_scratch::Qwen3VLHiddenResidualScratch,
    tied_logits_scratch::Qwen3VLTiedLogitsScratch,
    qkv_scratch::Qwen3VLQKVScratch,
) where {B,E,F,S}
    # Keep the blocks tuple in the pack. Copying it into this object was a
    # second allocation of the same immutable.
    cache.decode_pack = _Qwen3VLDecodeReady(
        pack,
        token_embedding_scratch,
        mrope_scratch,
        attention_scratch,
        mlp_scratch,
        rope_scratch,
        norm_scratch,
        hidden_norm_scratch,
        gate_up_scratch,
        hidden_residual_scratch,
        tied_logits_scratch,
        qkv_scratch,
    )
    return cache.decode_pack
end

@inline function _qwen3_vl_text_decode_ready!(
    ready::_Qwen3VLDecodeReady,
    token,
    cache::Qwen3VLStaticKVCache,
)
    spec = ready.pack.spec
    _validate_qwen3_vl_static_kv_cache(ready.pack, cache)
    isempty(cache) && throw(ArgumentError(
        "call hf_qwen3_vl_text_prefill_static before decode_step",
    ))
    cache.position < cache.capacity || throw(ArgumentError(
        "Qwen3-VL static KV cache has reached its capacity",
    ))
    coordinate = _qwen3_vl_decode_coordinate(
        cache.position,
        cache.rope_delta,
        spec.max_position_embeddings,
    )
    tokens = _qwen3_vl_decode_token_matrix(token, cache.batch_size)
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL decode token is outside the vocabulary",
    ))
    x = _qwen3_vl_decode_token_embedding!(
        ready.token_embedding, ready.pack.embedding, tokens[1],
    )
    cos_values, sin_values = _qwen3_vl_decode_mrope!(
        ready.mrope,
        spec.head_dim,
        spec.rope_theta,
        spec.mrope_section,
        coordinate,
    )
    write_position = cache.position + 1
    x = _qwen3_vl_static_decode_layers!(
        ready.pack,
        cache,
        x,
        cos_values,
        sin_values,
        write_position,
        ready.attention,
        ready.mlp,
        ready.rope,
        ready.qk_norm,
        ready.hidden,
        ready.gate_up,
        ready.residual,
        ready.qkv,
    )
    final_hidden = _qwen3_vl_decode_hidden_norm!(
        ready.hidden, x, ready.pack.final_norm, spec.rms_norm_eps,
    )
    logits = _qwen3_vl_decode_tied_logits!(
        ready.tied, ready.pack.embedding, final_hidden,
    )
    cache.position = write_position
    _validate_qwen3_vl_static_kv_cache(ready.pack, cache)
    return logits, cache
end

function _qwen3_vl_static_decode_layers!(
    pack,
    cache::Qwen3VLStaticKVCache,
    x::AbstractArray{BFloat16,3},
    cos_values,
    sin_values,
    write_position::Int,
    attention_scratch::Qwen3VLDecodeAttentionScratch,
    mlp_scratch::Qwen3VLMlpActivationScratch,
    rope_scratch::Qwen3VLQKRopeScratch,
    norm_scratch::Qwen3VLQKNormScratch,
    hidden_norm_scratch::Qwen3VLHiddenNormScratch,
    gate_up_scratch::Qwen3VLGateUpScratch,
    hidden_residual_scratch::Qwen3VLHiddenResidualScratch,
    qkv_scratch::Qwen3VLQKVScratch,
)
    spec = pack.spec
    size(x, 2) == 1 || throw(ArgumentError(
        "Qwen3-VL static decode block requires one token",
    ))
    for julia_layer in 1:spec.num_hidden_layers
        x, _ = _qwen3_vl_static_decode_block!(
            spec,
            pack.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            cache.layers[julia_layer],
            write_position,
            attention_scratch,
            mlp_scratch,
            rope_scratch,
            norm_scratch,
            hidden_norm_scratch,
            gate_up_scratch,
            hidden_residual_scratch,
            qkv_scratch,
        )
    end
    return x
end

function _qwen3_vl_static_cache_block!(
    spec,
    block,
    x,
    cos_values,
    sin_values,
    mask,
    layer_cache::Qwen3VLStaticLayerKVCache,
    write_start::Int,
    profile_runner=nothing,
    layer_index::Int=0;
    attention_scratch=nothing,
    mlp_scratch=nothing,
    rope_scratch=nothing,
    norm_scratch=nothing,
    hidden_norm_scratch=nothing,
    gate_up_scratch=nothing,
    hidden_residual_scratch=nothing,
    qkv_scratch=nothing,
)
    sequence_length = size(x, 2)
    write_stop = write_start + sequence_length - 1
    1 <= write_start <= write_stop <= size(layer_cache.keys, 3) ||
        throw(ArgumentError(
            "Qwen3-VL static cache write would exceed its capacity",
        ))

    store = function (keys, values)
        expected = (
            spec.head_dim,
            spec.num_key_value_heads,
            sequence_length,
            size(x, 3),
        )
        size(keys) == expected || throw(DimensionMismatch(
            "Qwen3-VL projected keys do not match the static cache layout",
        ))
        size(values) == expected || throw(DimensionMismatch(
            "Qwen3-VL projected values do not match the static cache layout",
        ))
        eltype(keys) == eltype(layer_cache.keys) || throw(ArgumentError(
            "Qwen3-VL projected keys do not match the static cache dtype",
        ))
        eltype(values) == eltype(layer_cache.values) || throw(ArgumentError(
            "Qwen3-VL projected values do not match the static cache dtype",
        ))
        get_device(keys) == get_device(layer_cache.keys) || throw(ArgumentError(
            "Qwen3-VL projected keys do not match the static cache device",
        ))
        get_device(values) == get_device(layer_cache.values) || throw(ArgumentError(
            "Qwen3-VL projected values do not match the static cache device",
        ))

        write_range = write_start:write_stop
        if sequence_length != 1 || !_qwen3_vl_device_write_kv!(
            layer_cache.keys,
            layer_cache.values,
            keys,
            values,
            write_start,
        )
            copyto!(view(layer_cache.keys, :, :, write_range, :), keys)
            copyto!(view(layer_cache.values, :, :, write_range, :), values)
        end
        valid_range = 1:write_stop
        return (
            view(layer_cache.keys, :, :, valid_range, :),
            view(layer_cache.values, :, :, valid_range, :),
            layer_cache,
        )
    end
    return _qwen3_vl_cache_block_core(
        spec,
        block,
        x,
        cos_values,
        sin_values,
        mask,
        store,
        profile_runner,
        layer_index;
        attention_scratch,
        mlp_scratch,
        rope_scratch,
        norm_scratch,
        hidden_norm_scratch,
        gate_up_scratch,
        hidden_residual_scratch,
        qkv_scratch,
    )
end

function _qwen3_vl_static_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    cache::Qwen3VLStaticKVCache,
    logits_to_keep::Int,
)
    spec = _qwen3_vl_cache_spec(parameters)
    sequence_length, batch_size = size(tokens)
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    isempty(cache) || throw(ArgumentError(
        "Qwen3-VL static cached prefill requires an empty cache",
    ))
    batch_size == cache.batch_size || throw(DimensionMismatch(
        "Qwen3-VL prompt batch size does not match the static cache",
    ))
    0 < sequence_length <= cache.capacity || throw(ArgumentError(
        "Qwen3-VL prompt length exceeds the static cache capacity",
    ))
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL input_ids contain an out-of-vocabulary id",
    ))
    size(rope_layout.visual_mask) == size(tokens) || throw(DimensionMismatch(
        "Qwen3-VL visual mask does not match input_ids",
    ))
    _qwen3_vl_prompt_visual_token_contract(
        parameters,
        tokens,
        rope_layout.visual_mask,
        rope_layout.attention_mask,
    )
    rope_deltas = _qwen3_vl_prompt_rope_deltas(
        spec,
        rope_layout,
        sequence_length,
        batch_size,
    )
    all(rope_layout.attention_mask) || throw(ArgumentError(
        "Qwen3-VL static generation currently requires an all-ones attention mask",
    ))
    rope_delta = only(rope_deltas)
    0 <= logits_to_keep <= sequence_length || throw(ArgumentError(
        "logits_to_keep must be between zero and the prefill length",
    ))
    return spec, rope_delta
end

"""
    hf_qwen3_vl_text_prefill_static(parameters, input_ids, rope_layout;
                                    vision_features=nothing, cache,
                                    logits_to_keep=1,
                                    capture_input_embeddings=true,
                                    capture_final_hidden=true)

Run one batch-1 Qwen3-VL prompt and write every layer's K/V tensors into the
fixed `1:prompt_length` prefix of `cache`. The returned cache is the same
mutable object passed by the caller, and all layer storage identities remain
unchanged. The `capture_*` keywords have the same lightweight diagnostic
semantics as dynamic cached prefill.
"""
function hf_qwen3_vl_text_prefill_static(
    parameters,
    input_ids,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    cache::Qwen3VLStaticKVCache,
    logits_to_keep::Integer=1,
    capture_input_embeddings::Bool=true,
    capture_final_hidden::Bool=true,
)
    options = _qwen3_vl_prefill_options(logits_to_keep)
    tokens = _qwen3_vl_token_matrix(input_ids)
    spec, rope_delta = _qwen3_vl_static_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        cache,
        options.logits_to_keep,
    )

    if vision_features === nothing
        any(rope_layout.visual_mask) && throw(ArgumentError(
            "Qwen3-VL visual placeholders require vision features",
        ))
    else
        length(vision_features.deepstack) == 3 || throw(DimensionMismatch(
            "Qwen3-VL static prefill requires exactly three DeepStack features",
        ))
        _validate_qwen3_vl_text_feature_geometry(
            parameters,
            vision_features,
            count(rope_layout.visual_mask),
        )
        _validate_qwen3_vl_text_feature_residency(
            parameters,
            vision_features,
        )
    end
    return _qwen3_vl_text_prefill_static_compute!(
        parameters,
        tokens,
        rope_layout,
        vision_features,
        cache,
        spec,
        rope_delta,
        options.logits_to_keep,
        capture_input_embeddings,
        capture_final_hidden,
    )
end

function _qwen3_vl_prevalidated_static_prompt_contract(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    vision_features,
    cache::Qwen3VLStaticKVCache,
    contract::_Qwen3VLGenerationPromptContract,
)
    spec = _qwen3_vl_cache_spec(parameters)
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    isempty(cache) || throw(ArgumentError(
        "Qwen3-VL static cached prefill requires an empty cache",
    ))
    contract.batch_size == cache.batch_size || throw(DimensionMismatch(
        "Qwen3-VL prompt batch size does not match the static cache",
    ))
    0 < contract.sequence_length <= cache.capacity || throw(ArgumentError(
        "Qwen3-VL prompt length exceeds the static cache capacity",
    ))
    sealed_spec = _qwen3_vl_validate_generation_prompt_sources(
        contract,
        parameters,
        tokens,
        rope_layout,
        vision_features,
    )
    spec === sealed_spec || throw(ArgumentError(
        "Qwen3-VL prevalidated generation decoder spec changed",
    ))
    return spec, contract.rope_delta
end

function _qwen3_vl_text_prefill_static_prevalidated(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout;
    vision_features=nothing,
    cache::Qwen3VLStaticKVCache,
    contract::_Qwen3VLGenerationPromptContract,
    capture_input_embeddings::Bool=true,
    capture_final_hidden::Bool=true,
)
    spec, rope_delta = _qwen3_vl_prevalidated_static_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        vision_features,
        cache,
        contract,
    )
    sealed_parameters = _qwen3_vl_prevalidated_text_parameters(contract)
    sealed_vision_features = _qwen3_vl_prevalidated_vision_features(contract)
    return _qwen3_vl_text_prefill_static_compute!(
        sealed_parameters,
        tokens,
        rope_layout,
        sealed_vision_features,
        cache,
        spec,
        rope_delta,
        1,
        capture_input_embeddings,
        capture_final_hidden,
    )
end

function _qwen3_vl_text_prefill_static_compute!(
    parameters,
    tokens,
    rope_layout::Qwen3VLRopeLayout,
    vision_features,
    cache::Qwen3VLStaticKVCache,
    spec::Qwen3VLTextSpec,
    rope_delta::Int,
    logits_to_keep::Int,
    capture_input_embeddings::Bool,
    capture_final_hidden::Bool,
)
    sequence_length, batch_size = size(tokens)
    x = reshape(
        gather(parameters.embedding, tokens),
        spec.hidden_size,
        sequence_length,
        batch_size,
    )
    if vision_features !== nothing
        x = _qwen3_vl_replace_visual_embeddings(
            x,
            vision_features.visual_embeddings,
            rope_layout.visual_mask,
        )
    end
    input_embeddings = capture_input_embeddings ? x : nothing
    cos_values, sin_values = _qwen3_vl_text_mrope(
        spec,
        parameters.embedding,
        rope_layout.position_ids,
    )
    mask = _qwen3_vl_causal_mask(
        parameters.embedding,
        Bool.(rope_layout.attention_mask),
    )

    for julia_layer in 1:spec.num_hidden_layers
        layer = julia_layer - 1
        x, _ = _qwen3_vl_static_cache_block!(
            spec,
            parameters.blocks[julia_layer],
            x,
            cos_values,
            sin_values,
            mask,
            cache.layers[julia_layer],
            1,
        )
        if vision_features !== nothing && layer < 3
            x = _qwen3_vl_add_deepstack(
                x,
                vision_features.deepstack[layer + 1],
                rope_layout.visual_mask,
            )
        end
    end

    final_hidden, projection = _qwen3_vl_prefill_projection(
        parameters,
        x,
        sequence_length,
        logits_to_keep,
        capture_final_hidden,
    )
    logits = _qwen3_vl_project_tied(parameters.embedding, projection)
    cache.position = sequence_length
    cache.rope_delta = rope_delta
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    result = Qwen3VLTextPrefill(
        input_embeddings,
        Dict{Int,Any}(),
        Dict{Int,Any}(),
        final_hidden,
        logits,
        rope_layout,
    )
    return result, cache
end

function _qwen3_vl_text_decode_step_static_impl(
    parameters,
    token,
    cache::Qwen3VLStaticKVCache,
    profile_runner,
)
    ready = cache.decode_pack
    if profile_runner === nothing &&
        ready isa _Qwen3VLDecodeReady &&
        ready.pack.embedding === parameters.embedding
        return _qwen3_vl_text_decode_ready!(ready, token, cache)
    end
    spec = _qwen3_vl_cache_spec(parameters)
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    isempty(cache) && throw(ArgumentError(
        "call hf_qwen3_vl_text_prefill_static before decode_step",
    ))
    cache.position < cache.capacity || throw(ArgumentError(
        "Qwen3-VL static KV cache has reached its capacity",
    ))
    coordinate = _qwen3_vl_decode_coordinate(
        cache.position,
        cache.rope_delta,
        spec.max_position_embeddings,
    )
    tokens = _qwen3_vl_decode_token_matrix(token, cache.batch_size)
    all(id -> 1 <= id <= spec.vocab_size, tokens) || throw(ArgumentError(
        "Qwen3-VL decode token is outside the vocabulary",
    ))
    if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_device_decode_workspace!(cache, spec, parameters.embedding)
    end
    token_embedding_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_token_embedding_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    mrope_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_mrope_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end

    x = @_qwen3_vl_stage profile_runner :token_embedding 0 begin
        if token_embedding_scratch !== nothing &&
            eltype(parameters.embedding) === BFloat16 &&
            size(tokens) == (1, 1)
            _qwen3_vl_decode_token_embedding!(
                token_embedding_scratch,
                parameters.embedding,
                tokens[1],
            )
        else
            reshape(
                gather(parameters.embedding, tokens),
                spec.hidden_size,
                1,
                cache.batch_size,
            )
        end
    end
    cos_values, sin_values = @_qwen3_vl_stage profile_runner :mrope_prepare 0 begin
        if mrope_scratch !== nothing && cache.batch_size == 1
            _qwen3_vl_decode_mrope!(
                mrope_scratch,
                spec.head_dim,
                spec.rope_theta,
                spec.mrope_section,
                coordinate,
            )
        else
            position_ids = fill(coordinate, 3, 1, cache.batch_size)
            _qwen3_vl_text_mrope(
                spec,
                parameters.embedding,
                position_ids,
            )
        end
    end
    write_position = cache.position + 1
    attention_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_decode_attention_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    mlp_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_mlp_activation_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    rope_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_qk_rope_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    norm_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_qk_norm_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    hidden_norm_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_hidden_norm_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    gate_up_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_gate_up_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    hidden_residual_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_hidden_residual_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    tied_logits_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_tied_logits_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    qkv_scratch = if eltype(parameters.embedding) === BFloat16
        _qwen3_vl_ensure_qkv_scratch!(
            cache,
            spec,
            parameters.embedding,
        )
    else
        nothing
    end
    ready_pack = nothing
    x = if profile_runner === nothing &&
        attention_scratch isa Qwen3VLDecodeAttentionScratch &&
        mlp_scratch isa Qwen3VLMlpActivationScratch &&
        rope_scratch isa Qwen3VLQKRopeScratch &&
        norm_scratch isa Qwen3VLQKNormScratch &&
        hidden_norm_scratch isa Qwen3VLHiddenNormScratch &&
        gate_up_scratch isa Qwen3VLGateUpScratch &&
        hidden_residual_scratch isa Qwen3VLHiddenResidualScratch &&
        qkv_scratch isa Qwen3VLQKVScratch &&
        x isa AbstractArray{BFloat16,3} &&
        size(x, 2) == 1
        ready_pack = _qwen3_vl_decode_pack(cache, parameters)
        _qwen3_vl_static_decode_layers!(
            ready_pack,
            cache,
            x,
            cos_values,
            sin_values,
            write_position,
            attention_scratch,
            mlp_scratch,
            rope_scratch,
            norm_scratch,
            hidden_norm_scratch,
            gate_up_scratch,
            hidden_residual_scratch,
            qkv_scratch,
        )
    else
        for julia_layer in 1:spec.num_hidden_layers
            x, _ = _qwen3_vl_static_cache_block!(
                spec,
                parameters.blocks[julia_layer],
                x,
                cos_values,
                sin_values,
                nothing,
                cache.layers[julia_layer],
                write_position,
                profile_runner,
                julia_layer;
                attention_scratch,
                mlp_scratch,
                rope_scratch,
                norm_scratch,
                hidden_norm_scratch,
                gate_up_scratch,
                hidden_residual_scratch,
                qkv_scratch,
            )
        end
        x
    end
    final_hidden = @_qwen3_vl_stage profile_runner :final_norm 0 begin
        if hidden_norm_scratch !== nothing &&
            eltype(x) === BFloat16 &&
            size(x) == size(hidden_norm_scratch.output)
            _qwen3_vl_decode_hidden_norm!(
                hidden_norm_scratch,
                x,
                parameters.final_norm,
                spec.rms_norm_eps,
            )
        else
            _qwen3_vl_text_rmsnorm(
                x,
                parameters.final_norm,
                spec.rms_norm_eps,
            )
        end
    end
    logits = @_qwen3_vl_stage profile_runner :vocab_logits 0 begin
        if tied_logits_scratch !== nothing &&
            eltype(final_hidden) === BFloat16 &&
            size(final_hidden) == (tied_logits_scratch.hidden_size, 1, 1) &&
            size(parameters.embedding) == (
                tied_logits_scratch.hidden_size,
                tied_logits_scratch.vocab_size,
            )
            _qwen3_vl_decode_tied_logits!(
                tied_logits_scratch,
                parameters.embedding,
                final_hidden,
            )
        else
            _qwen3_vl_project_tied(parameters.embedding, final_hidden)
        end
    end
    cache.position = write_position
    _validate_qwen3_vl_static_kv_cache(parameters, cache)
    if profile_runner === nothing &&
        attention_scratch isa Qwen3VLDecodeAttentionScratch &&
        mlp_scratch isa Qwen3VLMlpActivationScratch &&
        rope_scratch isa Qwen3VLQKRopeScratch &&
        norm_scratch isa Qwen3VLQKNormScratch &&
        hidden_norm_scratch isa Qwen3VLHiddenNormScratch &&
        gate_up_scratch isa Qwen3VLGateUpScratch &&
        hidden_residual_scratch isa Qwen3VLHiddenResidualScratch &&
        tied_logits_scratch isa Qwen3VLTiedLogitsScratch &&
        qkv_scratch isa Qwen3VLQKVScratch &&
        token_embedding_scratch isa Qwen3VLTokenEmbeddingScratch &&
        mrope_scratch isa Qwen3VLMropeScratch
        if ready_pack isa _Qwen3VLDecodePack
            _qwen3_vl_install_ready!(
                cache,
                ready_pack,
                token_embedding_scratch,
                mrope_scratch,
                attention_scratch,
                mlp_scratch,
                rope_scratch,
                norm_scratch,
                hidden_norm_scratch,
                gate_up_scratch,
                hidden_residual_scratch,
                tied_logits_scratch,
                qkv_scratch,
            )
        else
            cache.decode_pack = _Qwen3VLDecodeReady(
                _qwen3_vl_decode_pack(cache, parameters),
                token_embedding_scratch,
                mrope_scratch,
                attention_scratch,
                mlp_scratch,
                rope_scratch,
                norm_scratch,
                hidden_norm_scratch,
                gate_up_scratch,
                hidden_residual_scratch,
                tied_logits_scratch,
                qkv_scratch,
            )
        end
    end
    _qwen3_vl_after_impl!(parameters, token, cache)
    return logits, cache
end


"""
    hf_qwen3_vl_text_decode_step_static(parameters, token, cache)

Write one token in place at physical slot `cache.position + 1`, attend only to
the valid `1:cache.position+1` prefix, and return `(logits, cache)`. The next
mRoPE coordinate remains `cache.position + cache.rope_delta` before the
position increment.
"""
function hf_qwen3_vl_text_decode_step_static(
    parameters,
    token,
    cache::Qwen3VLStaticKVCache,
)
    ready = cache.decode_pack
    if ready isa _Qwen3VLDecodeReady &&
        ready.pack.embedding === parameters.embedding
        return _qwen3_vl_text_decode_ready!(ready, token, cache)
    end
    return _qwen3_vl_text_decode_step_static_impl(
        parameters,
        token,
        cache,
        nothing,
    )
end

"""
    _profile_qwen3_vl_text_decode_step_static(parameters, token, cache, runner)

Run one bounded-static Qwen3-VL decode step while routing each decoder stage
through `runner(stage, layer_index, thunk)`. Request-level stages use layer
index zero; decoder blocks use one-based layer indices. This diagnostic entry
point preserves the public decode path's validation, cache mutation, and
numerical operations. It is intended for allocation attribution, not latency
measurement, because a runner may synchronize between stages.
"""
function _profile_qwen3_vl_text_decode_step_static(
    parameters,
    token,
    cache::Qwen3VLStaticKVCache,
    runner,
)
    runner === nothing && throw(ArgumentError(
        "Qwen3-VL profiling requires a stage runner",
    ))
    return _qwen3_vl_text_decode_step_static_impl(
        parameters,
        token,
        cache,
        runner,
    )
end
