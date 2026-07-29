using BFloat16s: BFloat16
import Adapt

# Week 16 introduced RTN weight-only quantization. Week 17 adds deterministic
# reconstruction-MSE scale calibration and one shared, fine-grained plan for
# both in-memory and streamed quantization. Embeddings and norm scales stay
# BF16. Compute is unchanged: weights are dequantized back to BF16 right before
# the existing `_bf16a_linear`, so quantization only changes weight residency,
# never the compute contract.

const _QWEN3_QUANTIZATION_TARGETS = (
    :q_proj,
    :k_proj,
    :v_proj,
    :o_proj,
    :gate_proj,
    :up_proj,
    :down_proj,
    :lm_head,
)
const _DEFAULT_INT4_MSE_CLIP_RATIOS = (
    1.0f0,
    0.99f0,
    0.975f0,
    0.95f0,
    0.925f0,
    0.9f0,
    0.875f0,
    0.85f0,
    0.8f0,
)

"""
    LinearQuantizationSpec(
        scheme=:int4;
        group=128,
        calibration=:maxabs,
        clip_ratios=_DEFAULT_INT4_MSE_CLIP_RATIOS,
    )

Storage policy for one linear weight. Supported schemes are packed symmetric
`:int4`, symmetric per-output-channel `:int8`, and unquantized `:bf16`.
For INT4, `calibration=:maxabs` preserves the Week 16 RTN behavior, while
`:mse` chooses one clipping ratio per output-row/input-group by minimizing
weight reconstruction squared error over the frozen `clip_ratios`.
"""
struct LinearQuantizationSpec{C<:Tuple}
    scheme::Symbol
    group::Int
    calibration::Symbol
    clip_ratios::C

    function LinearQuantizationSpec(
        scheme::Symbol=:int4;
        group::Integer=128,
        calibration::Symbol=:maxabs,
        clip_ratios=_DEFAULT_INT4_MSE_CLIP_RATIOS,
    )
        scheme in (:int4, :int8, :bf16) || throw(ArgumentError(
            "unsupported quantization scheme $(repr(scheme))",
        ))
        group > 0 || throw(ArgumentError("quantization group must be positive"))
        if scheme === :int4
            iseven(group) || throw(ArgumentError("INT4 group size must be even"))
            calibration in (:maxabs, :mse) || throw(ArgumentError(
                "unsupported INT4 calibration $(repr(calibration))",
            ))
        elseif calibration !== :maxabs
            throw(ArgumentError(
                "calibration $(repr(calibration)) is only supported for INT4",
            ))
        end

        ratios = if scheme === :int4 && calibration === :mse
            Tuple(Float32(ratio) for ratio in clip_ratios)
        else
            (1.0f0,)
        end
        isempty(ratios) && throw(ArgumentError(
            "INT4 MSE calibration requires at least one clipping ratio",
        ))
        all(ratio -> isfinite(ratio) && 0.0f0 < ratio <= 1.0f0, ratios) ||
            throw(ArgumentError("INT4 clipping ratios must be finite in (0, 1]"))
        1.0f0 in ratios || throw(ArgumentError(
            "INT4 MSE clipping ratios must include 1.0 as the max-abs baseline",
        ))
        return new{typeof(ratios)}(scheme, Int(group), calibration, ratios)
    end
end

"""
    QuantizationPlan(; default, projection_overrides, layer_overrides)

Fine-grained Qwen3 linear-weight policy. Resolution priority is:

1. `(layer, projection)` override,
2. projection override,
3. default.

Layers are one-based. `:lm_head` is a projection-level target because it does
not belong to a transformer block.
"""
struct QuantizationPlan
    default::LinearQuantizationSpec
    projection_overrides::Dict{Symbol,LinearQuantizationSpec}
    layer_overrides::Dict{Tuple{Int,Symbol},LinearQuantizationSpec}
end

function QuantizationPlan(;
    default::LinearQuantizationSpec=LinearQuantizationSpec(),
    projection_overrides=Dict{Symbol,LinearQuantizationSpec}(),
    layer_overrides=Dict{Tuple{Int,Symbol},LinearQuantizationSpec}(),
)
    projections = Dict{Symbol,LinearQuantizationSpec}()
    for (projection, spec) in pairs(projection_overrides)
        projection isa Symbol || throw(ArgumentError(
            "quantization projection keys must be Symbols",
        ))
        projection in _QWEN3_QUANTIZATION_TARGETS || throw(ArgumentError(
            "unsupported quantization projection $(repr(projection))",
        ))
        spec isa LinearQuantizationSpec || throw(ArgumentError(
            "quantization projection overrides must be LinearQuantizationSpec values",
        ))
        projections[projection] = spec
    end

    layers = Dict{Tuple{Int,Symbol},LinearQuantizationSpec}()
    for (target, spec) in pairs(layer_overrides)
        target isa Tuple && length(target) == 2 || throw(ArgumentError(
            "layer override keys must be `(layer, projection)` tuples",
        ))
        layer, projection = target
        layer isa Integer && layer > 0 || throw(ArgumentError(
            "quantization override layers must be positive one-based integers",
        ))
        projection isa Symbol || throw(ArgumentError(
            "quantization projection keys must be Symbols",
        ))
        projection in _QWEN3_QUANTIZATION_TARGETS || throw(ArgumentError(
            "unsupported quantization projection $(repr(projection))",
        ))
        projection === :lm_head && throw(ArgumentError(
            "lm_head does not accept a layer-specific quantization override",
        ))
        spec isa LinearQuantizationSpec || throw(ArgumentError(
            "quantization layer overrides must be LinearQuantizationSpec values",
        ))
        layers[(Int(layer), projection)] = spec
    end
    return QuantizationPlan(default, projections, layers)
end

"""
    quantization_spec(plan, projection; layer=nothing)

Resolve one target using layer override > projection override > default.
"""
function quantization_spec(
    plan::QuantizationPlan,
    projection::Symbol;
    layer::Union{Nothing,Integer}=nothing,
)
    projection in _QWEN3_QUANTIZATION_TARGETS || throw(ArgumentError(
        "unsupported quantization projection $(repr(projection))",
    ))
    layer === nothing || layer > 0 || throw(ArgumentError(
        "quantization layers are positive and one-based",
    ))
    projection === :lm_head && layer !== nothing && throw(ArgumentError(
        "lm_head does not belong to a transformer layer",
    ))
    if layer !== nothing
        target = (Int(layer), projection)
        haskey(plan.layer_overrides, target) && return plan.layer_overrides[target]
    end
    return get(plan.projection_overrides, projection, plan.default)
end

function _validate_quantization_plan_layers(
    plan::QuantizationPlan,
    num_layers::Integer,
)
    num_layers > 0 || throw(ArgumentError("model must contain at least one layer"))
    for ((layer, projection), _) in plan.layer_overrides
        layer <= num_layers || throw(ArgumentError(
            "quantization override ($layer, $(repr(projection))) exceeds " *
            "model depth $num_layers",
        ))
    end
    return plan
end

function _legacy_quantization_plan(
    scheme::Symbol,
    group::Integer,
    int8_projections,
)
    overrides = Dict{Symbol,LinearQuantizationSpec}()
    for projection in int8_projections
        projection isa Symbol || throw(ArgumentError(
            "int8_projections entries must be Symbols",
        ))
        overrides[projection] = LinearQuantizationSpec(:int8; group)
    end
    return QuantizationPlan(
        default=LinearQuantizationSpec(scheme; group),
        projection_overrides=overrides,
    )
end

_resolve_quantization_plan(
    plan::Nothing,
    scheme::Symbol,
    group::Integer,
    int8_projections,
) = _legacy_quantization_plan(scheme, group, int8_projections)
_resolve_quantization_plan(
    plan::QuantizationPlan,
    ::Symbol,
    ::Integer,
    _,
) = plan

"""
    Int8ChannelWeight(q, scale)

Symmetric per-output-channel INT8 weight: `w ≈ Float32(q) .* scale` with
`q::Matrix{Int8}` of shape `(out, in)` and `scale::Vector{Float32}` per row.
"""
struct Int8ChannelWeight{Q,S}
    q::Q
    scale::S
end
Adapt.@adapt_structure Int8ChannelWeight

"""
    Int4GroupWeight(packed, scale, group, in_dim)

Symmetric group-wise INT4 weight. Adjacent input columns are packed two per
byte (`packed::Matrix{UInt8}` of shape `(out, in ÷ 2)`, low nibble first,
values offset by +8); `scale` has shape `(out, in ÷ group)`.
"""
struct Int4GroupWeight{Q,S}
    packed::Q
    scale::S
    group::Int
    in_dim::Int
end
Adapt.Adapt.adapt_structure(to, w::Int4GroupWeight) = Int4GroupWeight(
    Adapt.adapt(to, w.packed),
    Adapt.adapt(to, w.scale),
    w.group,
    w.in_dim,
)

function _quantize_int8_channel(weight::AbstractMatrix)
    w = Float32.(weight)
    scale = vec(maximum(abs, w; dims=2)) ./ 127.0f0
    scale = max.(scale, 1.0f-12)
    q = Int8.(clamp.(round.(w ./ scale), -127.0f0, 127.0f0))
    return Int8ChannelWeight(q, scale)
end

function _mse_calibrated_int4_scale(
    grouped::AbstractArray{Float32,3},
    maxabs_scale::AbstractMatrix{Float32},
    clip_ratios::Tuple,
)
    out_dim, group, groups = size(grouped)
    best_scale = similar(maxabs_scale)
    Threads.@threads :static for target_index in 1:(out_dim * groups)
        row = mod1(target_index, out_dim)
        group_index = div(target_index - 1, out_dim) + 1
        baseline_scale = maxabs_scale[row, group_index]
        selected_scale = baseline_scale
        selected_error = Inf32
        for ratio in clip_ratios
            candidate_scale = max(baseline_scale * ratio, 1.0f-12)
            candidate_error = 0.0f0
            @inbounds @simd for column in 1:group
                value = grouped[row, column, group_index]
                quantized = clamp(round(value / candidate_scale), -7.0f0, 7.0f0)
                reconstructed = Float32(BFloat16(quantized * candidate_scale))
                candidate_error += abs2(reconstructed - value)
            end
            if candidate_error < selected_error
                selected_error = candidate_error
                selected_scale = candidate_scale
            end
        end
        best_scale[row, group_index] = selected_scale
    end
    return best_scale
end

function _quantize_int4_group(
    weight::AbstractMatrix;
    group::Int=128,
    calibration::Symbol=:maxabs,
    clip_ratios=_DEFAULT_INT4_MSE_CLIP_RATIOS,
)
    out_dim, in_dim = size(weight)
    in_dim % group == 0 || throw(ArgumentError(
        "input dimension $in_dim is not divisible by group $group",
    ))
    iseven(group) || throw(ArgumentError("group size must be even"))
    spec = LinearQuantizationSpec(
        :int4;
        group,
        calibration,
        clip_ratios,
    )
    w = Float32.(weight)
    groups = in_dim ÷ group
    grouped = reshape(w, out_dim, group, groups)
    maxabs_scale = max.(
        reshape(maximum(abs, grouped; dims=2), out_dim, groups) ./ 7.0f0,
        1.0f-12,
    )
    scale = if spec.calibration === :maxabs
        maxabs_scale
    else
        _mse_calibrated_int4_scale(
            grouped,
            maxabs_scale,
            spec.clip_ratios,
        )
    end
    shifted = reshape(UInt8.(
        clamp.(
            round.(grouped ./ reshape(scale, out_dim, 1, groups)),
            -7.0f0,
            7.0f0,
        ) .+ 8.0f0,
    ), out_dim, in_dim)
    low = shifted[:, 1:2:end]
    high = shifted[:, 2:2:end]
    packed = low .| (high .<< 4)
    return Int4GroupWeight(packed, scale, group, in_dim)
end

_dequantize_bf16(weight::AbstractMatrix) = weight

function _dequantize_bf16(weight::Int8ChannelWeight)
    return BFloat16.(Float32.(weight.q) .* reshape(weight.scale, :, 1))
end

function _dequantize_bf16(weight::Int4GroupWeight)
    out_dim = size(weight.packed, 1)
    # Packed column c holds input columns (2c-1, 2c); both live in the same
    # group, so one gathered scale column serves the pair. Each half is fused
    # into a single broadcast so no full-width Float32 intermediate exists.
    packed_group_index = ((2 .* (1:(weight.in_dim ÷ 2)) .- 2) .÷ weight.group) .+ 1
    scale_half = weight.scale[:, packed_group_index]
    low = BFloat16.((Float32.(weight.packed .& 0x0f) .- 8.0f0) .* scale_half)
    high = BFloat16.((Float32.(weight.packed .>> 4) .- 8.0f0) .* scale_half)
    stacked = cat(
        reshape(low, out_dim, 1, weight.in_dim ÷ 2),
        reshape(high, out_dim, 1, weight.in_dim ÷ 2);
        dims=2,
    )
    return reshape(stacked, out_dim, weight.in_dim)
end

_quant_row_slice(weight::Int8ChannelWeight, rows) =
    Int8ChannelWeight(weight.q[rows, :], weight.scale[rows])
_quant_row_slice(weight::Int4GroupWeight, rows) = Int4GroupWeight(
    weight.packed[rows, :], weight.scale[rows, :], weight.group, weight.in_dim,
)
_quant_out_dim(weight::Int8ChannelWeight) = size(weight.q, 1)
_quant_out_dim(weight::Int4GroupWeight) = size(weight.packed, 1)

# Dequantize and multiply in output-row chunks: a monolithic BF16 lm_head
# (1.5 GiB for 14B) plus its dequant transients is exactly what pushed the
# GPU memory pool over the edge.
const _QUANT_LINEAR_CHUNK_ROWS = 8_192

function _bf16a_linear_quantized(weight, x::AbstractArray{<:Any,3})
    out_dim = _quant_out_dim(weight)
    out_dim <= _QUANT_LINEAR_CHUNK_ROWS &&
        return _bf16a_linear(_dequantize_bf16(weight), x)
    in_dim, num_tokens, batch_size = size(x)
    chunks = map(1:_QUANT_LINEAR_CHUNK_ROWS:out_dim) do row_start
        rows = row_start:min(row_start + _QUANT_LINEAR_CHUNK_ROWS - 1, out_dim)
        _bf16a_linear(_dequantize_bf16(_quant_row_slice(weight, rows)), x)
    end
    return cat(chunks...; dims=1)
end

_bf16a_linear(weight::Int8ChannelWeight, x::AbstractArray{<:Any,3}) =
    _bf16a_linear_quantized(weight, x)
_bf16a_linear(weight::Int4GroupWeight, x::AbstractArray{<:Any,3}) =
    _bf16a_linear_quantized(weight, x)

function _quantize_linear(weight, spec::LinearQuantizationSpec)
    spec.scheme === :bf16 && return BFloat16.(weight)
    spec.scheme === :int8 && return _quantize_int8_channel(weight)
    spec.scheme === :int4 && return _quantize_int4_group(
        weight;
        group=spec.group,
        calibration=spec.calibration,
        clip_ratios=spec.clip_ratios,
    )
    throw(ArgumentError("unsupported quantization scheme $(repr(spec.scheme))"))
end

_quantize_linear(weight, scheme::Symbol, group::Int) =
    _quantize_linear(weight, LinearQuantizationSpec(scheme; group))

"""
    load_hf_qwen3_quantized(
        model_dir;
        max_seq_len=64,
        scheme=:int8,
        group=128,
        plan=nothing,
        variant=nothing,
    )

Stream a HuggingFace Qwen3 checkpoint from disk and quantize it layer by
layer, so the full BF16 tree is never resident: peak host memory is the
quantized tree plus one layer of transients. Embedding and norm scales stay
BF16. With `plan=nothing`, the Week 16 `scheme`, `group`, and
`int8_projections` API is preserved. A `QuantizationPlan` can instead choose
INT4, INT8, or BF16 independently for each layer/projection and the LM head.
"""
function load_hf_qwen3_quantized(
    model_dir::AbstractString;
    max_seq_len=64,
    scheme::Symbol=:int8,
    group::Int=128,
    int8_projections::Tuple{Vararg{Symbol}}=(),
    plan::Union{Nothing,QuantizationPlan}=nothing,
    variant=nothing,
)
    quantization_plan = _resolve_quantization_plan(
        plan,
        scheme,
        group,
        int8_projections,
    )
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    config = load_hf_qwen3_config(
        joinpath(model_dir, "config.json");
        max_seq_len,
        variant,
    )
    model = GPTModel(config)
    _qwen3_validate_semantics(model)
    _validate_quantization_plan_layers(quantization_plan, model.num_layers)
    reader = open_safetensors_reader(model_dir)
    _qwen3_validate_tensor_names(model, Set(String.(collect(keys(reader)))))
    tensors = _StreamedTensors(reader)

    embedding_hf = _expect_tensor(
        tensors,
        "model.embed_tokens.weight",
        (model.vocab_size, model.d_model),
    )
    token_embedding = (; weight=BFloat16.(permutedims(embedding_hf, (2, 1))))
    embedding_hf = nothing
    GC.gc(false)

    d_model = model.d_model
    q_dim = model.num_heads * model.head_dim
    kv_dim = model.num_kv_heads * model.head_dim
    hidden_dim = model.mlp_hidden_dim
    # 逐投影读入→量化→释放：整块 Float32 常驻曾把 14B 混合精度加载的
    # 宿主峰值推过 22 GiB 触发 OOM。
    function _quantized_projection(prefix, name, shape, layer, projection)
        weight = _expect_tensor(tensors, "$prefix.$name", shape)
        quantized = _quantize_linear(
            weight,
            quantization_spec(quantization_plan, projection; layer),
        )
        weight = nothing
        GC.gc(false)
        return (; weight=quantized)
    end
    block_values = Vector{Any}(undef, model.num_layers)
    for index in 1:model.num_layers
        prefix = "model.layers.$(index - 1)"
        norm_scale(name, dims) = BFloat16.(reshape(
            _expect_tensor(tensors, "$prefix.$name", (dims,)),
            ntuple(i -> i == 1 ? dims : 1, 3)...,
        ))
        block_values[index] = (;
            norm1=(; scale=norm_scale("input_layernorm.weight", d_model)),
            attn=(;
                q_proj=_quantized_projection(prefix, "self_attn.q_proj.weight", (q_dim, d_model), index, :q_proj),
                k_proj=_quantized_projection(prefix, "self_attn.k_proj.weight", (kv_dim, d_model), index, :k_proj),
                v_proj=_quantized_projection(prefix, "self_attn.v_proj.weight", (kv_dim, d_model), index, :v_proj),
                o_proj=_quantized_projection(prefix, "self_attn.o_proj.weight", (d_model, q_dim), index, :o_proj),
                q_norm=(; scale=BFloat16.(_expect_tensor(
                    tensors, "$prefix.self_attn.q_norm.weight", (model.head_dim,),
                ))),
                k_norm=(; scale=BFloat16.(_expect_tensor(
                    tensors, "$prefix.self_attn.k_norm.weight", (model.head_dim,),
                ))),
            ),
            norm2=(; scale=norm_scale("post_attention_layernorm.weight", d_model)),
            mlp=(;
                gate_proj=_quantized_projection(prefix, "mlp.gate_proj.weight", (hidden_dim, d_model), index, :gate_proj),
                up_proj=_quantized_projection(prefix, "mlp.up_proj.weight", (hidden_dim, d_model), index, :up_proj),
                down_proj=_quantized_projection(prefix, "mlp.down_proj.weight", (d_model, hidden_dim), index, :down_proj),
            ),
        )
    end
    block_names = Tuple(Symbol("layer_$layer") for layer in 1:model.num_layers)
    blocks = NamedTuple{block_names}(Tuple(block_values))

    final_norm = (; scale=BFloat16.(reshape(_expect_tensor(
        tensors,
        "model.norm.weight",
        (model.d_model,),
    ), model.d_model, 1, 1)))
    lm_head = if model.tie_embeddings
        (;)
    else
        head = _quantize_linear(
            _expect_tensor(
                tensors,
                "lm_head.weight",
                (model.vocab_size, model.d_model),
            ),
            quantization_spec(quantization_plan, :lm_head),
        )
        GC.gc(false)
        (; weight=head)
    end
    dense_spec = config.qwen3_variant === nothing ?
        nothing : qwen3_dense_spec(config.qwen3_variant)
    return (;
        model,
        parameters=(; token_embedding, blocks, final_norm, lm_head),
        config,
        variant=dense_spec,
        scheme=quantization_plan.default.scheme,
        group=quantization_plan.default.group,
        plan=quantization_plan,
        source=abspath(model_dir),
    )
end

"""
    quantize_bf16_parameters(
        ps;
        scheme=:int8,
        group=128,
        int8_projections=(),
        plan=nothing,
    )

Quantize every linear weight (attention/MLP projections and an untied LM
head) of a BF16 Qwen3 parameter tree with round-to-nearest. Embedding and
norm scales stay BF16. The result feeds `hf_qwen3_bf16_accel_forward`
unchanged and can be moved to the GPU with `CUDA.cu`. A `QuantizationPlan`
enables reconstruction-calibrated INT4 and fine-grained mixed precision.
"""
function quantize_bf16_parameters(
    ps;
    scheme::Symbol=:int8,
    group::Int=128,
    int8_projections::Tuple{Vararg{Symbol}}=(),
    plan::Union{Nothing,QuantizationPlan}=nothing,
)
    quantization_plan = _resolve_quantization_plan(
        plan,
        scheme,
        group,
        int8_projections,
    )
    eltype(ps.token_embedding.weight) === BFloat16 || throw(ArgumentError(
        "quantize_bf16_parameters expects a BFloat16 parameter tree",
    ))
    block_values = values(ps.blocks)
    _validate_quantization_plan_layers(quantization_plan, length(block_values))
    blocks = ntuple(length(block_values)) do index
        block = block_values[index]
        (;
            norm1=block.norm1,
            attn=(;
                q_proj=(; weight=_quantize_linear(
                    block.attn.q_proj.weight,
                    quantization_spec(quantization_plan, :q_proj; layer=index),
                )),
                k_proj=(; weight=_quantize_linear(
                    block.attn.k_proj.weight,
                    quantization_spec(quantization_plan, :k_proj; layer=index),
                )),
                v_proj=(; weight=_quantize_linear(
                    block.attn.v_proj.weight,
                    quantization_spec(quantization_plan, :v_proj; layer=index),
                )),
                o_proj=(; weight=_quantize_linear(
                    block.attn.o_proj.weight,
                    quantization_spec(quantization_plan, :o_proj; layer=index),
                )),
                q_norm=block.attn.q_norm,
                k_norm=block.attn.k_norm,
            ),
            norm2=block.norm2,
            mlp=(;
                gate_proj=(; weight=_quantize_linear(
                    block.mlp.gate_proj.weight,
                    quantization_spec(quantization_plan, :gate_proj; layer=index),
                )),
                up_proj=(; weight=_quantize_linear(
                    block.mlp.up_proj.weight,
                    quantization_spec(quantization_plan, :up_proj; layer=index),
                )),
                down_proj=(; weight=_quantize_linear(
                    block.mlp.down_proj.weight,
                    quantization_spec(quantization_plan, :down_proj; layer=index),
                )),
            ),
        )
    end
    lm_head = if haskey(ps, :lm_head) && haskey(ps.lm_head, :weight)
        (; weight=_quantize_linear(
            ps.lm_head.weight,
            quantization_spec(quantization_plan, :lm_head),
        ))
    else
        ps.lm_head
    end
    return (;
        token_embedding=ps.token_embedding,
        blocks=NamedTuple{keys(ps.blocks)}(blocks),
        final_norm=ps.final_norm,
        lm_head,
    )
end

_tensor_storage_bytes(array::AbstractArray) =
    Int(length(array)) * Int(sizeof(eltype(array)))

"""
    quantized_parameter_bytes(parameters)

Count tensor payload bytes in a Qwen3 parameter tree. The result includes
embeddings, norm scales, packed values, quantization scales, and any BF16
linear overrides. It intentionally excludes Julia object metadata, allocator
slack, KV caches, activations, and dequantization temporaries.
"""
quantized_parameter_bytes(::Nothing) = 0
quantized_parameter_bytes(::Number) = 0
quantized_parameter_bytes(array::AbstractArray) = _tensor_storage_bytes(array)
quantized_parameter_bytes(weight::Int8ChannelWeight) =
    _tensor_storage_bytes(weight.q) + _tensor_storage_bytes(weight.scale)
quantized_parameter_bytes(weight::Int4GroupWeight) =
    _tensor_storage_bytes(weight.packed) + _tensor_storage_bytes(weight.scale)
quantized_parameter_bytes(values::NamedTuple) =
    sum(quantized_parameter_bytes, Base.values(values); init=0)
quantized_parameter_bytes(values::Tuple) =
    sum(quantized_parameter_bytes, values; init=0)
quantized_parameter_bytes(values::AbstractDict) =
    sum(quantized_parameter_bytes, Base.values(values); init=0)

function _linear_quantized_bytes(
    out_dim::Integer,
    in_dim::Integer,
    spec::LinearQuantizationSpec,
)
    out_dim > 0 && in_dim > 0 || throw(ArgumentError(
        "linear dimensions must be positive",
    ))
    if spec.scheme === :bf16
        return 2 * Int(out_dim) * Int(in_dim)
    elseif spec.scheme === :int8
        return Int(out_dim) * Int(in_dim) + 4 * Int(out_dim)
    elseif spec.scheme === :int4
        iseven(in_dim) || throw(ArgumentError(
            "INT4 input dimension $in_dim must be even for nibble packing",
        ))
        in_dim % spec.group == 0 || throw(ArgumentError(
            "INT4 input dimension $in_dim is not divisible by group $(spec.group)",
        ))
        return Int(out_dim) * (Int(in_dim) ÷ 2) +
            4 * Int(out_dim) * (Int(in_dim) ÷ spec.group)
    end
    throw(ArgumentError("unsupported quantization scheme $(repr(spec.scheme))"))
end

function _estimate_qwen3_quantized_bytes(
    vocab_size::Integer,
    d_model::Integer,
    hidden_dim::Integer,
    num_layers::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
    head_dim::Integer,
    tie_embeddings::Bool,
    plan::QuantizationPlan,
)
    q_dim = Int(num_heads) * Int(head_dim)
    kv_dim = Int(num_kv_heads) * Int(head_dim)

    # BF16 token embedding, final norm, and per-block input/post-attention
    # norms plus Q/K norm scales.
    bytes = 2 * Int(vocab_size) * Int(d_model)
    bytes += 2 * Int(d_model)
    bytes += Int(num_layers) * 2 * (
        2 * Int(d_model) + 2 * Int(head_dim)
    )

    for layer in 1:Int(num_layers)
        bytes += _linear_quantized_bytes(
            q_dim,
            d_model,
            quantization_spec(plan, :q_proj; layer),
        )
        bytes += _linear_quantized_bytes(
            kv_dim,
            d_model,
            quantization_spec(plan, :k_proj; layer),
        )
        bytes += _linear_quantized_bytes(
            kv_dim,
            d_model,
            quantization_spec(plan, :v_proj; layer),
        )
        bytes += _linear_quantized_bytes(
            d_model,
            q_dim,
            quantization_spec(plan, :o_proj; layer),
        )
        bytes += _linear_quantized_bytes(
            hidden_dim,
            d_model,
            quantization_spec(plan, :gate_proj; layer),
        )
        bytes += _linear_quantized_bytes(
            hidden_dim,
            d_model,
            quantization_spec(plan, :up_proj; layer),
        )
        bytes += _linear_quantized_bytes(
            d_model,
            hidden_dim,
            quantization_spec(plan, :down_proj; layer),
        )
    end
    if !tie_embeddings
        bytes += _linear_quantized_bytes(
            vocab_size,
            d_model,
            quantization_spec(plan, :lm_head),
        )
    end
    return bytes
end

"""
    estimate_qwen3_quantized_bytes(model_or_spec, plan)

Estimate tensor payload bytes without constructing quantized weights. Methods
accept a configured `GPTModel` or one frozen `Qwen3DenseSpec`. The estimate
uses the same one-based layer/projection resolution as the real loaders.
"""
function estimate_qwen3_quantized_bytes(
    model::GPTModel,
    plan::QuantizationPlan,
)
    _validate_quantization_plan_layers(plan, model.num_layers)
    return _estimate_qwen3_quantized_bytes(
        model.vocab_size,
        model.d_model,
        model.mlp_hidden_dim,
        model.num_layers,
        model.num_heads,
        model.num_kv_heads,
        model.head_dim,
        model.tie_embeddings,
        plan,
    )
end

function estimate_qwen3_quantized_bytes(
    spec::Qwen3DenseSpec,
    plan::QuantizationPlan,
)
    _validate_quantization_plan_layers(plan, spec.num_layers)
    return _estimate_qwen3_quantized_bytes(
        spec.vocab_size,
        spec.d_model,
        spec.mlp_hidden_dim,
        spec.num_layers,
        spec.num_heads,
        spec.num_kv_heads,
        spec.head_dim,
        spec.tie_embeddings,
        plan,
    )
end
