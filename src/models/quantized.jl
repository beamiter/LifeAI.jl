using BFloat16s: BFloat16
import Adapt

# Week 16 introduced RTN weight-only quantization. Week 17 added deterministic
# reconstruction-MSE scale calibration and one shared, fine-grained plan.
# Week 18 adds diagonal activation-aware calibration from streamed native-BF16
# input second moments. Embeddings and norm scales stay BF16. Compute is
# unchanged: weights are dequantized back to BF16 right before the existing
# `_bf16a_linear`, so calibration only changes stored scales, never the runtime
# compute contract.

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
For INT4, `calibration=:maxabs` preserves the Week 16 RTN behavior, `:mse`
minimizes weight reconstruction squared error, and `:activation_mse`
minimizes the same error weighted by calibration-input second moments.
Both calibrated modes search the frozen `clip_ratios`.
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
            calibration in (:maxabs, :mse, :activation_mse) || throw(ArgumentError(
                "unsupported INT4 calibration $(repr(calibration))",
            ))
        elseif calibration !== :maxabs
            throw(ArgumentError(
                "calibration $(repr(calibration)) is only supported for INT4",
            ))
        end

        ratios = if scheme === :int4 && calibration in (:mse, :activation_mse)
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

"""
    ActivationCalibration(
        layer_moments;
        lm_head_moment=nothing,
        token_count,
        num_layers,
        source="",
    )

Per-input-channel second moments from a separate calibration token set.
`layer_moments` keys are one-based `(layer, projection)` tuples for Qwen3
attention/MLP projections. An untied LM head uses `lm_head_moment`.

The object contains calibration-time statistics only. It guides INT4 scale
selection on the host and is not part of the deployed parameter tree.
"""
struct ActivationCalibration
    layer_moments::Dict{Tuple{Int,Symbol},Vector{Float32}}
    lm_head_moment::Union{Nothing,Vector{Float32}}
    token_count::Int
    num_layers::Int
    source::String
end

function _validated_activation_moment(values, target)
    values isa AbstractVector || throw(ArgumentError(
        "activation second moment for $target must be a vector",
    ))
    moment = Float32.(collect(values))
    isempty(moment) && throw(ArgumentError(
        "activation second moment for $target must not be empty",
    ))
    all(value -> isfinite(value) && value >= 0.0f0, moment) ||
        throw(ArgumentError(
            "activation second moment for $target must be finite and non-negative",
        ))
    any(>(0.0f0), moment) || throw(ArgumentError(
        "activation second moment for $target must contain positive mass",
    ))
    return moment
end

function ActivationCalibration(
    layer_moments;
    lm_head_moment=nothing,
    token_count::Integer,
    num_layers::Integer,
    source::AbstractString="",
)
    token_count > 0 || throw(ArgumentError(
        "activation calibration token_count must be positive",
    ))
    num_layers > 0 || throw(ArgumentError(
        "activation calibration num_layers must be positive",
    ))
    moments = Dict{Tuple{Int,Symbol},Vector{Float32}}()
    for (target, values) in pairs(layer_moments)
        target isa Tuple && length(target) == 2 || throw(ArgumentError(
            "activation calibration keys must be `(layer, projection)` tuples",
        ))
        layer, projection = target
        layer isa Integer && 1 <= layer <= num_layers || throw(ArgumentError(
            "activation calibration layer must be in 1:$num_layers",
        ))
        projection isa Symbol &&
            projection in _QWEN3_QUANTIZATION_TARGETS &&
            projection !== :lm_head || throw(ArgumentError(
                "unsupported activation calibration projection $(repr(projection))",
            ))
        key = (Int(layer), projection)
        haskey(moments, key) && throw(ArgumentError(
            "duplicate activation calibration target $key",
        ))
        moments[key] = _validated_activation_moment(values, key)
    end
    head_moment = lm_head_moment === nothing ?
        nothing : _validated_activation_moment(lm_head_moment, :lm_head)
    return ActivationCalibration(
        moments,
        head_moment,
        Int(token_count),
        Int(num_layers),
        String(source),
    )
end

"""
    activation_second_moment(calibration, projection; layer=nothing)

Resolve one calibration vector. Transformer projections require a one-based
`layer`; `:lm_head` requires `layer=nothing`. Missing statistics fail closed.
"""
function activation_second_moment(
    calibration::ActivationCalibration,
    projection::Symbol;
    layer::Union{Nothing,Integer}=nothing,
)
    projection in _QWEN3_QUANTIZATION_TARGETS || throw(ArgumentError(
        "unsupported activation calibration projection $(repr(projection))",
    ))
    if projection === :lm_head
        layer === nothing || throw(ArgumentError(
            "lm_head does not belong to a transformer layer",
        ))
        calibration.lm_head_moment === nothing && throw(ArgumentError(
            "activation calibration is missing lm_head statistics",
        ))
        return calibration.lm_head_moment
    end
    layer isa Integer && 1 <= layer <= calibration.num_layers ||
        throw(ArgumentError(
            "activation calibration layer must be in 1:$(calibration.num_layers)",
        ))
    target = (Int(layer), projection)
    haskey(calibration.layer_moments, target) || throw(ArgumentError(
        "activation calibration is missing target $target",
    ))
    return calibration.layer_moments[target]
end

_bf16_calibration_parameters(array::AbstractArray) = BFloat16.(array)
_bf16_calibration_parameters(values::NamedTuple) =
    NamedTuple{keys(values)}(Tuple(
        _bf16_calibration_parameters(value) for value in Base.values(values)
    ))
_bf16_calibration_parameters(values::Tuple) =
    Tuple(_bf16_calibration_parameters(value) for value in values)

"""
    calibrate_hf_qwen3_activations(
        model_dir,
        tokens;
        max_seq_len=max(64, size(tokens, 1)),
        variant=nothing,
        source="",
        accelerated=false,
        to_device=identity,
        to_host=identity,
    )

Stream a local Qwen3 checkpoint layer by layer and collect per-input-channel
second moments with the native BF16 operator contract. `tokens` uses LifeAI's
one-based ids and may contain multiple equal-length calibration sequences as
columns. The full BF16 parameter tree is never resident. With
`accelerated=true`, each single-layer BF16 parameter tree and the hidden state
are passed through `to_device`, while collected moment vectors are returned
through `to_host`; this permits CUDA calibration without a package-level CUDA
dependency.
"""
function calibrate_hf_qwen3_activations(
    model_dir::AbstractString,
    tokens::AbstractMatrix{<:Integer};
    max_seq_len::Integer=max(64, size(tokens, 1)),
    variant=nothing,
    source::AbstractString="",
    accelerated::Bool=false,
    to_device=identity,
    to_host=identity,
)
    isdir(model_dir) || throw(ArgumentError(
        "model directory does not exist: $model_dir",
    ))
    size(tokens, 1) > 0 && size(tokens, 2) > 0 || throw(ArgumentError(
        "activation calibration tokens must be a non-empty matrix",
    ))
    size(tokens, 1) <= max_seq_len || throw(ArgumentError(
        "activation calibration sequence exceeds max_seq_len",
    ))
    config = load_hf_qwen3_config(
        joinpath(model_dir, "config.json");
        max_seq_len,
        variant,
    )
    model = GPTModel(config)
    _qwen3_validate_semantics(model)
    token_matrix = Int.(collect(tokens))
    _validate_generation_ids(token_matrix, model.vocab_size)

    reader = open_safetensors_reader(model_dir)
    _qwen3_validate_tensor_names(model, Set(String.(collect(keys(reader)))))
    tensors = _StreamedTensors(reader)
    x = BFloat16.(_read_embedding_rows(
        reader,
        "model.embed_tokens.weight",
        token_matrix,
        model.d_model,
        model.vocab_size,
    ))
    cos_table, sin_table = _bf16_rope_tables(model)
    if accelerated
        x = to_device(x)
        cos_table = to_device(cos_table)
        sin_table = to_device(sin_table)
    end
    positions = 1:size(token_matrix, 1)
    cos_slice = cos_table[:, positions]
    sin_slice = sin_table[:, positions]
    mask = accelerated ?
        to_device(_bf16a_causal_mask(size(token_matrix, 1), size(token_matrix, 1))) :
        nothing
    layer_moments = Dict{Tuple{Int,Symbol},Vector{Float32}}()

    for layer in 1:model.num_layers
        block_f32 = _qwen3_block_parameters(model, tensors, layer - 1)
        block_bf16 = _bf16_calibration_parameters(block_f32)
        if accelerated
            block_bf16 = to_device(block_bf16)
            x, _, moments = _bf16a_block_activation_moments(
                model,
                block_bf16,
                x,
                cos_slice,
                sin_slice;
                mask,
            )
        else
            x, _, moments = _bf16_block_activation_moments(
                model,
                block_bf16,
                x,
                cos_table,
                sin_table;
                start_pos=1,
            )
        end
        for (projection, moment) in pairs(moments)
            layer_moments[(layer, projection)] =
                Float32.(collect(to_host(moment)))
        end
        block_f32 = nothing
        block_bf16 = nothing
        GC.gc(false)
    end

    final_scale = BFloat16.(reshape(_expect_tensor(
        tensors,
        "model.norm.weight",
        (model.d_model,),
    ), model.d_model, 1, 1))
    final_hidden = accelerated ?
        _bf16a_rmsnorm(x, to_device(final_scale), model.norm_epsilon) :
        _bf16_rmsnorm(x, final_scale, model.norm_epsilon)
    lm_head_moment = model.tie_embeddings ?
        nothing : Float32.(collect(to_host(
            accelerated ?
                _bf16a_input_second_moment(final_hidden) :
                _bf16_input_second_moment(final_hidden),
        )))
    resolved_source = isempty(source) ? abspath(model_dir) : String(source)
    calibration = ActivationCalibration(
        layer_moments;
        lm_head_moment,
        token_count=length(token_matrix),
        num_layers=model.num_layers,
        source=resolved_source,
    )
    dense_spec = config.qwen3_variant === nothing ?
        nothing : qwen3_dense_spec(config.qwen3_variant)
    return (; calibration, model, config, variant=dense_spec)
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
    grouped_activation_moment,
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
                importance = grouped_activation_moment === nothing ?
                    1.0f0 : grouped_activation_moment[column, group_index]
                candidate_error += importance * abs2(reconstructed - value)
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

function _grouped_activation_second_moment(
    values,
    in_dim::Int,
    group::Int,
)
    values isa AbstractVector || throw(ArgumentError(
        "activation-aware INT4 calibration requires a second-moment vector",
    ))
    length(values) == in_dim || throw(DimensionMismatch(
        "activation second moment has length $(length(values)); expected $in_dim",
    ))
    moment = Float32.(collect(values))
    all(value -> isfinite(value) && value >= 0.0f0, moment) ||
        throw(ArgumentError(
            "activation second moment must be finite and non-negative",
        ))
    groups = in_dim ÷ group
    grouped = reshape(moment, group, groups)
    for group_index in 1:groups
        any(>(0.0f0), view(grouped, :, group_index)) || throw(ArgumentError(
            "activation second moment group $group_index has no positive mass",
        ))
    end
    return grouped
end

function _quantize_int4_group(
    weight::AbstractMatrix;
    group::Int=128,
    calibration::Symbol=:maxabs,
    clip_ratios=_DEFAULT_INT4_MSE_CLIP_RATIOS,
    activation_second_moment=nothing,
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
    grouped_activation_moment = if spec.calibration === :activation_mse
        activation_second_moment === nothing && throw(ArgumentError(
            "activation-aware INT4 calibration requires activation statistics",
        ))
        _grouped_activation_second_moment(
            activation_second_moment,
            in_dim,
            group,
        )
    else
        activation_second_moment === nothing || throw(ArgumentError(
            "activation statistics are only valid with calibration=:activation_mse",
        ))
        nothing
    end
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
            grouped_activation_moment,
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

function _quantize_linear(
    weight,
    spec::LinearQuantizationSpec;
    activation_second_moment=nothing,
)
    if spec.calibration !== :activation_mse
        activation_second_moment === nothing || throw(ArgumentError(
            "activation statistics were provided for non-activation calibration",
        ))
    end
    spec.scheme === :bf16 && return BFloat16.(weight)
    spec.scheme === :int8 && return _quantize_int8_channel(weight)
    spec.scheme === :int4 && return _quantize_int4_group(
        weight;
        group=spec.group,
        calibration=spec.calibration,
        clip_ratios=spec.clip_ratios,
        activation_second_moment,
    )
    throw(ArgumentError("unsupported quantization scheme $(repr(spec.scheme))"))
end

_quantize_linear(weight, scheme::Symbol, group::Int) =
    _quantize_linear(weight, LinearQuantizationSpec(scheme; group))

function _activation_moment_for_quantization(
    calibration::Union{Nothing,ActivationCalibration},
    spec::LinearQuantizationSpec,
    projection::Symbol;
    layer::Union{Nothing,Integer}=nothing,
)
    spec.calibration === :activation_mse || return nothing
    calibration === nothing && throw(ArgumentError(
        "quantization plan requires an ActivationCalibration for " *
        "$(layer === nothing ? projection : (layer, projection))",
    ))
    return activation_second_moment(calibration, projection; layer)
end

function _validate_activation_calibration_depth(
    calibration::Union{Nothing,ActivationCalibration},
    num_layers::Integer,
)
    calibration === nothing && return nothing
    calibration.num_layers == num_layers || throw(ArgumentError(
        "activation calibration depth $(calibration.num_layers) does not " *
        "match model depth $num_layers",
    ))
    return calibration
end

function _validate_activation_calibration_usage(
    plan::QuantizationPlan,
    calibration::Union{Nothing,ActivationCalibration},
    num_layers::Integer;
    include_lm_head::Bool,
)
    _validate_activation_calibration_depth(calibration, num_layers)
    used = false
    for layer in 1:Int(num_layers), projection in _QWEN3_QUANTIZATION_TARGETS
        projection === :lm_head && continue
        spec = quantization_spec(plan, projection; layer)
        if spec.calibration === :activation_mse
            used = true
            _activation_moment_for_quantization(
                calibration,
                spec,
                projection;
                layer,
            )
        end
    end
    if include_lm_head
        head_spec = quantization_spec(plan, :lm_head)
        if head_spec.calibration === :activation_mse
            used = true
            _activation_moment_for_quantization(
                calibration,
                head_spec,
                :lm_head,
            )
        end
    end
    calibration !== nothing && !used && throw(ArgumentError(
        "ActivationCalibration was provided but the quantization plan does " *
        "not use calibration=:activation_mse",
    ))
    return calibration
end

"""
    load_hf_qwen3_quantized(
        model_dir;
        max_seq_len=64,
        scheme=:int8,
        group=128,
        plan=nothing,
        activation_calibration=nothing,
        variant=nothing,
    )

Stream a HuggingFace Qwen3 checkpoint from disk and quantize it layer by
layer, so the full BF16 tree is never resident: peak host memory is the
quantized tree plus one layer of transients. Embedding and norm scales stay
BF16. With `plan=nothing`, the Week 16 `scheme`, `group`, and
`int8_projections` API is preserved. A `QuantizationPlan` can instead choose
INT4, INT8, or BF16 independently for each layer/projection and the LM head.
Plans using `calibration=:activation_mse` additionally require an
`ActivationCalibration` with matching targets and input dimensions.
"""
function load_hf_qwen3_quantized(
    model_dir::AbstractString;
    max_seq_len=64,
    scheme::Symbol=:int8,
    group::Int=128,
    int8_projections::Tuple{Vararg{Symbol}}=(),
    plan::Union{Nothing,QuantizationPlan}=nothing,
    activation_calibration::Union{Nothing,ActivationCalibration}=nothing,
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
    _validate_activation_calibration_usage(
        quantization_plan,
        activation_calibration,
        model.num_layers,
        include_lm_head=!model.tie_embeddings,
    )
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
        spec = quantization_spec(quantization_plan, projection; layer)
        quantized = _quantize_linear(
            weight,
            spec;
            activation_second_moment=_activation_moment_for_quantization(
                activation_calibration,
                spec,
                projection;
                layer,
            ),
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
        head_spec = quantization_spec(quantization_plan, :lm_head)
        head = _quantize_linear(
            _expect_tensor(
                tensors,
                "lm_head.weight",
                (model.vocab_size, model.d_model),
            ),
            head_spec;
            activation_second_moment=_activation_moment_for_quantization(
                activation_calibration,
                head_spec,
                :lm_head,
            ),
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
        activation_calibration=nothing,
    )

Quantize every linear weight (attention/MLP projections and an untied LM
head) of a BF16 Qwen3 parameter tree with round-to-nearest. Embedding and
norm scales stay BF16. The result feeds `hf_qwen3_bf16_accel_forward`
unchanged and can be moved to the GPU with `CUDA.cu`. A `QuantizationPlan`
enables calibrated INT4 and fine-grained mixed precision. Plans using
`:activation_mse` require an `ActivationCalibration`.
"""
function quantize_bf16_parameters(
    ps;
    scheme::Symbol=:int8,
    group::Int=128,
    int8_projections::Tuple{Vararg{Symbol}}=(),
    plan::Union{Nothing,QuantizationPlan}=nothing,
    activation_calibration::Union{Nothing,ActivationCalibration}=nothing,
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
    has_untied_head = haskey(ps, :lm_head) && haskey(ps.lm_head, :weight)
    _validate_activation_calibration_usage(
        quantization_plan,
        activation_calibration,
        length(block_values),
        include_lm_head=has_untied_head,
    )
    function _quantized_weight(weight, projection; layer=nothing)
        spec = quantization_spec(quantization_plan, projection; layer)
        return _quantize_linear(
            weight,
            spec;
            activation_second_moment=_activation_moment_for_quantization(
                activation_calibration,
                spec,
                projection;
                layer,
            ),
        )
    end
    blocks = ntuple(length(block_values)) do index
        block = block_values[index]
        (;
            norm1=block.norm1,
            attn=(;
                q_proj=(; weight=_quantized_weight(
                    block.attn.q_proj.weight,
                    :q_proj;
                    layer=index,
                )),
                k_proj=(; weight=_quantized_weight(
                    block.attn.k_proj.weight,
                    :k_proj;
                    layer=index,
                )),
                v_proj=(; weight=_quantized_weight(
                    block.attn.v_proj.weight,
                    :v_proj;
                    layer=index,
                )),
                o_proj=(; weight=_quantized_weight(
                    block.attn.o_proj.weight,
                    :o_proj;
                    layer=index,
                )),
                q_norm=block.attn.q_norm,
                k_norm=block.attn.k_norm,
            ),
            norm2=block.norm2,
            mlp=(;
                gate_proj=(; weight=_quantized_weight(
                    block.mlp.gate_proj.weight,
                    :gate_proj;
                    layer=index,
                )),
                up_proj=(; weight=_quantized_weight(
                    block.mlp.up_proj.weight,
                    :up_proj;
                    layer=index,
                )),
                down_proj=(; weight=_quantized_weight(
                    block.mlp.down_proj.weight,
                    :down_proj;
                    layer=index,
                )),
            ),
        )
    end
    lm_head = if has_untied_head
        (; weight=_quantized_weight(
            ps.lm_head.weight,
            :lm_head,
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
