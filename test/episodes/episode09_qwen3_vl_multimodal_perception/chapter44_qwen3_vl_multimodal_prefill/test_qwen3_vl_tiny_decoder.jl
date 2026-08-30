using Base64: base64decode
using JSON3
using SHA: sha256
using Test
import LifeAI
import MLDataDevices
using LifeAI: Qwen3VLCheckpointSpec,
    Qwen3VLRopeLayout,
    Qwen3VLTextPrefill,
    Qwen3VLVisionInput,
    Qwen3VLTextSpec,
    hf_qwen3_vl_prefill,
    hf_qwen3_vl_text_prefill,
    qwen3_vl_checkpoint_spec

const _CH44_TINY_TEXT_SPEC = Qwen3VLTextSpec(
    32,             # vocab_size
    16,             # hidden_size
    32,             # intermediate_size
    4,              # num_hidden_layers
    2,              # num_attention_heads
    1,              # num_key_value_heads
    8,              # head_dim
    1.0e-6,         # rms_norm_eps
    10_000.0,       # rope_theta
    64,             # max_position_embeddings
    true,           # mrope_interleaved
    (2, 1, 1),      # mrope_section
    true,           # tie_word_embeddings
    "silu",
)

struct _Ch44ForeignDevice <: MLDataDevices.AbstractDevice end

struct _Ch44ForeignDeviceMatrix{T,A<:AbstractMatrix{T}} <: AbstractMatrix{T}
    data::A
end

Base.size(values::_Ch44ForeignDeviceMatrix) = size(values.data)
Base.axes(values::_Ch44ForeignDeviceMatrix) = axes(values.data)
Base.IndexStyle(::Type{<:_Ch44ForeignDeviceMatrix}) = IndexCartesian()
Base.getindex(values::_Ch44ForeignDeviceMatrix, indices...) =
    getindex(values.data, indices...)
MLDataDevices.get_device(::_Ch44ForeignDeviceMatrix) = _Ch44ForeignDevice()

function _ch44_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected the test call to fail")
end

function _ch44_tiny_values(count::Int, offset::Int; scale=0.02f0)
    return Float32[
        scale * sin(0.173f0 * Float32(offset + index))
        for index in 1:count
    ]
end

# Construct the mathematical matrix emitted by PyTorch's row-major
# `vals((rows, columns), offset)` fixture.
function _ch44_tiny_hf_matrix(rows::Int, columns::Int, offset::Int; scale=0.02f0)
    values = _ch44_tiny_values(rows * columns, offset; scale)
    return permutedims(reshape(values, columns, rows))
end

function _ch44_tiny_text_parameters()
    spec = _CH44_TINY_TEXT_SPEC
    embedding = permutedims(_ch44_tiny_hf_matrix(32, 16, 10))
    blocks = ntuple(spec.num_hidden_layers) do julia_layer
        offset = 10_000 * julia_layer
        return (;
            norm1=1.0f0 .+ _ch44_tiny_values(16, offset; scale=0.01f0),
            q_weight=_ch44_tiny_hf_matrix(16, 16, offset + 100),
            k_weight=_ch44_tiny_hf_matrix(8, 16, offset + 200),
            v_weight=_ch44_tiny_hf_matrix(8, 16, offset + 300),
            o_weight=_ch44_tiny_hf_matrix(16, 16, offset + 400),
            q_norm=1.0f0 .+ _ch44_tiny_values(8, offset + 500; scale=0.01f0),
            k_norm=1.0f0 .+ _ch44_tiny_values(8, offset + 600; scale=0.01f0),
            norm2=1.0f0 .+ _ch44_tiny_values(16, offset + 700; scale=0.01f0),
            gate_weight=_ch44_tiny_hf_matrix(32, 16, offset + 800),
            up_weight=_ch44_tiny_hf_matrix(32, 16, offset + 900),
            down_weight=_ch44_tiny_hf_matrix(16, 32, offset + 1_000),
        )
    end
    final_norm = 1.0f0 .+
        _ch44_tiny_values(16, 90_000; scale=0.01f0)
    return (; embedding, blocks, final_norm, spec)
end

function _ch44_tiny_prefill_inputs()
    position_ids = reshape(Int[
        0 1 2 2 2 2 4 5
        0 1 2 2 3 3 4 5
        0 1 2 3 2 3 4 5
    ], 3, 8, 1)
    visual_mask = falses(8, 1)
    visual_mask[3:6, 1] .= true
    # This explicit all-ones mask is part of the oracle. Passing `nothing` to
    # Transformers 4.57 takes its packed-sequence branch for this non-monotonic
    # temporal axis and freezes a different attention contract.
    attention_mask = trues(8, 1)
    rope_layout = Qwen3VLRopeLayout(
        position_ids,
        reshape(Int[-2], 1, 1),
        visual_mask,
        attention_mask,
    )
    visual_embeddings = permutedims(
        _ch44_tiny_hf_matrix(4, 16, 100_000; scale=0.1f0),
    )
    deepstack = ntuple(3) do index
        permutedims(_ch44_tiny_hf_matrix(
            4,
            16,
            110_000 + 1_000 * (index - 1);
            scale=0.1f0,
        ))
    end
    return (;
        input_ids=collect(1:8),
        rope_layout,
        vision_features=(; visual_embeddings, deepstack),
    )
end

function _ch44_hf_reference()
    path = joinpath(@__DIR__, "fixtures", "tiny_text_prefill_all_ones.json")
    return JSON3.read(read(path, String))
end

function _ch44_hf_tensor(reference, name::AbstractString)
    entry = reference.tensors[Symbol(name)]
    bytes = base64decode(String(entry.f32_le_base64))
    @test bytes2hex(sha256(bytes)) == String(entry.sha256)
    shape = Int.(collect(entry.shape))
    @test shape[1:2] == [1, 8]
    width = shape[3]
    values = collect(reinterpret(Float32, bytes))
    return reshape(values, width, 8, 1)
end

@testset "Chapter 44 — mRoPE section arithmetic is overflow-safe" begin
    names = fieldnames(Qwen3VLTextSpec)
    fields = ntuple(length(names)) do index
        names[index] === :mrope_section && return (
            typemax(Int),
            typemax(Int),
            6,
        )
        return getfield(_CH44_TINY_TEXT_SPEC, names[index])
    end
    overflow_spec = Qwen3VLTextSpec(fields...)
    # Ordinary Int addition wraps this tuple sum back to head_dim ÷ 2 == 4.
    @test sum(overflow_spec.mrope_section) == overflow_spec.head_dim ÷ 2
    @test sum(BigInt, overflow_spec.mrope_section) !=
        BigInt(overflow_spec.head_dim ÷ 2)
    @test_throws ArgumentError LifeAI._qwen3_vl_text_mrope(
        overflow_spec,
        zeros(Float32, 1),
        reshape(Int[0, 0, 0], 3, 1, 1),
    )

    cosine, sine = LifeAI._qwen3_vl_text_mrope(
        _CH44_TINY_TEXT_SPEC,
        zeros(Float32, 1),
        reshape(Int[1, 2, 3], 3, 1, 1),
    )
    @test size(cosine) == (4, 1, 1)
    @test size(sine) == (4, 1, 1)
    @test all(isfinite, cosine)
    @test all(isfinite, sine)
end

@testset "Chapter 44 — RoPE layout construction is strict" begin
    positions = reshape(Int32[0, 1, 0, 1, 0, 1], 3, 2, 1)
    deltas = reshape(Int16[0], 1, 1)
    visual = falses(2, 1)
    attention = trues(2, 1)
    layout = Qwen3VLRopeLayout(positions, deltas, visual, attention)
    @test layout.position_ids === positions
    @test layout.rope_deltas === deltas
    @test layout.visual_mask === visual
    @test layout.attention_mask === attention

    @test_throws ArgumentError Qwen3VLRopeLayout(1, 2, 3, 4)
    @test_throws DimensionMismatch Qwen3VLRopeLayout(
        zeros(Int, 3, 2),
        deltas,
        visual,
        attention,
    )
    @test_throws DimensionMismatch Qwen3VLRopeLayout(
        zeros(Int, 2, 2, 1),
        deltas,
        visual,
        attention,
    )
    @test_throws DimensionMismatch Qwen3VLRopeLayout(
        positions,
        Int[0],
        visual,
        attention,
    )
    @test_throws DimensionMismatch Qwen3VLRopeLayout(
        positions,
        deltas,
        falses(1, 1),
        trues(1, 1),
    )
    @test_throws MethodError Qwen3VLRopeLayout{
        typeof(positions),
        typeof(deltas),
        typeof(visual),
        typeof(attention),
    }(positions, deltas, visual, attention)
end

@testset "Chapter 44 — text prefill construction binds capture geometry" begin
    rope_layout = _ch44_tiny_prefill_inputs().rope_layout
    input_embeddings = zeros(Float32, 16, 8, 1)
    final_hidden = zeros(Float32, 16, 8, 1)
    last_state = zeros(Float32, 16, 8, 1)
    block_outputs = Dict{Int,Any}(
        0 => zeros(Float32, 16, 8, 1),
        3 => last_state,
    )
    layer_outputs = Dict{Int,Any}(
        0 => zeros(Float32, 16, 8, 1),
        3 => last_state,
    )
    logits = zeros(Float32, 32, 1, 1)
    prefill = Qwen3VLTextPrefill(
        input_embeddings,
        block_outputs,
        layer_outputs,
        final_hidden,
        logits,
        rope_layout,
    )
    @test prefill.input_embeddings === input_embeddings
    @test prefill.block_outputs === block_outputs
    @test prefill.layer_outputs === layer_outputs
    @test prefill.final_hidden === final_hidden
    @test prefill.logits === logits
    @test prefill.rope_layout === rope_layout
    @test prefill.block_outputs[3] === prefill.layer_outputs[3]

    @test Qwen3VLTextPrefill(
        input_embeddings,
        Dict{Int,Any}(),
        Dict{Int,Any}(),
        nothing,
        logits,
        rope_layout,
    ).input_embeddings === input_embeddings
    @test Qwen3VLTextPrefill(
        nothing,
        Dict{Int,Any}(),
        Dict{Int,Any}(),
        final_hidden,
        logits,
        rope_layout,
    ).final_hidden === final_hidden

    @test_throws MethodError Qwen3VLTextPrefill{
        typeof(input_embeddings),
        typeof(block_outputs),
        typeof(layer_outputs),
        typeof(final_hidden),
        typeof(logits),
        typeof(rope_layout),
    }(
        input_embeddings,
        block_outputs,
        layer_outputs,
        final_hidden,
        logits,
        rope_layout,
    )
    @test_throws DimensionMismatch Qwen3VLTextPrefill(
        input_embeddings,
        block_outputs,
        layer_outputs,
        final_hidden,
        zeros(Float32, 32, 1),
        rope_layout,
    )
    @test_throws DimensionMismatch Qwen3VLTextPrefill(
        input_embeddings,
        block_outputs,
        layer_outputs,
        final_hidden,
        zeros(Float32, 32, 9, 1),
        rope_layout,
    )
    @test_throws DimensionMismatch Qwen3VLTextPrefill(
        input_embeddings,
        block_outputs,
        layer_outputs,
        final_hidden,
        zeros(Float32, 32, 1, 2),
        rope_layout,
    )
    @test_throws DimensionMismatch Qwen3VLTextPrefill(
        zeros(Float32, 16, 7, 1),
        block_outputs,
        layer_outputs,
        final_hidden,
        logits,
        rope_layout,
    )
    @test_throws DimensionMismatch Qwen3VLTextPrefill(
        input_embeddings,
        block_outputs,
        layer_outputs,
        zeros(Float32, 15, 8, 1),
        logits,
        rope_layout,
    )
    @test_throws DimensionMismatch Qwen3VLTextPrefill(
        input_embeddings,
        block_outputs,
        Dict{Int,Any}(0 => last_state, 2 => last_state),
        final_hidden,
        logits,
        rope_layout,
    )
    for bad_key in (-1, true)
        captures = Dict{Any,Any}(bad_key => last_state)
        @test_throws ArgumentError Qwen3VLTextPrefill(
            input_embeddings,
            captures,
            captures,
            final_hidden,
            logits,
            rope_layout,
        )
    end
    @test_throws DimensionMismatch Qwen3VLTextPrefill(
        input_embeddings,
        Dict{Int,Any}(0 => zeros(Float32, 16, 7, 1)),
        Dict{Int,Any}(0 => zeros(Float32, 16, 7, 1)),
        final_hidden,
        logits,
        rope_layout,
    )
    @test_throws ArgumentError Qwen3VLTextPrefill(
        input_embeddings,
        Dict{Int,Any}(0 => zeros(Core.BFloat16, 16, 8, 1)),
        Dict{Int,Any}(0 => zeros(Core.BFloat16, 16, 8, 1)),
        final_hidden,
        logits,
        rope_layout,
    )
end

@testset "Chapter 44 — deterministic tiny Float32 decoder HF parity" begin
    reference = _ch44_hf_reference()
    @test String(reference.metadata.transformers) == "4.57.0"
    @test String(reference.metadata.torch) == "2.7.1+cpu"
    @test String(reference.metadata.attention_implementation) == "eager"
    @test String(reference.metadata.attention_mask) == "all_ones_i64_1x8"

    parameters = _ch44_tiny_text_parameters()
    inputs = _ch44_tiny_prefill_inputs()
    result = hf_qwen3_vl_text_prefill(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        logits_to_keep=Int128(0),
        capture_layers=(Int128(0), big(1), Int16(2), UInt8(3)),
        capture_input_embeddings=true,
        max_prefill_tokens=big(8),
    )

    @test size(result.input_embeddings) == (16, 8, 1)
    @test size(result.final_hidden) == (16, 8, 1)
    @test size(result.logits) == (32, 8, 1)
    @test Set(keys(result.block_outputs)) == Set(0:3)
    @test Set(keys(result.layer_outputs)) == Set(0:3)
    @test all(layer -> size(result.block_outputs[layer]) == (16, 8, 1), 0:3)
    @test all(layer -> size(result.layer_outputs[layer]) == (16, 8, 1), 0:3)

    # Full frozen tensors catch attention, mRoPE lane selection, residual order,
    # RMSNorm, SwiGLU, DeepStack placement, final norm, and tied projection.
    tolerance = 1.0f-6
    @test result.input_embeddings ≈
        _ch44_hf_tensor(reference, "input_embeddings") atol=tolerance rtol=tolerance
    for layer in 0:3
        @test result.block_outputs[layer] ≈
            _ch44_hf_tensor(reference, "block_$layer") atol=tolerance rtol=tolerance
        @test result.layer_outputs[layer] ≈
            _ch44_hf_tensor(reference, "layer_$layer") atol=tolerance rtol=tolerance
    end
    @test result.final_hidden ≈
        _ch44_hf_tensor(reference, "final_hidden") atol=tolerance rtol=tolerance
    @test result.logits ≈
        _ch44_hf_tensor(reference, "logits") atol=tolerance rtol=tolerance

    # Main vision features replace only the four image-pad embeddings.
    @test result.input_embeddings[:, 3:6, 1] ==
        inputs.vision_features.visual_embeddings
    text_positions = [1, 2, 7, 8]
    @test result.input_embeddings[:, text_positions, 1] ==
        parameters.embedding[:, inputs.input_ids[text_positions]]

    # DeepStack is added after decoder layers 0, 1, and 2, at visual positions
    # only. Layer 3 has no post-layer injection.
    for layer in 0:2
        difference = result.layer_outputs[layer] .- result.block_outputs[layer]
        @test difference[:, 3:6, 1] ≈
            inputs.vision_features.deepstack[layer + 1] atol=2.0f-7 rtol=2.0f-6
        @test all(iszero, difference[:, text_positions, 1])
    end
    @test result.layer_outputs[3] == result.block_outputs[3]
    @test result.layer_outputs[3] === result.block_outputs[3]
    @test result.rope_layout === inputs.rope_layout

    predicted = [argmax(view(result.logits, :, token, 1)) for token in 1:8]
    @test predicted == [1, 2, 13, 14, 15, 7, 32, 8]
end

@testset "Chapter 44 — text-consumed vision residency is preflighted" begin
    parameters = _ch44_tiny_text_parameters()
    inputs = _ch44_tiny_prefill_inputs()
    prefill_error = function (vision_features)
        return _ch44_captured_error(() -> hf_qwen3_vl_text_prefill(
            parameters,
            inputs.input_ids,
            inputs.rope_layout;
            vision_features,
        ))
    end

    dtype_visual = merge(inputs.vision_features, (
        visual_embeddings=Float64.(
            inputs.vision_features.visual_embeddings,
        ),
    ))
    error = prefill_error(dtype_visual)
    @test error isa ArgumentError
    @test sprint(showerror, error) ==
        "ArgumentError: Qwen3-VL visual_embeddings dtype must match " *
        "text parameter embedding"

    dtype_deepstack = merge(inputs.vision_features, (
        deepstack=Base.setindex(
            inputs.vision_features.deepstack,
            Float64.(inputs.vision_features.deepstack[1]),
            1,
        ),
    ))
    error = prefill_error(dtype_deepstack)
    @test error isa ArgumentError
    @test sprint(showerror, error) ==
        "ArgumentError: Qwen3-VL deepstack[1] dtype must match " *
        "text parameter embedding"

    foreign_visual = merge(inputs.vision_features, (
        visual_embeddings=_Ch44ForeignDeviceMatrix(
            inputs.vision_features.visual_embeddings,
        ),
    ))
    error = prefill_error(foreign_visual)
    @test error isa ArgumentError
    @test sprint(showerror, error) ==
        "ArgumentError: Qwen3-VL visual_embeddings device must match " *
        "text parameter embedding"

    foreign_deepstack = merge(inputs.vision_features, (
        deepstack=Base.setindex(
            inputs.vision_features.deepstack,
            _Ch44ForeignDeviceMatrix(inputs.vision_features.deepstack[1]),
            1,
        ),
    ))
    error = prefill_error(foreign_deepstack)
    @test error isa ArgumentError
    @test sprint(showerror, error) ==
        "ArgumentError: Qwen3-VL deepstack[1] device must match " *
        "text parameter embedding"

    # The long-standing DeepStack arity error remains ahead of residency.
    short_deepstack = (
        inputs.vision_features.deepstack[1],
        inputs.vision_features.deepstack[2],
    )
    error = prefill_error((;
        visual_embeddings=foreign_visual.visual_embeddings,
        deepstack=short_deepstack,
    ))
    @test error isa DimensionMismatch
    @test sprint(showerror, error) ==
        "DimensionMismatch: Qwen3-VL prefill requires exactly three " *
        "DeepStack features"

    # Shallow toy decoders retain the three-item public contract, but feature
    # residency matters only for the DeepStack entries their layers consume.
    for num_hidden_layers in 0:2
        shallow_parameters = (;
            embedding=parameters.embedding,
            spec=(; num_hidden_layers),
        )
        shallow_deepstack = ntuple(3) do index
            index <= num_hidden_layers &&
                return inputs.vision_features.deepstack[index]
            return _Ch44ForeignDeviceMatrix(Float64.(
                inputs.vision_features.deepstack[index],
            ))
        end
        shallow_features = merge(
            inputs.vision_features,
            (; deepstack=shallow_deepstack),
        )
        @test isnothing(LifeAI._validate_qwen3_vl_text_feature_residency(
            shallow_parameters,
            shallow_features,
        ))
    end
end

@testset "Chapter 44 — cache-free prompt layout contract" begin
    parameters = _ch44_tiny_text_parameters()
    inputs = _ch44_tiny_prefill_inputs()
    call_prefill = function (rope_layout; input_ids=inputs.input_ids, kwargs...)
        return hf_qwen3_vl_text_prefill(
            parameters,
            input_ids,
            rope_layout;
            vision_features=inputs.vision_features,
            logits_to_keep=1,
            kwargs...,
        )
    end

    overflow_integer = big(typemax(Int)) + 1
    option_cases = (
        (needle="logits_to_keep", options=(; logits_to_keep=true)),
        (
            needle="logits_to_keep",
            options=(; logits_to_keep=overflow_integer),
        ),
        (needle="logits_to_keep", options=(; logits_to_keep=-1)),
        (needle="max_prefill_tokens", options=(; max_prefill_tokens=true)),
        (
            needle="max_prefill_tokens",
            options=(; max_prefill_tokens=overflow_integer),
        ),
        (needle="max_prefill_tokens", options=(; max_prefill_tokens=0)),
    )
    vision_input = Qwen3VLVisionInput(
        zeros(Float32, 1_536, 4),
        reshape(Int[1, 2, 2], 3, 1),
    )
    base_checkpoint = qwen3_vl_checkpoint_spec()
    custom_checkpoint = Qwen3VLCheckpointSpec(
        base_checkpoint.variant,
        base_checkpoint.model_id,
        base_checkpoint.modelscope_revision,
        base_checkpoint.hf_revision,
        base_checkpoint.assets,
        base_checkpoint.tensor_count,
        base_checkpoint.tensor_bytes,
        base_checkpoint.parameter_count,
        base_checkpoint.image_token_id,
        6,
        base_checkpoint.vision_start_token_id,
        base_checkpoint.vision_end_token_id,
        base_checkpoint.bos_token_id,
        base_checkpoint.eos_token_id,
        base_checkpoint.text,
        base_checkpoint.vision,
    )
    checkpoint_error = try
        hf_qwen3_vl_prefill(
            42,
            (; spec=custom_checkpoint.text, checkpoint=custom_checkpoint),
            vision_input,
            Int[7],
        )
        nothing
    catch caught
        caught
    end
    @test checkpoint_error isa ArgumentError
    @test occursin(
        "video mRoPE requires the timestamp video processor",
        sprint(showerror, checkpoint_error),
    )

    for case in option_cases
        text_error = try
            hf_qwen3_vl_text_prefill(
                42,
                42,
                inputs.rope_layout;
                case.options...,
            )
            nothing
        catch caught
            caught
        end
        @test text_error isa ArgumentError
        @test occursin(case.needle, sprint(showerror, text_error))

        combined_error = try
            hf_qwen3_vl_prefill(
                42,
                42,
                vision_input,
                42;
                rope_layout=inputs.rope_layout,
                case.options...,
            )
            nothing
        catch caught
            caught
        end
        @test combined_error isa ArgumentError
        @test occursin(case.needle, sprint(showerror, combined_error))
    end

    capture_cases = (
        (value=(true,), message="capture layer must be an integer"),
        (value=(1.0,), message="capture layer must be an integer"),
        (
            value=(overflow_integer,),
            message="capture layer is outside the host integer range",
        ),
        (value=(-1,), message="capture layer is outside the decoder"),
        (
            value=(parameters.spec.num_hidden_layers,),
            message="capture layer is outside the decoder",
        ),
    )
    for case in capture_cases
        text_error = try
            call_prefill(
                inputs.rope_layout;
                capture_layers=case.value,
            )
            nothing
        catch caught
            caught
        end
        @test text_error isa ArgumentError
        @test text_error isa Exception &&
            occursin(case.message, sprint(showerror, text_error))

        combined_error = try
            hf_qwen3_vl_prefill(
                42,
                parameters,
                vision_input,
                inputs.input_ids;
                rope_layout=inputs.rope_layout,
                capture_layers=case.value,
            )
            nothing
        catch caught
            caught
        end
        @test combined_error isa ArgumentError
        @test combined_error isa Exception &&
            occursin(case.message, sprint(showerror, combined_error))
    end

    text_capture_preflight = try
        hf_qwen3_vl_text_prefill(
            42,
            42,
            inputs.rope_layout;
            capture_layers=(true,),
        )
        nothing
    catch caught
        caught
    end
    @test text_capture_preflight isa ArgumentError
    @test text_capture_preflight isa Exception && occursin(
        "capture layer must be an integer",
        sprint(showerror, text_capture_preflight),
    )

    combined_capture_preflight = try
        hf_qwen3_vl_prefill(
            42,
            42,
            vision_input,
            42;
            rope_layout=inputs.rope_layout,
            capture_layers=(true,),
        )
        nothing
    catch caught
        caught
    end
    @test combined_capture_preflight isa ArgumentError
    @test combined_capture_preflight isa Exception && occursin(
        "capture layer must be an integer",
        sprint(showerror, combined_capture_preflight),
    )

    @test_throws ArgumentError call_prefill(
        inputs.rope_layout;
        max_prefill_tokens=0,
    )
    @test_throws ArgumentError call_prefill(
        inputs.rope_layout;
        max_prefill_tokens=length(inputs.input_ids) - 1,
    )
    unlimited = call_prefill(
        inputs.rope_layout;
        max_prefill_tokens=typemax(Int),
    )
    @test size(unlimited.logits) == (parameters.spec.vocab_size, 1, 1)

    wide_layout = Qwen3VLRopeLayout(
        UInt128.(inputs.rope_layout.position_ids),
        Int128.(inputs.rope_layout.rope_deltas),
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    )
    wide = call_prefill(wide_layout)
    @test wide.final_hidden == unlimited.final_hidden
    @test wide.logits == unlimited.logits

    negative_positions = copy(inputs.rope_layout.position_ids)
    negative_positions[1, 1, 1] = -1
    @test_throws ArgumentError call_prefill(Qwen3VLRopeLayout(
        negative_positions,
        inputs.rope_layout.rope_deltas,
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    ))

    context_positions = copy(inputs.rope_layout.position_ids)
    context_positions[1, 1, 1] = parameters.spec.max_position_embeddings
    @test_throws ArgumentError call_prefill(Qwen3VLRopeLayout(
        context_positions,
        inputs.rope_layout.rope_deltas,
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    ))
    invalid_integer_layouts = (
        (
            layout=() -> Qwen3VLRopeLayout(
                Bool.(inputs.rope_layout.position_ids .> 0),
                reshape(Int[-6], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL position_ids must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                Float32.(inputs.rope_layout.position_ids),
                inputs.rope_layout.rope_deltas,
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL position_ids must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                inputs.rope_layout.position_ids,
                reshape(Bool[true], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL rope_delta must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                inputs.rope_layout.position_ids,
                reshape(Float64[-2.0], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL rope_delta must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                inputs.rope_layout.position_ids,
                reshape(BigInt[overflow_integer], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL rope_delta is outside the host integer range",
        ),
    )
    for case in invalid_integer_layouts
        layout_error = try
            call_prefill(case.layout())
            nothing
        catch caught
            caught
        end
        @test layout_error isa ArgumentError
        @test layout_error isa Exception &&
            occursin(case.message, sprint(showerror, layout_error))
    end
    @test_throws DimensionMismatch call_prefill(Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        Int[-2],
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    ))
    @test_throws ArgumentError call_prefill(Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        reshape(Int[-1], 1, 1),
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    ))
    @test_throws ArgumentError call_prefill(Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        inputs.rope_layout.rope_deltas,
        Int.(inputs.rope_layout.visual_mask),
        inputs.rope_layout.attention_mask,
    ))
    @test_throws ArgumentError call_prefill(Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        inputs.rope_layout.rope_deltas,
        inputs.rope_layout.visual_mask,
        Int.(inputs.rope_layout.attention_mask),
    ))

    empty_attention = falses(length(inputs.input_ids), 1)
    @test_throws ArgumentError call_prefill(Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        inputs.rope_layout.rope_deltas,
        falses(length(inputs.input_ids), 1),
        empty_attention,
    ))

    padded_ids = Int[1, 1, 11, 12]
    padded_positions = repeat(reshape(Int[1, 1, 0, 1], 1, 4, 1), 3, 1, 1)
    padded_layout = Qwen3VLRopeLayout(
        padded_positions,
        reshape(Int[-2], 1, 1),
        falses(4, 1),
        reshape(Bool[false, false, true, true], 4, 1),
    )
    padded = hf_qwen3_vl_text_prefill(
        parameters,
        padded_ids,
        padded_layout;
        logits_to_keep=1,
    )
    @test size(padded.logits) == (parameters.spec.vocab_size, 1, 1)
    @test all(isfinite, padded.logits)

    singleton_ids = Int[1, 11]
    singleton_layout = Qwen3VLRopeLayout(
        repeat(reshape(Int[1, 0], 1, 2, 1), 3, 1, 1),
        reshape(Int[-1], 1, 1),
        falses(2, 1),
        reshape(Bool[false, true], 2, 1),
    )
    singleton = hf_qwen3_vl_text_prefill(
        parameters,
        singleton_ids,
        singleton_layout;
        logits_to_keep=1,
    )
    @test all(isfinite, singleton.logits)

    invalid_visual_mask = copy(padded_layout.visual_mask)
    invalid_visual_mask[1, 1] = true
    @test_throws ArgumentError hf_qwen3_vl_text_prefill(
        parameters,
        padded_ids,
        Qwen3VLRopeLayout(
            padded_positions,
            reshape(Int[-2], 1, 1),
            invalid_visual_mask,
            padded_layout.attention_mask,
        );
        logits_to_keep=1,
    )

    long_ids = fill(1, parameters.spec.max_position_embeddings + 1)
    long_positions = repeat(
        reshape(collect(0:(length(long_ids) - 1)), 1, length(long_ids), 1),
        3,
        1,
        1,
    )
    long_layout = Qwen3VLRopeLayout(
        long_positions,
        reshape(Int[0], 1, 1),
        falses(length(long_ids), 1),
        trues(length(long_ids), 1),
    )
    @test_throws ArgumentError hf_qwen3_vl_text_prefill(
        parameters,
        long_ids,
        long_layout;
        logits_to_keep=1,
        max_prefill_tokens=length(long_ids),
    )
end
