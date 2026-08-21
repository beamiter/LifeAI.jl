using Base64: base64decode
using JSON3
using SHA: sha256
using Test
using LifeAI: Qwen3VLRopeLayout,
    Qwen3VLTextSpec,
    hf_qwen3_vl_text_prefill

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
        logits_to_keep=0,
        capture_layers=(0, 1, 2, 3),
        capture_input_embeddings=true,
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

    predicted = [argmax(view(result.logits, :, token, 1)) for token in 1:8]
    @test predicted == [1, 2, 13, 14, 15, 7, 32, 8]
end
