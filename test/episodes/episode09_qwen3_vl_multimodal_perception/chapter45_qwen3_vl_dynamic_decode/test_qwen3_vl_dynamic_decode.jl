using Base64: base64decode
using JSON3
using SHA: sha256
using Test
using LifeAI: Qwen3VLRopeLayout,
    Qwen3VLTextSpec,
    generate_hf_qwen3_vl_tokens,
    hf_qwen3_vl_text_decode_step,
    hf_qwen3_vl_text_prefill_cached,
    init_qwen3_vl_kv_cache

const _CH45_TINY_TEXT_SPEC = Qwen3VLTextSpec(
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

function _ch45_tiny_values(count::Int, offset::Int; scale=0.02f0)
    return Float32[
        scale * sin(0.173f0 * Float32(offset + index))
        for index in 1:count
    ]
end

# Mathematical matrix emitted by the exporter's row-major
# `tiny_values((rows, columns), offset)` construction.
function _ch45_tiny_hf_matrix(rows::Int, columns::Int, offset::Int; scale=0.02f0)
    values = _ch45_tiny_values(rows * columns, offset; scale)
    return permutedims(reshape(values, columns, rows))
end

function _ch45_tiny_text_parameters()
    spec = _CH45_TINY_TEXT_SPEC
    embedding = permutedims(_ch45_tiny_hf_matrix(32, 16, 10))
    blocks = ntuple(spec.num_hidden_layers) do julia_layer
        offset = 10_000 * julia_layer
        return (;
            norm1=1.0f0 .+ _ch45_tiny_values(16, offset; scale=0.01f0),
            q_weight=_ch45_tiny_hf_matrix(16, 16, offset + 100),
            k_weight=_ch45_tiny_hf_matrix(8, 16, offset + 200),
            v_weight=_ch45_tiny_hf_matrix(8, 16, offset + 300),
            o_weight=_ch45_tiny_hf_matrix(16, 16, offset + 400),
            q_norm=1.0f0 .+ _ch45_tiny_values(8, offset + 500; scale=0.01f0),
            k_norm=1.0f0 .+ _ch45_tiny_values(8, offset + 600; scale=0.01f0),
            norm2=1.0f0 .+ _ch45_tiny_values(16, offset + 700; scale=0.01f0),
            gate_weight=_ch45_tiny_hf_matrix(32, 16, offset + 800),
            up_weight=_ch45_tiny_hf_matrix(32, 16, offset + 900),
            down_weight=_ch45_tiny_hf_matrix(16, 32, offset + 1_000),
        )
    end
    final_norm = 1.0f0 .+
        _ch45_tiny_values(16, 90_000; scale=0.01f0)
    return (; embedding, blocks, final_norm, spec)
end

function _ch45_tiny_prefill_inputs()
    position_ids = reshape(Int[
        0 1 2 2 2 2 4 5
        0 1 2 2 3 3 4 5
        0 1 2 3 2 3 4 5
    ], 3, 8, 1)
    visual_mask = falses(8, 1)
    visual_mask[3:6, 1] .= true
    attention_mask = trues(8, 1)
    rope_layout = Qwen3VLRopeLayout(
        position_ids,
        reshape(Int[-2], 1, 1),
        visual_mask,
        attention_mask,
    )
    visual_embeddings = permutedims(
        _ch45_tiny_hf_matrix(4, 16, 100_000; scale=0.1f0),
    )
    deepstack = ntuple(3) do index
        permutedims(_ch45_tiny_hf_matrix(
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

function _ch45_reference()
    path = joinpath(@__DIR__, "fixtures", "tiny_text_dynamic_decode.json")
    return JSON3.read(read(path, String))
end

function _ch45_reference_bytes(reference, name::AbstractString)
    entry = reference.tensors[Symbol(name)]
    bytes = base64decode(String(entry.f32_le_base64))
    @test bytes2hex(sha256(bytes)) == String(entry.sha256)
    return bytes, Int.(collect(entry.shape))
end

function _ch45_hf_hidden(reference, name::AbstractString)
    bytes, shape = _ch45_reference_bytes(reference, name)
    @test length(shape) == 3
    batch, tokens, width = shape
    values = collect(reinterpret(Float32, bytes))
    return reshape(values, width, tokens, batch)
end

function _ch45_hf_cache(reference, phase::AbstractString, layer::Int, kind::String)
    name = "cache.$phase.layer.$layer.$kind"
    bytes, shape = _ch45_reference_bytes(reference, name)
    @test length(shape) == 4
    batch, kv_heads, tokens, head_dim = shape
    values = collect(reinterpret(Float32, bytes))
    # C-order HF (B, KV, L, D) bytes first reshape to Julia (D, L, KV, B),
    # then place KV before L to obtain LifeAI's (D, KV, L, B) contract.
    reversed_hf = reshape(values, head_dim, tokens, kv_heads, batch)
    return permutedims(reversed_hf, (1, 3, 2, 4))
end

function _ch45_top_two(logits)
    scores = vec(Array(view(logits, :, size(logits, 2), 1)))
    ids = partialsortperm(scores, 1:2; rev=true)
    return (; ids, margin=scores[ids[1]] - scores[ids[2]])
end

function _ch45_assert_cache_matches(reference, cache, phase::String, tokens::Int)
    @test cache.position == tokens
    @test length(cache.layers) == 4
    for layer in 0:3
        actual = cache.layers[layer + 1]
        expected_key = _ch45_hf_cache(reference, phase, layer, "key")
        expected_value = _ch45_hf_cache(reference, phase, layer, "value")
        @test size(actual.keys) == (8, 1, tokens, 1)
        @test size(actual.values) == (8, 1, tokens, 1)
        @test actual.keys ≈ expected_key atol=1.0f-6 rtol=1.0f-6
        @test actual.values ≈ expected_value atol=1.0f-6 rtol=1.0f-6
    end
end

@testset "Chapter 45 — frozen HF DynamicCache fixture contract" begin
    reference = _ch45_reference()
    metadata = reference.metadata
    @test String(metadata.oracle) == "qwen3_vl_tiny_dynamic_cache_greedy_decode"
    @test String(metadata.python) == "3.10.12"
    @test String(metadata.numpy) == "1.26.4"
    @test String(metadata.transformers) == "4.57.0"
    @test String(metadata.torch) == "2.7.1+cpu"
    @test String(metadata.attention_implementation) == "eager"
    @test String(metadata.attention_mask_contract) ==
        "explicit_all_ones_every_call"
    @test String(metadata.cache_capture_contract) ==
        "detach_clone_cpu_before_next_dynamic_cache_update"
    @test String(metadata.hf_cache_layout) == "batch,kv_heads,tokens,head_dim"
    @test String(metadata.julia_cache_layout) == "head_dim,kv_heads,tokens,batch"
    @test Int.(collect(metadata.hf_to_julia_permutation_1_based)) == [4, 2, 3, 1]

    @test Int.(collect(metadata.cache_lengths)) == [8, 9, 10]
    @test Int(metadata.prefill_length) == 8
    @test Int(metadata.decode_forward_calls) == 2
    @test Int(metadata.rope_delta) == -2
    @test Int.(collect(metadata.greedy_token_ids_0_based)) == [7, 7, 7]
    @test Int.(collect(metadata.greedy_token_ids_1_based)) == [8, 8, 8]
    @test Int.(collect(metadata.phases[2].mrope_position_ids_thw_0_based)) ==
        [6, 6, 6]
    @test Int.(collect(metadata.phases[3].mrope_position_ids_thw_0_based)) ==
        [7, 7, 7]
    @test Int(metadata.phases[2].physical_cache_position_0_based) == 8
    @test Int(metadata.phases[3].physical_cache_position_0_based) == 9
    @test Int.(collect(metadata.phases[2].attention_mask_shape)) == [1, 9]
    @test Int.(collect(metadata.phases[3].attention_mask_shape)) == [1, 10]

    expected_margins = [
        0.0004043877124786377,
        0.0006773620843887329,
        0.0008579939603805542,
    ]
    @test [Float64(phase.top_two.margin_f32) for phase in metadata.phases] ==
        expected_margins
    @test all(>(0), expected_margins)

    @test size(_ch45_hf_hidden(reference, "prefill.final_hidden")) == (16, 8, 1)
    @test size(_ch45_hf_hidden(reference, "prefill.logits")) == (32, 8, 1)
    for step in 0:1
        @test size(_ch45_hf_hidden(reference, "decode.$step.final_hidden")) ==
            (16, 1, 1)
        @test size(_ch45_hf_hidden(reference, "decode.$step.logits")) ==
            (32, 1, 1)
    end

    # The fixture itself proves snapshot discipline independently of LifeAI:
    # every later HF DynamicCache tensor has a byte-identical old prefix.
    for layer in 0:3, kind in ("key", "value")
        prefill = _ch45_hf_cache(reference, "prefill", layer, kind)
        decode0 = _ch45_hf_cache(reference, "decode.0", layer, kind)
        decode1 = _ch45_hf_cache(reference, "decode.1", layer, kind)
        @test decode0[:, :, 1:8, :] == prefill
        @test decode1[:, :, 1:9, :] == decode0
    end

    # Per-phase hidden tensors are independently tied to the frozen logits.
    parameters = _ch45_tiny_text_parameters()
    for phase in ("prefill", "decode.0", "decode.1")
        hidden = _ch45_hf_hidden(reference, "$phase.final_hidden")
        logits = _ch45_hf_hidden(reference, "$phase.logits")
        projected = reshape(
            transpose(parameters.embedding) * reshape(hidden, 16, :),
            size(logits),
        )
        @test projected ≈ logits atol=2.0f-7 rtol=2.0f-6
    end
end

@testset "Chapter 45 — tiny cached prefill and two decode steps" begin
    reference = _ch45_reference()
    parameters = _ch45_tiny_text_parameters()
    inputs = _ch45_tiny_prefill_inputs()
    cache0 = init_qwen3_vl_kv_cache(parameters; batch_size=1)
    @test isempty(cache0)
    @test cache0.position == 0
    @test cache0.rope_delta == 0
    @test all(layer -> layer.keys === nothing && layer.values === nothing, cache0.layers)
    @test_throws ArgumentError init_qwen3_vl_kv_cache(parameters; batch_size=2)
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step(parameters, 8, cache0)
    malformed_layout = Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        reshape(Int[-1], 1, 1),
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    )
    @test_throws ArgumentError hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        malformed_layout;
        vision_features=inputs.vision_features,
        cache=cache0,
    )

    prefill, cache8 = hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=cache0,
        logits_to_keep=0,
    )
    @test cache8.position == 8
    @test cache8.rope_delta == -2
    @test size(prefill.final_hidden) == (16, 8, 1)
    @test size(prefill.logits) == (32, 8, 1)
    @test prefill.final_hidden ≈
        _ch45_hf_hidden(reference, "prefill.final_hidden") atol=1.0f-6 rtol=1.0f-6
    @test prefill.logits ≈
        _ch45_hf_hidden(reference, "prefill.logits") atol=1.0f-6 rtol=1.0f-6
    _ch45_assert_cache_matches(reference, cache8, "prefill", 8)
    prefill_keys = map(layer -> copy(layer.keys), cache8.layers)
    prefill_values = map(layer -> copy(layer.values), cache8.layers)

    choice = _ch45_top_two(prefill.logits)
    @test choice.ids == [8, 24]
    @test choice.margin ≈ 0.0004043877f0 atol=2.0f-7 rtol=2.0f-5

    logits9, cache9 = hf_qwen3_vl_text_decode_step(parameters, choice.ids[1], cache8)
    @test cache9.position == 9
    @test cache9.rope_delta == -2
    @test size(logits9) == (32, 1, 1)
    @test logits9 ≈
        _ch45_hf_hidden(reference, "decode.0.logits") atol=1.0f-6 rtol=1.0f-6
    _ch45_assert_cache_matches(reference, cache9, "decode.0", 9)
    for layer in 1:4
        @test cache8.layers[layer].keys == prefill_keys[layer]
        @test cache8.layers[layer].values == prefill_values[layer]
        @test cache9.layers[layer].keys[:, :, 1:8, :] == prefill_keys[layer]
        @test cache9.layers[layer].values[:, :, 1:8, :] == prefill_values[layer]
    end
    decode0_keys = map(layer -> copy(layer.keys), cache9.layers)
    decode0_values = map(layer -> copy(layer.values), cache9.layers)

    choice9 = _ch45_top_two(logits9)
    @test choice9.ids == [8, 24]
    @test choice9.margin ≈ 0.0006773621f0 atol=2.0f-7 rtol=2.0f-5
    logits10, cache10 = hf_qwen3_vl_text_decode_step(
        parameters,
        choice9.ids[1],
        cache9,
    )
    @test cache10.position == 10
    @test cache10.rope_delta == -2
    @test size(logits10) == (32, 1, 1)
    @test logits10 ≈
        _ch45_hf_hidden(reference, "decode.1.logits") atol=1.0f-6 rtol=1.0f-6
    _ch45_assert_cache_matches(reference, cache10, "decode.1", 10)
    for layer in 1:4
        @test cache9.layers[layer].keys == decode0_keys[layer]
        @test cache9.layers[layer].values == decode0_values[layer]
        @test cache10.layers[layer].keys[:, :, 1:9, :] == decode0_keys[layer]
        @test cache10.layers[layer].values[:, :, 1:9, :] == decode0_values[layer]
    end
    choice10 = _ch45_top_two(logits10)
    @test choice10.ids == [8, 24]
    @test choice10.margin ≈ 0.0008579940f0 atol=2.0f-7 rtol=2.0f-5

    # A cache is request state, and cached prefill is legal only once.
    @test_throws ArgumentError hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=cache8,
    )
end

@testset "Chapter 45 — greedy generation cache timeline" begin
    reference = _ch45_reference()
    parameters = _ch45_tiny_text_parameters()
    inputs = _ch45_tiny_prefill_inputs()

    generated = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        capture_logits=true,
    )
    @test generated.prompt_ids == collect(1:8)
    @test generated.generated_ids == [8, 8, 8]
    @test generated.token_ids == vcat(collect(1:8), [8, 8, 8])
    @test generated.stop_reason === :length
    @test generated.strategy === :greedy
    @test generated.cache.position == 10
    @test generated.cache.rope_delta == -2
    @test length(generated.trace) == 3
    @test [step.step for step in generated.trace] == [1, 2, 3]
    @test [step.token_id for step in generated.trace] == [8, 8, 8]
    @test [step.hf_token_id for step in generated.trace] == [7, 7, 7]
    @test [step.second_token_id for step in generated.trace] == [24, 24, 24]
    @test all(step -> length(step.logits) == 32, generated.trace)
    @test [Float64(step.margin) for step in generated.trace] ≈ [
        Float64(phase.top_two.margin_f32) for phase in reference.metadata.phases
    ] atol=2.0e-7 rtol=2.0e-5

    # The first token is selected directly from prefill. With three output
    # tokens only two decode calls occur, so the final cache length is 10.
    @test generated.prefill !== nothing
    @test size(generated.prefill.logits) == (32, 1, 1)
    @test generated.trace[1].logits ≈
        vec(_ch45_hf_hidden(reference, "prefill.logits")[:, end, 1])
    @test generated.trace[2].logits ≈
        vec(_ch45_hf_hidden(reference, "decode.0.logits"))
    @test generated.trace[3].logits ≈
        vec(_ch45_hf_hidden(reference, "decode.1.logits"))

    stopped = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=[8],
    )
    @test stopped.generated_ids == [8]
    @test stopped.stop_reason === :eos
    @test stopped.cache.position == 8
    @test length(stopped.trace) == 1

    zero = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=0,
    )
    @test isempty(zero.generated_ids)
    @test zero.token_ids == collect(1:8)
    @test zero.prefill === nothing
    @test isempty(zero.cache)
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=-1,
    )
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=58,
    )
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=1,
        stop_token_ids=[33],
    )

    # A prompt may occupy the complete physical context when only the first
    # token from prefill logits is requested. The selected token is returned
    # without being appended; an actual decode attempt remains illegal.
    full_context_ids = fill(1, parameters.spec.max_position_embeddings)
    full_positions = repeat(
        reshape(collect(0:(length(full_context_ids) - 1)), 1, :, 1),
        3,
        1,
        1,
    )
    full_layout = Qwen3VLRopeLayout(
        full_positions,
        reshape(Int[0], 1, 1),
        falses(length(full_context_ids), 1),
        trues(length(full_context_ids), 1),
    )
    boundary = generate_hf_qwen3_vl_tokens(
        parameters,
        full_context_ids,
        full_layout;
        max_new_tokens=1,
        stop_token_ids=Int[],
    )
    @test length(boundary.generated_ids) == 1
    @test boundary.cache.position == parameters.spec.max_position_embeddings
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step(
        parameters,
        only(boundary.generated_ids),
        boundary.cache,
    )
end
