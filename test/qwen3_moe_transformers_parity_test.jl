using Test
using JSON3
using SHA
using LifeAI:
    decode_step,
    hf_qwen3_moe_forward_trace,
    hf_token_ids,
    init_kv_cache,
    init_static_kv_cache,
    load_hf_qwen3_moe_model,
    load_safetensors,
    prefill,
    qwen3_topk_routing

const _QWEN3_MOE_TINY_PARITY_DIR = joinpath(
    @__DIR__,
    "fixtures",
    "qwen3_moe_tiny_parity",
)

_qwen3_moe_fixture_sha256(path) = bytes2hex(SHA.sha256(read(path)))
_qwen3_moe_hf_layout(array) = permutedims(array, (3, 2, 1))

@testset "Qwen3 MoE Transformers tiny checkpoint parity" begin
    metadata_path = joinpath(_QWEN3_MOE_TINY_PARITY_DIR, "reference.json")
    metadata = JSON3.read(read(metadata_path, String))
    @test Int(metadata.schema_version) == 1
    @test String(metadata.model_type) == "qwen3_moe"
    @test String(metadata.dtype) == "float32"
    @test String(metadata.versions.torch) == "2.7.1+cpu"
    @test String(metadata.versions.transformers) == "4.51.3"

    fixture_assets = (
        ("config", "config.json"),
        ("generation_config", "generation_config.json"),
        ("model", "model.safetensors"),
        ("reference", "reference.safetensors"),
    )
    for (name, filename) in fixture_assets
        @test _qwen3_moe_fixture_sha256(
            joinpath(_QWEN3_MOE_TINY_PARITY_DIR, filename),
        ) == String(getproperty(metadata.checksums, Symbol("$(name)_sha256")))
    end
    @test _qwen3_moe_fixture_sha256(joinpath(
        @__DIR__,
        "..",
        "scripts",
        "export_qwen3_moe_tiny_reference.py",
    )) == String(metadata.checksums.script_sha256)

    loaded = load_hf_qwen3_moe_model(
        _QWEN3_MOE_TINY_PARITY_DIR;
        max_seq_len=16,
    )
    reference = load_safetensors(joinpath(
        _QWEN3_MOE_TINY_PARITY_DIR,
        "reference.safetensors",
    ))
    @test loaded.model.num_layers == 2
    @test loaded.model.num_experts == 4
    @test loaded.model.experts_per_token == 2
    @test loaded.model.normalize_routing

    tokens = reshape(hf_token_ids(
        Int.(collect(metadata.input_ids_0_based));
        vocab_size=loaded.model.vocab_size,
    ), :, 1)
    trace = hf_qwen3_moe_forward_trace(
        loaded.model,
        tokens,
        loaded.parameters,
        loaded.states,
    )
    @test trace.embedding == _qwen3_moe_hf_layout(reference["embedding"])
    for layer in 0:(loaded.model.num_layers - 1)
        @test trace.router_logits[layer + 1] ≈ permutedims(
            reference["router_logits.$layer"],
            (2, 1),
        ) atol = 2.0f-6 rtol = 2.0f-6
        @test trace.blocks[layer + 1] ≈ _qwen3_moe_hf_layout(
            reference["block.$layer"],
        ) atol = 5.0f-6 rtol = 5.0f-6

        routing = qwen3_topk_routing(
            trace.router_logits[layer + 1],
            loaded.model.experts_per_token;
            normalize=loaded.model.normalize_routing,
        )
        expected_selected = metadata.selected_experts_0_based[string(layer)]
        expected_weights = reference["routing_weights.$layer"]
        for token in axes(routing, 2)
            selected = partialsortperm(
                view(trace.router_logits[layer + 1], :, token),
                1:loaded.model.experts_per_token;
                rev=true,
            )
            @test selected .- 1 == Int.(collect(expected_selected[token]))
            @test routing[selected, token] ≈ expected_weights[token, :] atol = 2.0f-7 rtol = 2.0f-6
        end
    end
    @test trace.final_hidden ≈ _qwen3_moe_hf_layout(
        reference["final_hidden"],
    ) atol = 5.0f-6 rtol = 5.0f-6
    @test trace.logits ≈ _qwen3_moe_hf_layout(
        reference["logits"],
    ) atol = 5.0f-6 rtol = 5.0f-6
    direct_logits, _ = loaded.model(
        tokens,
        loaded.parameters,
        loaded.states,
    )
    @test direct_logits == trace.logits

    prompt_tokens = reshape(hf_token_ids(
        Int.(collect(metadata.prompt_ids_0_based));
        vocab_size=loaded.model.vocab_size,
    ), :, 1)
    decode_token = hf_token_ids(
        [Int(metadata.decode_id_0_based)];
        vocab_size=loaded.model.vocab_size,
    )
    expected_prompt = _qwen3_moe_hf_layout(reference["prompt_logits"])
    expected_decode = _qwen3_moe_hf_layout(reference["decode_logits"])

    dynamic = init_kv_cache(loaded.model; batch_size=1)
    prompt_logits, dynamic, dynamic_states = prefill(
        loaded.model,
        loaded.parameters,
        loaded.states,
        prompt_tokens,
        dynamic,
    )
    decode_logits, dynamic, _ = decode_step(
        loaded.model,
        loaded.parameters,
        dynamic_states,
        decode_token,
        dynamic,
    )
    @test prompt_logits ≈ expected_prompt atol = 5.0f-6 rtol = 5.0f-6
    @test decode_logits ≈ expected_decode atol = 5.0f-6 rtol = 5.0f-6
    @test argmax(vec(decode_logits)) == argmax(vec(expected_decode))

    static = init_static_kv_cache(loaded.model; batch_size=1)
    static_prompt, static, static_states = prefill(
        loaded.model,
        loaded.parameters,
        loaded.states,
        prompt_tokens,
        static,
    )
    static_decode, static, _ = decode_step(
        loaded.model,
        loaded.parameters,
        static_states,
        decode_token,
        static,
    )
    @test static_prompt ≈ expected_prompt atol = 5.0f-6 rtol = 5.0f-6
    @test static_decode ≈ expected_decode atol = 5.0f-6 rtol = 5.0f-6
    @test argmax(vec(static_decode)) == argmax(vec(expected_decode))
end
