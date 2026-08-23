using Test
using LifeAI

include(joinpath(
    pkgdir(LifeAI),
    "scripts",
    "qwen3_vl_long_reference_contract.jl",
))

isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || error(
    "Chapter 47 profiling tests require the Chapter 46 tiny decode fixture",
)

const _CH47_BLOCK_STAGES = Symbol[
    :pre_attention_norm,
    :qkv_projection,
    :qk_norm,
    :qk_rope,
    :kv_write,
    :attention,
    :attention_output_projection_residual,
    :post_attention_norm,
    :mlp_gate_up_projection,
    :mlp_activation,
    :mlp_down_projection_residual,
]

function _ch47_expected_stages(layer_count::Int)
    stages = Tuple{Symbol,Int}[
        (:token_embedding, 0),
        (:mrope_prepare, 0),
    ]
    for layer in 1:layer_count, stage in _CH47_BLOCK_STAGES
        push!(stages, (stage, layer))
    end
    append!(stages, [(:final_norm, 0), (:vocab_logits, 0)])
    return stages
end

function _ch47_test_long_reference(asset_sha256)
    generated = mod.(collect(1:_CH47_LONG_REFERENCE_TOKEN_COUNT), 32)
    decisions = Any[]
    for step in eachindex(generated)
        physical_position = step == 1 ? nothing : 76 + step - 2
        push!(decisions, Dict(
            "generated_step" => step,
            "phase" => step == 1 ? "prefill" : "decode.$(step - 2)",
            "input_token_id_0_based" => step == 1 ? nothing : generated[step - 1],
            "physical_cache_position_0_based" => physical_position,
            "mrope_position_ids_thw_0_based" => step == 1 ? Int[] :
                fill(physical_position - 56, 3),
            "attention_mask_shape" => [1, 76 + step - 1],
            "cache_length" => 76 + step - 1,
            "top_two" => Dict(
                "top1_token_id_0_based" => generated[step],
                "top2_token_id_0_based" => mod(generated[step] + 1, 32),
                "top1_logit_f32" => 2.0,
                "top2_logit_f32" => 1.0,
                "margin_f32" => 1.0,
            ),
            "logits_shape" => [1, 1, 32],
            "logits_dtype" => "torch.float32",
            "logits_raw_sha256" => repeat("a", 64),
        ))
    end
    payload = Dict(
        "schema_version" => 1,
        "oracle" => _CH47_LONG_REFERENCE_ORACLE,
        "claim" => _CH47_LONG_REFERENCE_CLAIM,
        "model_id" => "test/model",
        "modelscope_revision" => repeat("b", 40),
        "huggingface_revision" => repeat("c", 40),
        "asset_sha256" => asset_sha256,
        "compute_dtype" => "float32",
        "compute_device" => "cpu",
        "compute_device_type" => "cpu",
        "cuda_device" => "",
        "attention_implementation" => "eager",
        "attention_mask_contract" => "explicit_all_ones_every_call",
        "cache_contract" =>
            "streaming_hf_dynamic_cache_geometry_no_kv_snapshots",
        "greedy" => true,
        "termination_contract" => _CH47_LONG_REFERENCE_TERMINATION,
        "stop_token_ids_0_based" => Int[],
        "prompt" => "Describe.",
        "rendered_prompt" => "test rendered prompt",
        "image_sha256" => repeat("e", 64),
        "rendered_prompt_sha256" =>
            _ch47_long_text_sha256("test rendered prompt"),
        "image_shape_hwc" => [256, 256, 3],
        "grid_thw" => [[1, 16, 16]],
        "image_token_count" => 64,
        "prompt_tokens" => 76,
        "rope_delta" => -56,
        "input_ids_0_based" => mod.(collect(0:75), 32),
        "input_ids_u32le_sha256" =>
            "a740c20852b884e65287bc08751de2a6661650ef339854570bb76b4d18134fb4",
        "chapter45_reference_sha256" => Dict(
            "reference.safetensors" => repeat("1", 64),
            "reference.json" => repeat("2", 64),
        ),
        "generated_token_count" => length(generated),
        "generated_token_ids_0_based" => generated,
        "token_timeline_u32le_sha256" =>
            "ee166616daae0fc1c8cc168fc3359a09379979724186a9727ed2baad0e2159b1",
        "checkpoint_prefix_u32le_sha256" => Dict(
            "32" => "56d7f20b1fc768a415fbc7a92e10516cfefb16e1303463567ca2cb9973b7ac76",
            "128" => "56db589a75413e7cdc3415a5cff475ca5dd438d94a6b5110d52b983675c5eb16",
            "256" => "ee166616daae0fc1c8cc168fc3359a09379979724186a9727ed2baad0e2159b1",
        ),
        "checkpoint_cache_geometry" => Dict(
            string(token_count) => Dict(
                "length" => 76 + token_count - 1,
                "layer_count" => 28,
                "key_shapes_hf" => [
                    [1, 8, 76 + token_count - 1, 128] for _ in 1:28
                ],
                "value_shapes_hf" => [
                    [1, 8, 76 + token_count - 1, 128] for _ in 1:28
                ],
            ) for token_count in (32, 128, 256)
        ),
        "decisions" => decisions,
        "torch_num_threads" => 24,
        "torch_num_interop_threads" => 24,
        "torch_seed" => 0,
        "deterministic_algorithms" => true,
        "deterministic_algorithms_warn_only" => false,
        "float32_matmul_precision" => "highest",
        "mkldnn_available" => true,
        "mkldnn_enabled" => true,
        "mkldnn_deterministic" => false,
        "mkl_available" => true,
        "openmp_available" => true,
        "source_sha256" => Dict(
            "export_qwen3_vl_long_generation_reference.py" => repeat("3", 64),
            "export_qwen3_vl_decode_reference.py" => repeat("4", 64),
            "export_qwen3_vl_prefill_reference.py" => repeat("5", 64),
        ),
        "generated_text" => "test",
    )
    merge!(payload, _CH47_LONG_REFERENCE_ENVIRONMENT)
    return payload
end

function _ch47_write_test_long_reference(path, payload)
    open(path, "w") do io
        JSON3.pretty(io, payload)
        println(io)
    end
    return _ch47_long_file_sha256(path)
end

function _ch47_load_test_long_reference(
    path,
    digest,
    assets;
    lengths=[4, 6, 256],
)
    return _ch47_load_long_generation_reference(
        path,
        digest;
        model_id="test/model",
        modelscope_revision=repeat("b", 40),
        huggingface_revision=repeat("c", 40),
        asset_sha256=assets,
        required_lengths=lengths,
        expected_vocab_size=32,
        expected_image_sha256=repeat("e", 64),
        expected_rendered_prompt_sha256=
            _ch47_long_text_sha256("test rendered prompt"),
        expected_chapter45_reference_sha256=repeat("1", 64),
        expected_chapter45_metadata_sha256=repeat("2", 64),
        expected_source_sha256=Dict(
            "export_qwen3_vl_long_generation_reference.py" => repeat("3", 64),
            "export_qwen3_vl_decode_reference.py" => repeat("4", 64),
            "export_qwen3_vl_prefill_reference.py" => repeat("5", 64),
        ),
    )
end

@testset "Chapter 47 — decode profiling hook is numerically transparent" begin
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()
    baseline_cache = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
    profiled_cache = init_qwen3_vl_static_kv_cache(parameters; capacity=10)

    baseline_prefill, _ = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=baseline_cache,
    )
    profiled_prefill, _ = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=profiled_cache,
    )
    token = _ch46_top_two(baseline_prefill.logits).ids[1]
    @test profiled_prefill.logits == baseline_prefill.logits

    baseline_logits, returned_baseline = hf_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        baseline_cache,
    )
    observed = Tuple{Symbol,Int}[]
    runner = function (stage, layer, thunk)
        push!(observed, (stage, layer))
        return thunk()
    end
    profiled_logits, returned_profiled =
        LifeAI._profile_qwen3_vl_text_decode_step_static(
            parameters,
            token,
            profiled_cache,
            runner,
        )

    @test returned_baseline === baseline_cache
    @test returned_profiled === profiled_cache
    @test profiled_logits == baseline_logits
    @test profiled_cache.position == baseline_cache.position == 9
    @test profiled_cache.rope_delta == baseline_cache.rope_delta == -2
    @test observed == _ch47_expected_stages(parameters.spec.num_hidden_layers)
    @test length(observed) == 48
    @test count(==((:kv_write, 1)), observed) == 1
    for layer in eachindex(profiled_cache.layers)
        @test profiled_cache.layers[layer].keys ==
            baseline_cache.layers[layer].keys
        @test profiled_cache.layers[layer].values ==
            baseline_cache.layers[layer].values
    end

    @test_throws ArgumentError LifeAI._profile_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        profiled_cache,
        nothing,
    )
end

@testset "Chapter 47 — pinned HF long-reference contract fails closed" begin
    mktempdir() do directory
        assets = Dict("model.safetensors" => repeat("d", 64))
        payload = _ch47_test_long_reference(assets)
        path = joinpath(directory, "long_generation_reference.json")
        digest = _ch47_write_test_long_reference(path, payload)
        loaded = _ch47_load_test_long_reference(path, digest, assets)

        @test loaded.file_sha256 == digest
        @test loaded.generated_ids_0_based[1:6] == collect(1:6)
        @test loaded.generated_ids_1_based[1:6] == collect(2:7)
        @test loaded.generated_token_count == 256
        @test loaded.top_two[1].top1_token_id_0_based == 1
        @test loaded.token_timeline_u32le_sha256 ==
            "ee166616daae0fc1c8cc168fc3359a09379979724186a9727ed2baad0e2159b1"
        @test loaded.cuda_device == ""
        @test loaded.compute_device == "cpu"
        @test loaded.input_ids_0_based == mod.(collect(0:75), 32)
        @test _ch47_long_first_divergence(
            loaded.generated_ids_1_based,
            loaded,
        ) === nothing
        divergent = copy(loaded.generated_ids_1_based)
        divergent[7] += 1
        first_divergence = _ch47_long_first_divergence(divergent, loaded)
        @test first_divergence.generated_step == 7
        @test first_divergence.lifeai_token_id_1_based == divergent[7]
        @test first_divergence.hf_token_id_0_based == 7
        @test first_divergence.hf_margin_f32 == 1.0f0
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            repeat("0", 64),
            assets,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            digest,
            assets;
            lengths=[4, 257],
        )

        changed = deepcopy(payload)
        changed["decisions"][5]["top_two"]["top1_token_id_0_based"] = 9
        changed_digest = _ch47_write_test_long_reference(path, changed)
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            changed_digest,
            assets,
        )

        zero_margin = deepcopy(payload)
        zero_margin["decisions"][6]["top_two"]["margin_f32"] = 0.0
        zero_margin_digest = _ch47_write_test_long_reference(path, zero_margin)
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            zero_margin_digest,
            assets,
        )

        forged_margin = deepcopy(payload)
        forged_margin["decisions"][6]["top_two"]["margin_f32"] = 0.5
        forged_margin_digest = _ch47_write_test_long_reference(
            path,
            forged_margin,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            forged_margin_digest,
            assets,
        )

        wrong_claim = deepcopy(payload)
        wrong_claim["claim"] = "self_determinism_only"
        wrong_claim_digest = _ch47_write_test_long_reference(path, wrong_claim)
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            wrong_claim_digest,
            assets,
        )

        wrong_environment = deepcopy(payload)
        wrong_environment["transformers"] = "4.57.3"
        wrong_environment_digest = _ch47_write_test_long_reference(
            path,
            wrong_environment,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            wrong_environment_digest,
            assets,
        )

        wrong_backend = deepcopy(payload)
        wrong_backend["deterministic_algorithms"] = false
        wrong_backend_digest = _ch47_write_test_long_reference(
            path,
            wrong_backend,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            wrong_backend_digest,
            assets,
        )

        wrong_prompt_hash = deepcopy(payload)
        wrong_prompt_hash["input_ids_u32le_sha256"] = repeat("0", 64)
        wrong_prompt_digest = _ch47_write_test_long_reference(
            path,
            wrong_prompt_hash,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            wrong_prompt_digest,
            assets,
        )

        missing_checkpoint = deepcopy(payload)
        delete!(missing_checkpoint["checkpoint_prefix_u32le_sha256"], "32")
        missing_digest = _ch47_write_test_long_reference(
            path,
            missing_checkpoint,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            missing_digest,
            assets,
        )

        wrong_geometry = deepcopy(payload)
        wrong_geometry["checkpoint_cache_geometry"]["128"][
            "key_shapes_hf"
        ][3][3] -= 1
        wrong_geometry_digest = _ch47_write_test_long_reference(
            path,
            wrong_geometry,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            wrong_geometry_digest,
            assets,
        )

        wrong_source = deepcopy(payload)
        wrong_source["source_sha256"][
            "export_qwen3_vl_long_generation_reference.py"
        ] = repeat("6", 64)
        wrong_source_digest = _ch47_write_test_long_reference(
            path,
            wrong_source,
        )
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            wrong_source_digest,
            assets,
        )

        short_timeline = deepcopy(payload)
        pop!(short_timeline["generated_token_ids_0_based"])
        pop!(short_timeline["decisions"])
        short_timeline["generated_token_count"] = 255
        short_timeline["token_timeline_u32le_sha256"] =
            _ch47_long_token_timeline_sha256(
                short_timeline["generated_token_ids_0_based"],
            )
        short_digest = _ch47_write_test_long_reference(path, short_timeline)
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            short_digest,
            assets,
        )

        wrong_assets = deepcopy(payload)
        wrong_assets["asset_sha256"]["model.safetensors"] = repeat("e", 64)
        wrong_asset_digest = _ch47_write_test_long_reference(path, wrong_assets)
        @test_throws ArgumentError _ch47_load_test_long_reference(
            path,
            wrong_asset_digest,
            assets,
        )
    end
end
