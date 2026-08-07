using Test
using BFloat16s: BFloat16
using JSON3
using SHA: sha256
using LifeAI:
    HFQwen3BF16XLASession,
    generate_hf_qwen3_bf16_xla!,
    load_hf_qwen3_compact_model,
    load_hf_qwen3_model,
    open_safetensors_reader,
    plan_qwen3_xla_window,
    qwen3_xla_key_positions,
    qwen3_xla_pad_prompt,
    read_safetensors_tensor

isdefined(@__MODULE__, :repository_test_asset) ||
    include("repository_test_assets.jl")

isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
    include("qwen3_tiny_model_fixture.jl")

@testset "exact XLA 4K window planning" begin
    plan = plan_qwen3_xla_window(3584, 512)
    @test plan.context_tokens == 4096
    @test plan.prompt_bucket_tokens == 3584
    @test plan.left_padding_tokens == 0
    @test plan.sequence_tokens == 4096
    @test plan.cache_tokens == 4095

    padded = plan_qwen3_xla_window(
        17,
        4;
        context_tokens=64,
        chunk_tokens=16,
    )
    @test padded.prompt_bucket_tokens == 32
    @test padded.left_padding_tokens == 15
    @test qwen3_xla_pad_prompt(collect(2:18), padded; pad_token_id=7) ==
        vcat(fill(7, 15), collect(2:18))
    key_positions = qwen3_xla_key_positions(padded)
    @test key_positions[1:15] == fill(typemax(Int32), 15)
    @test key_positions[16:end] == Int32.(16:64)

    @test_throws ArgumentError plan_qwen3_xla_window(0, 1)
    @test_throws ArgumentError plan_qwen3_xla_window(1, 0)
    @test_throws ArgumentError plan_qwen3_xla_window(3584, 513)
    @test_throws ArgumentError plan_qwen3_xla_window(3585, 512)
    @test_throws ArgumentError plan_qwen3_xla_window(
        49,
        15;
        context_tokens=64,
        chunk_tokens=16,
    )
    @test_throws ArgumentError plan_qwen3_xla_window(
        1,
        1;
        context_tokens=typemax(Int),
    )
end

@testset "frozen Qwen3 XLA deployment hardware evidence is self-consistent" begin
    reference_path = repository_test_asset("qwen3_8b_cuda_greedy_reference.json")
    report_path = repository_test_asset("qwen3_8b_4090d_bf16_xla_daily.json")
    trace_path = repository_test_asset(
        "qwen3_8b_4090d_bf16_xla_daily_nvidia_smi.csv",
    )
    reference_sha =
        "83f62afbbb470b695b6990a3b86a8860407a37874354d6b039e1ce19917e2747"
    report_sha =
        "075dc76a023a8143213f640bfb354d6e38c10b5747b5ef3e8c7e6baeb1c730dc"
    trace_sha =
        "72fd3fe80b56647714604a85499a3a9d8e5833412f7a50864d8ea1aa6588b586"
    profile_sha =
        "0638eecce7864d261770c8af1698575f055cf3149d6e5d70605c7cf35dbb8d01"

    @test bytes2hex(sha256(read(reference_path))) == reference_sha
    reference = JSON3.read(read(reference_path, String))
    @test Int(reference["schema_version"]) == 2
    @test Set(String(case["name"]) for case in reference["cases"]) ==
        Set((
            "single_chunk_64",
            "left_padded_65",
            "full_prompt_3584",
        ))
    @test sum(
        Int(case["generated_tokens"]) for case in reference["cases"]
    ) == 96
    @test all(
        !haskey(case, "elapsed_seconds") for case in reference["cases"]
    )

    @test bytes2hex(sha256(read(report_path))) == report_sha
    report = JSON3.read(read(report_path, String))
    @test Bool(report["closed"])
    @test String(report["reference_sha256"]) == reference_sha
    @test String(report["profile_sha256"]) == profile_sha
    @test Int(report["acceptance"]["cuda_bf16_parity_match_tokens"]) == 96
    @test Int(report["acceptance"]["cuda_bf16_parity_expected_tokens"]) == 96
    @test Int(report["acceptance"]["full_window_sequence_tokens"]) == 4096
    @test Int(report["acceptance"]["full_window_cache_tokens"]) == 4095
    passed_fields = [
        Bool(value)
        for (key, value) in pairs(report["acceptance"])
        if endswith(String(key), "_passed")
    ]
    @test !isempty(passed_fields)
    @test all(passed_fields)

    @test bytes2hex(sha256(read(trace_path))) == trace_sha
    trace_lines = readlines(trace_path)
    @test length(trace_lines) == Int(report["physical_trace"]["samples"])
    trace_minimum_free = minimum(trace_lines) do line
        fields = strip.(split(line, ','))
        parse(Int, fields[3]) * 1024^2
    end
    @test trace_minimum_free ==
        Int(report["physical_trace"]["minimum_free_bytes"])
    @test String(report["physical_trace"]["sha256"]) == trace_sha
    @test trace_minimum_free >=
        Int(report["acceptance"]["physical_minimum_free_limit_bytes"])
end

@testset "streamed compact tree is one-transfer ready" begin
    mktempdir() do directory
        _qwen3_tiny_model_fixture_dir(directory; tie=false)
        reader = open_safetensors_reader(directory)
        embedding = read_safetensors_tensor(
            reader,
            "model.embed_tokens.weight";
            target_dtype=BFloat16,
        )
        @test eltype(embedding) === BFloat16
        @test_throws ArgumentError read_safetensors_tensor(
            reader,
            "model.embed_tokens.weight";
            target_dtype=Float16,
        )

        ordinary = load_hf_qwen3_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        streamed = load_hf_qwen3_compact_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        expected = LifeAI._bf16a_compact_parameters(ordinary.parameters)
        @test streamed.parameter_layout === :compact_packed
        @test streamed.parameters == expected
        @test LifeAI._qwen3_xla_tensor_leaves(streamed.parameters) ==
            8 * streamed.model.num_layers + 3
        @test LifeAI._qwen3_xla_tensor_bytes(streamed.parameters) ==
            LifeAI._qwen3_xla_tensor_bytes(ordinary.parameters)

        for block in values(streamed.parameters.blocks)
            @test hasproperty(block, :qkv_weight)
            @test hasproperty(block, :gate_up_weight)
            @test !hasproperty(block.attn, :q_proj)
            @test !hasproperty(block.attn, :k_proj)
            @test !hasproperty(block.attn, :v_proj)
            @test !hasproperty(block.mlp, :gate_proj)
            @test !hasproperty(block.mlp, :up_proj)
        end
    end
end

@testset "fixed chunk packed prefill matches full packed prefill" begin
    mktempdir() do directory
        _qwen3_tiny_model_fixture_dir(directory; tie=false)
        loaded = load_hf_qwen3_compact_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        model = loaded.model
        parameters = loaded.parameters
        rope = first(values(model.blocks.layers)).attn.rope
        cos_table = BFloat16.(rope.cos_cache)
        sin_table = BFloat16.(rope.sin_cache)
        cache_shape = (
            model.head_dim,
            model.num_kv_heads,
            model.max_seq_len,
            1,
        )
        chunk_keys = Tuple(
            zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
        )
        chunk_values = Tuple(
            zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
        )
        tokens = reshape(collect(2:9), :, 1)
        position = Int32[0]
        key_positions = Int32.(collect(1:model.max_seq_len))
        logits = nothing
        for columns in (1:4, 5:8)
            logits, position = LifeAI._bf16a_static_prefill_chunk_core(
                model,
                parameters,
                tokens[columns, :],
                chunk_keys,
                chunk_values,
                position,
                cos_table,
                sin_table,
                key_positions,
            )
        end

        full_keys = Tuple(
            zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
        )
        full_values = Tuple(
            zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
        )
        full_logits = LifeAI._bf16a_static_prefill(
            model,
            parameters,
            tokens,
            full_keys,
            full_values,
            cos_table,
            sin_table,
            LifeAI._bf16a_causal_mask(8, 8),
        )
        @test logits == full_logits[:, end:end, :]
        @test position == Int32[8]
        for layer in 1:model.num_layers
            @test chunk_keys[layer][:, :, 1:8, :] ==
                full_keys[layer][:, :, 1:8, :]
            @test chunk_values[layer][:, :, 1:8, :] ==
                full_values[layer][:, :, 1:8, :]
        end

        # Left-padding cache entries are masked, so their arbitrary token id
        # cannot change the real prompt's final logits.
        function padded_logits(pad_token_id)
            keys = Tuple(
                zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
            )
            values = Tuple(
                zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
            )
            padded_plan = plan_qwen3_xla_window(
                5,
                1;
                context_tokens=16,
                chunk_tokens=8,
            )
            padded_tokens = qwen3_xla_pad_prompt(
                vec(tokens[1:5, :]),
                padded_plan;
                pad_token_id,
            )
            result, _ = LifeAI._bf16a_static_prefill_chunk_core(
                model,
                parameters,
                reshape(padded_tokens, :, 1),
                keys,
                values,
                Int32[0],
                cos_table,
                sin_table,
                qwen3_xla_key_positions(padded_plan),
            )
            return result
        end
        @test padded_logits(1) == padded_logits(19)
    end
end

@testset "XLA deployment source enforces a single parameter-tree transfer" begin
    source = read(joinpath(
        @__DIR__,
        "..",
        "src",
        "generation",
        "qwen3_xla_deployment.jl",
    ), String)
    @test length(collect(eachmatch(
        r"Reactant\.to_rarray\(host_parameters\)",
        source,
    ))) == 1
    @test !occursin("to_rarray(loaded.parameters)", source)
    @test occursin("original_parameter_tree_constructed=false", source)
    @test occursin("device_parameter_tree_count=1", source)
end

function _qwen3_xla_deployment_script_source(name::AbstractString)
    return read(joinpath(@__DIR__, "..", "scripts", name), String)
end

function _qwen3_xla_deployment_required_match(pattern::Regex, source::AbstractString)
    result = match(pattern, source)
    @test result !== nothing
    return result
end

function _qwen3_xla_deployment_last_match(pattern::Regex, source::AbstractString)
    result = nothing
    for candidate in eachmatch(pattern, source)
        result = candidate
    end
    return result
end

@testset "XLA CLI acceptance contract is fail-closed" begin
    benchmark = _qwen3_xla_deployment_script_source(
        "benchmark_qwen3_xla_deployment.jl",
    )
    chat = _qwen3_xla_deployment_script_source("run_qwen3_xla_chat.jl")

    function allocator_fraction_default(source)
        result = _qwen3_xla_deployment_required_match(
            r"""get\(\s*ENV,\s*"XLA_REACTANT_GPU_MEM_FRACTION",\s*"([^"]+)"\s*\)"""s,
            source,
        )
        return result === nothing ? nothing : only(result.captures)
    end
    @test allocator_fraction_default(benchmark) == "0.87"
    @test allocator_fraction_default(chat) == "0.87"

    pinned_reference = _qwen3_xla_deployment_required_match(
        r"""const\s+QWEN3_XLA_DEPLOYMENT_CUDA_REFERENCE_SHA256\s*=\s*"([0-9a-f]{64})"""s,
        benchmark,
    )
    @test pinned_reference !== nothing
    if pinned_reference !== nothing
        @test only(pinned_reference.captures) ==
            "83f62afbbb470b695b6990a3b86a8860407a37874354d6b039e1ce19917e2747"
    end
    @test occursin(
        r"reference_sha256\s*==\s*QWEN3_XLA_DEPLOYMENT_CUDA_REFERENCE_SHA256\s*\|\|\s*error"s,
        benchmark,
    )
    @test occursin(
        r"""Int\(reference\["schema_version"\]\)\s*==\s*2\s*\|\|\s*error"""s,
        benchmark,
    )

    frozen_cases = _qwen3_xla_deployment_required_match(
        r"Set\(keys\(reference_cases\)\)\s*==\s*Set\(\((.*?)\)\)\s*\|\|\s*error"s,
        benchmark,
    )
    @test frozen_cases !== nothing
    if frozen_cases !== nothing
        names = Set(
            match.captures[1]
            for match in eachmatch(
                r"\"([^\"]+)\"",
                only(frozen_cases.captures),
            )
        )
        @test names == Set((
            "single_chunk_64",
            "left_padded_65",
            "full_prompt_3584",
        ))
    end
    @test occursin(
        r"""Int\(case\["generated_tokens"\]\)\s*==\s*32"""s,
        benchmark,
    )
    @test occursin(
        r"""length\(case\["generated_ids_0_based"\]\)\s*==\s*32"""s,
        benchmark,
    )

    @test occursin(r"for\s+index\s+in\s+1:8", benchmark)
    @test occursin(
        r"steady_runs\s*=\s*short_runs\[2:end\]"s,
        benchmark,
    )
    @test occursin(
        r"median\(\s*\[run\.prefill_seconds\s+for\s+run\s+in\s+steady_runs\]\s*\)"s,
        benchmark,
    )
    @test occursin(
        r"median\(\s*\[run\.decode_tokens_per_second\s+for\s+run\s+in\s+steady_runs\]\s*\)"s,
        benchmark,
    )
    @test !occursin(
        r"steady_(?:prefill_seconds|decode_tps)\s*=\s*short_runs\[end\]"s,
        benchmark,
    )

    @test occursin("--loop-ms=200", benchmark)
    @test occursin(
        r"physical_monitor\s*=\s*run\(.*?wait\s*=\s*false"s,
        benchmark,
    )
    @test occursin(r"atexit\(stop_physical_monitor\)", benchmark)
    @test occursin(
        r"sample_interval_ms\s*=\s*200",
        benchmark,
    )
    @test occursin(
        r"minimum_physical_free\s*=\s*physical_trace\.minimum_free_bytes",
        benchmark,
    )
    @test occursin(
        r"allocator_minimum_free_limit_bytes\s*=\s*profile\.workspace_reserve_bytes",
        benchmark,
    )
    @test occursin(
        r"physical_minimum_free_limit_bytes\s*=\s*profile\.workspace_reserve_bytes",
        benchmark,
    )
    @test occursin(
        r"minimum_physical_free\s*>=\s*profile\.workspace_reserve_bytes",
        benchmark,
    )

    fail_closed = _qwen3_xla_deployment_last_match(
        r"closed\s*\|\|\s*error\s*\(",
        benchmark,
    )
    report_write = _qwen3_xla_deployment_last_match(r"JSON3\.pretty\(", benchmark)
    @test fail_closed !== nothing
    @test report_write !== nothing
    if fail_closed !== nothing && report_write !== nothing
        @test report_write.offset < fail_closed.offset
    end
end

@testset "XLA request validation precedes device execution" begin
    prefill_reached = Ref(false)
    decode_reached = Ref(false)
    compiled_prefill = function (_...)
        prefill_reached[] = true
        error("compiled prefill must not run for invalid request metadata")
    end
    compiled_decode = function (_...)
        decode_reached[] = true
        error("compiled decode must not run for invalid request metadata")
    end
    session = HFQwen3BF16XLASession(
        (; vocab_size=16),
        nothing,
        (; eos_ids=Int[]),
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        compiled_prefill,
        compiled_decode,
        :greedy,
        16,
        8,
        0,
        7,
        nothing,
    )

    @test_throws ArgumentError generate_hf_qwen3_bf16_xla!(
        session,
        [2, 3];
        max_new_tokens=1,
        pad_token_id=17,
        stop_token_ids=Int[],
    )
    @test session.position == 7
    @test !prefill_reached[]
    @test !decode_reached[]

    @test_throws ArgumentError generate_hf_qwen3_bf16_xla!(
        session,
        [2, 3];
        max_new_tokens=1,
        pad_token_id=1,
        stop_token_ids=[17],
    )
    @test session.position == 7
    @test !prefill_reached[]
    @test !decode_reached[]
end
