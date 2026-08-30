using Test
using BFloat16s: BFloat16
using JSON3
using Lux
using Random: Xoshiro
import LifeAI
using LifeAI:
    GPTModel,
    Qwen3DeploymentProfile,
    decode_hf_qwen3_bf16!,
    fit_qwen3_chat_context,
    generate_hf_qwen3_bf16!,
    generate_hf_text!,
    hf_generation_config,
    hf_qwen3_bf16_accel_forward,
    init_hf_qwen3_bf16_session,
    load_hf_qwen3_bf16_session,
    load_hf_qwen3_tokenizer,
    load_qwen3_deployment_profile,
    prefill_hf_qwen3_bf16!,
    qwen3_dense_spec,
    qwen3_kv_cache_bytes,
    reset_hf_qwen3_bf16_session!,
    verify_qwen3_deployment_assets

isdefined(@__MODULE__, :write_qwen3_tokenizer_fixture) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tokenizer_fixture.jl"))
isdefined(@__MODULE__, :LIFEAI_REPO_ROOT) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

const _QWEN3_CUDA_DEPLOYMENT_PROFILE_PATH = joinpath(
    LIFEAI_REPO_ROOT,
    "configs",
    "deployment",
    "qwen3_8b_4090d_bf16_daily.json",
)

function _qwen3_deployment_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected Qwen3 deployment call to fail")
end

function _qwen3_cuda_deployment_tiny_bundle(directory; max_seq_len=128)
    write_qwen3_tokenizer_fixture(directory)
    tokenizer = load_hf_qwen3_tokenizer(directory; revision="qwen3_cuda_deployment-test")
    model = GPTModel(
        263,
        8,
        4,
        2;
        num_kv_heads=2,
        head_dim=4,
        mlp_hidden_dim=12,
        use_bias=false,
        lm_head_bias=false,
        is_causal=true,
        use_rope=true,
        use_qk_norm=true,
        qk_norm_epsilon=1.0f-6,
        max_seq_len,
        rope_theta=1_000_000.0,
        rope_style=:rotate_half,
        norm_epsilon=1.0f-6,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=false,
    )
    parameters = Lux.fmap(
        value -> value isa AbstractArray ? BFloat16.(value) : value,
        Lux.initialparameters(Xoshiro(19), model),
    )
    return (;
        model,
        parameters,
        tokenizer,
        generation_config=hf_generation_config(tokenizer),
    )
end

@testset "4090D profile and exact context budget" begin
    profile = load_qwen3_deployment_profile(_QWEN3_CUDA_DEPLOYMENT_PROFILE_PATH)
    @test profile.schema_version == 1
    @test profile.variant === :qwen3_8b
    @test profile.context_tokens == 4096
    @test profile.max_prompt_tokens == 3584
    @test profile.max_new_tokens == 512
    @test profile.prefill_chunk_tokens == 64
    @test profile.prefill_reclaim_interval_chunks == 1
    @test profile.decode_reclaim_interval_tokens == 8
    @test profile.strategy === :sample
    @test !profile.enable_thinking
    @test profile.workspace_reserve_bytes == 5 * 1024^3
    @test profile.asset_manifest == "qwen3_8b_frozen_assets.json"

    qwen8 = qwen3_dense_spec(:qwen3_8b)
    qwen14 = qwen3_dense_spec(:qwen3_14b)
    @test qwen3_kv_cache_bytes(qwen8, 1024) == 150_994_944
    @test qwen3_kv_cache_bytes(qwen8, 4096) == 603_979_776
    @test qwen3_kv_cache_bytes(qwen14, 1024) == 167_772_160
    @test qwen3_kv_cache_bytes(qwen14, 4096) == 671_088_640
    @test qwen3_kv_cache_bytes(qwen8, 0) == 0
    @test qwen3_kv_cache_bytes(qwen8, 16; batch_size=2, dtype_bytes=4) ==
        4 * qwen3_kv_cache_bytes(qwen8, 16)
    narrow_bytes = qwen3_kv_cache_bytes(
        qwen8,
        Int32(16);
        batch_size=Int128(2),
        dtype_bytes=big(4),
    )
    @test narrow_bytes isa Int
    @test narrow_bytes == qwen3_kv_cache_bytes(
        qwen8,
        16;
        batch_size=2,
        dtype_bytes=4,
    )
    @test_throws ArgumentError qwen3_kv_cache_bytes(qwen8, -1)
    @test_throws ArgumentError qwen3_kv_cache_bytes(qwen8, typemax(Int))
    too_large = big(typemax(Int)) + 1
    for invalid_context in (true, too_large)
        @test_throws ArgumentError qwen3_kv_cache_bytes(qwen8, invalid_context)
    end
    for invalid_batch in (true, too_large)
        @test_throws ArgumentError qwen3_kv_cache_bytes(
            qwen8,
            16;
            batch_size=invalid_batch,
        )
    end
    for invalid_dtype in (true, too_large)
        @test_throws ArgumentError qwen3_kv_cache_bytes(
            qwen8,
            16;
            dtype_bytes=invalid_dtype,
        )
    end
    @test_throws ArgumentError qwen3_kv_cache_bytes(
        (; num_layers=true, head_dim=128, num_kv_heads=8),
        16,
    )
    @test_throws ArgumentError qwen3_kv_cache_bytes(
        (; num_layers=36, head_dim=too_large, num_kv_heads=8),
        16,
    )
    @test_throws ArgumentError qwen3_kv_cache_bytes(
        qwen8,
        16;
        batch_size=typemax(Int),
    )
    @test_throws ArgumentError qwen3_kv_cache_bytes(
        qwen8,
        16;
        dtype_bytes=typemax(Int),
    )

    mktempdir() do directory
        object = JSON3.read(read(_QWEN3_CUDA_DEPLOYMENT_PROFILE_PATH, String), Dict{String,Any})
        object["surprise"] = true
        invalid = joinpath(directory, "unknown.json")
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        write(invalid, "[]")
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        object = JSON3.read(read(_QWEN3_CUDA_DEPLOYMENT_PROFILE_PATH, String), Dict{String,Any})
        object["enable_thinking"] = 1
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        object["enable_thinking"] = false
        object["prefill_chunk_tokens"] = 5000
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        object["prefill_chunk_tokens"] = 64
        object["prefill_reclaim_interval_chunks"] = 0
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        object["prefill_reclaim_interval_chunks"] = 1
        object["decode_reclaim_interval_tokens"] = 0
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        delete!(object, "surprise")
        object["decode_reclaim_interval_tokens"] = 8
        object["max_prompt_tokens"] = 4000
        invalid = joinpath(directory, "overflow.json")
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        object["max_prompt_tokens"] = 3584
        object["revision"] = "moving-target"
        invalid = joinpath(directory, "revision.json")
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        object = JSON3.read(read(_QWEN3_CUDA_DEPLOYMENT_PROFILE_PATH, String), Dict{String,Any})
        object["context_tokens"] = typemax(Int)
        object["max_prompt_tokens"] = typemax(Int)
        object["max_new_tokens"] = 1
        invalid = joinpath(directory, "overflow-int.json")
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)
    end

    mktempdir() do directory
        write(joinpath(directory, "hello.bin"), "hello")
        manifest = Dict(
            "schema_version" => 1,
            "model_id" => "Qwen/test",
            "revision" => "frozen",
            "files" => [Dict(
                "name" => "hello.bin",
                "size" => 5,
                "sha256" => "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824",
            )],
        )
        path = joinpath(directory, "assets.json")
        write(path, JSON3.write(manifest))
        report = verify_qwen3_deployment_assets(
            directory,
            path;
            model_id="Qwen/test",
            revision="frozen",
        )
        @test report.total_bytes == 5
        @test only(report.files).name == "hello.bin"
        @test LifeAI._qwen3_add_asset_bytes(0, 5) == 5
        @test LifeAI._qwen3_add_asset_bytes(typemax(Int) - 1, 1) ==
            typemax(Int)
        overflow_failure = _qwen3_deployment_captured_error() do
            LifeAI._qwen3_add_asset_bytes(typemax(Int), 1)
        end
        @test overflow_failure isa ArgumentError
        @test sprint(showerror, overflow_failure) ==
            "ArgumentError: asset manifest total byte count exceeds the host integer range"
        substring_report = verify_qwen3_deployment_assets(
            directory,
            path;
            model_id=SubString("xQwen/test", 2),
            revision=SubString("xfrozen", 2),
        )
        @test substring_report.model_id == "Qwen/test"
        for (options, message) in (
            ((; model_id=Symbol("Qwen/test")), "model_id must be a string"),
            ((; model_id=true), "model_id must be a string"),
            ((; revision=:frozen), "revision must be a string"),
            ((; revision=true), "revision must be a string"),
        )
            failure = _qwen3_deployment_captured_error() do
                verify_qwen3_deployment_assets(
                    directory,
                    path;
                    options...,
                )
            end
            @test failure isa ArgumentError
            @test sprint(showerror, failure) == "ArgumentError: $message"
        end

        write(joinpath(directory, "model.safetensors"), "unverified")
        @test_throws ArgumentError verify_qwen3_deployment_assets(directory, path)
        rm(joinpath(directory, "model.safetensors"))

        manifest["files"][1]["size"] = 4
        write(path, JSON3.write(manifest))
        @test_throws ArgumentError verify_qwen3_deployment_assets(directory, path)
        manifest["files"][1]["size"] = 5
        manifest["files"][1]["name"] = "../hello.bin"
        write(path, JSON3.write(manifest))
        @test_throws ArgumentError verify_qwen3_deployment_assets(directory, path)
    end
end

@testset "Qwen3 deployment profile constructor is strict" begin
    loaded = load_qwen3_deployment_profile(_QWEN3_CUDA_DEPLOYMENT_PROFILE_PATH)
    base = ntuple(
        index -> getfield(loaded, index),
        fieldcount(Qwen3DeploymentProfile),
    )
    replace_field = (values, index, value) -> ntuple(
        current -> current == index ? value : values[current],
        length(values),
    )
    integer_fields = (
        1 => "schema_version",
        7 => "context_tokens",
        8 => "max_prompt_tokens",
        9 => "max_new_tokens",
        10 => "prefill_chunk_tokens",
        11 => "prefill_reclaim_interval_chunks",
        12 => "decode_reclaim_interval_tokens",
        16 => "top_k",
        18 => "minimum_gpu_bytes",
        19 => "workspace_reserve_bytes",
    )
    string_fields = (
        2 => "name",
        3 => "model_id",
        4 => "revision",
        20 => "asset_manifest",
    )
    symbol_fields = (5 => "variant", 6 => "weight_dtype", 14 => "strategy")

    normalized_values = ntuple(length(base)) do index
        if any(first(field) == index for field in integer_fields)
            big(base[index])
        elseif any(first(field) == index for field in string_fields)
            SubString("x$(base[index])", 2)
        elseif index in (15, 17)
            BigFloat(base[index])
        else
            base[index]
        end
    end
    normalized = Qwen3DeploymentProfile(normalized_values...)
    @test ntuple(index -> getfield(normalized, index), length(base)) == base
    @test all(
        getfield(normalized, index) isa Int for (index, _) in integer_fields
    )
    @test all(
        getfield(normalized, index) isa String for (index, _) in string_fields
    )
    @test normalized.temperature isa Float32
    @test normalized.top_p isa Float32

    too_large = big(typemax(Int)) + 1
    for (index, label) in integer_fields
        for (value, message) in (
            (true, "$label must be an integer"),
            (1.0, "$label must be an integer"),
            (too_large, "$label is outside the host integer range"),
        )
            failure = _qwen3_deployment_captured_error() do
                Qwen3DeploymentProfile(replace_field(base, index, value)...)
            end
            @test failure isa ArgumentError
            @test sprint(showerror, failure) == "ArgumentError: $message"
        end
    end

    for (index, label) in string_fields
        failure = _qwen3_deployment_captured_error() do
            Qwen3DeploymentProfile(replace_field(base, index, Symbol(label))...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: $label must be a string"
    end
    for (index, label) in symbol_fields
        failure = _qwen3_deployment_captured_error() do
            Qwen3DeploymentProfile(replace_field(base, index, label)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: $label must be a Symbol"
    end
    thinking_failure = _qwen3_deployment_captured_error() do
        Qwen3DeploymentProfile(replace_field(base, 13, 0)...)
    end
    @test thinking_failure isa ArgumentError
    @test sprint(showerror, thinking_failure) ==
        "ArgumentError: enable_thinking must be a boolean"

    for (index, label) in ((15, "temperature"), (17, "top_p"))
        for (value, message) in (
            (true, "$label must be a real number"),
            (
                big(10)^1000,
                "$label must be positive and finite at Float32 precision",
            ),
            (
                BigFloat("1e-1000"),
                "$label must be positive and finite at Float32 precision",
            ),
        )
            failure = _qwen3_deployment_captured_error() do
                Qwen3DeploymentProfile(replace_field(base, index, value)...)
            end
            @test failure isa ArgumentError
            @test sprint(showerror, failure) == "ArgumentError: $message"
        end
    end

    semantic_failures = (
        (1, 2, "unsupported Qwen3 deployment profile schema_version 2"),
        (2, "", "deployment profile name must not be empty"),
        (
            3,
            "wrong-model",
            "profile model_id does not match the frozen qwen3_8b spec",
        ),
        (
            4,
            "moving-target",
            "profile revision does not match the frozen qwen3_8b spec",
        ),
        (
            6,
            :f16,
            "the Week 19 deployment runtime currently requires weight_dtype=bf16",
        ),
        (14, :config, "strategy must be greedy or sample"),
        (16, 0, "top_k must be positive"),
        (17, 1.5, "top_p must be finite and in (0, 1]"),
        (19, -1, "workspace_reserve_bytes must be non-negative"),
        (20, "", "asset_manifest must not be empty"),
    )
    for (index, value, message) in semantic_failures
        failure = _qwen3_deployment_captured_error() do
            Qwen3DeploymentProfile(replace_field(base, index, value)...)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end
end

@testset "dense session options fail before checkpoint I/O" begin
    mktempdir() do directory
        too_large = big(typemax(Int)) + 1
        cases = (
            (
                needle="context_tokens must be an integer",
                options=(; context_tokens=true, prefill_chunk_tokens=1),
            ),
            (
                needle="context_tokens is outside the host integer range",
                options=(; context_tokens=too_large, prefill_chunk_tokens=1),
            ),
            (
                needle="context_tokens must be positive",
                options=(; context_tokens=0, prefill_chunk_tokens=1),
            ),
            (
                needle="prefill_chunk_tokens must be an integer",
                options=(; context_tokens=8, prefill_chunk_tokens=true),
            ),
            (
                needle="prefill_chunk_tokens is outside the host integer range",
                options=(; context_tokens=8, prefill_chunk_tokens=too_large),
            ),
            (
                needle="prefill_chunk_tokens must be in 1:context_tokens",
                options=(; context_tokens=8, prefill_chunk_tokens=0),
            ),
            (
                needle="prefill_chunk_tokens must be in 1:context_tokens",
                options=(; context_tokens=8, prefill_chunk_tokens=9),
            ),
        )
        for case in cases
            failure = _qwen3_deployment_captured_error() do
                load_hf_qwen3_bf16_session(
                    directory;
                    case.options...,
                )
            end
            @test failure isa ArgumentError
            @test occursin(case.needle, sprint(showerror, failure))
            @test !occursin("tokenizer", sprint(showerror, failure))
        end

        variant_failure = _qwen3_deployment_captured_error() do
            load_hf_qwen3_bf16_session(
                directory;
                context_tokens=8,
                prefill_chunk_tokens=1,
                variant=:not_qwen3,
            )
        end
        @test variant_failure isa ArgumentError
        @test occursin(
            "unknown Qwen3 dense variant",
            sprint(showerror, variant_failure),
        )

        io_failure = _qwen3_deployment_captured_error() do
            load_hf_qwen3_bf16_session(
                directory;
                context_tokens=Int32(8),
                prefill_chunk_tokens=big(1),
            )
        end
        @test io_failure isa ArgumentError
        @test occursin(
            "required Qwen3 tokenizer file",
            sprint(showerror, io_failure),
        )
    end
end

@testset "chunked last-logit prefill and reusable static cache" begin
    mktempdir() do directory
        bundle = _qwen3_cuda_deployment_tiny_bundle(directory; max_seq_len=32)
        @test_throws ArgumentError init_hf_qwen3_bf16_session(
            bundle;
            context_tokens=true,
            prefill_chunk_tokens=1,
        )
        @test_throws ArgumentError init_hf_qwen3_bf16_session(
            bundle;
            context_tokens=16,
            prefill_chunk_tokens=true,
        )
        @test_throws ArgumentError init_hf_qwen3_bf16_session(
            bundle;
            context_tokens=big(typemax(Int)) + 1,
            prefill_chunk_tokens=1,
        )
        session = init_hf_qwen3_bf16_session(
            bundle;
            context_tokens=16,
            prefill_chunk_tokens=3,
        )
        too_large = big(typemax(Int)) + 1
        session.position = 7
        for invalid_prompt in (Bool[true], BigInt[too_large])
            @test_throws ArgumentError prefill_hf_qwen3_bf16!(
                session,
                invalid_prompt,
            )
            @test session.position == 7
        end
        for integer_type in (Int8, Int32, Int128, BigInt)
            normalized_tokens = LifeAI._qwen3_session_token_vector(
                session,
                integer_type.([1, 2]),
            )
            @test normalized_tokens == [1, 2]
            @test eltype(normalized_tokens) === Int
        end
        reset_hf_qwen3_bf16_session!(session)
        tokens = [1, 5, 8, 3, 12]
        skipped = LifeAI._bf16a_forward_pass(
            session.model,
            session.parameters,
            reshape(tokens[1:3], :, 1),
            session.caches,
            session.cos_table,
            session.sin_table,
            LifeAI._bf16a_causal_mask(3, 3);
            start_pos=1,
            project_last_token_only=true,
            project_logits=false,
            capture_trace=false,
        )
        @test skipped.embedding === nothing
        @test skipped.block_outputs === nothing
        @test skipped.final_hidden === nothing
        @test skipped.logits === nothing

        baseline = hf_qwen3_bf16_accel_forward(
            bundle.model,
            bundle.parameters,
            reshape(tokens, :, 1);
            decode_token=[7],
            greedy_steps=4,
        )
        chunk_positions = Int[]
        logits = prefill_hf_qwen3_bf16!(
            session,
            tokens;
            on_chunk=position -> push!(chunk_positions, position),
        )
        @test logits == baseline.logits[:, end:end, :]
        @test chunk_positions == [3, 5]
        @test session.position == length(tokens)
        for invalid_token in (true, too_large)
            @test_throws ArgumentError decode_hf_qwen3_bf16!(
                session,
                invalid_token,
            )
            @test session.position == length(tokens)
        end
        @test decode_hf_qwen3_bf16!(session, Int128(7)) == baseline.decode_logits
        @test session.position == length(tokens) + 1

        first_run = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=4,
            strategy=:greedy,
            stop_token_ids=Int[],
        )
        @test first_run.generated_ids == baseline.greedy_tokens
        @test length(first_run.trace) == 4
        @test first_run.stop_reason === :length
        @test first_run.prefill_seconds >= 0
        @test first_run.decode_seconds >= 0
        generation_prefill_positions = Int[]
        generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=1,
            strategy=:greedy,
            stop_token_ids=Int[],
            on_prefill_chunk=position -> push!(
                generation_prefill_positions,
                position,
            ),
        )
        @test generation_prefill_positions == [3, 5]
        @test_throws ArgumentError generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=typemax(Int),
        )
        full_context = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=session.context_tokens - length(tokens),
            strategy=:greedy,
            stop_token_ids=Int[],
        )
        @test length(full_context.generated_ids) ==
            session.context_tokens - length(tokens)
        @test session.position == session.context_tokens - 1

        second_run = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=4,
            strategy=:greedy,
            stop_token_ids=Int[],
        )
        @test second_run.generated_ids == first_run.generated_ids
        @test session.position == length(tokens) + 3

        eos_run = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=4,
            strategy=:greedy,
            stop_token_ids=[first(first_run.generated_ids)],
        )
        @test length(eos_run.generated_ids) == 1
        @test eos_run.stop_reason === :eos
        @test session.position == length(tokens)

        callback_ids = Int[]
        sampled = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=4,
            strategy=:sample,
            temperature=0.7,
            top_k=5,
            top_p=0.8,
            rng=Xoshiro(91),
            stop_token_ids=Int[],
            on_token=token -> push!(callback_ids, token),
        )
        sampled_repeat = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=4,
            strategy=:sample,
            temperature=0.7,
            top_k=5,
            top_p=0.8,
            rng=Xoshiro(91),
            stop_token_ids=Int[],
        )
        @test sampled.generated_ids == sampled_repeat.generated_ids
        @test callback_ids == sampled.generated_ids

        preserved_sampling_position = session.position
        invalid_sampling_options = (
            (; temperature=true, top_k=5, top_p=0.8),
            (;
                temperature=big(floatmax(Float32)) * 2,
                top_k=5,
                top_p=0.8,
            ),
            (; temperature=0.7, top_k=true, top_p=0.8),
            (; temperature=0.7, top_k=too_large, top_p=0.8),
            (; temperature=0.7, top_k=5, top_p=true),
            (; temperature=0.7, top_k=5, top_p=big"1e-1000"),
        )
        for options in invalid_sampling_options
            @test_throws ArgumentError generate_hf_qwen3_bf16!(
                session,
                tokens;
                max_new_tokens=4,
                strategy=:sample,
                options...,
                stop_token_ids=Int[],
            )
            @test session.position == preserved_sampling_position
        end

        wide_sample = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=1,
            strategy=:sample,
            temperature=Float64(0.7),
            top_k=Int128(5),
            top_p=Float64(0.8),
            rng=Xoshiro(91),
            stop_token_ids=Int[],
        )
        @test length(wide_sample.generated_ids) == 1
        @test only(wide_sample.trace).temperature isa Float32
        @test only(wide_sample.trace).top_k === 5
        @test only(wide_sample.trace).top_p isa Float32

        preserved_position = session.position
        for invalid_stops in (Bool[true], BigInt[too_large])
            @test_throws ArgumentError generate_hf_qwen3_bf16!(
                session,
                tokens;
                max_new_tokens=0,
                stop_token_ids=invalid_stops,
            )
            @test session.position == preserved_position
        end
        for invalid_output in (true, too_large)
            @test_throws ArgumentError generate_hf_qwen3_bf16!(
                session,
                tokens;
                max_new_tokens=invalid_output,
                stop_token_ids=Int[],
            )
            @test session.position == preserved_position
        end
        output_preflight = _qwen3_deployment_captured_error() do
            generate_hf_qwen3_bf16!(
                session,
                nothing;
                max_new_tokens=true,
                stop_token_ids=Int[],
            )
        end
        @test output_preflight isa ArgumentError
        @test occursin(
            "max_new_tokens must be an integer",
            sprint(showerror, output_preflight),
        )
        @test session.position == preserved_position

        zero = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=Int128(0),
        )
        @test isempty(zero.generated_ids)
        @test session.position == 0

        reset_hf_qwen3_bf16_session!(session)
        @test session.position == 0
        @test_throws ArgumentError decode_hf_qwen3_bf16!(session, 1)
        @test_throws ArgumentError prefill_hf_qwen3_bf16!(session, fill(1, 17))
        @test_throws ArgumentError generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=12,
        )
        @test_throws ArgumentError generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=1,
            strategy=:beam,
        )
    end
end

@testset "daily chat history compaction preserves newest request" begin
    mktempdir() do directory
        bundle = _qwen3_cuda_deployment_tiny_bundle(directory; max_seq_len=256)
        session = init_hf_qwen3_bf16_session(
            bundle;
            context_tokens=256,
            prefill_chunk_tokens=8,
        )
        messages = [
            (role="system", content="S"),
            (role="user", content=repeat("old ", 12)),
            (role="assistant", content=repeat("answer ", 12)),
            (role="user", content="new"),
        ]
        full = fit_qwen3_chat_context(
            session,
            messages;
            max_prompt_tokens=255,
            enable_thinking=false,
        )
        wide_full = fit_qwen3_chat_context(
            session,
            messages;
            max_prompt_tokens=Int128(255),
            enable_thinking=false,
        )
        @test wide_full == full

        overflow_integer = big(typemax(Int)) + 1
        invalid_fit_limits = (
            (value=true, message="max_prompt_tokens must be an integer"),
            (value=255.0, message="max_prompt_tokens must be an integer"),
            (
                value=overflow_integer,
                message="max_prompt_tokens is outside the host integer range",
            ),
        )
        for case in invalid_fit_limits
            failure = _qwen3_deployment_captured_error() do
                fit_qwen3_chat_context(
                    session,
                    nothing;
                    max_prompt_tokens=case.value,
                )
            end
            @test failure isa ArgumentError
            @test occursin(case.message, sprint(showerror, failure))
            @test session.position == 0
        end

        fitted = fit_qwen3_chat_context(
            session,
            messages;
            max_prompt_tokens=50,
            enable_thinking=false,
        )
        @test full.dropped_messages == 0
        @test fitted.dropped_messages == 2
        @test first(fitted.messages) == first(messages)
        @test last(fitted.messages) == last(messages)
        @test length(fitted.prompt_ids) <= 50
        @test occursin("new", fitted.prompt)
        text_result = generate_hf_text!(
            session,
            "hi";
            chat=true,
            enable_thinking=false,
            max_prompt_tokens=240,
            max_new_tokens=2,
            strategy=:greedy,
            stop_token_ids=Int[],
        )
        @test length(text_result.generated_ids) == 2
        @test occursin("<|im_start|>user", text_result.prompt)
        raw_result = generate_hf_text!(
            session,
            "hi";
            chat=false,
            max_prompt_tokens=240,
            max_new_tokens=1,
            strategy=:greedy,
            stop_token_ids=Int[],
        )
        @test raw_result.prompt == "hi"
        @test raw_result.dropped_messages == 0

        preserved_position = session.position
        invalid_generation_budgets = (
            (
                options=(; max_new_tokens=true),
                message="max_new_tokens must be an integer",
            ),
            (
                options=(; max_new_tokens=1.0),
                message="max_new_tokens must be an integer",
            ),
            (
                options=(; max_new_tokens=overflow_integer),
                message="max_new_tokens is outside the host integer range",
            ),
            (
                options=(; max_new_tokens=typemin(Int)),
                message="max_new_tokens must be non-negative",
            ),
            (
                options=(; max_new_tokens=Int128(session.context_tokens)),
                message="max_new_tokens must be less than session.context_tokens",
            ),
            (
                options=(; max_new_tokens=1, max_prompt_tokens=true),
                message="max_prompt_tokens must be an integer",
            ),
            (
                options=(;
                    max_new_tokens=1,
                    max_prompt_tokens=overflow_integer,
                ),
                message="max_prompt_tokens is outside the host integer range",
            ),
            (
                options=(; max_new_tokens=1, max_prompt_tokens=0),
                message="max_prompt_tokens must be in",
            ),
            (
                options=(;
                    max_new_tokens=1,
                    max_prompt_tokens=session.context_tokens,
                ),
                message="max_prompt_tokens must be in",
            ),
        )
        for case in invalid_generation_budgets
            failure = _qwen3_deployment_captured_error() do
                generate_hf_text!(
                    session,
                    nothing;
                    chat=false,
                    strategy=:greedy,
                    stop_token_ids=Int[],
                    case.options...,
                )
            end
            @test failure isa ArgumentError
            @test occursin(case.message, sprint(showerror, failure))
            @test session.position == preserved_position
        end

        chat_budget_preflight = _qwen3_deployment_captured_error() do
            generate_hf_text!(
                session,
                nothing;
                chat=true,
                max_new_tokens=true,
            )
        end
        @test chat_budget_preflight isa ArgumentError
        @test occursin(
            "max_new_tokens must be an integer",
            sprint(showerror, chat_budget_preflight),
        )
        @test session.position == preserved_position

        explicit_wide_budget = generate_hf_text!(
            session,
            "hi";
            chat=false,
            max_new_tokens=Int128(0),
            max_prompt_tokens=big(session.context_tokens),
            strategy=:greedy,
            stop_token_ids=Int[],
        )
        @test isempty(explicit_wide_budget.generated_ids)
        @test explicit_wide_budget.prompt == "hi"
        @test session.position == 0

        default_budget = generate_hf_text!(
            session,
            "hi";
            chat=false,
            max_new_tokens=Int32(0),
            strategy=:greedy,
            stop_token_ids=Int[],
        )
        @test isempty(default_budget.generated_ids)
        @test default_budget.prompt_ids == explicit_wide_budget.prompt_ids
        @test session.position == 0

        @test_throws ArgumentError fit_qwen3_chat_context(
            session,
            [(role="user", content=repeat("x", 80))];
            max_prompt_tokens=10,
        )
    end
end

@testset "interactive CLI avoids soft-scope history reassignment" begin
    script = read(
        joinpath(LIFEAI_REPO_ROOT, "scripts", "run_qwen3_cuda_chat.jl"),
        String,
    )
    @test !occursin("history = convert(", script)
    @test occursin(
        "empty!(history)\n    append!(history, result.messages)",
        script,
    )
    @test occursin("isempty(line) && eof(stdin) && break", script)
end
