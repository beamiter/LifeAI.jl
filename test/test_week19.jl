using Test
using BFloat16s: BFloat16
using JSON3
using Lux
using Random: Xoshiro
import LifeAI
using LifeAI:
    GPTModel,
    decode_hf_qwen3_bf16!,
    fit_qwen3_chat_context,
    generate_hf_qwen3_bf16!,
    generate_hf_text!,
    hf_generation_config,
    hf_qwen3_bf16_accel_forward,
    init_hf_qwen3_bf16_session,
    load_hf_qwen3_tokenizer,
    load_qwen3_deployment_profile,
    prefill_hf_qwen3_bf16!,
    qwen3_dense_spec,
    qwen3_kv_cache_bytes,
    reset_hf_qwen3_bf16_session!,
    verify_qwen3_deployment_assets

isdefined(@__MODULE__, :write_week08_tokenizer_fixture) ||
    include("week08_fixture.jl")

const _WEEK19_PROFILE_PATH = joinpath(
    @__DIR__,
    "..",
    "configs",
    "deployment",
    "qwen3_8b_4090d_bf16_daily.json",
)

function _week19_tiny_bundle(directory; max_seq_len=128)
    write_week08_tokenizer_fixture(directory)
    tokenizer = load_hf_qwen3_tokenizer(directory; revision="week19-test")
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
    profile = load_qwen3_deployment_profile(_WEEK19_PROFILE_PATH)
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
    @test_throws ArgumentError qwen3_kv_cache_bytes(qwen8, -1)
    @test_throws ArgumentError qwen3_kv_cache_bytes(qwen8, typemax(Int))

    mktempdir() do directory
        object = JSON3.read(read(_WEEK19_PROFILE_PATH, String), Dict{String,Any})
        object["surprise"] = true
        invalid = joinpath(directory, "unknown.json")
        write(invalid, JSON3.write(object))
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        write(invalid, "[]")
        @test_throws ArgumentError load_qwen3_deployment_profile(invalid)

        object = JSON3.read(read(_WEEK19_PROFILE_PATH, String), Dict{String,Any})
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

        object = JSON3.read(read(_WEEK19_PROFILE_PATH, String), Dict{String,Any})
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

@testset "chunked last-logit prefill and reusable static cache" begin
    mktempdir() do directory
        bundle = _week19_tiny_bundle(directory; max_seq_len=32)
        session = init_hf_qwen3_bf16_session(
            bundle;
            context_tokens=16,
            prefill_chunk_tokens=3,
        )
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
        @test decode_hf_qwen3_bf16!(session, 7) == baseline.decode_logits
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

        zero = generate_hf_qwen3_bf16!(
            session,
            tokens;
            max_new_tokens=0,
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
        bundle = _week19_tiny_bundle(directory; max_seq_len=256)
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
        @test_throws ArgumentError fit_qwen3_chat_context(
            session,
            [(role="user", content=repeat("x", 80))];
            max_prompt_tokens=10,
        )
    end
end

@testset "interactive CLI avoids soft-scope history reassignment" begin
    script = read(joinpath(
        @__DIR__,
        "..",
        "scripts",
        "run_qwen3_cuda_chat.jl",
    ), String)
    @test !occursin("history = convert(", script)
    @test occursin(
        "empty!(history)\n    append!(history, result.messages)",
        script,
    )
    @test occursin("isempty(line) && eof(stdin) && break", script)
end
