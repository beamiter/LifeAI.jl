using BFloat16s: BFloat16
using LifeAI
using JSON3
using SHA: sha256
using Test

const QWEN3_MOE_CHAPTER24_FIXTURE = joinpath(
    dirname(@__DIR__),
    "chapter24_qwen3_moe_architecture",
    "fixtures",
    "qwen3_moe_real_checkpoint",
    "config.json",
)
const QWEN3_MOE_TINY_OFFLOAD_FIXTURE = joinpath(
    dirname(dirname(QWEN3_MOE_CHAPTER24_FIXTURE)),
    "qwen3_moe_tiny_parity",
)

@testset "Qwen3 MoE GPU offload contract" begin
    config = load_hf_qwen3_moe_config(
        QWEN3_MOE_CHAPTER24_FIXTURE;
        max_seq_len=40_960,
    )
    model = GPTModel(config)
    plan = qwen3_moe_offload_plan(model, 40_960)

    @test plan.context_tokens == 40_960
    @test plan.batch_size == 1
    @test plan.max_active_experts == 128
    @test plan.dtype_bytes == sizeof(BFloat16)
    @test plan.resident_parameter_bytes == 2_459_856_896
    @test plan.active_expert_layer_bytes == 1_207_959_552
    @test plan.kv_cache_bytes == 4_026_531_840
    @test plan.working_set_floor_bytes == 7_694_348_288
    @test plan.working_set_floor_bytes < 8 * 2^30

    routes = Int32[5 2 8; 2 5 2]
    remapped = LifeAI._qwen3_local_expert_routes(routes, 8)
    @test remapped.active_experts == [2, 5, 8]
    @test remapped.local_indices == Int32[2 1 3; 1 2 1]
    @test size(remapped.local_indices) == size(routes)

    @test_throws ArgumentError qwen3_moe_offload_plan(model, 0)
    @test_throws ArgumentError qwen3_moe_offload_plan(
        model,
        40_960;
        max_active_experts=129,
    )
    @test_throws ArgumentError LifeAI._qwen3_local_expert_routes(
        Int32[0 1],
        8,
    )
end

@testset "Qwen3 MoE offload chunked prefill and decode" begin
    session = load_hf_qwen3_moe_offload_session(
        QWEN3_MOE_TINY_OFFLOAD_FIXTURE;
        context_tokens=8,
        prefill_chunk_tokens=1,
        grouped_experts=false,
    )
    prefill = prefill_hf_qwen3_moe_offload!(session, [2, 3])
    @test prefill.position == 2
    @test length(prefill.chunks) == 2
    @test size(prefill.logits) == (17, 1, 1)
    @test sum(chunk.expert_bytes_read for chunk in prefill.chunks) ==
        prefill.expert_bytes_read
    @test all(length(chunk.active_experts) == 2 for chunk in prefill.chunks)

    decode = decode_hf_qwen3_moe_offload!(session, 4)
    @test decode.position == 3
    @test size(decode.logits) == (17, 1, 1)
    @test decode.expert_bytes_read > 0
    @test reset_hf_qwen3_moe_offload_session!(session).position == 0
end

@testset "Qwen3 MoE real GPU offload result contract" begin
    repo_root = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
    summary_path = joinpath(
        repo_root,
        "benchmark_results",
        "qwen3_moe_cuda_offload",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary["schema_version"]) == 1
    @test String(summary["model_id"]) == "Qwen/Qwen3-30B-A3B"
    @test Int(summary["session"]["context_tokens"]) == 40_960
    @test Bool(summary["session"]["full_context_cache_allocated"])
    @test Int(summary["session"]["kv_cache_bytes"]) == 4_026_531_840
    @test Int(summary["session"]["working_set_floor_bytes"]) == 7_694_348_288

    two_token = summary["two_token_reference"]
    @test Bool(two_token["grouped"]["prefill_argmax_match"])
    @test Bool(two_token["grouped"]["decode_argmax_match"])
    @test Float64(two_token["grouped_over_scalar_speedup"]["prefill"]) < 1
    @test Float64(two_token["grouped_over_scalar_speedup"]["decode"]) < 1

    wide = summary["thirty_two_token_reference"]
    @test Bool(wide["path_argmax_match"])
    @test Float64(wide["grouped_over_scalar_speedup"]["prefill"]) > 1
    @test Float64(wide["grouped_over_scalar_speedup"]["decode"]) > 1
    @test Int(wide["grouped"]["prompt_active_experts_maximum"]) < 128
    @test !Bool(summary["decision"]["full_window_prefill_executed"])

    for (name, relative_path) in (
        "benchmark_script" => joinpath(
            "scripts",
            "benchmark_qwen3_moe_cuda_offload.jl",
        ),
        "offload_implementation" => joinpath(
            "src",
            "generation",
            "qwen3_moe_offload.jl",
        ),
    )
        expected = String(summary["source_sha256"][name])
        actual = bytes2hex(sha256(read(joinpath(repo_root, relative_path))))
        @test actual == expected
    end
end
