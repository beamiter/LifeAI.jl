using Test
using JSON3
using SHA

isdefined(@__MODULE__, :LIFEAI_REPO_ROOT) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "repository_test_assets.jl"))

_qwen3_moe_real_parity_sha256(path) = bytes2hex(SHA.sha256(read(path)))

@testset "Qwen3-30B-A3B real streamed parity report contract" begin
    summary_path = joinpath(
        LIFEAI_REPO_ROOT,
        "benchmark_results",
        "qwen3_moe_real_parity",
        "summary.json",
    )
    summary = JSON3.read(read(summary_path, String))
    @test Int(summary.schema_version) == 1
    @test String(summary.model_id) == "Qwen/Qwen3-30B-A3B"
    @test String(summary.revision) ==
        "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39"
    @test Int(summary.checkpoint.shard_count) == 16
    @test Int(summary.checkpoint.shard_bytes) == 61_066_575_648
    @test Int(summary.case.layers) == 48
    @test Int(summary.case.experts) == 128
    @test Int(summary.case.experts_per_token) == 8
    @test Int(summary.case.routing_slots) == 1_152

    for name in (:exporter, :verifier)
        path = joinpath(
            LIFEAI_REPO_ROOT,
            String(getproperty(summary.scripts, name)),
        )
        expected = String(getproperty(
            summary.scripts,
            Symbol("$(name)_sha256"),
        ))
        @test _qwen3_moe_real_parity_sha256(path) == expected
    end

    float32 = summary.float32
    @test Bool(float32.passed)
    @test Int(float32.routing_slots_matched) ==
        Int(float32.routing_slots_total) == 1_152
    @test Int(float32.token_layer_expert_sets_matched) ==
        Int(float32.token_layer_cases) == 144
    @test Float64(float32.logits_max_abs) <= 5.0e-5
    @test Float64(float32.decode_logits_max_abs) <= 2.0e-5
    @test Bool(float32.prompt_argmax_equal)
    @test Bool(float32.decode_argmax_equal)

    bfloat16 = summary.bfloat16
    @test Bool(bfloat16.passed)
    @test Int(bfloat16.routing_slots_matched) == 1_105
    @test Int(bfloat16.routing_slots_total) == 1_152
    @test Float64(bfloat16.routing_overlap_fraction) >= 0.95
    @test Float64(bfloat16.common_routing_weight_max_abs) <= 0.02
    @test Float64(bfloat16.logits_max_abs) <= 0.3125
    @test Float64(bfloat16.decode_logits_max_abs) <= 0.3125
    @test Bool(bfloat16.prompt_argmax_equal)
    @test Bool(bfloat16.decode_argmax_equal)
    @test Int(bfloat16.process_maxrss_bytes) < Int(float32.process_maxrss_bytes)

    expert_reads = summary.expert_reads
    @test Int(expert_reads.decode_unique_experts_across_layers) == 48 * 8
    @test Int(expert_reads.prompt_unique_experts_across_layers) <
        Int(expert_reads.all_experts_per_pass)
end
