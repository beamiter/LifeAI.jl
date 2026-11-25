using Test
using CUDA
using Random
using LinearAlgebra
using LifeAI

@testset "Ripple native Blackwell NVFP4 cuBLASLt GEMM" begin
    @test CUDA.functional()
    @test native_nvfp4_supported()
    CUDA.allowscalar(false)

    fixture_dir = joinpath(@__DIR__, "fixtures")
    bundle_ref = load_hf_ripple_model(fixture_dir; device=CUDA.cu, scheme=:nvfp4, quantize_on_device=true)
    bundle_native = prepare_ripple_nvfp4(bundle_ref; activation_scaling=:token)
    stats_before = ripple_native_fp4_stats(bundle_native)
    @test stats_before.backend === :cublaslt_nvfp4
    @test stats_before.prepared_weights > 0
    @test stats_before.native_calls == 0

    prompt = Int[1, 5, 11, 23, 7, 19]
    cache_ref = RippleKVCache(bundle_ref.model)
    cache_nat = RippleKVCache(bundle_native.model)
    logits_ref = Array(hf_ripple_forward(bundle_ref.model, bundle_ref.parameters, prompt; cache=cache_ref, last_only=true))
    logits_nat = Array(hf_ripple_forward(bundle_native.model, bundle_native.parameters, prompt; cache=cache_nat, last_only=true))
    @test all(isfinite, logits_nat)
    @test size(logits_nat) == size(logits_ref)
    cos_sim = dot(vec(logits_ref), vec(logits_nat)) / (norm(vec(logits_ref)) * norm(vec(logits_nat)))
    @test cos_sim > 0.95

    # Single-projection native cuBLASLt FP4 vs dequantized reference FP4 parity.
    w_ref = bundle_ref.parameters["model.layers.0.self_attn.q_proj.weight"]
    w_nat = bundle_native.parameters["model.layers.0.self_attn.q_proj.weight"]
    x_probe = CUDA.cu(randn(MersenneTwister(42), Float32, size(w_ref, 2), 4))
    y_ref = Array(LifeAI._ripple_linear(w_ref, x_probe))
    y_nat = Array(LifeAI._ripple_linear(w_nat, x_probe))
    proj_cos = dot(vec(y_ref), vec(y_nat)) / (norm(vec(y_ref)) * norm(vec(y_nat)))
    @test proj_cos > 0.985

    # Decode several steps to exercise shared-input multi-projection native FP4 GEMM (Q/K and gate/up).
    for _ in 1:4
        tok = argmax(vec(logits_nat))
        logits_nat = Array(hf_ripple_forward(bundle_native.model, bundle_native.parameters, [tok]; cache=cache_nat, last_only=true))
        @test all(isfinite, logits_nat)
    end
    stats_after = ripple_native_fp4_stats(bundle_native)
    @test stats_after.native_calls > stats_before.prepared_weights

    # Also test offload bundle with native NVFP4 prepared on GPU-resident weights.
    tmp = mktempdir()
    artifact = joinpath(tmp, "ripple-nvfp4")
    quantize_ripple_checkpoint(fixture_dir, artifact; scheme=:nvfp4, device=CUDA.cu)
    offload_bundle = load_ripple_quantized(artifact; device=CUDA.cu, resident_bytes=1024 * 1024)
    offload_native = prepare_ripple_nvfp4(offload_bundle; activation_scaling=:token)
    @test ripple_native_fp4_stats(offload_native).prepared_weights > 0
    out_offload = Array(hf_ripple_forward(offload_native.model, offload_native.parameters, prompt; last_only=true))
    @test all(isfinite, out_offload)
    @test ripple_native_fp4_stats(offload_native).native_calls > 0
    rm(tmp; recursive=true)
end
