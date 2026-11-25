#!/usr/bin/env julia
# Blackwell RTX PRO 5000 72GB microbenchmark of Ripple-65B projection shapes:
# Native cuBLASLt NVFP4 (:global & :token, plus shared-input multi-projection)
# vs reference NVFP4 vs packed INT4 GEMV vs BF16 dense GEMM.

using CUDA
using Random
using LinearAlgebra
using Statistics
using JSON3
using BFloat16s: BFloat16
using LifeAI

function _bench_ms(f; warmup::Int=3, iters::Int=15)
    for _ in 1:warmup
        f()
    end
    CUDA.synchronize()
    times = Vector{Float64}(undef, iters)
    for i in 1:iters
        t0 = time_ns()
        f()
        CUDA.synchronize()
        times[i] = (time_ns() - t0) / 1e6
    end
    return median(times)
end

function _cosine(a, b)
    va = Float64.(vec(Array(a)))
    vb = Float64.(vec(Array(b)))
    return dot(va, vb) / (norm(va) * norm(vb))
end

function main(args)
    summary_path = length(args) >= 1 ? abspath(args[1]) : nothing
    CUDA.functional() || error("CUDA is not functional")
    native_nvfp4_supported() || error("native NVFP4 is not supported on this GPU")
    CUDA.allowscalar(false)

    rng = MersenneTwister(20261125)
    # Exact Ripple-65B-Preview decoder projection geometries (hidden=5120, ffn=16384/40960).
    shapes = [
        ("q_proj", 10240, 5120),
        ("o_proj", 5120, 10240),
        ("gate_proj_16k", 16384, 5120),
        ("down_proj_16k", 5120, 16384),
    ]
    token_counts = [1, 64, 256]
    records = Any[]

    for (name, rows, cols) in shapes
        w_f32_host = Float32.(randn(rng, Float32, rows, cols) .* (1.0f0 / sqrt(Float32(cols))))
        w_bf16 = CUDA.cu(BFloat16.(w_f32_host))
        w_nvfp4_ref = LifeAI._flint_weight_to_device(CUDA.cu, quantize_nvfp4(w_f32_host))
        w_nvfp4_glob = prepare_nvfp4_gemm(w_nvfp4_ref; activation_scaling=:global)
        w_nvfp4_tok = prepare_nvfp4_gemm(w_nvfp4_ref; activation_scaling=:token)
        w_int4_dev = LifeAI._flint_weight_to_device(CUDA.cu, LifeAI._quantize_int4_group(w_f32_host; group=128))

        for ntok in token_counts
            x_host = randn(rng, Float32, cols, ntok)
            x_f32 = CUDA.cu(x_host)
            x_bf16 = CUDA.cu(BFloat16.(x_host))
            y_exact = w_f32_host * x_host

            y_bf16 = Array(w_bf16 * x_bf16)
            y_glob = Array(LifeAI._ripple_linear(w_nvfp4_glob, x_f32))
            y_tok = Array(LifeAI._ripple_linear(w_nvfp4_tok, x_f32))
            y_int4 = Array(LifeAI._ripple_linear(w_int4_dev, x_f32))

            bf16_ms = _bench_ms(() -> w_bf16 * x_bf16)
            nvfp4_glob_ms = _bench_ms(() -> LifeAI._ripple_linear(w_nvfp4_glob, x_f32))
            nvfp4_tok_ms = _bench_ms(() -> LifeAI._ripple_linear(w_nvfp4_tok, x_f32))
            int4_ms = _bench_ms(() -> LifeAI._ripple_linear(w_int4_dev, x_f32))

            flops = 2.0 * rows * cols * ntok
            push!(records, (;
                projection=name,
                rows, cols, tokens=ntok,
                bf16_ms=round(bf16_ms; digits=4),
                int4_ms=round(int4_ms; digits=4),
                nvfp4_global_ms=round(nvfp4_glob_ms; digits=4),
                nvfp4_token_ms=round(nvfp4_tok_ms; digits=4),
                nvfp4_global_vs_bf16_speedup=round(bf16_ms / nvfp4_glob_ms; digits=3),
                nvfp4_global_tflops=round(flops / (nvfp4_glob_ms * 1e9); digits=3),
                cosine_bf16=round(_cosine(y_exact, y_bf16); digits=6),
                cosine_int4=round(_cosine(y_exact, y_int4); digits=6),
                cosine_nvfp4_global=round(_cosine(y_exact, y_glob); digits=6),
                cosine_nvfp4_token=round(_cosine(y_exact, y_tok); digits=6),
            ))
        end
        w_bf16 = nothing; w_nvfp4_ref = nothing; w_nvfp4_glob = nothing; w_nvfp4_tok = nothing; w_int4_dev = nothing
        GC.gc(true); CUDA.reclaim()
    end

    # Shared-input gate + up multi-projection (16384 x 5120 each) at decode=1 and prefill=256.
    w_g = prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu, quantize_nvfp4(randn(rng, Float32, 16384, 5120) .* 0.01f0)); activation_scaling=:global)
    w_u = prepare_nvfp4_gemm(LifeAI._flint_weight_to_device(CUDA.cu, quantize_nvfp4(randn(rng, Float32, 16384, 5120) .* 0.01f0)); activation_scaling=:global)
    multi_records = Any[]
    for ntok in (1, 256)
        x = CUDA.cu(randn(rng, Float32, 5120, ntok))
        sep_ms = _bench_ms(() -> (LifeAI._ripple_linear(w_g, x), LifeAI._ripple_linear(w_u, x)))
        fused_ms = _bench_ms(() -> LifeAI._ripple_linear_multi((w_g, w_u), x))
        push!(multi_records, (;
            pair="gate_up_16k", tokens=ntok,
            separate_ms=round(sep_ms; digits=4),
            shared_input_multi_ms=round(fused_ms; digits=4),
            speedup=round(sep_ms / fused_ms; digits=3),
        ))
    end

    summary = (;
        device=CUDA.name(CUDA.device()),
        compute_capability=string(CUDA.capability(CUDA.device())),
        note="Synthetic Ripple-65B projection-shape microbenchmark on Blackwell; not full 65B checkpoint quality.",
        projections=records,
        multi_projection=multi_records,
    )
    payload = JSON3.write(summary)
    println(payload)
    if summary_path !== nothing
        mkpath(dirname(summary_path))
        write(summary_path, payload)
    end
end

main(ARGS)
