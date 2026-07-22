# Week 06 GQA benchmark: KV cache memory and decode throughput.
#
# Fixed model shape; only num_kv_heads (and QK-norm) vary. Reports parameter
# counts, theoretical/observed cache bytes, and eager/dynamic/static decode
# throughput, plus the full/dynamic/static correctness verdict per profile.
#
#     julia --project=. examples/benchmark_week06_gqa.jl

using Dates
using Lux
using Printf
using Random
using LifeAI
using LifeAI: benchmark_kv_cache, gpt_config, kv_cache_correctness

const VOCAB_SIZE = 256
const D_MODEL = 128
const NUM_HEADS = 8
const HEAD_DIM = 16
const NUM_LAYERS = 4
const MAX_SEQ_LEN = 256
const PROMPT_TOKENS = 64
const DECODE_TOKENS = 64
const SAMPLES = 5
const SEED = 20260721

function build_model(num_kv_heads::Int, use_qk_norm::Bool)
    return GPTModel(
        VOCAB_SIZE,
        D_MODEL,
        NUM_HEADS,
        NUM_LAYERS;
        num_kv_heads,
        head_dim=HEAD_DIM,
        use_qk_norm,
        use_rope=true,
        rope_theta=1.0f6,
        norm_type=:rmsnorm,
        norm_epsilon=1.0f-6,
        mlp_type=:swiglu,
        tie_embeddings=true,
        max_seq_len=MAX_SEQ_LEN,
    )
end

function run_profile(label, num_kv_heads, use_qk_norm)
    model = build_model(num_kv_heads, use_qk_norm)
    ps, st = Lux.setup(MersenneTwister(SEED), model)

    rng = MersenneTwister(SEED + 1)
    prompt = rand(rng, 1:VOCAB_SIZE, PROMPT_TOKENS)
    decode_tokens = rand(rng, 1:VOCAB_SIZE, DECODE_TOKENS)

    correctness = kv_cache_correctness(model, ps, st, prompt, decode_tokens)
    report = benchmark_kv_cache(
        model, ps, st, prompt, decode_tokens; samples=SAMPLES,
    )

    return (;
        label,
        num_kv_heads,
        use_qk_norm,
        parameters=LuxCore.parameterlength(ps),
        correctness=correctness.passed,
        static_cache_bytes=report.static.theoretical_cache_bytes,
        dynamic_cache_bytes=report.dynamic.theoretical_cache_bytes,
        eager_decode_tps=report.eager.steady.decode_tokens_per_second,
        dynamic_decode_tps=report.dynamic.steady.decode_tokens_per_second,
        static_decode_tps=report.static.steady.decode_tokens_per_second,
    )
end

function main()
    profiles = (
        ("mha_baseline", NUM_HEADS, false),
        ("gqa_half", NUM_HEADS ÷ 2, false),
        ("gqa_half_qk_norm", NUM_HEADS ÷ 2, true),
        ("mqa", 1, true),
    )

    rows = [run_profile(label, kv, qk) for (label, kv, qk) in profiles]

    output_dir = joinpath(@__DIR__, "..", "benchmark_results", "week06")
    mkpath(output_dir)

    tsv_path = joinpath(output_dir, "gqa_cpu.tsv")
    open(tsv_path, "w") do io
        println(io, join((
            "profile", "num_kv_heads", "use_qk_norm", "parameters",
            "correctness", "static_cache_bytes", "dynamic_cache_bytes",
            "eager_decode_tps", "dynamic_decode_tps", "static_decode_tps",
        ), '\t'))
        for r in rows
            println(io, join((
                r.label, r.num_kv_heads, r.use_qk_norm, r.parameters,
                r.correctness, r.static_cache_bytes, r.dynamic_cache_bytes,
                round(r.eager_decode_tps; digits=1),
                round(r.dynamic_decode_tps; digits=1),
                round(r.static_decode_tps; digits=1),
            ), '\t'))
        end
    end

    md_path = joinpath(output_dir, "gqa_summary.md")
    open(md_path, "w") do io
        println(io, "# Week 06 GQA benchmark (CPU)")
        println(io)
        println(io, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io)
        println(io, "Fixed shape: d_model=$(D_MODEL), heads=$(NUM_HEADS), " *
            "head_dim=$(HEAD_DIM), layers=$(NUM_LAYERS), " *
            "max_seq_len=$(MAX_SEQ_LEN); RMSNorm + SwiGLU + tied + RoPE 1e6. " *
            "Prompt $(PROMPT_TOKENS), decode $(DECODE_TOKENS), " *
            "$(SAMPLES) steady samples, batch 1. " *
            "Only num_kv_heads and QK-norm vary.")
        println(io)
        println(io, "| Profile | KV heads | QK-norm | Parameters | Correct | " *
            "Static cache KiB | Dynamic cache KiB | Eager tok/s | " *
            "Dynamic tok/s | Static tok/s |")
        println(io, "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | " *
            "---: | ---: |")
        for r in rows
            @printf(io,
                "| %s | %d | %s | %d | %s | %.1f | %.1f | %.1f | %.1f | %.1f |\n",
                r.label, r.num_kv_heads, r.use_qk_norm, r.parameters,
                r.correctness, r.static_cache_bytes / 1024,
                r.dynamic_cache_bytes / 1024, r.eager_decode_tps,
                r.dynamic_decode_tps, r.static_decode_tps)
        end
    end

    println("wrote $(tsv_path)")
    println("wrote $(md_path)")
    for r in rows
        @printf("%-18s kv=%d params=%-7d cache=%6.1fKiB dyn=%8.1f tok/s correct=%s\n",
            r.label, r.num_kv_heads, r.parameters,
            r.static_cache_bytes / 1024, r.dynamic_decode_tps, r.correctness)
    end

    return nothing
end

main()
