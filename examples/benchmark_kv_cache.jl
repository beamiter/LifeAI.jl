using LifeAI
using Lux
using Random

rng = Xoshiro(20260715)
model = GPTModel(
    64,
    64,
    4,
    4;
    max_seq_len=128,
    use_rope=true,
)
ps, st = Lux.setup(rng, model)

prompt = collect(1:32)
decode_tokens = collect(33:64)

correctness = kv_cache_correctness(
    model,
    ps,
    st,
    prompt,
    decode_tokens,
)
@assert correctness.passed "KV cache correctness matrix failed"

println("Correctness:")
display(correctness)

println()
println("CPU eager/dynamic/static benchmark:")
report = benchmark_kv_cache(
    model,
    ps,
    st,
    prompt,
    decode_tokens;
    samples=10,
)
display(report)

if lowercase(get(ENV, "LIFEAI_BENCH_XLA", "false")) in ("1", "true", "yes")
    backend = get(ENV, "LIFEAI_XLA_BACKEND", "gpu")
    mode_decode_count = parse(
        Int,
        get(
            ENV,
            "LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS",
            string(min(length(decode_tokens), 4)),
        ),
    )
    1 <= mode_decode_count <= length(decode_tokens) || error(
        "LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS must be in " *
        "1:$(length(decode_tokens))",
    )
    println()
    println("Reactant/XLA no-cache/dynamic/static benchmark ($backend):")
    xla_report = benchmark_xla_cache_modes(
        model,
        ps,
        st,
        prompt,
        first(decode_tokens, mode_decode_count);
        xla_backend=backend,
        samples=10,
    )
    display(xla_report)
else
    println()
    println(
        "XLA benchmark skipped. Set LIFEAI_BENCH_XLA=true and optionally " *
        "LIFEAI_XLA_BACKEND=cpu|gpu|tpu.",
    )
end
