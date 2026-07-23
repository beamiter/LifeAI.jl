#!/usr/bin/env julia

using Dates: now
using JSON3
using LifeAI
using LinearAlgebra: BLAS

2 <= length(ARGS) <= 3 || error(
    "usage: julia --project=. scripts/benchmark_gpt2_inference.jl " *
    "MODEL_DIR OUTPUT_JSON [SAMPLES]",
)

model_dir = abspath(ARGS[1])
output_path = abspath(ARGS[2])
samples = length(ARGS) == 3 ? parse(Int, ARGS[3]) : 3
samples > 0 || error("SAMPLES must be positive")
prompt_lengths = parse.(
    Int,
    split(get(ENV, "LIFEAI_GPT2_PROMPT_LENGTHS", "16,64,256"), ','),
)
decode_count = parse(Int, get(ENV, "LIFEAI_GPT2_DECODE_TOKENS", "8"))
revision = get(ENV, "LIFEAI_GPT2_REVISION", "")
isempty(revision) && error("LIFEAI_GPT2_REVISION must be the immutable source revision")
all(>(0), prompt_lengths) || error("prompt lengths must be positive")
decode_count > 0 || error("decode token count must be positive")
max_seq_len = maximum(prompt_lengths) + decode_count

elapsed_seconds(start_ns) = Float64(time_ns() - start_ns) / 1.0e9
load_start = time_ns()
bundle = load_hf_gpt2_bundle(model_dir; max_seq_len, revision)
load_seconds = elapsed_seconds(load_start)
rss_after_load_bytes = Sys.maxrss()

seed_text = repeat(
    "LifeAI reproduces GPT-2 with frozen weights and exact byte BPE. " *
    "生命来自观察、记忆与反馈。\n",
    max_seq_len,
)
token_stream = encode(bundle.tokenizer, seed_text; add_special_tokens=false)
length(token_stream) >= max_seq_len || error("benchmark seed encoded too few tokens")

function timing_view(mode)
    return (;
        warmup_prefill_seconds=mode.warmup.prefill_seconds,
        warmup_decode_seconds=mode.warmup.decode_seconds,
        steady_prefill_seconds=mode.steady.prefill_seconds,
        steady_decode_seconds=mode.steady.decode_seconds,
        steady_decode_tokens_per_second=mode.steady.decode_tokens_per_second,
        samples=mode.samples,
    )
end

results = []
for prompt_length in prompt_lengths
    prompt = token_stream[1:prompt_length]
    decode_tokens = token_stream[(prompt_length + 1):(prompt_length + decode_count)]
    correctness = kv_cache_correctness(
        bundle.model,
        bundle.parameters,
        bundle.states,
        prompt,
        decode_tokens;
        atol=5.0f-4,
        rtol=5.0f-4,
    )
    correctness.passed ||
        error("cache correctness failed for prompt length $prompt_length")
    benchmark = benchmark_kv_cache(
        bundle.model,
        bundle.parameters,
        bundle.states,
        prompt,
        decode_tokens;
        samples,
    )
    push!(results, (;
        prompt_tokens=prompt_length,
        decode_tokens=decode_count,
        correctness=(;
            passed=correctness.passed,
            dynamic_prefill_max_abs=correctness.prefill.dynamic_max_error,
            static_prefill_max_abs=correctness.prefill.static_max_error,
            dynamic_decode_max_abs=correctness.decode.dynamic_max_error,
            static_decode_max_abs=correctness.decode.static_max_error,
        ),
        eager=timing_view(benchmark.eager),
        dynamic=merge(timing_view(benchmark.dynamic), (;
            theoretical_cache_bytes=benchmark.dynamic.theoretical_cache_bytes,
            observed_summarysize=benchmark.dynamic.observed_summarysize,
        )),
        static=merge(timing_view(benchmark.static), (;
            theoretical_cache_bytes=benchmark.static.theoretical_cache_bytes,
            observed_summarysize=benchmark.static.observed_summarysize,
        )),
    ))
end

report = (;
    schema_version=1,
    recorded_at=string(now()),
    model_id="openai-community/gpt2",
    model_source=bundle.source,
    revision,
    file_checksums=bundle.checksums,
    julia_version=string(VERSION),
    cpu_model=first(Sys.cpu_info()).model,
    cpu_target=Sys.CPU_NAME,
    logical_cpus=Sys.CPU_THREADS,
    julia_threads=Threads.nthreads(),
    blas_threads=BLAS.get_num_threads(),
    samples,
    synchronization=(
        "Every timed model call materializes the last logits on the host through " *
        "the shared benchmark harness."
    ),
    dtype_contract="F32 safetensors storage -> Float32 parameters/compute",
    max_seq_len,
    load_seconds,
    rss_after_load_bytes,
    process_peak_rss_bytes=Sys.maxrss(),
    model=(;
        parameter_count=124_439_808,
        vocab_size=bundle.model.vocab_size,
        d_model=bundle.model.d_model,
        num_layers=bundle.model.num_layers,
        num_heads=bundle.model.num_heads,
        num_kv_heads=bundle.model.num_kv_heads,
        head_dim=bundle.model.head_dim,
        position_embedding_type=String(bundle.model.position_embedding_type),
        mlp_type=String(bundle.model.mlp_type),
    ),
    results,
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println("wrote $output_path")
