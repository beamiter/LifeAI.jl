#!/usr/bin/env julia

using CUDA
using Dates: now
using JSON3
using LifeAI
using LinearAlgebra: BLAS
using Lux
using MLDataDevices: CUDADevice, cpu_device
using Statistics: median

3 <= length(ARGS) <= 4 || error(
    "usage: julia --project=. scripts/benchmark_qwen3_accelerator.jl " *
    "MODEL_DIR OUTPUT_JSON BACKEND [SAMPLES]",
)

model_dir = abspath(ARGS[1])
output_path = abspath(ARGS[2])
backend = ARGS[3]
backend in ("cuda", "xla_gpu") || error("BACKEND must be cuda or xla_gpu")
samples = length(ARGS) == 4 ? parse(Int, ARGS[4]) : 3
samples > 0 || error("SAMPLES must be positive")
default_prompt_lengths = backend == "cuda" ? "16,64,256" : "16"
default_decode_count = backend == "cuda" ? "8" : "2"
prompt_lengths = parse.(
    Int,
    split(get(ENV, "LIFEAI_QWEN3_PROMPT_LENGTHS", default_prompt_lengths), ','),
)
decode_count = parse(
    Int,
    get(ENV, "LIFEAI_QWEN3_DECODE_TOKENS", default_decode_count),
)
all(>(0), prompt_lengths) || error("prompt lengths must be positive")
decode_count > 0 || error("decode token count must be positive")
backend == "xla_gpu" && length(prompt_lengths) != 1 &&
    error("xla_gpu feasibility run requires exactly one prompt length")
max_seq_len = maximum(prompt_lengths) + decode_count
revision = get(ENV, "LIFEAI_QWEN3_REVISION", basename(model_dir))

CUDA.functional() || error("CUDA.jl is not functional on this machine")
cuda_device = CUDA.device()
device_report = (;
    name=CUDA.name(cuda_device),
    capability=string(CUDA.capability(cuda_device)),
    driver_version=string(CUDA.driver_version()),
    runtime_version=string(CUDA.runtime_version()),
    total_memory_bytes=CUDA.total_memory(),
    free_memory_bytes_before_load=CUDA.free_memory(),
)

elapsed_seconds(start_ns) = Float64(time_ns() - start_ns) / 1.0e9
host_array(value) = Array(cpu_device()(value))
function synchronize(value)
    host = host_array(value)
    return sum(host), host
end

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

load_start = time_ns()
bundle = load_hf_qwen3_bundle(model_dir; max_seq_len, revision)
load_seconds = elapsed_seconds(load_start)
rss_after_load_bytes = Sys.maxrss()

seed_text = repeat(
    "LifeAI validates Qwen3 inference with reproducible prompts. 生命来自观察、记忆与反馈。\n",
    max_seq_len,
)
token_stream = encode(bundle.tokenizer, seed_text; add_special_tokens=false)
length(token_stream) >= max_seq_len || error("benchmark seed text encoded too few tokens")

results, setup = if backend == "cuda"
    device = CUDADevice()
    setup_start = time_ns()
    parameters, states = device((bundle.parameters, Lux.testmode(bundle.states)))
    CUDA.synchronize()
    setup_seconds = elapsed_seconds(setup_start)
    setup_report = (;
        parameter_transfer_seconds=setup_seconds,
        used_memory_bytes_after_transfer=CUDA.used_memory(),
        cached_memory_bytes_after_transfer=CUDA.cached_memory(),
        free_memory_bytes_after_transfer=CUDA.free_memory(),
    )

    cuda_results = []
    for prompt_length in prompt_lengths
        local prompt = token_stream[1:prompt_length]
        local decode_tokens =
            token_stream[(prompt_length + 1):(prompt_length + decode_count)]
        correctness = kv_cache_correctness(
            bundle.model,
            parameters,
            states,
            prompt,
            decode_tokens;
            device,
            atol=5.0f-3,
            rtol=5.0f-4,
        )
        correctness.passed || error(
            "CUDA cache correctness failed for prompt length $prompt_length",
        )
        benchmark = benchmark_kv_cache(
            bundle.model,
            parameters,
            states,
            prompt,
            decode_tokens;
            device,
            samples,
        )
        push!(cuda_results, (;
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
    cuda_results, setup_report
else
    prompt_length = only(prompt_lengths)
    prompt = token_stream[1:prompt_length]
    decode_tokens = token_stream[(prompt_length + 1):(prompt_length + decode_count)]
    host_state = Lux.testmode(bundle.states)
    reference_prefill, _ = bundle.model(
        reshape(prompt, :, 1),
        bundle.parameters,
        host_state,
    )
    reference_decode = Vector{Float32}[]
    context = copy(prompt)
    for token in decode_tokens
        push!(context, token)
        logits, _ = bundle.model(
            reshape(context, :, 1),
            bundle.parameters,
            host_state,
        )
        push!(reference_decode, copy(@view logits[:, end, 1]))
    end

    setup_start = time_ns()
    decoder = XLAKVDecoder(
        bundle.model,
        bundle.parameters,
        host_state;
        xla_backend="gpu",
    )
    setup_seconds = elapsed_seconds(setup_start)

    first_prefill_start = time_ns()
    first_prefill, _, _ = xla_prefill!(decoder, prompt)
    _, first_prefill_host = synchronize(first_prefill)
    first_prefill_seconds = elapsed_seconds(first_prefill_start)
    prefill_error = Float32(maximum(abs.(first_prefill_host .- reference_prefill)))
    prefill_argmax_passed =
        argmax(@view first_prefill_host[:, end, 1]) ==
        argmax(@view reference_prefill[:, end, 1])

    first_decode_start = time_ns()
    first_decode, _, _ = xla_decode_step!(decoder, first(decode_tokens))
    _, first_decode_host = synchronize(first_decode)
    first_decode_seconds = elapsed_seconds(first_decode_start)
    decode_error = Float32(maximum(abs.(
        vec(first_decode_host) .- first(reference_decode),
    )))
    decode_argmax_passed =
        argmax(vec(first_decode_host)) == argmax(first(reference_decode))

    prefill_samples = Float64[]
    decode_samples = Float64[]
    for _ in 1:samples
        sample_start = time_ns()
        logits, _, _ = xla_prefill!(decoder, prompt)
        synchronize(logits)
        push!(prefill_samples, elapsed_seconds(sample_start))

        sample_start = time_ns()
        for (index, token) in enumerate(decode_tokens)
            logits, _, _ = xla_decode_step!(decoder, token)
            _, logits_host = synchronize(logits)
            global decode_error = max(
                decode_error,
                Float32(maximum(abs.(
                    vec(logits_host) .- reference_decode[index],
                ))),
            )
            global decode_argmax_passed &= (
                argmax(vec(logits_host)) == argmax(reference_decode[index])
            )
        end
        push!(decode_samples, elapsed_seconds(sample_start))
    end
    correctness_passed = isapprox(
        first_prefill_host,
        reference_prefill;
        atol=2.0f-2,
        rtol=5.0f-3,
    ) && decode_error <= 2.0f-2 &&
        prefill_argmax_passed && decode_argmax_passed
    correctness_passed || error(
        "XLA GPU correctness failed: prefill=$prefill_error decode=$decode_error",
    )
    xla_result = (;
        prompt_tokens=prompt_length,
        decode_tokens=decode_count,
        correctness=(;
            passed=correctness_passed,
            prefill_max_abs=prefill_error,
            decode_max_abs=decode_error,
            prefill_argmax_passed,
            decode_argmax_passed,
            atol=2.0f-2,
            rtol=5.0f-3,
        ),
        setup_seconds,
        compile_and_first_run=(;
            prefill_seconds=first_prefill_seconds,
            decode_seconds=first_decode_seconds,
        ),
        steady=(;
            prefill_seconds=median(prefill_samples),
            decode_seconds=median(decode_samples),
            decode_tokens_per_second=decode_count / median(decode_samples),
            prefill_samples,
            decode_samples,
        ),
        executable_count=length(decoder.prefill_thunks) + 1,
        theoretical_cache_bytes=2 * bundle.model.num_layers *
            bundle.model.head_dim * bundle.model.num_kv_heads *
            bundle.model.max_seq_len * sizeof(decoder.cache_eltype),
    )
    [xla_result], (;
        decoder_setup_seconds=setup_seconds,
        free_memory_bytes_after_setup=CUDA.free_memory(),
    )
end

report = (;
    schema_version=1,
    recorded_at=string(now()),
    backend,
    model_source=bundle.source,
    revision,
    julia_version=string(VERSION),
    cpu_model=first(Sys.cpu_info()).model,
    cpu_target=Sys.CPU_NAME,
    logical_cpus=Sys.CPU_THREADS,
    julia_threads=Threads.nthreads(),
    blas_threads=BLAS.get_num_threads(),
    samples,
    synchronization="materialize logits on host after every timed model call",
    dtype_contract="BF16 safetensors storage -> Float32 parameters/compute",
    max_seq_len,
    load_seconds,
    rss_after_load_bytes,
    process_peak_rss_bytes=Sys.maxrss(),
    device=device_report,
    setup,
    model=(;
        vocab_size=bundle.model.vocab_size,
        d_model=bundle.model.d_model,
        num_layers=bundle.model.num_layers,
        num_heads=bundle.model.num_heads,
        num_kv_heads=bundle.model.num_kv_heads,
        head_dim=bundle.model.head_dim,
    ),
    results,
)

mkpath(dirname(output_path))
open(output_path, "w") do io
    JSON3.pretty(io, report)
    write(io, '\n')
end
println("wrote $output_path")
