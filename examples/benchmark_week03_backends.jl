using Dates
using LinearAlgebra
using Lux
using LifeAI
using MLDataDevices: CUDADevice
using Printf
using Random
using Statistics
import CUDA

const SUPPORTED_BACKENDS = ("cpu", "gpu", "xla_cpu", "xla_gpu")

function parse_options(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument, "--") || error("unknown argument: $argument")
        parts = split(argument[3:end], "="; limit=2)
        length(parts) == 2 || error("expected --name=value, got: $argument")
        options[parts[1]] = parts[2]
    end
    return options
end

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))
function env_bool(name, default)
    value = lowercase(get(ENV, name, string(default)))
    value in ("1", "true", "yes") && return true
    value in ("0", "false", "no") && return false
    error("$name must be true/false, 1/0, or yes/no")
end

function benchmark_config()
    profile = get(ENV, "LIFEAI_BENCH_PROFILE", "baseline")
    norm_type = Symbol(lowercase(get(ENV, "LIFEAI_BENCH_NORM_TYPE", "layernorm")))
    mlp_type = Symbol(lowercase(get(ENV, "LIFEAI_BENCH_MLP_TYPE", "gelu")))
    tie_embeddings = env_bool("LIFEAI_BENCH_TIE_EMBEDDINGS", false)
    vocab_size = env_int("LIFEAI_BENCH_VOCAB_SIZE", 512)
    embed_dim = env_int("LIFEAI_BENCH_EMBED_DIM", 128)
    num_heads = env_int("LIFEAI_BENCH_NUM_HEADS", 4)
    num_layers = env_int("LIFEAI_BENCH_NUM_LAYERS", 4)
    seq_len = env_int("LIFEAI_BENCH_SEQ_LEN", 128)
    batch_size = env_int("LIFEAI_BENCH_BATCH_SIZE", 8)
    prompt_tokens = env_int("LIFEAI_BENCH_PROMPT_TOKENS", 128)
    decode_tokens = env_int("LIFEAI_BENCH_DECODE_TOKENS", 64)
    xla_mode_decode_tokens = env_int(
        "LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS",
        min(decode_tokens, 4),
    )
    warmup_steps = env_int("LIFEAI_BENCH_WARMUP_STEPS", 3)
    samples = env_int("LIFEAI_BENCH_SAMPLES", 30)
    learning_rate = env_float("LIFEAI_BENCH_LEARNING_RATE", 3.0e-4)
    correctness_atol = env_float("LIFEAI_BENCH_ATOL", 5.0e-3)
    correctness_rtol = env_float("LIFEAI_BENCH_RTOL", 5.0e-3)
    seed = env_int("LIFEAI_BENCH_SEED", 20260718)

    vocab_size > 1 || error("LIFEAI_BENCH_VOCAB_SIZE must be greater than 1")
    embed_dim > 0 || error("LIFEAI_BENCH_EMBED_DIM must be positive")
    num_heads > 0 || error("LIFEAI_BENCH_NUM_HEADS must be positive")
    embed_dim % num_heads == 0 ||
        error("LIFEAI_BENCH_EMBED_DIM must be divisible by LIFEAI_BENCH_NUM_HEADS")
    num_layers > 0 || error("LIFEAI_BENCH_NUM_LAYERS must be positive")
    seq_len > 0 || error("LIFEAI_BENCH_SEQ_LEN must be positive")
    batch_size > 0 || error("LIFEAI_BENCH_BATCH_SIZE must be positive")
    prompt_tokens > 0 || error("LIFEAI_BENCH_PROMPT_TOKENS must be positive")
    decode_tokens > 0 || error("LIFEAI_BENCH_DECODE_TOKENS must be positive")
    1 <= xla_mode_decode_tokens <= decode_tokens || error(
        "LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS must be in 1:$decode_tokens",
    )
    warmup_steps >= 0 || error("LIFEAI_BENCH_WARMUP_STEPS must be non-negative")
    samples > 0 || error("LIFEAI_BENCH_SAMPLES must be positive")
    correctness_atol >= 0 || error("LIFEAI_BENCH_ATOL must be non-negative")
    correctness_rtol >= 0 || error("LIFEAI_BENCH_RTOL must be non-negative")
    norm_type in (:layernorm, :rmsnorm) ||
        error("LIFEAI_BENCH_NORM_TYPE must be layernorm or rmsnorm")
    mlp_type in (:gelu, :swiglu) ||
        error("LIFEAI_BENCH_MLP_TYPE must be gelu or swiglu")

    max_seq_len = max(seq_len, prompt_tokens + decode_tokens)
    return (;
        profile,
        norm_type,
        mlp_type,
        tie_embeddings,
        vocab_size,
        embed_dim,
        num_heads,
        num_layers,
        seq_len,
        batch_size,
        prompt_tokens,
        decode_tokens,
        xla_mode_decode_tokens,
        max_seq_len,
        warmup_steps,
        samples,
        learning_rate,
        correctness_atol,
        correctness_rtol,
        seed,
    )
end

function build_model(config)
    return GPTModel(
        config.vocab_size,
        config.embed_dim,
        config.num_heads,
        config.num_layers;
        max_seq_len=config.max_seq_len,
        use_rope=true,
        norm_type=config.norm_type,
        mlp_type=config.mlp_type,
        tie_embeddings=config.tie_embeddings,
    )
end

function host_array(value)
    return Array(Lux.cpu_device()(value))
end

function host_number(value)
    host = Lux.cpu_device()(value)
    host isa Number && return Float64(host)
    return Float64(only(Array(host)))
end

function synchronize_logits(logits)
    return sum(host_array(logits))
end

elapsed_seconds(start_ns) = Float64(time_ns() - start_ns) / 1.0e9

function timing_statistics(seconds)
    isempty(seconds) && error("cannot summarize empty timings")
    return (;
        minimum=minimum(seconds),
        median=median(seconds),
        p90=quantile(seconds, 0.90),
        maximum=maximum(seconds),
        mean=mean(seconds),
    )
end

function trainer_for(backend)
    if backend == "cpu"
        return TrainerGPT(
            learning_rate=env_float("LIFEAI_BENCH_LEARNING_RATE", 3.0e-4),
            backend=:zygote,
            device=Lux.cpu_device(),
            return_gradients=false,
        )
    elseif backend == "gpu"
        CUDA.functional() || error("CUDA.jl is not functional on this machine")
        return TrainerGPT(
            learning_rate=env_float("LIFEAI_BENCH_LEARNING_RATE", 3.0e-4),
            backend=:zygote,
            device=CUDADevice(),
            return_gradients=false,
        )
    elseif backend == "xla_cpu"
        return TrainerGPT(
            learning_rate=env_float("LIFEAI_BENCH_LEARNING_RATE", 3.0e-4),
            backend=:xla,
            xla_backend="cpu",
            return_gradients=false,
            static_shapes=true,
        )
    elseif backend == "xla_gpu"
        return TrainerGPT(
            learning_rate=env_float("LIFEAI_BENCH_LEARNING_RATE", 3.0e-4),
            backend=:xla,
            xla_backend="gpu",
            return_gradients=false,
            static_shapes=true,
        )
    end
    error("unsupported backend: $backend")
end

function benchmark_training(backend, model, config)
    trainer_start = time_ns()
    trainer = trainer_for(backend)
    trainer_seconds = elapsed_seconds(trainer_start)

    state_start = time_ns()
    train_state = init_train_state(Xoshiro(config.seed), model, trainer)
    state_seconds = elapsed_seconds(state_start)

    data_rng = Xoshiro(config.seed + 1)
    x = rand(data_rng, 1:config.vocab_size, config.seq_len, config.batch_size)
    y = rand(data_rng, 1:config.vocab_size, config.seq_len, config.batch_size)
    batch = (x, y)
    tokens_per_step = length(y)

    GC.gc()
    first_start = time_ns()
    train_state, first_loss, _ = train_step!(trainer, train_state, batch)
    first_loss_value = host_number(first_loss)
    first_seconds = elapsed_seconds(first_start)

    final_loss = first_loss_value
    post_compile_gc_start = time_ns()
    GC.gc()
    post_compile_gc_seconds = elapsed_seconds(post_compile_gc_start)

    warmup_seconds = Float64[]
    for _ in 1:config.warmup_steps
        sample_start = time_ns()
        train_state, loss, _ = train_step!(trainer, train_state, batch)
        final_loss = host_number(loss)
        push!(warmup_seconds, elapsed_seconds(sample_start))
    end

    pre_measure_gc_start = time_ns()
    GC.gc()
    pre_measure_gc_seconds = elapsed_seconds(pre_measure_gc_start)

    steady_seconds = Float64[]
    for _ in 1:config.samples
        sample_start = time_ns()
        train_state, loss, _ = train_step!(trainer, train_state, batch)
        final_loss = host_number(loss)
        push!(steady_seconds, elapsed_seconds(sample_start))
    end

    stats = timing_statistics(steady_seconds)
    return (;
        trainer_seconds,
        state_seconds,
        first_seconds,
        post_compile_gc_seconds,
        warmup_seconds,
        pre_measure_gc_seconds,
        steady=stats,
        steady_samples=copy(steady_seconds),
        steady_tokens_per_second=tokens_per_step / stats.median,
        tokens_per_step,
        first_loss=first_loss_value,
        final_loss,
        completed_steps=train_state.step,
    )
end

function reference_logits(model, ps, st, prompt, decode_token)
    prefill_logits, _ = model(reshape(prompt, :, 1), ps, st)
    context = vcat(prompt, decode_token)
    decode_logits, _ = model(reshape(context, :, 1), ps, st)
    return prefill_logits, decode_logits[:, end:end, :]
end

function compare_logits(left, right, config)
    left_host = host_array(left)
    right_host = host_array(right)
    return (;
        max_abs_error=maximum(abs.(left_host .- right_host)),
        passed=isapprox(
            left_host,
            right_host;
            atol=config.correctness_atol,
            rtol=config.correctness_rtol,
        ),
    )
end

function benchmark_eager_kv(backend, model, host_ps, host_st, prompt, tokens, config)
    trainer = trainer_for(backend)
    device = trainer.device

    setup_start = time_ns()
    ps, st = device((host_ps, Lux.testmode(host_st)))
    setup_seconds = elapsed_seconds(setup_start)

    reference_prefill, reference_decode = reference_logits(
        model,
        host_ps,
        Lux.testmode(host_st),
        prompt,
        first(tokens),
    )

    first_cache = init_static_kv_cache(
        model;
        batch_size=1,
        dtype=Float32,
        device,
    )
    first_prefill_start = time_ns()
    logits, first_cache, first_state = prefill(
        model,
        ps,
        st,
        prompt,
        first_cache;
        device,
    )
    synchronize_logits(logits)
    first_prefill_seconds = elapsed_seconds(first_prefill_start)
    prefill_comparison = compare_logits(logits, reference_prefill, config)

    first_decode_start = time_ns()
    logits, _, _ = decode_step(
        model,
        ps,
        first_state,
        first(tokens),
        first_cache;
        device,
    )
    synchronize_logits(logits)
    first_decode_seconds = elapsed_seconds(first_decode_start)
    decode_comparison = compare_logits(logits, reference_decode, config)

    prefill_seconds = Float64[]
    decode_seconds = Float64[]
    for _ in 1:config.samples
        cache = init_static_kv_cache(
            model;
            batch_size=1,
            dtype=Float32,
            device,
        )
        sample_start = time_ns()
        logits, cache, state = prefill(model, ps, st, prompt, cache; device)
        synchronize_logits(logits)
        push!(prefill_seconds, elapsed_seconds(sample_start))

        sample_start = time_ns()
        for token in tokens
            logits, cache, state = decode_step(
                model,
                ps,
                state,
                token,
                cache;
                device,
            )
        end
        synchronize_logits(logits)
        push!(decode_seconds, elapsed_seconds(sample_start))
    end

    prefill_stats = timing_statistics(prefill_seconds)
    decode_stats = timing_statistics(decode_seconds)
    return (;
        setup_seconds,
        first_prefill_seconds,
        first_decode_seconds,
        prefill=prefill_stats,
        decode=decode_stats,
        prefill_samples=copy(prefill_seconds),
        decode_samples=copy(decode_seconds),
        decode_tokens_per_second=length(tokens) / decode_stats.median,
        prefill_max_abs_error=prefill_comparison.max_abs_error,
        decode_max_abs_error=decode_comparison.max_abs_error,
        correctness_passed=prefill_comparison.passed && decode_comparison.passed,
    )
end

function benchmark_xla_kv(backend, model, host_ps, host_st, prompt, tokens, config)
    xla_backend = backend == "xla_cpu" ? "cpu" : "gpu"
    reference_prefill, reference_decode = reference_logits(
        model,
        host_ps,
        Lux.testmode(host_st),
        prompt,
        first(tokens),
    )

    setup_start = time_ns()
    decoder = XLAKVDecoder(model, host_ps, host_st; xla_backend)
    setup_seconds = elapsed_seconds(setup_start)

    first_prefill_start = time_ns()
    logits, _, _ = xla_prefill!(decoder, prompt)
    synchronize_logits(logits)
    first_prefill_seconds = elapsed_seconds(first_prefill_start)
    prefill_comparison = compare_logits(logits, reference_prefill, config)

    first_decode_start = time_ns()
    logits, _, _ = xla_decode_step!(decoder, first(tokens))
    synchronize_logits(logits)
    first_decode_seconds = elapsed_seconds(first_decode_start)
    decode_comparison = compare_logits(logits, reference_decode, config)

    prefill_seconds = Float64[]
    decode_seconds = Float64[]
    for _ in 1:config.samples
        sample_start = time_ns()
        logits, _, _ = xla_prefill!(decoder, prompt)
        synchronize_logits(logits)
        push!(prefill_seconds, elapsed_seconds(sample_start))

        sample_start = time_ns()
        for token in tokens
            logits, _, _ = xla_decode_step!(decoder, token)
        end
        synchronize_logits(logits)
        push!(decode_seconds, elapsed_seconds(sample_start))
    end

    prefill_stats = timing_statistics(prefill_seconds)
    decode_stats = timing_statistics(decode_seconds)
    return (;
        setup_seconds,
        first_prefill_seconds,
        first_decode_seconds,
        prefill=prefill_stats,
        decode=decode_stats,
        prefill_samples=copy(prefill_seconds),
        decode_samples=copy(decode_seconds),
        decode_tokens_per_second=length(tokens) / decode_stats.median,
        prefill_max_abs_error=prefill_comparison.max_abs_error,
        decode_max_abs_error=decode_comparison.max_abs_error,
        correctness_passed=prefill_comparison.passed && decode_comparison.passed,
    )
end

function add_metric!(metrics, name, value, unit="")
    push!(metrics, (String(name), string(value), String(unit)))
    return metrics
end

function add_xla_cache_mode_metrics!(metrics, label, report)
    prefix = "xla_gpu_$(label)"
    add_metric!(metrics, "$(prefix)_setup_ms", report.setup_seconds * 1.0e3, "ms")
    add_metric!(
        metrics,
        "$(prefix)_first_prefill_ms",
        report.compile_and_first_run.prefill_seconds * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "$(prefix)_first_decode_ms_per_token",
        report.compile_and_first_run.first_decode_seconds * 1.0e3,
        "ms/token",
    )
    add_metric!(
        metrics,
        "$(prefix)_cold_decode_total_ms",
        report.compile_and_first_run.decode_total_seconds * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "$(prefix)_steady_prefill_p50_ms",
        report.steady.prefill.p50_seconds * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "$(prefix)_steady_prefill_p90_ms",
        report.steady.prefill.p90_seconds * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "$(prefix)_steady_prefill_samples_ms",
        join(
            (seconds * 1.0e3 for seconds in report.steady.prefill_samples_seconds),
            ',',
        ),
        "ms",
    )
    add_metric!(
        metrics,
        "$(prefix)_steady_decode_p50_ms_per_token",
        report.steady.decode_p50_seconds_per_token * 1.0e3,
        "ms/token",
    )
    add_metric!(
        metrics,
        "$(prefix)_steady_decode_p90_ms_per_token",
        report.steady.decode_p90_seconds_per_token * 1.0e3,
        "ms/token",
    )
    add_metric!(
        metrics,
        "$(prefix)_steady_decode_samples_ms",
        join(
            (seconds * 1.0e3 for seconds in report.steady.decode_samples_seconds),
            ',',
        ),
        "ms",
    )
    add_metric!(
        metrics,
        "$(prefix)_decode_tokens_per_second",
        report.steady.decode_tokens_per_second,
        "tokens/s",
    )
    add_metric!(
        metrics,
        "$(prefix)_executable_count",
        report.executable_count,
        "executables",
    )
    add_metric!(
        metrics,
        "$(prefix)_theoretical_cache_bytes",
        report.theoretical_cache_bytes,
        "bytes",
    )
    add_metric!(
        metrics,
        "$(prefix)_prefill_max_abs_error",
        report.correctness.prefill_max_abs_error,
    )
    add_metric!(
        metrics,
        "$(prefix)_decode_max_abs_error",
        report.correctness.decode_max_abs_error,
    )
    add_metric!(
        metrics,
        "$(prefix)_correctness_passed",
        report.correctness.passed,
    )
    return metrics
end

function collect_metrics(backend)
    backend in SUPPORTED_BACKENDS ||
        error("backend must be one of: $(join(SUPPORTED_BACKENDS, ", "))")
    config = benchmark_config()
    model = build_model(config)
    host_ps, host_st = Lux.setup(Xoshiro(config.seed), model)
    prompt = collect(mod1.(1:config.prompt_tokens, config.vocab_size))
    tokens = collect(
        mod1.((config.prompt_tokens + 1):(config.prompt_tokens + config.decode_tokens),
        config.vocab_size),
    )

    println(
        "[$backend] training: first step + $(config.warmup_steps) warm-up + " *
        "$(config.samples) steady samples",
    )
    training = benchmark_training(backend, model, config)
    println("[$backend] KV inference: first calls + $(config.samples) steady samples")
    inference = startswith(backend, "xla_") ?
        benchmark_xla_kv(backend, model, host_ps, host_st, prompt, tokens, config) :
        benchmark_eager_kv(backend, model, host_ps, host_st, prompt, tokens, config)
    xla_cache_modes = if backend == "xla_gpu"
        mode_tokens = first(tokens, config.xla_mode_decode_tokens)
        println(
            "[$backend] no-cache/dynamic/static: " *
            "$(length(mode_tokens)) decode tokens",
        )
        benchmark_xla_cache_modes(
            model,
            host_ps,
            host_st,
            prompt,
            mode_tokens;
            xla_backend="gpu",
            samples=config.samples,
            atol=config.correctness_atol,
            rtol=config.correctness_rtol,
        )
    else
        nothing
    end

    element_bytes = sizeof(Float32)
    parameter_count = Lux.parameterlength(host_ps)
    cache_bytes = (
        2 * config.num_layers * (config.embed_dim ÷ config.num_heads) *
        config.num_heads * config.max_seq_len * element_bytes
    )

    metrics = Tuple{String,String,String}[]
    add_metric!(metrics, "timestamp", Dates.format(now(), dateformat"yyyy-mm-ddTHH:MM:SS"))
    add_metric!(metrics, "backend", backend)
    add_metric!(metrics, "profile", config.profile)
    add_metric!(metrics, "norm_type", config.norm_type)
    add_metric!(metrics, "mlp_type", config.mlp_type)
    add_metric!(metrics, "tie_embeddings", config.tie_embeddings)
    add_metric!(metrics, "mlp_hidden_dim", model.mlp_hidden_dim)
    add_metric!(
        metrics,
        "execution_stack",
        startswith(backend, "xla_") ? "Reactant+Enzyme+XLA" : "Zygote eager",
    )
    add_metric!(metrics, "julia_version", VERSION)
    add_metric!(metrics, "cpu_name", Sys.CPU_NAME)
    add_metric!(metrics, "julia_threads", Threads.nthreads(), "threads")
    add_metric!(metrics, "blas_threads", BLAS.get_num_threads(), "threads")
    add_metric!(metrics, "vocab_size", config.vocab_size, "tokens")
    add_metric!(metrics, "embed_dim", config.embed_dim)
    add_metric!(metrics, "num_heads", config.num_heads)
    add_metric!(metrics, "num_layers", config.num_layers)
    add_metric!(metrics, "seq_len", config.seq_len, "tokens")
    add_metric!(metrics, "batch_size", config.batch_size)
    add_metric!(metrics, "prompt_tokens", config.prompt_tokens, "tokens")
    add_metric!(metrics, "decode_tokens", config.decode_tokens, "tokens")
    add_metric!(
        metrics,
        "xla_mode_decode_tokens",
        config.xla_mode_decode_tokens,
        "tokens",
    )
    add_metric!(metrics, "warmup_steps", config.warmup_steps, "steps")
    add_metric!(metrics, "samples", config.samples)
    add_metric!(metrics, "learning_rate", config.learning_rate)
    add_metric!(metrics, "correctness_atol", config.correctness_atol)
    add_metric!(metrics, "correctness_rtol", config.correctness_rtol)
    add_metric!(metrics, "seed", config.seed)
    add_metric!(metrics, "parameter_count", parameter_count, "parameters")
    add_metric!(metrics, "parameter_bytes", parameter_count * element_bytes, "bytes")
    add_metric!(metrics, "theoretical_kv_cache_bytes", cache_bytes, "bytes")

    add_metric!(metrics, "training_trainer_setup_ms", training.trainer_seconds * 1.0e3, "ms")
    add_metric!(metrics, "training_state_setup_ms", training.state_seconds * 1.0e3, "ms")
    add_metric!(metrics, "training_first_step_ms", training.first_seconds * 1.0e3, "ms")
    add_metric!(
        metrics,
        "training_post_compile_gc_ms",
        training.post_compile_gc_seconds * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "training_warmup_samples_ms",
        join((seconds * 1.0e3 for seconds in training.warmup_seconds), ','),
        "ms",
    )
    add_metric!(
        metrics,
        "training_pre_measure_gc_ms",
        training.pre_measure_gc_seconds * 1.0e3,
        "ms",
    )
    add_metric!(metrics, "training_steady_min_ms", training.steady.minimum * 1.0e3, "ms")
    add_metric!(metrics, "training_steady_p50_ms", training.steady.median * 1.0e3, "ms")
    add_metric!(metrics, "training_steady_p90_ms", training.steady.p90 * 1.0e3, "ms")
    add_metric!(metrics, "training_steady_max_ms", training.steady.maximum * 1.0e3, "ms")
    add_metric!(
        metrics,
        "training_steady_samples_ms",
        join((seconds * 1.0e3 for seconds in training.steady_samples), ','),
        "ms",
    )
    add_metric!(
        metrics,
        "training_steady_tokens_per_second",
        training.steady_tokens_per_second,
        "tokens/s",
    )
    add_metric!(metrics, "training_tokens_per_step", training.tokens_per_step, "tokens")
    add_metric!(metrics, "training_first_loss", training.first_loss)
    add_metric!(metrics, "training_final_loss", training.final_loss)

    add_metric!(metrics, "inference_setup_ms", inference.setup_seconds * 1.0e3, "ms")
    add_metric!(
        metrics,
        "inference_first_prefill_ms",
        inference.first_prefill_seconds * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "inference_first_decode_ms_per_token",
        inference.first_decode_seconds * 1.0e3,
        "ms/token",
    )
    add_metric!(
        metrics,
        "inference_prefill_steady_p50_ms",
        inference.prefill.median * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "inference_prefill_steady_p90_ms",
        inference.prefill.p90 * 1.0e3,
        "ms",
    )
    add_metric!(
        metrics,
        "inference_prefill_steady_samples_ms",
        join((seconds * 1.0e3 for seconds in inference.prefill_samples), ','),
        "ms",
    )
    add_metric!(
        metrics,
        "inference_decode_steady_p50_ms_per_token",
        inference.decode.median * 1.0e3 / config.decode_tokens,
        "ms/token",
    )
    add_metric!(
        metrics,
        "inference_decode_steady_p90_ms_per_token",
        inference.decode.p90 * 1.0e3 / config.decode_tokens,
        "ms/token",
    )
    add_metric!(
        metrics,
        "inference_decode_steady_samples_ms",
        join((seconds * 1.0e3 for seconds in inference.decode_samples), ','),
        "ms",
    )
    add_metric!(
        metrics,
        "inference_decode_tokens_per_second",
        inference.decode_tokens_per_second,
        "tokens/s",
    )
    add_metric!(
        metrics,
        "inference_prefill_max_abs_error",
        inference.prefill_max_abs_error,
    )
    add_metric!(
        metrics,
        "inference_decode_max_abs_error",
        inference.decode_max_abs_error,
    )
    add_metric!(metrics, "inference_correctness_passed", inference.correctness_passed)
    if xla_cache_modes !== nothing
        add_metric!(
            metrics,
            "xla_gpu_cache_modes_runtime_warmup_ms",
            xla_cache_modes.runtime_warmup_seconds * 1.0e3,
            "ms",
        )
        add_xla_cache_mode_metrics!(
            metrics,
            "no_cache",
            xla_cache_modes.no_cache,
        )
        add_xla_cache_mode_metrics!(
            metrics,
            "dynamic_cache",
            xla_cache_modes.dynamic_cache,
        )
        add_xla_cache_mode_metrics!(
            metrics,
            "static_cache",
            xla_cache_modes.static_cache,
        )
    end
    add_metric!(metrics, "process_peak_rss_mb", Sys.maxrss() / 2.0^20, "MiB")
    return metrics
end

function write_metrics(path, backend, metrics)
    mkpath(dirname(abspath(path)))
    open(path, "w") do io
        println(io, "backend\tmetric\tvalue\tunit")
        for (metric, value, unit) in metrics
            clean_value = replace(value, '\t' => ' ', '\n' => ' ')
            println(io, backend, '\t', metric, '\t', clean_value, '\t', unit)
        end
    end
    return path
end

function read_metrics(path)
    metrics = Dict{String,String}()
    lines = readlines(path)
    isempty(lines) && return metrics
    for line in Iterators.drop(lines, 1)
        columns = split(line, '\t'; keepempty=true)
        length(columns) >= 3 || continue
        metrics[columns[2]] = columns[3]
    end
    return metrics
end

function format_metric(metrics, key; digits=2)
    value = get(metrics, key, "")
    isempty(value) && return "—"
    parsed = tryparse(Float64, value)
    parsed === nothing && return value
    return @sprintf("%.*f", digits, parsed)
end

function format_metric_ratio(metrics, numerator_key, denominator_key; digits=2)
    numerator = tryparse(Float64, get(metrics, numerator_key, ""))
    denominator = tryparse(Float64, get(metrics, denominator_key, ""))
    (numerator === nothing || denominator === nothing || denominator == 0) && return "—"
    return @sprintf("%.*f", digits, numerator / denominator)
end

function read_statuses(directory)
    path = joinpath(directory, "status.tsv")
    isfile(path) || return Dict{String,Tuple{String,String}}()
    statuses = Dict{String,Tuple{String,String}}()
    for line in Iterators.drop(readlines(path), 1)
        columns = split(line, '\t'; limit=3, keepempty=true)
        length(columns) >= 2 || continue
        statuses[columns[1]] = (
            columns[2],
            length(columns) == 3 ? columns[3] : "",
        )
    end
    return statuses
end

function summarize(directory)
    statuses = read_statuses(directory)
    output = IOBuffer()
    println(
        output,
        "# ",
        get(ENV, "LIFEAI_BENCH_TITLE", "Week 03 backend benchmark"),
    )
    println(output)
    println(output, "生成时间：", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    println(output)
    println(
        output,
        "| 后端 | 状态 | 正确性 | 训练首步 ms | 训练稳态 p50/p90 ms | 训练 tokens/s | ",
        "Prefill 首次/稳态 ms | Decode 首次/稳态 ms/token | Decode tokens/s | ",
        "峰值 RSS MiB | 峰值 GPU MiB |",
    )
    println(
        output,
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    )

    successful_metrics = Dict{String,Dict{String,String}}()
    for backend in SUPPORTED_BACKENDS
        status, message = get(statuses, backend, ("missing", "未运行"))
        path = joinpath(directory, "$backend.tsv")
        if status == "ok" && isfile(path)
            metrics = read_metrics(path)
            successful_metrics[backend] = metrics
            println(
                output,
                "| ", backend,
                " | ok",
                " | ", get(metrics, "inference_correctness_passed", "—"),
                " | ", format_metric(metrics, "training_first_step_ms"),
                " | ", format_metric(metrics, "training_steady_p50_ms"),
                " / ", format_metric(metrics, "training_steady_p90_ms"),
                " | ", format_metric(metrics, "training_steady_tokens_per_second"; digits=1),
                " | ", format_metric(metrics, "inference_first_prefill_ms"),
                " / ", format_metric(metrics, "inference_prefill_steady_p50_ms"),
                " | ", format_metric(metrics, "inference_first_decode_ms_per_token"),
                " / ", format_metric(metrics, "inference_decode_steady_p50_ms_per_token"),
                " | ", format_metric(metrics, "inference_decode_tokens_per_second"; digits=1),
                " | ", format_metric(metrics, "process_peak_rss_mb"; digits=1),
                " | ", format_metric(metrics, "gpu_peak_memory_mb"; digits=1),
                " |",
            )
        else
            clean_message = replace(message, '|' => '/', '\n' => ' ')
            println(
                output,
                "| ", backend, " | ", status, ": ", clean_message,
                " | — | — | — | — | — | — | — | — | — |",
            )
        end
    end

    if !isempty(successful_metrics)
        println(output)
        println(output, "## 训练稳态尾延迟")
        println(output)
        println(
            output,
            "| 后端 | p50 ms | p90 ms | max ms | p90/p50 | 原始样本 |",
        )
        println(output, "| --- | ---: | ---: | ---: | ---: | --- |")
        for backend in SUPPORTED_BACKENDS
            haskey(successful_metrics, backend) || continue
            metrics = successful_metrics[backend]
            println(
                output,
                "| ", backend,
                " | ", format_metric(metrics, "training_steady_p50_ms"),
                " | ", format_metric(metrics, "training_steady_p90_ms"),
                " | ", format_metric(metrics, "training_steady_max_ms"),
                " | ", format_metric_ratio(
                    metrics,
                    "training_steady_p90_ms",
                    "training_steady_p50_ms",
                ),
                " | `training_steady_samples_ms` |",
            )
        end
        println(output)
        println(
            output,
            "Cold compile 后先执行 ",
            get(first(values(successful_metrics)), "warmup_steps", "?"),
            " 个不计入统计的训练 warm-up step；GC 时间和 warm-up 原始耗时单独写入 TSV。",
        )
    end

    xla_gpu_metrics = get(successful_metrics, "xla_gpu", nothing)
    if xla_gpu_metrics !== nothing &&
            haskey(xla_gpu_metrics, "xla_gpu_no_cache_executable_count")
        println(output)
        println(output, "## XLA+GPU Cache 模式对比")
        println(output)
        println(
            output,
            "| 模式 | 正确性 | Cold prefill ms | Cold decode 总计 ms | ",
            "Steady prefill p50 ms | Steady decode p50 ms/token | ",
            "Decode tokens/s | Executables | Cache KiB |",
        )
        println(
            output,
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        )
        for (label, title) in (
            ("no_cache", "No cache"),
            ("dynamic_cache", "Dynamic cache"),
            ("static_cache", "Static cache"),
        )
            prefix = "xla_gpu_$(label)"
            cache_bytes = tryparse(
                Float64,
                get(xla_gpu_metrics, "$(prefix)_theoretical_cache_bytes", ""),
            )
            cache_kib = cache_bytes === nothing ?
                "—" : @sprintf("%.2f", cache_bytes / 1024)
            println(
                output,
                "| ", title,
                " | ", get(xla_gpu_metrics, "$(prefix)_correctness_passed", "—"),
                " | ", format_metric(xla_gpu_metrics, "$(prefix)_first_prefill_ms"),
                " | ", format_metric(xla_gpu_metrics, "$(prefix)_cold_decode_total_ms"),
                " | ", format_metric(
                    xla_gpu_metrics,
                    "$(prefix)_steady_prefill_p50_ms",
                ),
                " | ", format_metric(
                    xla_gpu_metrics,
                    "$(prefix)_steady_decode_p50_ms_per_token",
                ),
                " | ", format_metric(
                    xla_gpu_metrics,
                    "$(prefix)_decode_tokens_per_second";
                    digits=1,
                ),
                " | ", format_metric(
                    xla_gpu_metrics,
                    "$(prefix)_executable_count";
                    digits=0,
                ),
                " | ", cache_kib,
                " |",
            )
        end
        println(output)
        println(
            output,
            "公共 Reactant/Lux runtime warmup：",
            format_metric(
                xla_gpu_metrics,
                "xla_gpu_cache_modes_runtime_warmup_ms",
            ),
            " ms（使用不同 batch shape，不复用三模式的目标 executable）。",
        )
        println(
            output,
            "三模式使用相同的 ",
            get(xla_gpu_metrics, "xla_mode_decode_tokens", "?"),
            " 个 decode token。No cache 与 dynamic cache 的 shape 会逐 token ",
            "变化，因此 cold pass 会为各长度分别编译 executable。",
        )
    end

    println(output)
    println(output, "## 测试配置")
    println(output)
    if isempty(successful_metrics)
        println(output, "没有成功结果。查看同目录下的 `*.log`。")
    else
        metrics = first(values(successful_metrics))
        println(
            output,
            "- 模型：vocab=", get(metrics, "vocab_size", "?"),
            "，embed=", get(metrics, "embed_dim", "?"),
            "，heads=", get(metrics, "num_heads", "?"),
            "，layers=", get(metrics, "num_layers", "?"),
            "，norm=", get(metrics, "norm_type", "?"),
            "，mlp=", get(metrics, "mlp_type", "?"),
            "，tied=", get(metrics, "tie_embeddings", "?"),
            "，parameters=", get(metrics, "parameter_count", "?"), "。",
        )
        println(
            output,
            "- 训练：seq_len=", get(metrics, "seq_len", "?"),
            "，batch_size=", get(metrics, "batch_size", "?"),
            "；推理：prompt=", get(metrics, "prompt_tokens", "?"),
            "，decode=", get(metrics, "decode_tokens", "?"),
            "；warm-up steps=", get(metrics, "warmup_steps", "?"),
            "，稳态样本数=", get(metrics, "samples", "?"), "。",
        )
        println(
            output,
            "- CPU=", get(metrics, "cpu_name", "?"),
            "，Julia threads=", get(metrics, "julia_threads", "?"),
            "，BLAS threads=", get(metrics, "blas_threads", "?"), "。",
        )
    end

    println(output)
    println(output, "## 口径")
    println(output)
    println(
        output,
        "- 每个后端使用独立进程；首步/首次调用包含该进程内的后端编译与首次执行开销。",
    )
    println(
        output,
        "- 四后端主表统一使用固定形状 KV Cache；额外的 XLA+GPU 表比较 no-cache、dynamic cache 和 static cache，batch size 均为 1。",
    )
    if !isempty(successful_metrics)
        metrics = first(values(successful_metrics))
        println(
            output,
            "- 正确性使用 host full-forward reference 与 `isapprox`；atol=",
            get(metrics, "correctness_atol", "?"),
            "，rtol=", get(metrics, "correctness_rtol", "?"),
            "，同时保留 prefill/decode 最大绝对误差。",
        )
    end
    println(
        output,
        "- GPU 显存由 `nvidia-smi` 轮询采样，空值表示工具或设备不可用；RSS 是整个 Julia 进程峰值，不等于模型显存。",
    )
    println(
        output,
        "- 吞吐越大越好，延迟越小越好。不同后端的数值只有在配置、线程数和机器相同时才可直接比较。",
    )

    summary = String(take!(output))
    path = joinpath(directory, "summary.md")
    open(path, "w") do io
        write(io, summary)
    end
    print(summary)
    return path
end

function main(args)
    options = parse_options(args)
    if haskey(options, "summarize")
        summarize(abspath(options["summarize"]))
        return
    end

    backend = get(options, "backend", "")
    output = get(options, "output", "")
    isempty(backend) && error("missing --backend=$(join(SUPPORTED_BACKENDS, "|"))")
    isempty(output) && error("missing --output=PATH")
    metrics = collect_metrics(backend)
    write_metrics(output, backend, metrics)
    println("[$backend] result: $(abspath(output))")
end

main(ARGS)
