using MLDataDevices: cpu_device, get_device
using Statistics: median

_host_array(x) = Array(cpu_device()(x))

function _max_abs_error(left, right)
    left_host = _host_array(left)
    right_host = _host_array(right)
    return Float32(maximum(abs.(left_host .- right_host)))
end

"""
    kv_cache_correctness(model, ps, st, prompt_tokens, decode_tokens; kwargs...)

Compare full forward, dynamic KV cache, and fixed-shape KV cache logits for one
prompt and a sequence of incremental decode tokens.

The current harness is intentionally batch-size 1 so every decode token is an
integer and the report is easy to inspect.
"""
function kv_cache_correctness(
    model::GPTModel,
    ps,
    st::NamedTuple,
    prompt_tokens,
    decode_tokens;
    device=get_device(ps),
    atol::Real=1.0f-5,
    rtol::Real=1.0f-4,
)
    prompt = _prefill_token_matrix(prompt_tokens)
    size(prompt, 2) == 1 ||
        throw(ArgumentError("correctness harness currently requires batch_size=1"))
    tokens = Int.(collect(decode_tokens))
    size(prompt, 1) + length(tokens) <= model.max_seq_len || throw(ArgumentError(
        "prompt plus decode tokens exceeds model.max_seq_len",
    ))
    _validate_generation_ids(prompt, model.vocab_size)
    _validate_generation_ids(tokens, model.vocab_size)

    reference_prefill, _ = model(device(prompt), ps, st)

    dynamic_cache = init_kv_cache(model; batch_size=1)
    dynamic_prefill, dynamic_cache, dynamic_state = prefill(
        model,
        ps,
        st,
        prompt,
        dynamic_cache;
        device,
    )

    static_cache = init_static_kv_cache(
        model;
        batch_size=1,
        dtype=_parameter_cache_eltype(ps),
        device,
    )
    static_prefill, static_cache, static_state = prefill(
        model,
        ps,
        st,
        prompt,
        static_cache;
        device,
    )

    prefill_dynamic_error = _max_abs_error(dynamic_prefill, reference_prefill)
    prefill_static_error = _max_abs_error(static_prefill, reference_prefill)
    dynamic_decode_error = 0.0f0
    static_decode_error = 0.0f0
    dynamic_decode_passed = true
    static_decode_passed = true

    context = vec(prompt)
    for token in tokens
        push!(context, token)
        reference_logits, _ = model(
            device(reshape(context, :, 1)),
            ps,
            st,
        )
        reference_last = reference_logits[:, end:end, :]

        dynamic_logits, dynamic_cache, dynamic_state = decode_step(
            model,
            ps,
            dynamic_state,
            token,
            dynamic_cache;
            device,
        )
        static_logits, static_cache, static_state = decode_step(
            model,
            ps,
            static_state,
            token,
            static_cache;
            device,
        )

        dynamic_decode_error = max(
            dynamic_decode_error,
            _max_abs_error(dynamic_logits, reference_last),
        )
        static_decode_error = max(
            static_decode_error,
            _max_abs_error(static_logits, reference_last),
        )
        dynamic_decode_passed &= isapprox(
            _host_array(dynamic_logits),
            _host_array(reference_last);
            atol,
            rtol,
        )
        static_decode_passed &= isapprox(
            _host_array(static_logits),
            _host_array(reference_last);
            atol,
            rtol,
        )
    end

    prefill_dynamic_passed = isapprox(
        _host_array(dynamic_prefill),
        _host_array(reference_prefill);
        atol,
        rtol,
    )
    prefill_static_passed = isapprox(
        _host_array(static_prefill),
        _host_array(reference_prefill);
        atol,
        rtol,
    )

    return (;
        passed=prefill_dynamic_passed &&
            prefill_static_passed &&
            dynamic_decode_passed &&
            static_decode_passed,
        prefill=(;
            dynamic_passed=prefill_dynamic_passed,
            static_passed=prefill_static_passed,
            dynamic_max_error=prefill_dynamic_error,
            static_max_error=prefill_static_error,
        ),
        decode=(;
            dynamic_passed=dynamic_decode_passed,
            static_passed=static_decode_passed,
            dynamic_max_error=dynamic_decode_error,
            static_max_error=static_decode_error,
        ),
        prompt_tokens=size(prompt, 1),
        decode_tokens=length(tokens),
        batch_size=1,
        atol=Float32(atol),
        rtol=Float32(rtol),
    )
end

function _synchronize_logits(logits)
    # Materialization is also the synchronization boundary for asynchronous
    # device backends, making the timing semantics explicit.
    return sum(_host_array(logits))
end

function _run_eager_timing(model, ps, st, prompt, decode_tokens, device)
    context = vec(prompt)

    start = time_ns()
    logits, _ = model(device(reshape(context, :, 1)), ps, st)
    _synchronize_logits(logits)
    prefill_seconds = Float64(time_ns() - start) / 1.0e9

    start = time_ns()
    for token in decode_tokens
        push!(context, token)
        logits, _ = model(device(reshape(context, :, 1)), ps, st)
        _synchronize_logits(logits)
    end
    decode_seconds = Float64(time_ns() - start) / 1.0e9

    return (; prefill_seconds, decode_seconds)
end

function _run_dynamic_timing(model, ps, st, prompt, decode_tokens, device)
    cache = init_kv_cache(model; batch_size=1)

    start = time_ns()
    logits, cache, state = prefill(model, ps, st, prompt, cache; device)
    _synchronize_logits(logits)
    prefill_seconds = Float64(time_ns() - start) / 1.0e9

    start = time_ns()
    for token in decode_tokens
        logits, cache, state = decode_step(
            model,
            ps,
            state,
            token,
            cache;
            device,
        )
        _synchronize_logits(logits)
    end
    decode_seconds = Float64(time_ns() - start) / 1.0e9

    return (; prefill_seconds, decode_seconds, cache)
end

function _run_static_timing(model, ps, st, prompt, decode_tokens, device)
    cache = init_static_kv_cache(
        model;
        batch_size=1,
        dtype=_parameter_cache_eltype(ps),
        device,
    )

    start = time_ns()
    logits, cache, state = prefill(model, ps, st, prompt, cache; device)
    _synchronize_logits(logits)
    prefill_seconds = Float64(time_ns() - start) / 1.0e9

    start = time_ns()
    for token in decode_tokens
        logits, cache, state = decode_step(
            model,
            ps,
            state,
            token,
            cache;
            device,
        )
        _synchronize_logits(logits)
    end
    decode_seconds = Float64(time_ns() - start) / 1.0e9

    return (; prefill_seconds, decode_seconds, cache)
end

function _steady_summary(runs, decode_token_count::Int)
    prefill_seconds = median(map(run -> run.prefill_seconds, runs))
    decode_seconds = median(map(run -> run.decode_seconds, runs))
    tokens_per_second = decode_token_count == 0 ?
        Inf : decode_token_count / decode_seconds

    return (;
        prefill_seconds,
        decode_seconds,
        decode_tokens_per_second=tokens_per_second,
    )
end

function _timing_samples(runs)
    return Tuple((;
        prefill_seconds=run.prefill_seconds,
        decode_seconds=run.decode_seconds,
    ) for run in runs)
end

function _benchmark_sample_runs(runner, samples::Int)
    runs = NamedTuple[]
    for _ in 1:samples
        # A sample should not retain the previous sample's cache. Collecting
        # outside the timing windows prevents cross-sample cache allocations
        # from becoming an accidental GC benchmark.
        GC.gc(false)
        run = runner()
        push!(runs, (;
            prefill_seconds=run.prefill_seconds,
            decode_seconds=run.decode_seconds,
        ))
    end
    return runs
end

"""
    benchmark_kv_cache(model, ps, st, prompt_tokens, decode_tokens; samples=5)

Measure eager full-recompute generation, dynamic KV cache, and fixed-shape KV
cache. The report separates first-run warmup from steady-state medians and keeps
cache storage calculations explicit.

This is a microbenchmark harness, not a substitute for BenchmarkTools or a
production serving benchmark.
"""
function benchmark_kv_cache(
    model::GPTModel,
    ps,
    st::NamedTuple,
    prompt_tokens,
    decode_tokens;
    device=get_device(ps),
    samples::Int=5,
)
    samples > 0 || throw(ArgumentError("`samples` must be positive"))

    prompt = _prefill_token_matrix(prompt_tokens)
    size(prompt, 2) == 1 ||
        throw(ArgumentError("benchmark currently requires batch_size=1"))
    tokens = Int.(collect(decode_tokens))
    size(prompt, 1) + length(tokens) <= model.max_seq_len || throw(ArgumentError(
        "prompt plus decode tokens exceeds model.max_seq_len",
    ))
    _validate_generation_ids(prompt, model.vocab_size)
    _validate_generation_ids(tokens, model.vocab_size)

    GC.gc(false)
    eager_warmup = _run_eager_timing(model, ps, st, prompt, tokens, device)
    GC.gc(false)
    dynamic_warmup = _run_dynamic_timing(model, ps, st, prompt, tokens, device)
    GC.gc(false)
    static_warmup = _run_static_timing(model, ps, st, prompt, tokens, device)

    eager_runs = _benchmark_sample_runs(samples) do
        _run_eager_timing(model, ps, st, prompt, tokens, device)
    end
    dynamic_runs = _benchmark_sample_runs(samples) do
        _run_dynamic_timing(model, ps, st, prompt, tokens, device)
    end
    static_runs = _benchmark_sample_runs(samples) do
        _run_static_timing(model, ps, st, prompt, tokens, device)
    end

    element_bytes = sizeof(_parameter_cache_eltype(ps))
    final_tokens = size(prompt, 1) + length(tokens)
    per_token_cache_bytes =
        2 * model.num_layers * model.head_dim * model.num_kv_heads * element_bytes
    theoretical_dynamic_bytes = per_token_cache_bytes * final_tokens
    theoretical_static_bytes = per_token_cache_bytes * model.max_seq_len

    return (;
        configuration=(;
            prompt_tokens=size(prompt, 1),
            decode_tokens=length(tokens),
            batch_size=1,
            model=gpt_config(model),
            samples,
            element_bytes,
        ),
        eager=(;
            warmup=eager_warmup,
            steady=_steady_summary(eager_runs, length(tokens)),
            samples=_timing_samples(eager_runs),
        ),
        dynamic=(;
            warmup=(;
                prefill_seconds=dynamic_warmup.prefill_seconds,
                decode_seconds=dynamic_warmup.decode_seconds,
            ),
            steady=_steady_summary(dynamic_runs, length(tokens)),
            samples=_timing_samples(dynamic_runs),
            theoretical_cache_bytes=theoretical_dynamic_bytes,
            observed_summarysize=Base.summarysize(dynamic_warmup.cache),
        ),
        static=(;
            warmup=(;
                prefill_seconds=static_warmup.prefill_seconds,
                decode_seconds=static_warmup.decode_seconds,
            ),
            steady=_steady_summary(static_runs, length(tokens)),
            samples=_timing_samples(static_runs),
            theoretical_cache_bytes=theoretical_static_bytes,
            observed_summarysize=Base.summarysize(static_warmup.cache),
        ),
    )
end

"""
    benchmark_xla_kv_cache(model, ps, st, prompt_tokens, decode_tokens; kwargs...)

Measure Reactant/XLA setup, first prefill/decode calls (compile + execute), and
steady-state executable reuse. Materializing logits on the host is used as the
synchronization boundary.
"""
function benchmark_xla_kv_cache(
    model::GPTModel,
    ps,
    st::NamedTuple,
    prompt_tokens,
    decode_tokens;
    xla_backend::AbstractString="gpu",
    samples::Int=5,
)
    samples > 0 || throw(ArgumentError("`samples` must be positive"))

    prompt = Int.(collect(prompt_tokens))
    tokens = Int.(collect(decode_tokens))
    isempty(prompt) && throw(ArgumentError("prompt must not be empty"))
    isempty(tokens) && throw(ArgumentError("decode_tokens must not be empty"))
    length(prompt) + length(tokens) <= model.max_seq_len || throw(ArgumentError(
        "prompt plus decode tokens exceeds model.max_seq_len",
    ))

    start = time_ns()
    decoder = XLAKVDecoder(model, ps, st; xla_backend)
    setup_seconds = Float64(time_ns() - start) / 1.0e9

    start = time_ns()
    logits, _, _ = xla_prefill!(decoder, prompt)
    _synchronize_logits(logits)
    first_prefill_seconds = Float64(time_ns() - start) / 1.0e9

    start = time_ns()
    logits, _, _ = xla_decode_step!(decoder, first(tokens))
    _synchronize_logits(logits)
    first_decode_seconds = Float64(time_ns() - start) / 1.0e9

    runs = map(1:samples) do _
        start = time_ns()
        logits, _, _ = xla_prefill!(decoder, prompt)
        _synchronize_logits(logits)
        prefill_seconds = Float64(time_ns() - start) / 1.0e9

        start = time_ns()
        for token in tokens
            logits, _, _ = xla_decode_step!(decoder, token)
            _synchronize_logits(logits)
        end
        decode_seconds = Float64(time_ns() - start) / 1.0e9

        (; prefill_seconds, decode_seconds)
    end

    return (;
        backend=String(xla_backend),
        setup_seconds,
        compile_and_first_run=(;
            prefill_seconds=first_prefill_seconds,
            decode_seconds=first_decode_seconds,
        ),
        steady=_steady_summary(runs, length(tokens)),
        prompt_tokens=length(prompt),
        decode_tokens=length(tokens),
        samples,
    )
end
