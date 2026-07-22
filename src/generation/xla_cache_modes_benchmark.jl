using Statistics: median, quantile

function _xla_mode_inputs(model, prompt_tokens, decode_tokens, samples)
    samples > 0 || throw(ArgumentError("`samples` must be positive"))
    prompt = Int.(collect(prompt_tokens))
    tokens = Int.(collect(decode_tokens))
    isempty(prompt) && throw(ArgumentError("prompt must not be empty"))
    isempty(tokens) && throw(ArgumentError("decode_tokens must not be empty"))
    length(prompt) + length(tokens) <= model.max_seq_len || throw(ArgumentError(
        "prompt plus decode tokens exceeds model.max_seq_len",
    ))
    _validate_generation_ids(prompt, model.vocab_size)
    _validate_generation_ids(tokens, model.vocab_size)
    return prompt, tokens
end

function _xla_reference_logits(model, ps, st, prompt, tokens)
    state = Lux.testmode(st)
    prefill_logits, _ = model(reshape(prompt, :, 1), ps, state)
    references = Vector{Array{Float32,3}}(undef, length(tokens))
    context = copy(prompt)
    for (index, token) in enumerate(tokens)
        push!(context, token)
        logits, _ = model(reshape(context, :, 1), ps, state)
        references[index] = Float32.(logits[:, end:end, :])
    end
    return Float32.(prefill_logits), references
end

function _xla_compare_logits(logits, reference; atol, rtol)
    host = Float32.(_host_array(logits))
    return (;
        max_abs_error=Float32(maximum(abs.(host .- reference))),
        passed=isapprox(host, reference; atol, rtol),
    )
end

function _xla_compare_last_logits(logits, reference; atol, rtol)
    host = Float32.(_host_array(logits))
    last_logits = host[:, end:end, :]
    return (;
        max_abs_error=Float32(maximum(abs.(last_logits .- reference))),
        passed=isapprox(last_logits, reference; atol, rtol),
    )
end

function _xla_merge_comparison(current, comparison)
    return (;
        max_abs_error=max(current.max_abs_error, comparison.max_abs_error),
        passed=current.passed && comparison.passed,
    )
end

function _xla_latency_distribution(values)
    return (;
        minimum_seconds=minimum(values),
        p50_seconds=median(values),
        p90_seconds=quantile(values, 0.90),
        maximum_seconds=maximum(values),
    )
end

function _xla_mode_steady(prefill_runs, decode_runs, decode_token_count)
    prefill = _xla_latency_distribution(prefill_runs)
    decode = _xla_latency_distribution(decode_runs)
    return (;
        prefill,
        decode,
        prefill_samples_seconds=copy(prefill_runs),
        decode_samples_seconds=copy(decode_runs),
        decode_p50_seconds_per_token=decode.p50_seconds / decode_token_count,
        decode_p90_seconds_per_token=decode.p90_seconds / decode_token_count,
        decode_tokens_per_second=decode_token_count / decode.p50_seconds,
    )
end

function _xla_mode_setup(ps, st, xla_backend)
    Reactant.set_default_backend(String(xla_backend))
    device = Lux.reactant_device(; force=true)
    ps_xla, st_xla = device((ps, Lux.testmode(st)))
    return device, ps_xla, st_xla
end

function _compile_xla_full_forward(model, tokens, ps, st)
    return Reactant.@compile model(tokens, ps, st)
end

function _compile_xla_dynamic_prefill(model, tokens, ps, st, cache)
    return Reactant.@compile _dynamic_gpt_prefill_kernel(
        model,
        tokens,
        ps,
        st,
        cache,
    )
end

function _compile_xla_dynamic_decode(model, tokens, ps, st, cache)
    return Reactant.@compile _dynamic_gpt_decode_kernel(
        model,
        tokens,
        ps,
        st,
        cache,
    )
end

function _warm_xla_mode_runtime(model, ps, st, prompt, xla_backend)
    start = time_ns()
    device, ps_xla, st_xla = _xla_mode_setup(ps, st, xla_backend)
    # The benchmark itself is batch-size 1. A two-column input initializes the
    # Reactant/Lux tracing and compiler stack without creating an executable
    # that any measured mode can reuse.
    warmup_tokens = device(reshape([first(prompt), first(prompt)], 1, 2))
    thunk = _compile_xla_full_forward(
        model,
        warmup_tokens,
        ps_xla,
        st_xla,
    )
    logits, _ = thunk(warmup_tokens, ps_xla, st_xla)
    _synchronize_logits(logits)
    return Float64(time_ns() - start) / 1.0e9
end

function _benchmark_xla_no_cache(
    model,
    ps,
    st,
    prompt,
    tokens,
    reference_prefill,
    reference_decode;
    xla_backend,
    samples,
    atol,
    rtol,
)
    setup_start = time_ns()
    device, ps_xla, st_xla = _xla_mode_setup(ps, st, xla_backend)
    setup_seconds = Float64(time_ns() - setup_start) / 1.0e9
    thunks = Dict{Int,Any}()

    prompt_matrix = reshape(prompt, :, 1)
    prompt_xla = device(prompt_matrix)
    prefill_start = time_ns()
    prefill_thunk = _compile_xla_full_forward(model, prompt_xla, ps_xla, st_xla)
    thunks[length(prompt)] = prefill_thunk
    logits, _ = prefill_thunk(prompt_xla, ps_xla, st_xla)
    _synchronize_logits(logits)
    first_prefill_seconds = Float64(time_ns() - prefill_start) / 1.0e9
    prefill_comparison = _xla_compare_logits(
        logits,
        reference_prefill;
        atol,
        rtol,
    )

    context = copy(prompt)
    decode_comparison = (; max_abs_error=0.0f0, passed=true)
    first_decode_seconds = 0.0
    first_decode_total_seconds = 0.0
    for (index, token) in enumerate(tokens)
        step_start = time_ns()
        push!(context, token)
        context_xla = device(reshape(context, :, 1))
        thunk = _compile_xla_full_forward(model, context_xla, ps_xla, st_xla)
        thunks[length(context)] = thunk
        logits, _ = thunk(context_xla, ps_xla, st_xla)
        _synchronize_logits(logits)
        step_seconds = Float64(time_ns() - step_start) / 1.0e9
        first_decode_total_seconds += step_seconds
        index == 1 && (first_decode_seconds = step_seconds)
        comparison = _xla_compare_last_logits(
            logits,
            reference_decode[index];
            atol,
            rtol,
        )
        decode_comparison = _xla_merge_comparison(
            decode_comparison,
            comparison,
        )
    end

    prefill_runs = Float64[]
    decode_runs = Float64[]
    for _ in 1:samples
        prompt_xla = device(prompt_matrix)
        sample_start = time_ns()
        logits, _ = thunks[length(prompt)](prompt_xla, ps_xla, st_xla)
        _synchronize_logits(logits)
        push!(prefill_runs, Float64(time_ns() - sample_start) / 1.0e9)

        context = copy(prompt)
        sample_start = time_ns()
        for token in tokens
            push!(context, token)
            context_xla = device(reshape(context, :, 1))
            logits, _ = thunks[length(context)](
                context_xla,
                ps_xla,
                st_xla,
            )
            _synchronize_logits(logits)
        end
        push!(decode_runs, Float64(time_ns() - sample_start) / 1.0e9)
    end

    return (;
        mode=:no_cache,
        setup_seconds,
        compile_and_first_run=(;
            prefill_seconds=first_prefill_seconds,
            first_decode_seconds,
            decode_total_seconds=first_decode_total_seconds,
        ),
        steady=_xla_mode_steady(prefill_runs, decode_runs, length(tokens)),
        executable_count=length(thunks),
        theoretical_cache_bytes=0,
        correctness=(;
            passed=prefill_comparison.passed && decode_comparison.passed,
            prefill_max_abs_error=prefill_comparison.max_abs_error,
            decode_max_abs_error=decode_comparison.max_abs_error,
            atol=Float32(atol),
            rtol=Float32(rtol),
        ),
    )
end

function _benchmark_xla_dynamic_cache(
    model,
    ps,
    st,
    prompt,
    tokens,
    reference_prefill,
    reference_decode;
    xla_backend,
    samples,
    atol,
    rtol,
)
    setup_start = time_ns()
    device, ps_xla, st_xla = _xla_mode_setup(ps, st, xla_backend)
    setup_seconds = Float64(time_ns() - setup_start) / 1.0e9
    decode_thunks = Dict{Int,Any}()

    empty_cache = init_kv_cache(model; batch_size=1)
    prompt_xla = device(reshape(prompt, :, 1))
    prefill_start = time_ns()
    prefill_thunk = _compile_xla_dynamic_prefill(
        model,
        prompt_xla,
        ps_xla,
        st_xla,
        empty_cache,
    )
    logits, cache, state = prefill_thunk(
        model,
        prompt_xla,
        ps_xla,
        st_xla,
        empty_cache,
    )
    _synchronize_logits(logits)
    first_prefill_seconds = Float64(time_ns() - prefill_start) / 1.0e9
    prefill_comparison = _xla_compare_logits(
        logits,
        reference_prefill;
        atol,
        rtol,
    )

    decode_comparison = (; max_abs_error=0.0f0, passed=true)
    first_decode_seconds = 0.0
    first_decode_total_seconds = 0.0
    for (index, token) in enumerate(tokens)
        step_start = time_ns()
        cached_length = cache.position
        token_xla = device(reshape([token], 1, 1))
        thunk = _compile_xla_dynamic_decode(
            model,
            token_xla,
            ps_xla,
            state,
            cache,
        )
        decode_thunks[cached_length] = thunk
        logits, cache, state = thunk(
            model,
            token_xla,
            ps_xla,
            state,
            cache,
        )
        _synchronize_logits(logits)
        step_seconds = Float64(time_ns() - step_start) / 1.0e9
        first_decode_total_seconds += step_seconds
        index == 1 && (first_decode_seconds = step_seconds)
        comparison = _xla_compare_logits(
            logits,
            reference_decode[index];
            atol,
            rtol,
        )
        decode_comparison = _xla_merge_comparison(
            decode_comparison,
            comparison,
        )
    end

    prefill_runs = Float64[]
    decode_runs = Float64[]
    for _ in 1:samples
        empty_cache = init_kv_cache(model; batch_size=1)
        prompt_xla = device(reshape(prompt, :, 1))
        sample_start = time_ns()
        logits, cache, state = prefill_thunk(
            model,
            prompt_xla,
            ps_xla,
            st_xla,
            empty_cache,
        )
        _synchronize_logits(logits)
        push!(prefill_runs, Float64(time_ns() - sample_start) / 1.0e9)

        sample_start = time_ns()
        for token in tokens
            cached_length = cache.position
            token_xla = device(reshape([token], 1, 1))
            logits, cache, state = decode_thunks[cached_length](
                model,
                token_xla,
                ps_xla,
                state,
                cache,
            )
            _synchronize_logits(logits)
        end
        push!(decode_runs, Float64(time_ns() - sample_start) / 1.0e9)
    end

    element_bytes = sizeof(_parameter_cache_eltype(ps_xla))
    final_length = length(prompt) + length(tokens)
    cache_bytes =
        2 * model.num_layers * model.head_dim * model.num_kv_heads *
        final_length * element_bytes

    return (;
        mode=:dynamic_cache,
        setup_seconds,
        compile_and_first_run=(;
            prefill_seconds=first_prefill_seconds,
            first_decode_seconds,
            decode_total_seconds=first_decode_total_seconds,
        ),
        steady=_xla_mode_steady(prefill_runs, decode_runs, length(tokens)),
        executable_count=1 + length(decode_thunks),
        theoretical_cache_bytes=cache_bytes,
        correctness=(;
            passed=prefill_comparison.passed && decode_comparison.passed,
            prefill_max_abs_error=prefill_comparison.max_abs_error,
            decode_max_abs_error=decode_comparison.max_abs_error,
            atol=Float32(atol),
            rtol=Float32(rtol),
        ),
    )
end

function _benchmark_xla_static_cache(
    model,
    ps,
    st,
    prompt,
    tokens,
    reference_prefill,
    reference_decode;
    xla_backend,
    samples,
    atol,
    rtol,
)
    setup_start = time_ns()
    decoder = XLAKVDecoder(model, ps, st; xla_backend)
    setup_seconds = Float64(time_ns() - setup_start) / 1.0e9

    prefill_start = time_ns()
    logits, _, _ = xla_prefill!(decoder, prompt)
    _synchronize_logits(logits)
    first_prefill_seconds = Float64(time_ns() - prefill_start) / 1.0e9
    prefill_comparison = _xla_compare_logits(
        logits,
        reference_prefill;
        atol,
        rtol,
    )

    decode_comparison = (; max_abs_error=0.0f0, passed=true)
    first_decode_seconds = 0.0
    first_decode_total_seconds = 0.0
    for (index, token) in enumerate(tokens)
        step_start = time_ns()
        logits, _, _ = xla_decode_step!(decoder, token)
        _synchronize_logits(logits)
        step_seconds = Float64(time_ns() - step_start) / 1.0e9
        first_decode_total_seconds += step_seconds
        index == 1 && (first_decode_seconds = step_seconds)
        comparison = _xla_compare_logits(
            logits,
            reference_decode[index];
            atol,
            rtol,
        )
        decode_comparison = _xla_merge_comparison(
            decode_comparison,
            comparison,
        )
    end

    prefill_runs = Float64[]
    decode_runs = Float64[]
    for _ in 1:samples
        sample_start = time_ns()
        logits, _, _ = xla_prefill!(decoder, prompt)
        _synchronize_logits(logits)
        push!(prefill_runs, Float64(time_ns() - sample_start) / 1.0e9)

        sample_start = time_ns()
        for token in tokens
            logits, _, _ = xla_decode_step!(decoder, token)
            _synchronize_logits(logits)
        end
        push!(decode_runs, Float64(time_ns() - sample_start) / 1.0e9)
    end

    element_bytes = sizeof(decoder.cache_eltype)
    cache_bytes =
        2 * model.num_layers * model.head_dim * model.num_kv_heads *
        model.max_seq_len * element_bytes

    return (;
        mode=:static_cache,
        setup_seconds,
        compile_and_first_run=(;
            prefill_seconds=first_prefill_seconds,
            first_decode_seconds,
            decode_total_seconds=first_decode_total_seconds,
        ),
        steady=_xla_mode_steady(prefill_runs, decode_runs, length(tokens)),
        executable_count=length(decoder.prefill_thunks) + 1,
        theoretical_cache_bytes=cache_bytes,
        correctness=(;
            passed=prefill_comparison.passed && decode_comparison.passed,
            prefill_max_abs_error=prefill_comparison.max_abs_error,
            decode_max_abs_error=decode_comparison.max_abs_error,
            atol=Float32(atol),
            rtol=Float32(rtol),
        ),
    )
end

"""
    benchmark_xla_cache_modes(model, ps, st, prompt_tokens, decode_tokens; kwargs...)

Compare three XLA autoregressive inference strategies under the same model,
tokens, synchronization boundaries, and sample count:

- `no_cache`: re-run full forward on every growing context; one executable per
  context length.
- `dynamic_cache`: append K/V tensors as the cache grows; one decode executable
  per cache length because shapes change.
- `static_cache`: preallocate the maximum cache and reuse one fixed-shape
  decode executable for every token position.

The cold decode total includes compilation and execution for the complete
decode sequence. Steady-state timings reuse every executable compiled during
the cold pass. Every decode step materializes logits on the host, matching the
synchronization requirements of token-by-token generation.
"""
function benchmark_xla_cache_modes(
    model::GPTModel,
    ps,
    st::NamedTuple,
    prompt_tokens,
    decode_tokens;
    xla_backend::AbstractString="gpu",
    samples::Int=5,
    atol::Real=5.0f-3,
    rtol::Real=5.0f-3,
)
    xla_backend in ("cpu", "gpu", "tpu") ||
        throw(ArgumentError("`xla_backend` must be \"cpu\", \"gpu\", or \"tpu\""))
    atol >= 0 || throw(ArgumentError("`atol` must be non-negative"))
    rtol >= 0 || throw(ArgumentError("`rtol` must be non-negative"))
    prompt, tokens = _xla_mode_inputs(
        model,
        prompt_tokens,
        decode_tokens,
        samples,
    )
    reference_prefill, reference_decode = _xla_reference_logits(
        model,
        ps,
        st,
        prompt,
        tokens,
    )
    runtime_warmup_seconds = _warm_xla_mode_runtime(
        model,
        ps,
        st,
        prompt,
        xla_backend,
    )

    no_cache = _benchmark_xla_no_cache(
        model,
        ps,
        st,
        prompt,
        tokens,
        reference_prefill,
        reference_decode;
        xla_backend,
        samples,
        atol,
        rtol,
    )
    dynamic_cache = _benchmark_xla_dynamic_cache(
        model,
        ps,
        st,
        prompt,
        tokens,
        reference_prefill,
        reference_decode;
        xla_backend,
        samples,
        atol,
        rtol,
    )
    static_cache = _benchmark_xla_static_cache(
        model,
        ps,
        st,
        prompt,
        tokens,
        reference_prefill,
        reference_decode;
        xla_backend,
        samples,
        atol,
        rtol,
    )

    return (;
        backend=String(xla_backend),
        runtime_warmup_seconds,
        configuration=(;
            prompt_tokens=length(prompt),
            decode_tokens=length(tokens),
            batch_size=1,
            samples,
            model=gpt_config(model),
        ),
        no_cache,
        dynamic_cache,
        static_cache,
    )
end
