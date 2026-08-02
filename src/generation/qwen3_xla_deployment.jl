using BFloat16s: BFloat16
using Random: AbstractRNG, default_rng
using Reactant

"""
    Qwen3XLAWindowPlan

Host-side request plan for the fixed-shape XLA prefill path. Prompts are
left-padded to a multiple of `chunk_tokens`; padding cache slots are excluded
through a per-request key-position vector.
"""
struct Qwen3XLAWindowPlan
    context_tokens::Int
    prompt_tokens::Int
    max_new_tokens::Int
    chunk_tokens::Int
    prompt_bucket_tokens::Int
    left_padding_tokens::Int
    sequence_tokens::Int
    cache_tokens::Int
end

"""
    plan_qwen3_xla_window(
        prompt_tokens, max_new_tokens;
        context_tokens=4096, chunk_tokens=64,
    )

Validate the exact logical and physical context budget without overflow.
`cache_tokens` is one less than the padded prompt plus requested output because
the final selected token does not need to be written back to K/V.
"""
function plan_qwen3_xla_window(
    prompt_tokens::Integer,
    max_new_tokens::Integer;
    context_tokens::Integer=4096,
    chunk_tokens::Integer=64,
)
    context = Int(context_tokens)
    prompt = Int(prompt_tokens)
    output = Int(max_new_tokens)
    chunk = Int(chunk_tokens)
    context > 0 || throw(ArgumentError("context_tokens must be positive"))
    context <= typemax(Int32) || throw(ArgumentError(
        "context_tokens must fit in Int32 device positions",
    ))
    prompt > 0 || throw(ArgumentError("prompt_tokens must be positive"))
    output > 0 || throw(ArgumentError("max_new_tokens must be positive"))
    0 < chunk <= context || throw(ArgumentError(
        "chunk_tokens must be in 1:context_tokens",
    ))
    output <= context || throw(ArgumentError(
        "max_new_tokens exceeds context_tokens",
    ))
    prompt <= context - output || throw(ArgumentError(
        "prompt plus requested output exceeds context_tokens",
    ))

    bucket = cld(prompt, chunk) * chunk
    bucket <= context - output || throw(ArgumentError(
        "padded XLA prompt bucket plus requested output exceeds " *
        "context_tokens",
    ))
    padding = bucket - prompt
    return Qwen3XLAWindowPlan(
        context,
        prompt,
        output,
        chunk,
        bucket,
        padding,
        prompt + output,
        bucket + output - 1,
    )
end

function qwen3_xla_pad_prompt(
    prompt_tokens,
    plan::Qwen3XLAWindowPlan;
    pad_token_id::Integer=1,
)
    prompt = Int.(vec(collect(prompt_tokens)))
    length(prompt) == plan.prompt_tokens || throw(ArgumentError(
        "prompt token count does not match the XLA window plan",
    ))
    pad = Int(pad_token_id)
    pad > 0 || throw(ArgumentError("pad_token_id must be positive"))
    return vcat(fill(pad, plan.left_padding_tokens), prompt)
end

function qwen3_xla_key_positions(plan::Qwen3XLAWindowPlan)
    positions = Int32.(collect(1:plan.context_tokens))
    if plan.left_padding_tokens > 0
        positions[1:plan.left_padding_tokens] .= typemax(Int32)
    end
    return positions
end

_qwen3_xla_tensor_bytes(x::AbstractArray) =
    length(x) * sizeof(eltype(x))
_qwen3_xla_tensor_bytes(x::NamedTuple) =
    sum(_qwen3_xla_tensor_bytes, values(x); init=0)
_qwen3_xla_tensor_bytes(x::Tuple) =
    sum(_qwen3_xla_tensor_bytes, values(x); init=0)
_qwen3_xla_tensor_bytes(x) = 0

_qwen3_xla_tensor_leaves(x::AbstractArray) = 1
_qwen3_xla_tensor_leaves(x::NamedTuple) =
    sum(_qwen3_xla_tensor_leaves, values(x); init=0)
_qwen3_xla_tensor_leaves(x::Tuple) =
    sum(_qwen3_xla_tensor_leaves, values(x); init=0)
_qwen3_xla_tensor_leaves(x) = 0

function _qwen3_xla_allocator_snapshot()
    stats = try
        Reactant.XLA.allocatorstats()
    catch
        return nothing
    end
    return (;
        num_allocs=stats.num_allocs,
        bytes_in_use=stats.bytes_in_use,
        peak_bytes_in_use=stats.peak_bytes_in_use,
        largest_alloc_size=stats.largest_alloc_size,
        bytes_limit=stats.bytes_limit,
        bytes_reserved=stats.bytes_reserved,
        peak_bytes_reserved=stats.peak_bytes_reserved,
        bytes_reservable_limit=stats.bytes_reservable_limit,
        largest_free_block_bytes=stats.largest_free_block_bytes,
        pool_bytes=stats.pool_bytes,
        peak_pool_bytes=stats.peak_pool_bytes,
    )
end

"""
    HFQwen3BF16XLASession

Reusable batch-1 XLA runtime with one packed parameter tree, one compiled
64-token prefill executable, one compiled single-token decode executable and a
fixed-capacity BF16 K/V cache.
"""
mutable struct HFQwen3BF16XLASession
    model::Any
    parameters::Any
    tokenizer::Any
    generation_config::Any
    cos_table::Any
    sin_table::Any
    key_caches::Any
    value_caches::Any
    compiled_prefill::Any
    compiled_decode::Any
    strategy::Symbol
    context_tokens::Int
    prefill_chunk_tokens::Int
    sample_top_k::Int
    position::Int
    load_metrics::Any
end

"""
    load_hf_qwen3_bf16_xla_session(
        model_dir;
        context_tokens=4096,
        prefill_chunk_tokens=64,
        strategy=:greedy,
        revision="",
        variant=nothing,
    )

Stream a compact host parameter tree, recursively transfer that tree exactly
once, allocate static K/V, and compile fixed-shape prefill/decode executables.
The ordinary unpacked parameter topology is never built or transferred.
Callers must select the desired Reactant backend before invoking this function.
"""
function load_hf_qwen3_bf16_xla_session(
    model_dir::AbstractString;
    context_tokens::Integer=4096,
    prefill_chunk_tokens::Integer=64,
    strategy::Symbol=:greedy,
    sample_top_k=nothing,
    revision::AbstractString="",
    variant=nothing,
)
    strategy in (:greedy, :sample, :device_sample) || throw(ArgumentError(
        "XLA strategy must be :greedy, :sample, or :device_sample",
    ))
    if sample_top_k !== nothing
        strategy === :device_sample || throw(ArgumentError(
            "sample_top_k only applies to the :device_sample strategy",
        ))
        sample_top_k isa Integer && sample_top_k > 0 || throw(ArgumentError(
            "sample_top_k must be a positive integer",
        ))
    end
    context = Int(context_tokens)
    chunk = Int(prefill_chunk_tokens)
    0 < chunk <= context || throw(ArgumentError(
        "prefill_chunk_tokens must be in 1:context_tokens",
    ))
    context % chunk == 0 || throw(ArgumentError(
        "context_tokens must be divisible by prefill_chunk_tokens",
    ))

    allocator_before_load = _qwen3_xla_allocator_snapshot()
    load_started = time_ns()
    loaded = load_hf_qwen3_compact_bundle(
        model_dir;
        max_seq_len=context,
        weight_dtype=BFloat16,
        revision,
        variant,
    )
    host_load_seconds = (time_ns() - load_started) / 1.0e9
    model = loaded.model
    tokenizer = loaded.tokenizer
    generation_config = loaded.generation_config
    host_parameters = loaded.parameters
    parameter_logical_bytes = _qwen3_xla_tensor_bytes(host_parameters)
    parameter_tensor_count = _qwen3_xla_tensor_leaves(host_parameters)
    rope = first(values(model.blocks.layers)).attn.rope

    transfer_started = time_ns()
    parameters = Reactant.to_rarray(host_parameters)
    parameter_transfer_seconds = (time_ns() - transfer_started) / 1.0e9
    allocator_after_parameter_transfer = _qwen3_xla_allocator_snapshot()

    # Remove every host reference to the large tree before allocating K/V or
    # compiling. `parameters` is the only device parameter topology.
    loaded = nothing
    host_parameters = nothing
    GC.gc(true)

    runtime_started = time_ns()
    cos_table = Reactant.to_rarray(BFloat16.(rope.cos_cache[:, 1:context]))
    sin_table = Reactant.to_rarray(BFloat16.(rope.sin_cache[:, 1:context]))
    cache_shape = (
        model.head_dim,
        model.num_kv_heads,
        context,
        1,
    )
    key_caches = Tuple(
        Reactant.to_rarray(zeros(BFloat16, cache_shape))
        for _ in 1:model.num_layers
    )
    value_caches = Tuple(
        Reactant.to_rarray(zeros(BFloat16, cache_shape))
        for _ in 1:model.num_layers
    )
    runtime_allocation_seconds = (time_ns() - runtime_started) / 1.0e9
    allocator_after_runtime_allocation = _qwen3_xla_allocator_snapshot()

    compile_tokens = Reactant.to_rarray(ones(Int, chunk, 1))
    compile_token = Reactant.to_rarray(ones(Int, 1))
    compile_position = Reactant.to_rarray(zeros(Int32, 1))
    compile_key_positions =
        Reactant.to_rarray(Int32.(collect(1:context)))

    # `top_k` fixes the number of extraction passes, so it is the one sampling
    # option baked into the executable; temperature and top-p stay as device
    # scalars and can change per request without recompiling.
    top_k_static = 0
    if strategy === :device_sample
        configured = sample_top_k === nothing ?
            generation_config.top_k : sample_top_k
        configured isa Integer && configured > 0 || throw(ArgumentError(
            "device sampling requires a positive top_k; the generation " *
            "config did not supply one, pass sample_top_k explicitly",
        ))
        top_k_static = min(Int(configured), model.vocab_size)
    end
    compile_uniform = Reactant.to_rarray(zeros(Float32, 1))
    compile_temperature = Reactant.to_rarray(ones(Float32, 1))
    compile_top_p = Reactant.to_rarray(ones(Float32, 1))

    prefill_kernel = if strategy === :device_sample
        (ps, tokens, kc, vc, position, cos_t, sin_t, kp, u, t, p) ->
            _bf16a_static_prefill_chunk_sample(
                model,
                ps,
                tokens,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
                u,
                t,
                p,
                top_k_static,
            )
    elseif strategy === :greedy
        (ps, tokens, kc, vc, position, cos_t, sin_t, kp) ->
            _bf16a_static_prefill_chunk_greedy(
                model,
                ps,
                tokens,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
    else
        (ps, tokens, kc, vc, position, cos_t, sin_t, kp) ->
            _bf16a_static_prefill_chunk_core(
                model,
                ps,
                tokens,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
    end
    decode_kernel = if strategy === :device_sample
        (ps, token, kc, vc, position, cos_t, sin_t, kp, u, t, p) ->
            _bf16a_static_decode_sample_step_packed(
                model,
                ps,
                token,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
                u,
                t,
                p,
                top_k_static,
            )
    elseif strategy === :greedy
        (ps, token, kc, vc, position, cos_t, sin_t, kp) ->
            _bf16a_static_decode_greedy_step_packed(
                model,
                ps,
                token,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
    else
        function (ps, token, kc, vc, position, cos_t, sin_t, kp)
            logits = _bf16a_static_decode_step_packed(
                model,
                ps,
                token,
                kc,
                vc,
                position,
                cos_t,
                sin_t,
                kp,
            )
            return logits, position .+ one(Int32)
        end
    end

    prefill_compile_started = time_ns()
    compiled_prefill = if strategy === :device_sample
        Reactant.@compile prefill_kernel(
            parameters,
            compile_tokens,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
            compile_uniform,
            compile_temperature,
            compile_top_p,
        )
    else
        Reactant.@compile prefill_kernel(
            parameters,
            compile_tokens,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
        )
    end
    prefill_compile_seconds =
        (time_ns() - prefill_compile_started) / 1.0e9

    decode_compile_started = time_ns()
    compiled_decode = if strategy === :device_sample
        Reactant.@compile decode_kernel(
            parameters,
            compile_token,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
            compile_uniform,
            compile_temperature,
            compile_top_p,
        )
    else
        Reactant.@compile decode_kernel(
            parameters,
            compile_token,
            key_caches,
            value_caches,
            compile_position,
            cos_table,
            sin_table,
            compile_key_positions,
        )
    end
    decode_compile_seconds =
        (time_ns() - decode_compile_started) / 1.0e9
    allocator_ready = _qwen3_xla_allocator_snapshot()
    load_metrics = (;
        host_load_seconds,
        parameter_transfer_seconds,
        runtime_allocation_seconds,
        prefill_compile_seconds,
        decode_compile_seconds,
        parameter_logical_bytes,
        parameter_tensor_count,
        device_parameter_tree_count=1,
        device_parameter_tree_transfer_count=1,
        original_parameter_tree_constructed=false,
        original_parameter_tree_transferred=false,
        separate_packed_projection_tree_transferred=false,
        allocator_before_load,
        allocator_after_parameter_transfer,
        allocator_after_runtime_allocation,
        allocator_ready,
    )
    return HFQwen3BF16XLASession(
        model,
        parameters,
        tokenizer,
        generation_config,
        cos_table,
        sin_table,
        key_caches,
        value_caches,
        compiled_prefill,
        compiled_decode,
        strategy,
        context,
        chunk,
        top_k_static,
        0,
        load_metrics,
    )
end

function reset_hf_qwen3_bf16_xla_session!(
    session::HFQwen3BF16XLASession,
)
    session.position = 0
    return session
end

"""
    generate_hf_qwen3_bf16_xla!(
        session, prompt_tokens; max_new_tokens=512, kwargs...
    )

Generate from a reusable single-tree XLA session.

  * `:greedy` keeps argmax on device and transfers one token per step.
  * `:device_sample` keeps the whole temperature/top-k/top-p policy on device;
    the host sends one uniform and receives one token per step.
  * `:sample` transfers one logits vector per step and applies the host
    temperature/top-k/top-p policy of the eager deployment path.

`sample_uniforms` replaces the RNG draws with a fixed sequence, which is how
the two sampling strategies are compared token by token.
"""
function generate_hf_qwen3_bf16_xla!(
    session::HFQwen3BF16XLASession,
    prompt_tokens;
    max_new_tokens::Integer=512,
    temperature=nothing,
    top_k=nothing,
    top_p=nothing,
    rng::AbstractRNG=default_rng(),
    stop_token_ids=nothing,
    pad_token_id::Integer=1,
    on_token=nothing,
    sample_uniforms=nothing,
)
    prompt_ids = Int.(vec(collect(prompt_tokens)))
    _validate_generation_ids(prompt_ids, session.model.vocab_size)
    plan = plan_qwen3_xla_window(
        length(prompt_ids),
        max_new_tokens;
        context_tokens=session.context_tokens,
        chunk_tokens=session.prefill_chunk_tokens,
    )
    padded = qwen3_xla_pad_prompt(
        prompt_ids,
        plan;
        pad_token_id,
    )
    1 <= pad_token_id <= session.model.vocab_size || throw(ArgumentError(
        "pad_token_id is outside the model vocabulary",
    ))
    stops = stop_token_ids === nothing ?
        Set(session.tokenizer.eos_ids) : Set(Int.(stop_token_ids))
    all(id -> 1 <= id <= session.model.vocab_size, stops) ||
        throw(ArgumentError("stop token id is outside the model vocabulary"))

    resolved_temperature = temperature === nothing ?
        session.generation_config.temperature : temperature
    resolved_top_k = top_k === nothing ?
        session.generation_config.top_k : top_k
    resolved_top_p = top_p === nothing ?
        session.generation_config.top_p : top_p

    uniforms = nothing
    if sample_uniforms !== nothing
        session.strategy === :greedy && throw(ArgumentError(
            "sample_uniforms requires a sampling strategy",
        ))
        uniforms = Float32[
            _validate_device_sampling_uniform(value)
            for value in vec(collect(sample_uniforms))
        ]
        length(uniforms) >= plan.max_new_tokens || throw(ArgumentError(
            "sample_uniforms must supply one value per requested token",
        ))
    end
    draw_uniform(step) = uniforms === nothing ?
        rand(rng, Float32) : uniforms[step]

    temperature_state = nothing
    top_p_state = nothing
    if session.strategy === :device_sample
        options = _validate_device_sampling_options(;
            temperature=resolved_temperature,
            top_k=resolved_top_k,
            top_p=resolved_top_p,
            vocab_size=session.model.vocab_size,
        )
        min(options.top_k, session.model.vocab_size) == session.sample_top_k ||
            throw(ArgumentError(
                "top_k must match the compiled device sampling constant " *
                "$(session.sample_top_k); recompile the session to change it",
            ))
        temperature_state = Reactant.to_rarray(Float32[options.temperature])
        top_p_state = Reactant.to_rarray(Float32[options.top_p])
    end

    key_positions = Reactant.to_rarray(qwen3_xla_key_positions(plan))
    position_state = Reactant.to_rarray(zeros(Int32, 1))
    reset_hf_qwen3_bf16_xla_session!(session)
    allocator_before = _qwen3_xla_allocator_snapshot()

    prefill_started = time_ns()
    output_state = nothing
    # Every prefill chunk runs the same executable but only the final chunk
    # selects a token, so one uniform is consumed for the first output token
    # regardless of how many chunks the prompt needs.
    prefill_uniform_state = session.strategy === :device_sample ?
        Reactant.to_rarray(Float32[draw_uniform(1)]) : nothing
    for first_index in 1:session.prefill_chunk_tokens:length(padded)
        last_index = first_index + session.prefill_chunk_tokens - 1
        token_state = Reactant.to_rarray(reshape(
            padded[first_index:last_index],
            :,
            1,
        ))
        output_state, position_state = if session.strategy === :device_sample
            session.compiled_prefill(
                session.parameters,
                token_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
                prefill_uniform_state,
                temperature_state,
                top_p_state,
            )
        else
            session.compiled_prefill(
                session.parameters,
                token_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
            )
        end
    end
    first_output = if session.strategy === :sample
        logits = Array(output_state)
        choice, _ = _qwen3_session_choice(
            logits,
            identity,
            :sample,
            rng;
            temperature=resolved_temperature,
            top_k=resolved_top_k,
            top_p=resolved_top_p,
            sample_uniform=uniforms === nothing ? nothing : uniforms[1],
        )
        choice
    else
        Int(Array(output_state)[1])
    end
    prefill_seconds = (time_ns() - prefill_started) / 1.0e9

    generated_ids = Int[first_output]
    on_token === nothing || on_token(first_output, 1)
    stop_reason = first_output in stops ? :eos : :length

    decode_started = time_ns()
    while length(generated_ids) < plan.max_new_tokens &&
            !(last(generated_ids) in stops)
        step = length(generated_ids) + 1
        input_state = session.strategy === :sample ?
            Reactant.to_rarray([last(generated_ids)]) :
            output_state
        output_state, position_state = if session.strategy === :device_sample
            session.compiled_decode(
                session.parameters,
                input_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
                Reactant.to_rarray(Float32[draw_uniform(step)]),
                temperature_state,
                top_p_state,
            )
        else
            session.compiled_decode(
                session.parameters,
                input_state,
                session.key_caches,
                session.value_caches,
                position_state,
                session.cos_table,
                session.sin_table,
                key_positions,
            )
        end
        next_token = if session.strategy === :sample
            logits = Array(output_state)
            choice, _ = _qwen3_session_choice(
                logits,
                identity,
                :sample,
                rng;
                temperature=resolved_temperature,
                top_k=resolved_top_k,
                top_p=resolved_top_p,
                sample_uniform=uniforms === nothing ? nothing : uniforms[step],
            )
            choice
        else
            Int(Array(output_state)[1])
        end
        push!(generated_ids, next_token)
        on_token === nothing || on_token(next_token, length(generated_ids))
    end
    decode_seconds = (time_ns() - decode_started) / 1.0e9
    last(generated_ids) in stops && (stop_reason = :eos)
    session.position =
        plan.prompt_bucket_tokens + length(generated_ids) - 1
    allocator_after = _qwen3_xla_allocator_snapshot()
    completion = decode(
        session.tokenizer,
        generated_ids;
        errors=:replace,
        skip_special_tokens=true,
    )
    return (;
        prompt_ids,
        generated_ids,
        token_ids=vcat(prompt_ids, generated_ids),
        completion,
        stop_reason,
        strategy=session.strategy,
        window_plan=plan,
        prefill_seconds,
        decode_seconds,
        tokens_per_second=length(generated_ids) <= 1 ?
            0.0 : (length(generated_ids) - 1) / decode_seconds,
        allocator_before,
        allocator_after,
    )
end
