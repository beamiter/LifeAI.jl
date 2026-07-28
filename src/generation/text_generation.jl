using MLDataDevices: cpu_device, get_device
using Random: AbstractRNG, default_rng

"""
    generate(model, ps, st, prompt_tokens; kwargs...)

Autoregressively generate token ids. The returned token vector includes the prompt.
"""
function generate(
    model,
    ps,
    st,
    prompt_tokens;
    max_new_tokens::Int=100,
    temperature::Real=1.0f0,
    top_k=nothing,
    top_p=nothing,
    rng::AbstractRNG=default_rng(),
    device=get_device(ps),
)
    max_new_tokens >= 0 || throw(ArgumentError(
        "`max_new_tokens` must be non-negative",
    ))

    generated = Int.(collect(prompt_tokens))
    isempty(generated) && throw(ArgumentError("`prompt_tokens` must not be empty"))
    all(id -> 1 <= id <= model.vocab_size, generated) || throw(ArgumentError(
        "prompt token id is outside 1:$(model.vocab_size)",
    ))

    st_current = st
    host = cpu_device()

    for _ in 1:max_new_tokens
        context_start = max(1, length(generated) - model.max_seq_len + 1)
        context = generated[context_start:end]
        input_tokens = device(reshape(context, length(context), 1))

        logits, st_current = model(
            input_tokens,
            ps,
            st_current,
        )

        last_logits = vec(host(@view(logits[:, end, 1])))
        next_id = _sample_token(
            last_logits,
            rng;
            temperature,
            top_k,
            top_p,
        )

        push!(generated, next_id)
    end

    return generated, st_current
end

"""
    generate(model, ps, st, tokenizer, prompt; kwargs...)

String convenience overload for every `AbstractTokenizer`. Byte-based generation may
end on an incomplete UTF-8 sequence, so display decoding defaults to `:replace` while
the underlying token ids and `decode_bytes` remain lossless.
"""
function generate(
    model,
    ps,
    st,
    tokenizer::AbstractTokenizer,
    prompt::AbstractString;
    add_special_tokens::Bool=false,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    kwargs...,
)
    prompt_tokens = encode(tokenizer, prompt; add_special_tokens)
    generated_tokens, st_new = generate(
        model,
        ps,
        st,
        prompt_tokens;
        kwargs...,
    )

    return decode(
        tokenizer,
        generated_tokens;
        errors=decode_errors,
        skip_special_tokens,
    ), st_new
end

"""String convenience overload for KV-cached generation with any tokenizer."""
function generate_cached(
    model::GPTModel,
    ps,
    st::NamedTuple,
    tokenizer::AbstractTokenizer,
    prompt::AbstractString;
    add_special_tokens::Bool=false,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    kwargs...,
)
    generated_tokens, st_new = generate_cached(
        model,
        ps,
        st,
        encode(tokenizer, prompt; add_special_tokens);
        kwargs...,
    )
    return decode(
        tokenizer,
        generated_tokens;
        errors=decode_errors,
        skip_special_tokens,
    ), st_new
end

"""String convenience overload for compiled KV-cached generation with any tokenizer."""
function generate_xla_cached!(
    decoder,
    tokenizer::AbstractTokenizer,
    prompt::AbstractString;
    add_special_tokens::Bool=false,
    decode_errors::Symbol=:replace,
    skip_special_tokens::Bool=true,
    kwargs...,
)
    generated_tokens, state = generate_xla_cached!(
        decoder,
        encode(tokenizer, prompt; add_special_tokens);
        kwargs...,
    )
    return decode(
        tokenizer,
        generated_tokens;
        errors=decode_errors,
        skip_special_tokens,
    ), state
end
