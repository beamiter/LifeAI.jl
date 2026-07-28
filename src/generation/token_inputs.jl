# Host-side generation input normalization and validation shared by eager,
# dynamic-cache, static-cache, streamed, and accelerated inference paths.

function _prefill_token_matrix(prompt_tokens)
    if prompt_tokens isa AbstractVector
        return reshape(Int.(collect(prompt_tokens)), :, 1)
    elseif prompt_tokens isa AbstractMatrix
        return Int.(collect(prompt_tokens))
    end

    throw(DimensionMismatch(
        "`prompt_tokens` must be a vector or a (seq_len, batch) matrix",
    ))
end

function _decode_token_matrix(token, batch_size::Int)
    if token isa Integer
        batch_size == 1 ||
            throw(DimensionMismatch("a scalar token is only valid for batch_size=1"))
        return reshape([Int(token)], 1, 1)
    elseif token isa AbstractVector
        length(token) == batch_size ||
            throw(DimensionMismatch("token vector length must equal cache.batch_size"))
        return reshape(Int.(collect(token)), 1, batch_size)
    elseif token isa AbstractMatrix
        size(token) == (1, batch_size) ||
            throw(DimensionMismatch("token matrix must have shape (1, cache.batch_size)"))
        return Int.(collect(token))
    end

    throw(DimensionMismatch("`token` must be an integer, vector, or one-row matrix"))
end

function _validate_generation_ids(tokens, vocab_size::Int)
    all(id -> 1 <= id <= vocab_size, tokens) ||
        throw(ArgumentError("token id is outside 1:$vocab_size"))
    return nothing
end
