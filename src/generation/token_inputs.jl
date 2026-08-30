# Host-side generation input normalization and validation shared by eager,
# dynamic-cache, static-cache, streamed, and accelerated inference paths.

function _strict_host_int(value, label::AbstractString)
    value isa Integer && !(value isa Bool) || throw(ArgumentError(
        "$label must be an integer",
    ))
    return try
        Int(value)
    catch error
        error isa Union{InexactError,OverflowError,DomainError,MethodError} ||
            rethrow()
        throw(ArgumentError("$label is outside the host integer range"))
    end
end

function _strict_host_int_array(values, label::AbstractString)
    return map(value -> _strict_host_int(value, label), collect(values))
end

function _strict_preserving_int_array(values, label::AbstractString)
    element_type = eltype(values)
    element_type <: Integer && !(element_type <: Bool) || throw(ArgumentError(
        "$label must contain integers",
    ))
    values isa Array && return map(
        value -> _strict_host_int(value, label),
        values,
    )
    return try
        Int.(values)
    catch error
        error isa Union{InexactError,OverflowError,DomainError,MethodError} ||
            rethrow()
        throw(ArgumentError("$label contains an integer outside the host range"))
    end
end

function _prefill_token_matrix(prompt_tokens)
    if prompt_tokens isa AbstractVector
        return reshape(
            _strict_host_int_array(prompt_tokens, "`prompt_tokens`"),
            :,
            1,
        )
    elseif prompt_tokens isa AbstractMatrix
        return _strict_host_int_array(prompt_tokens, "`prompt_tokens`")
    end

    throw(DimensionMismatch(
        "`prompt_tokens` must be a vector or a (seq_len, batch) matrix",
    ))
end

function _decode_token_matrix(token, batch_size::Int)
    if token isa Integer
        batch_size == 1 ||
            throw(DimensionMismatch("a scalar token is only valid for batch_size=1"))
        return reshape([_strict_host_int(token, "`token`")], 1, 1)
    elseif token isa AbstractVector
        length(token) == batch_size ||
            throw(DimensionMismatch("token vector length must equal cache.batch_size"))
        return reshape(
            _strict_host_int_array(token, "`token`"),
            1,
            batch_size,
        )
    elseif token isa AbstractMatrix
        size(token) == (1, batch_size) ||
            throw(DimensionMismatch("token matrix must have shape (1, cache.batch_size)"))
        return _strict_host_int_array(token, "`token`")
    end

    throw(DimensionMismatch("`token` must be an integer, vector, or one-row matrix"))
end

function _validate_generation_ids(tokens, vocab_size::Int)
    all(id -> 1 <= id <= vocab_size, tokens) ||
        throw(ArgumentError("token id is outside 1:$vocab_size"))
    return nothing
end
