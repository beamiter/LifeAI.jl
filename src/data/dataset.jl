"""
    DatasetLoader(token_ids; seq_len, batch_size=1, stride=seq_len, drop_last=true)

A minimal next-token dataset loader for GPT training.

Each sample is a pair of shifted token sequences:

    x = token_ids[start:start + seq_len - 1]
    y = token_ids[start + 1:start + seq_len]

Batches follow the tensor convention used by `GPTModel`:

    x: (seq_len, batch)
    y: (seq_len, batch)

`stride` controls the distance between neighboring windows. Use `stride=1`
for all overlapping windows, or `stride=seq_len` for non-overlapping chunks.
"""
struct DatasetLoader{T<:Integer, V<:AbstractVector{T}}
    token_ids::V
    seq_len::Int
    batch_size::Int
    stride::Int
    drop_last::Bool
    starts::Vector{Int}
end

function DatasetLoader(
    token_ids::AbstractVector{T};
    seq_len::Int,
    batch_size::Int=1,
    stride::Int=seq_len,
    drop_last::Bool=true,
) where {T<:Integer}
    @assert seq_len > 0 "`seq_len` must be positive"
    @assert batch_size > 0 "`batch_size` must be positive"
    @assert stride > 0 "`stride` must be positive"
    @assert length(token_ids) >= seq_len + 1 "`token_ids` must contain at least seq_len + 1 tokens"
    @assert all(>(0), token_ids) "`token_ids` must contain positive 1-based ids"

    last_start = length(token_ids) - seq_len
    starts = collect(1:stride:last_start)

    return DatasetLoader(token_ids, seq_len, batch_size, stride, drop_last, starts)
end

"""
    DatasetLoader(tokenizer, text; kwargs...)

Encode `text` with `tokenizer` and construct a next-token dataset loader.
"""
function DatasetLoader(tokenizer::Tokenizer, text::AbstractString; kwargs...)
    return DatasetLoader(encode(tokenizer, text); kwargs...)
end

"""Return the number of sequence windows in the loader."""
num_samples(loader::DatasetLoader) = length(loader.starts)

"""Return the number of batches produced by the loader."""
function num_batches(loader::DatasetLoader)
    samples = num_samples(loader)
    return loader.drop_last ? samples ÷ loader.batch_size : cld(samples, loader.batch_size)
end

Base.length(loader::DatasetLoader) = num_batches(loader)
Base.eltype(::Type{<:DatasetLoader{T}}) where {T} = Tuple{Matrix{T}, Matrix{T}}

function Base.getindex(loader::DatasetLoader{T}, batch_index::Integer) where {T}
    1 <= batch_index <= length(loader) || throw(BoundsError(loader, batch_index))

    first_sample = (batch_index - 1) * loader.batch_size + 1
    last_sample = min(first_sample + loader.batch_size - 1, num_samples(loader))
    current_batch_size = last_sample - first_sample + 1

    x = Matrix{T}(undef, loader.seq_len, current_batch_size)
    y = Matrix{T}(undef, loader.seq_len, current_batch_size)

    @inbounds for (column, sample_index) in enumerate(first_sample:last_sample)
        start = loader.starts[sample_index]
        for position in 1:loader.seq_len
            x[position, column] = loader.token_ids[start + position - 1]
            y[position, column] = loader.token_ids[start + position]
        end
    end

    return x, y
end

function Base.iterate(loader::DatasetLoader, batch_index::Int=1)
    batch_index > length(loader) && return nothing
    return loader[batch_index], batch_index + 1
end

function _validation_count(
    total::Int;
    validation_fraction::Real,
    validation_size,
)
    total >= 2 || throw(ArgumentError("the stream must contain at least two items"))

    if validation_size === nothing
        0 < validation_fraction < 1 ||
            throw(ArgumentError("`validation_fraction` must be between 0 and 1"))
        count = round(Int, total * validation_fraction)
        return clamp(count, 1, total - 1)
    end

    validation_size isa Integer ||
        throw(ArgumentError("`validation_size` must be an integer or `nothing`"))
    1 <= validation_size < total ||
        throw(ArgumentError("`validation_size` must be in 1:$(total - 1)"))
    return Int(validation_size)
end

"""
    split_token_stream(token_ids; validation_fraction=0.1, validation_size=nothing)

Split a raw token stream *before* constructing sliding windows. This is the
important ordering for avoiding train/validation leakage: train and validation
loaders later create windows from two disjoint vectors.

Returns a named tuple with `train`, `validation`, and `split_index`.
"""
function split_token_stream(
    token_ids::AbstractVector{T};
    validation_fraction::Real=0.1,
    validation_size=nothing,
) where {T<:Integer}
    count = _validation_count(
        length(token_ids);
        validation_fraction,
        validation_size,
    )
    split_index = length(token_ids) - count

    return (;
        train=collect(@view(token_ids[1:split_index])),
        validation=collect(@view(token_ids[(split_index + 1):end])),
        split_index,
    )
end

"""
    split_text_stream(text; validation_fraction=0.1, validation_size=nothing)

Character-boundary counterpart of [`split_token_stream`](@ref). The tokenizer
can then be fitted only on `train`, making unknown validation characters follow
the tokenizer's explicit unknown-token policy.
"""
function split_text_stream(
    text::AbstractString;
    validation_fraction::Real=0.1,
    validation_size=nothing,
)
    characters = collect(text)
    count = _validation_count(
        length(characters);
        validation_fraction,
        validation_size,
    )
    split_index = length(characters) - count

    return (;
        train=join(@view(characters[1:split_index])),
        validation=join(@view(characters[(split_index + 1):end])),
        split_index,
    )
end

"""
    train_validation_loaders(text; seq_len, kwargs...)

Create a leakage-safe character-level experiment:

1. split the raw character stream;
2. fit the tokenizer on the train split only;
3. encode train and validation independently;
4. construct independent sliding-window loaders.

Returns `(; tokenizer, train, validation, text_split, token_split)`.
"""
function train_validation_loaders(
    text::AbstractString;
    seq_len::Int,
    batch_size::Int=1,
    stride::Int=seq_len,
    drop_last::Bool=true,
    validation_fraction::Real=0.1,
    validation_size=nothing,
    add_unk::Bool=true,
    unk_token::Char='�',
)
    text_split = split_text_stream(
        text;
        validation_fraction,
        validation_size,
    )
    tokenizer = fit_tokenizer(
        text_split.train;
        add_unk,
        unk_token,
    )
    train_tokens = encode(tokenizer, text_split.train)
    validation_tokens = encode(tokenizer, text_split.validation)

    minimum_length = seq_len + 1
    length(train_tokens) >= minimum_length || throw(ArgumentError(
        "train split must contain at least $minimum_length tokens for seq_len=$seq_len",
    ))
    length(validation_tokens) >= minimum_length || throw(ArgumentError(
        "validation split must contain at least $minimum_length tokens for seq_len=$seq_len",
    ))

    train_loader = DatasetLoader(
        train_tokens;
        seq_len,
        batch_size,
        stride,
        drop_last,
    )
    validation_loader = DatasetLoader(
        validation_tokens;
        seq_len,
        batch_size,
        stride,
        drop_last,
    )

    return (;
        tokenizer,
        train=train_loader,
        validation=validation_loader,
        text_split,
        token_split=(;
            train=train_tokens,
            validation=validation_tokens,
            split_index=length(train_tokens),
        ),
    )
end
