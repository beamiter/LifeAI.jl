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
