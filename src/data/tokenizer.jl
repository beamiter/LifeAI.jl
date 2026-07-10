"""
    Tokenizer

A minimal character-level tokenizer for tiny GPT experiments.

Token ids are 1-based so they can be passed directly to `Lux.Embedding` and
`GPTModel`.

Fields:

- `token_to_id`: maps each character to its integer token id.
- `id_to_token`: maps each integer token id back to its character.
- `unk_id`: id used for unknown characters, or `nothing` when unknown
  characters should raise an error.
"""
struct Tokenizer
    token_to_id::Dict{Char, Int}
    id_to_token::Vector{Char}
    unk_id::Union{Nothing, Int}

    function Tokenizer(
        token_to_id::Dict{Char, Int},
        id_to_token::Vector{Char},
        unk_id::Union{Nothing, Int}=nothing,
    )
        @assert !isempty(id_to_token) "`id_to_token` must not be empty"
        @assert length(token_to_id) == length(id_to_token) "token maps must have equal sizes"

        for (id, token) in enumerate(id_to_token)
            @assert get(token_to_id, token, 0) == id "token maps must be exact inverses"
        end

        if unk_id !== nothing
            @assert 1 <= unk_id <= length(id_to_token) "`unk_id` is outside the vocabulary"
        end

        new(token_to_id, id_to_token, unk_id)
    end
end

"""
    fit_tokenizer(text; add_unk=false, unk_token='�')

Build a deterministic character-level tokenizer from `text`.

Characters are sorted before ids are assigned, which makes the vocabulary
stable regardless of their first occurrence order. When `add_unk=true`,
`unk_token` is assigned id 1 and unseen characters encode to that id.
"""
function fit_tokenizer(
    text::AbstractString;
    add_unk::Bool=false,
    unk_token::Char='�',
)
    tokens = sort!(unique(collect(text)))
    @assert !isempty(tokens) "`text` must contain at least one character"

    if add_unk
        filter!(token -> token != unk_token, tokens)
        pushfirst!(tokens, unk_token)
    end

    token_to_id = Dict(token => id for (id, token) in enumerate(tokens))
    unk_id = add_unk ? token_to_id[unk_token] : nothing

    return Tokenizer(token_to_id, tokens, unk_id)
end

"""
    encode(tokenizer, text)

Encode a string as a vector of 1-based token ids.

If the tokenizer has no unknown token, an unseen character raises
`ArgumentError`.
"""
function encode(tokenizer::Tokenizer, text::AbstractString)
    ids = Vector{Int}(undef, length(text))

    for (index, token) in enumerate(text)
        id = get(tokenizer.token_to_id, token, tokenizer.unk_id)

        if id === nothing
            throw(ArgumentError("character $(repr(token)) is not in the tokenizer vocabulary"))
        end

        ids[index] = id
    end

    return ids
end

"""
    decode(tokenizer, ids)

Decode an iterable of token ids into a `String`.
"""
function decode(tokenizer::Tokenizer, ids)
    output = IOBuffer()
    vocabulary_size = length(tokenizer.id_to_token)

    for id in ids
        id isa Integer || throw(ArgumentError("token id $(repr(id)) is not an integer"))
        1 <= id <= vocabulary_size ||
            throw(ArgumentError("token id $id is outside 1:$vocabulary_size"))

        print(output, tokenizer.id_to_token[id])
    end

    return String(take!(output))
end

"""
    vocab_size(tokenizer)

Return the number of tokens in the vocabulary.
"""
vocab_size(tokenizer::Tokenizer) = length(tokenizer.id_to_token)

Base.length(tokenizer::Tokenizer) = vocab_size(tokenizer)
Base.in(token::Char, tokenizer::Tokenizer) = haskey(tokenizer.token_to_id, token)
