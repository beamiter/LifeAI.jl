using SHA: sha256
using TOML
using Unicode

const TOKENIZER_ARTIFACT_VERSION = 1
const BYTE_ALPHABET_SIZE = 256
const DEFAULT_BYTE_SPECIAL_TOKENS = (
    :bos => "<|bos|>",
    :eos => "<|eos|>",
    :pad => "<|pad|>",
)

abstract type AbstractTokenizer end

function _validate_normalization(normalization::Symbol)
    normalized = Symbol(lowercase(String(normalization)))
    normalized in (:none, :nfc, :nfd, :nfkc, :nfkd) || throw(ArgumentError(
        "`normalization` must be :none, :nfc, :nfd, :nfkc, or :nfkd",
    ))
    return normalized
end

"""Apply the explicitly configured Unicode normalization policy."""
function normalize_text(text::AbstractString, normalization::Symbol=:none)
    normalized = _validate_normalization(normalization)
    normalized === :none && return String(text)
    return Unicode.normalize(String(text), Symbol(uppercase(String(normalized))))
end

function _sha256_hex(data)
    bytes = data isa AbstractString ? Vector{UInt8}(codeunits(data)) : Vector{UInt8}(data)
    return bytes2hex(sha256(bytes))
end

function _special_token_tables(
    entries;
    first_id::Int=BYTE_ALPHABET_SIZE + 1,
)
    first_id > 0 || throw(ArgumentError("`first_id` must be positive"))
    ids = Dict{Symbol,Int}()
    texts = Dict{Symbol,String}()
    seen_texts = Set{String}()

    for (offset, entry) in enumerate(entries)
        entry isa Pair || throw(ArgumentError(
            "special tokens must be supplied as `name => text` pairs",
        ))
        name = Symbol(first(entry))
        text = String(last(entry))
        isempty(String(name)) && throw(ArgumentError("special token name must not be empty"))
        isempty(text) && throw(ArgumentError("special token text must not be empty"))
        haskey(ids, name) && throw(ArgumentError("duplicate special token name: $name"))
        text in seen_texts && throw(ArgumentError("duplicate special token text: $(repr(text))"))

        ids[name] = first_id + offset - 1
        texts[name] = text
        push!(seen_texts, text)
    end

    return ids, texts
end

function _validate_special_token_tables(
    ids::Dict{Symbol,Int},
    texts::Dict{Symbol,String};
    first_id::Int=BYTE_ALPHABET_SIZE + 1,
)
    Set(keys(ids)) == Set(keys(texts)) || throw(ArgumentError(
        "special token id and text maps must contain the same names",
    ))
    expected_ids = collect(first_id:(first_id + length(ids) - 1))
    actual_ids = sort!(collect(values(ids)))
    actual_ids == expected_ids || throw(ArgumentError(
        "special token ids must be contiguous starting at $first_id",
    ))
    length(unique(values(texts))) == length(texts) || throw(ArgumentError(
        "special token texts must be unique",
    ))
    all(text -> !isempty(text), values(texts)) || throw(ArgumentError(
        "special token texts must not be empty",
    ))
    return nothing
end

function _ordered_special_tokens(tokenizer)
    names = sort!(collect(keys(tokenizer.special_tokens)); by=name -> tokenizer.special_tokens[name])
    return [
        (;
            name,
            id=tokenizer.special_tokens[name],
            text=tokenizer.special_token_strings[name],
        ) for name in names
    ]
end

function _special_token_name(tokenizer, id::Int)
    for (name, token_id) in tokenizer.special_tokens
        token_id == id && return name
    end
    return nothing
end

function _append_special_token_bytes!(output::Vector{UInt8}, tokenizer, id::Int)
    name = _special_token_name(tokenizer, id)
    name === nothing && throw(ArgumentError("token id $id is not a known special token"))
    append!(output, codeunits(tokenizer.special_token_strings[name]))
    return output
end

function _add_boundary_special_tokens(tokenizer, ids::Vector{Int}, add_special_tokens::Bool)
    add_special_tokens || return ids
    output = Int[]
    bos_id = special_token_id(tokenizer, :bos)
    eos_id = special_token_id(tokenizer, :eos)
    bos_id === nothing || push!(output, bos_id)
    append!(output, ids)
    eos_id === nothing || push!(output, eos_id)
    return output
end

function _decode_utf8(bytes::Vector{UInt8}; errors::Symbol=:strict)
    errors in (:strict, :replace) || throw(ArgumentError(
        "`errors` must be :strict or :replace",
    ))
    text = String(copy(bytes))
    if errors === :strict
        isvalid(text) || throw(ArgumentError("decoded bytes are not valid UTF-8"))
        return text
    end
    isvalid(text) && return text

    output = IOBuffer()
    for character in text
        print(output, isvalid(character) ? character : '�')
    end
    return String(take!(output))
end

"""
    Tokenizer

Legacy deterministic character-level tokenizer. The type and its field layout remain
unchanged so existing code keeps working; it now implements `AbstractTokenizer`.
"""
struct Tokenizer <: AbstractTokenizer
    token_to_id::Dict{Char,Int}
    id_to_token::Vector{Char}
    unk_id::Union{Nothing,Int}

    function Tokenizer(
        token_to_id::Dict{Char,Int},
        id_to_token::Vector{Char},
        unk_id::Union{Nothing,Int}=nothing,
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

"""Build a deterministic character-level tokenizer from `text`."""
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

function encode(
    tokenizer::Tokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)
    add_special_tokens && throw(ArgumentError(
        "legacy character tokenizer has no BOS/EOS boundary tokens",
    ))
    ids = Vector{Int}(undef, length(text))

    for (index, token) in enumerate(text)
        id = get(tokenizer.token_to_id, token, tokenizer.unk_id)
        id === nothing && throw(ArgumentError(
            "character $(repr(token)) is not in the tokenizer vocabulary",
        ))
        ids[index] = id
    end
    return ids
end

function decode_bytes(
    tokenizer::Tokenizer,
    ids;
    skip_special_tokens::Bool=false,
)
    output = IOBuffer()
    vocabulary_size = length(tokenizer.id_to_token)

    for id in ids
        id isa Integer || throw(ArgumentError("token id $(repr(id)) is not an integer"))
        1 <= id <= vocabulary_size || throw(ArgumentError(
            "token id $id is outside 1:$vocabulary_size",
        ))
        if skip_special_tokens && tokenizer.unk_id !== nothing && id == tokenizer.unk_id
            continue
        end
        print(output, tokenizer.id_to_token[Int(id)])
    end
    return Vector{UInt8}(take!(output))
end

function decode(
    tokenizer::Tokenizer,
    ids;
    errors::Symbol=:strict,
    skip_special_tokens::Bool=false,
)
    return _decode_utf8(
        decode_bytes(tokenizer, ids; skip_special_tokens);
        errors,
    )
end

vocab_size(tokenizer::Tokenizer) = length(tokenizer.id_to_token)
Base.length(tokenizer::Tokenizer) = vocab_size(tokenizer)
Base.in(token::Char, tokenizer::Tokenizer) = haskey(tokenizer.token_to_id, token)
special_token_id(tokenizer::Tokenizer, name) =
    Symbol(name) === :unk ? tokenizer.unk_id : nothing
_normalization_mode(::Tokenizer) = :none

"""A reversible UTF-8 byte tokenizer with 1-based byte ids and explicit specials."""
struct ByteTokenizer <: AbstractTokenizer
    normalization::Symbol
    special_tokens::Dict{Symbol,Int}
    special_token_strings::Dict{Symbol,String}

    function ByteTokenizer(
        normalization::Symbol,
        special_tokens::Dict{Symbol,Int},
        special_token_strings::Dict{Symbol,String},
    )
        normalized = _validate_normalization(normalization)
        _validate_special_token_tables(special_tokens, special_token_strings)
        new(normalized, copy(special_tokens), copy(special_token_strings))
    end
end

function ByteTokenizer(;
    normalization::Symbol=:none,
    special_tokens=DEFAULT_BYTE_SPECIAL_TOKENS,
)
    ids, texts = _special_token_tables(special_tokens)
    return ByteTokenizer(normalization, ids, texts)
end

_normalization_mode(tokenizer::ByteTokenizer) = tokenizer.normalization
vocab_size(tokenizer::ByteTokenizer) = BYTE_ALPHABET_SIZE + length(tokenizer.special_tokens)
Base.length(tokenizer::ByteTokenizer) = vocab_size(tokenizer)
special_token_id(tokenizer::ByteTokenizer, name) =
    get(tokenizer.special_tokens, Symbol(name), nothing)

function encode(
    tokenizer::ByteTokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)
    normalized = normalize_text(text, tokenizer.normalization)
    ids = [Int(byte) + 1 for byte in codeunits(normalized)]
    return _add_boundary_special_tokens(tokenizer, ids, add_special_tokens)
end

function decode_bytes(
    tokenizer::ByteTokenizer,
    ids;
    skip_special_tokens::Bool=false,
)
    output = UInt8[]
    vocabulary_size = vocab_size(tokenizer)

    for raw_id in ids
        raw_id isa Integer || throw(ArgumentError(
            "token id $(repr(raw_id)) is not an integer",
        ))
        id = Int(raw_id)
        1 <= id <= vocabulary_size || throw(ArgumentError(
            "token id $id is outside 1:$vocabulary_size",
        ))
        if id <= BYTE_ALPHABET_SIZE
            push!(output, UInt8(id - 1))
        elseif !skip_special_tokens
            _append_special_token_bytes!(output, tokenizer, id)
        end
    end
    return output
end

function decode(
    tokenizer::ByteTokenizer,
    ids;
    errors::Symbol=:strict,
    skip_special_tokens::Bool=false,
)
    return _decode_utf8(
        decode_bytes(tokenizer, ids; skip_special_tokens);
        errors,
    )
end

function _merge_pair(sequence::Vector{Int}, pair::Tuple{Int,Int}, new_id::Int)
    output = Int[]
    sizehint!(output, length(sequence))
    index = 1
    while index <= length(sequence)
        if index < length(sequence) &&
           sequence[index] == pair[1] &&
           sequence[index + 1] == pair[2]
            push!(output, new_id)
            index += 2
        else
            push!(output, sequence[index])
            index += 1
        end
    end
    return output
end

function _build_bpe_tables(
    special_tokens::Dict{Symbol,Int},
    special_token_strings::Dict{Symbol,String},
    merges::Vector{Tuple{Int,Int}},
)
    _validate_special_token_tables(special_tokens, special_token_strings)
    merge_offset = BYTE_ALPHABET_SIZE + length(special_tokens)
    total_vocabulary = merge_offset + length(merges)
    token_bytes = [UInt8[] for _ in 1:total_vocabulary]
    merge_ranks = Dict{Tuple{Int,Int},Int}()

    for id in 1:BYTE_ALPHABET_SIZE
        token_bytes[id] = UInt8[UInt8(id - 1)]
    end

    for (rank, pair) in enumerate(merges)
        haskey(merge_ranks, pair) && throw(ArgumentError(
            "duplicate BPE merge pair: $pair",
        ))
        new_id = merge_offset + rank
        left, right = pair
        1 <= left < new_id || throw(ArgumentError("invalid left token id in merge $pair"))
        1 <= right < new_id || throw(ArgumentError("invalid right token id in merge $pair"))
        isempty(token_bytes[left]) && throw(ArgumentError(
            "BPE merges cannot consume special token id $left",
        ))
        isempty(token_bytes[right]) && throw(ArgumentError(
            "BPE merges cannot consume special token id $right",
        ))
        token_bytes[new_id] = vcat(token_bytes[left], token_bytes[right])
        merge_ranks[pair] = rank
    end
    return merge_ranks, token_bytes
end

"""Deterministic byte-level BPE tokenizer."""
struct ByteBPETokenizer <: AbstractTokenizer
    normalization::Symbol
    special_tokens::Dict{Symbol,Int}
    special_token_strings::Dict{Symbol,String}
    merges::Vector{Tuple{Int,Int}}
    merge_ranks::Dict{Tuple{Int,Int},Int}
    token_bytes::Vector{Vector{UInt8}}
    trainer_config::NamedTuple
    corpus_fingerprint::String
end

function ByteBPETokenizer(
    normalization::Symbol,
    special_tokens::Dict{Symbol,Int},
    special_token_strings::Dict{Symbol,String},
    merges::Vector{Tuple{Int,Int}};
    trainer_config::NamedTuple=(;
        algorithm=:byte_bpe,
        algorithm_version=1,
        target_vocab_size=BYTE_ALPHABET_SIZE + length(special_tokens) + length(merges),
        min_frequency=1,
        tie_break=:lexicographic_token_id,
    ),
    corpus_fingerprint::AbstractString="",
)
    normalized = _validate_normalization(normalization)
    ranks, bytes = _build_bpe_tables(
        special_tokens,
        special_token_strings,
        merges,
    )
    return ByteBPETokenizer(
        normalized,
        copy(special_tokens),
        copy(special_token_strings),
        copy(merges),
        ranks,
        bytes,
        trainer_config,
        String(corpus_fingerprint),
    )
end

_normalization_mode(tokenizer::ByteBPETokenizer) = tokenizer.normalization
vocab_size(tokenizer::ByteBPETokenizer) = length(tokenizer.token_bytes)
Base.length(tokenizer::ByteBPETokenizer) = vocab_size(tokenizer)
special_token_id(tokenizer::ByteBPETokenizer, name) =
    get(tokenizer.special_tokens, Symbol(name), nothing)

function _corpus_fingerprint(texts::Vector{String}, normalization::Symbol)
    output = IOBuffer()
    println(output, "normalization=", normalization)
    for text in sort(texts)
        bytes = codeunits(text)
        println(output, "bytes=", length(bytes))
        write(output, bytes)
        write(output, UInt8('\n'))
    end
    return bytes2hex(sha256(take!(output)))
end

function _pair_counts(sequences::Vector{Vector{Int}})
    counts = Dict{Tuple{Int,Int},Int}()
    for sequence in sequences
        for index in 1:(length(sequence) - 1)
            pair = (sequence[index], sequence[index + 1])
            counts[pair] = get(counts, pair, 0) + 1
        end
    end
    return counts
end

"""
    fit_byte_bpe(texts; vocab_size=512, min_frequency=2, ...)

Train deterministic byte BPE. Equal-frequency pairs are resolved by ascending token-id
pair, so the same normalized corpus and configuration produce identical merges.
"""
function fit_byte_bpe(
    texts::AbstractVector{<:AbstractString};
    vocab_size::Int=512,
    min_frequency::Int=2,
    normalization::Symbol=:none,
    special_tokens=DEFAULT_BYTE_SPECIAL_TOKENS,
)
    isempty(texts) && throw(ArgumentError("BPE training corpus must not be empty"))
    min_frequency > 0 || throw(ArgumentError("`min_frequency` must be positive"))
    normalized_mode = _validate_normalization(normalization)
    normalized_texts = [normalize_text(text, normalized_mode) for text in texts]
    all(isempty, normalized_texts) && throw(ArgumentError(
        "BPE training corpus must contain at least one byte",
    ))

    special_ids, special_strings = _special_token_tables(special_tokens)
    base_vocabulary = BYTE_ALPHABET_SIZE + length(special_ids)
    vocab_size >= base_vocabulary || throw(ArgumentError(
        "`vocab_size` must be at least $base_vocabulary",
    ))

    sequences = [
        [Int(byte) + 1 for byte in codeunits(text)] for text in normalized_texts
    ]
    merges = Tuple{Int,Int}[]

    while base_vocabulary + length(merges) < vocab_size
        counts = _pair_counts(sequences)
        isempty(counts) && break
        best_frequency = maximum(values(counts))
        best_frequency < min_frequency && break
        candidates = sort!([
            pair for (pair, frequency) in counts if frequency == best_frequency
        ])
        best_pair = first(candidates)
        new_id = base_vocabulary + length(merges) + 1
        sequences = [_merge_pair(sequence, best_pair, new_id) for sequence in sequences]
        push!(merges, best_pair)
    end

    trainer_config = (;
        algorithm=:byte_bpe,
        algorithm_version=1,
        target_vocab_size=vocab_size,
        min_frequency,
        tie_break=:lexicographic_token_id,
    )
    return ByteBPETokenizer(
        normalized_mode,
        special_ids,
        special_strings,
        merges;
        trainer_config,
        corpus_fingerprint=_corpus_fingerprint(normalized_texts, normalized_mode),
    )
end

fit_byte_bpe(text::AbstractString; kwargs...) = fit_byte_bpe([String(text)]; kwargs...)

function encode(
    tokenizer::ByteBPETokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)
    normalized = normalize_text(text, tokenizer.normalization)
    sequence = [Int(byte) + 1 for byte in codeunits(normalized)]
    merge_offset = BYTE_ALPHABET_SIZE + length(tokenizer.special_tokens)
    for (rank, pair) in enumerate(tokenizer.merges)
        sequence = _merge_pair(sequence, pair, merge_offset + rank)
    end
    return _add_boundary_special_tokens(tokenizer, sequence, add_special_tokens)
end

function decode_bytes(
    tokenizer::ByteBPETokenizer,
    ids;
    skip_special_tokens::Bool=false,
)
    output = UInt8[]
    vocabulary_size = vocab_size(tokenizer)
    for raw_id in ids
        raw_id isa Integer || throw(ArgumentError(
            "token id $(repr(raw_id)) is not an integer",
        ))
        id = Int(raw_id)
        1 <= id <= vocabulary_size || throw(ArgumentError(
            "token id $id is outside 1:$vocabulary_size",
        ))
        bytes = tokenizer.token_bytes[id]
        if isempty(bytes)
            skip_special_tokens || _append_special_token_bytes!(output, tokenizer, id)
        else
            append!(output, bytes)
        end
    end
    return output
end

function decode(
    tokenizer::ByteBPETokenizer,
    ids;
    errors::Symbol=:strict,
    skip_special_tokens::Bool=false,
)
    return _decode_utf8(
        decode_bytes(tokenizer, ids; skip_special_tokens);
        errors,
    )
end

function token_byte_length(tokenizer::Tokenizer, id::Integer)
    1 <= id <= vocab_size(tokenizer) || throw(ArgumentError("token id is outside vocabulary"))
    return ncodeunits(string(tokenizer.id_to_token[Int(id)]))
end

function token_byte_length(tokenizer::ByteTokenizer, id::Integer)
    1 <= id <= vocab_size(tokenizer) || throw(ArgumentError("token id is outside vocabulary"))
    return id <= BYTE_ALPHABET_SIZE ? 1 : 0
end

function token_byte_length(tokenizer::ByteBPETokenizer, id::Integer)
    1 <= id <= vocab_size(tokenizer) || throw(ArgumentError("token id is outside vocabulary"))
    return length(tokenizer.token_bytes[Int(id)])
end

function encoded_byte_length(tokenizer::AbstractTokenizer, ids)
    return sum(token_byte_length(tokenizer, id) for id in ids)
end

function tokenizer_statistics(
    tokenizer::AbstractTokenizer,
    texts::AbstractVector{<:AbstractString},
)
    normalized_texts = [normalize_text(text, _normalization_mode(tokenizer)) for text in texts]
    token_count = sum(length(encode(tokenizer, text)) for text in normalized_texts)
    byte_count = sum(ncodeunits, normalized_texts)
    character_count = sum(length, normalized_texts)
    return (;
        documents=length(normalized_texts),
        vocab_size=vocab_size(tokenizer),
        tokens=token_count,
        bytes=byte_count,
        characters=character_count,
        tokens_per_byte=byte_count == 0 ? 0.0 : token_count / byte_count,
        bytes_per_token=token_count == 0 ? 0.0 : byte_count / token_count,
        tokens_per_character=character_count == 0 ? 0.0 : token_count / character_count,
    )
end

tokenizer_statistics(tokenizer::AbstractTokenizer, text::AbstractString) =
    tokenizer_statistics(tokenizer, [String(text)])

function tokenizer_config(tokenizer::Tokenizer)
    return (;
        type=:character,
        id_base=1,
        normalization=:none,
        unk_id=tokenizer.unk_id,
        vocabulary_size=vocab_size(tokenizer),
    )
end

function tokenizer_config(tokenizer::ByteTokenizer)
    return (;
        type=:byte,
        id_base=1,
        normalization=tokenizer.normalization,
        byte_alphabet_size=BYTE_ALPHABET_SIZE,
        special_tokens=_ordered_special_tokens(tokenizer),
        vocabulary_size=vocab_size(tokenizer),
    )
end

function tokenizer_config(tokenizer::ByteBPETokenizer)
    return (;
        type=:byte_bpe,
        id_base=1,
        normalization=tokenizer.normalization,
        byte_alphabet_size=BYTE_ALPHABET_SIZE,
        special_tokens=_ordered_special_tokens(tokenizer),
        vocabulary_size=vocab_size(tokenizer),
        merge_count=length(tokenizer.merges),
        trainer_config=tokenizer.trainer_config,
        corpus_fingerprint=tokenizer.corpus_fingerprint,
    )
end

function _write_special_token_fingerprint!(output::IO, tokenizer)
    for entry in _ordered_special_tokens(tokenizer)
        println(
            output,
            "special=", entry.id, ":", String(entry.name), ":",
            bytes2hex(codeunits(entry.text)),
        )
    end
end

function tokenizer_fingerprint(tokenizer::Tokenizer)
    output = IOBuffer()
    println(output, "schema=", TOKENIZER_ARTIFACT_VERSION)
    println(output, "type=character")
    println(output, "id_base=1")
    println(output, "unk_id=", tokenizer.unk_id === nothing ? 0 : tokenizer.unk_id)
    for (id, token) in enumerate(tokenizer.id_to_token)
        println(output, "token=", id, ":", Int(token))
    end
    return bytes2hex(sha256(take!(output)))
end

function tokenizer_fingerprint(tokenizer::ByteTokenizer)
    output = IOBuffer()
    println(output, "schema=", TOKENIZER_ARTIFACT_VERSION)
    println(output, "type=byte")
    println(output, "id_base=1")
    println(output, "normalization=", tokenizer.normalization)
    _write_special_token_fingerprint!(output, tokenizer)
    return bytes2hex(sha256(take!(output)))
end

function tokenizer_fingerprint(tokenizer::ByteBPETokenizer)
    output = IOBuffer()
    println(output, "schema=", TOKENIZER_ARTIFACT_VERSION)
    println(output, "type=byte_bpe")
    println(output, "id_base=1")
    println(output, "normalization=", tokenizer.normalization)
    _write_special_token_fingerprint!(output, tokenizer)
    println(output, "corpus=", tokenizer.corpus_fingerprint)
    for (name, value) in pairs(tokenizer.trainer_config)
        println(output, "trainer=", name, ":", value)
    end
    for (rank, pair) in enumerate(tokenizer.merges)
        println(output, "merge=", rank, ":", pair[1], ":", pair[2])
    end
    return bytes2hex(sha256(take!(output)))
end

function _special_tokens_artifact(tokenizer)
    return [
        Dict(
            "name" => String(entry.name),
            "id" => entry.id,
            "text_hex" => bytes2hex(codeunits(entry.text)),
        ) for entry in _ordered_special_tokens(tokenizer)
    ]
end

function _tokenizer_artifact(tokenizer::Tokenizer)
    return Dict{String,Any}(
        "schema_version" => TOKENIZER_ARTIFACT_VERSION,
        "type" => "character",
        "id_base" => 1,
        "normalization" => "none",
        "unk_id" => tokenizer.unk_id === nothing ? 0 : tokenizer.unk_id,
        "vocabulary_codepoints" => Int.(tokenizer.id_to_token),
        "fingerprint" => tokenizer_fingerprint(tokenizer),
    )
end

function _tokenizer_artifact(tokenizer::ByteTokenizer)
    return Dict{String,Any}(
        "schema_version" => TOKENIZER_ARTIFACT_VERSION,
        "type" => "byte",
        "id_base" => 1,
        "normalization" => String(tokenizer.normalization),
        "byte_alphabet_size" => BYTE_ALPHABET_SIZE,
        "special_tokens" => _special_tokens_artifact(tokenizer),
        "fingerprint" => tokenizer_fingerprint(tokenizer),
    )
end

function _trainer_artifact(config::NamedTuple)
    return Dict(String(name) => (value isa Symbol ? String(value) : value) for (name, value) in pairs(config))
end

function _tokenizer_artifact(tokenizer::ByteBPETokenizer)
    return Dict{String,Any}(
        "schema_version" => TOKENIZER_ARTIFACT_VERSION,
        "type" => "byte_bpe",
        "id_base" => 1,
        "normalization" => String(tokenizer.normalization),
        "byte_alphabet_size" => BYTE_ALPHABET_SIZE,
        "special_tokens" => _special_tokens_artifact(tokenizer),
        "merges" => [[pair[1], pair[2]] for pair in tokenizer.merges],
        "trainer" => _trainer_artifact(tokenizer.trainer_config),
        "corpus_fingerprint" => tokenizer.corpus_fingerprint,
        "fingerprint" => tokenizer_fingerprint(tokenizer),
    )
end

"""Save a stable, versioned TOML tokenizer artifact atomically."""
function save_tokenizer(path::AbstractString, tokenizer::AbstractTokenizer)
    isempty(path) && throw(ArgumentError("tokenizer path must not be empty"))
    absolute_path = abspath(path)
    mkpath(dirname(absolute_path))
    temporary_path = tempname(dirname(absolute_path))
    try
        open(temporary_path, "w") do io
            TOML.print(io, _tokenizer_artifact(tokenizer); sorted=true)
        end
        mv(temporary_path, absolute_path; force=true)
    finally
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return absolute_path
end

function _special_tokens_from_artifact(entries)
    ids = Dict{Symbol,Int}()
    texts = Dict{Symbol,String}()
    for entry in entries
        name = Symbol(entry["name"])
        ids[name] = Int(entry["id"])
        texts[name] = String(hex2bytes(entry["text_hex"]))
    end
    _validate_special_token_tables(ids, texts)
    return ids, texts
end

function _trainer_from_artifact(payload)
    return (;
        algorithm=Symbol(payload["algorithm"]),
        algorithm_version=Int(payload["algorithm_version"]),
        target_vocab_size=Int(payload["target_vocab_size"]),
        min_frequency=Int(payload["min_frequency"]),
        tie_break=Symbol(payload["tie_break"]),
    )
end

"""Load and verify a tokenizer artifact."""
function load_tokenizer(path::AbstractString)
    isfile(path) || throw(ArgumentError("tokenizer artifact does not exist: $path"))
    payload = TOML.parsefile(path)
    schema_version = get(payload, "schema_version", 0)
    schema_version == TOKENIZER_ARTIFACT_VERSION || throw(ArgumentError(
        "unsupported tokenizer artifact schema version $schema_version",
    ))
    get(payload, "id_base", 0) == 1 || throw(ArgumentError(
        "tokenizer artifact must use 1-based token ids",
    ))

    tokenizer_type = get(payload, "type", "")
    tokenizer = if tokenizer_type == "character"
        tokens = Char.(Int.(payload["vocabulary_codepoints"]))
        token_to_id = Dict(token => id for (id, token) in enumerate(tokens))
        raw_unk_id = Int(get(payload, "unk_id", 0))
        Tokenizer(token_to_id, tokens, raw_unk_id == 0 ? nothing : raw_unk_id)
    elseif tokenizer_type == "byte"
        ids, texts = _special_tokens_from_artifact(get(payload, "special_tokens", Any[]))
        ByteTokenizer(Symbol(payload["normalization"]), ids, texts)
    elseif tokenizer_type == "byte_bpe"
        ids, texts = _special_tokens_from_artifact(get(payload, "special_tokens", Any[]))
        merges = Tuple{Int,Int}[
            (Int(pair[1]), Int(pair[2])) for pair in payload["merges"]
        ]
        ByteBPETokenizer(
            Symbol(payload["normalization"]),
            ids,
            texts,
            merges;
            trainer_config=_trainer_from_artifact(payload["trainer"]),
            corpus_fingerprint=String(payload["corpus_fingerprint"]),
        )
    elseif tokenizer_type == "hf_qwen3_bpe"
        _hf_qwen3_tokenizer_from_json(
            String(hex2bytes(payload["tokenizer_json_hex"])),
            String(hex2bytes(payload["tokenizer_config_json_hex"])),
            String(hex2bytes(payload["generation_config_json_hex"]));
            revision=String(get(payload, "revision", "")),
        )
    else
        throw(ArgumentError("unsupported tokenizer artifact type $(repr(tokenizer_type))"))
    end

    expected = String(get(payload, "fingerprint", ""))
    actual = tokenizer_fingerprint(tokenizer)
    expected == actual || throw(ArgumentError(
        "tokenizer artifact fingerprint mismatch: expected $expected, computed $actual",
    ))
    return tokenizer
end

_model_tokenizer_vocab_compatible(model_vocab_size::Int, tokenizer::AbstractTokenizer) =
    model_vocab_size == vocab_size(tokenizer)
