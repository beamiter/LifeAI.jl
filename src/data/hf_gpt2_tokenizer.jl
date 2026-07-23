using JSON3

const _GPT2_TOKENIZER_REGEX =
    raw"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

"""Validated inference settings from GPT-2 `generation_config.json`."""
struct HFGPT2GenerationConfig
    bos_id::Int
    eos_ids::Vector{Int}
    pad_id::Int
    do_sample::Bool
    temperature::Float32
    top_k::Int
    top_p::Float32
    transformers_version::String
end

"""Strict imported tokenizer for the original GPT-2 byte-level BPE."""
struct HFGPT2Tokenizer <: AbstractTokenizer
    vocabulary::Dict{String,Int}
    id_to_token::Vector{String}
    token_bytes::Vector{Vector{UInt8}}
    merge_ranks::Dict{Tuple{String,String},Int}
    pretokenizer_regex::Regex
    endoftext_id::Int
    eos_ids::Vector{Int}
    special_ids::Set{Int}
    generation::HFGPT2GenerationConfig
    model_max_length::Int
    revision::String
    raw_tokenizer_json::String
    raw_tokenizer_config_json::String
    raw_generation_config_json::String
    tokenizer_sha256::String
    tokenizer_config_sha256::String
    generation_config_sha256::String
end

function _gpt2_exact_keys(object, expected, label)
    actual = Set(String.(collect(keys(object))))
    expected_set = Set(expected)
    actual == expected_set || throw(ArgumentError(
        "$label fields differ; missing=$(sort!(collect(setdiff(expected_set, actual)))) " *
        "unexpected=$(sort!(collect(setdiff(actual, expected_set))))",
    ))
    return nothing
end

function _gpt2_exact_byte_level(object, prefix::Bool, trim::Bool, label)
    object isa JSON3.Object || throw(ArgumentError("$label must be an object"))
    _gpt2_exact_keys(object, ["type", "add_prefix_space", "trim_offsets"], label)
    _hf_exact_value(object, "type", "ByteLevel", label)
    _hf_exact_bool(object, "add_prefix_space", prefix, label)
    _hf_exact_bool(object, "trim_offsets", trim, label)
    return nothing
end

function _gpt2_parse_model(model, char_to_byte)
    model isa JSON3.Object || throw(ArgumentError("tokenizer model must be an object"))
    _gpt2_exact_keys(
        model,
        [
            "dropout",
            "unk_token",
            "continuing_subword_prefix",
            "end_of_word_suffix",
            "fuse_unk",
            "vocab",
            "merges",
        ],
        "tokenizer.json model",
    )
    _hf_exact_value(model, "dropout", nothing, "tokenizer.json model")
    _hf_exact_value(model, "unk_token", nothing, "tokenizer.json model")
    _hf_exact_value(model, "continuing_subword_prefix", "", "tokenizer.json model")
    _hf_exact_value(model, "end_of_word_suffix", "", "tokenizer.json model")
    _hf_exact_bool(model, "fuse_unk", false, "tokenizer.json model")

    raw_vocabulary = _hf_required(model, "vocab", "tokenizer.json model")
    raw_vocabulary isa JSON3.Object || throw(ArgumentError("BPE vocab must be an object"))
    vocabulary = Dict{String,Int}()
    for raw_token in keys(raw_vocabulary)
        token = String(raw_token)
        id0 = raw_vocabulary[raw_token]
        id0 isa Integer && id0 >= 0 || throw(ArgumentError(
            "GPT-2 vocabulary ids must be non-negative integers",
        ))
        haskey(vocabulary, token) && throw(ArgumentError("duplicate GPT-2 vocabulary token"))
        vocabulary[token] = Int(id0) + 1
    end
    ids = sort!(collect(values(vocabulary)))
    ids == collect(1:length(ids)) || throw(ArgumentError(
        "GPT-2 vocabulary ids must be contiguous from zero",
    ))

    raw_merges = _hf_required(model, "merges", "tokenizer.json model")
    raw_merges isa JSON3.Array || throw(ArgumentError("BPE merges must be an array"))
    merge_ranks = Dict{Tuple{String,String},Int}()
    for (rank, raw_merge) in enumerate(raw_merges)
        raw_merge isa AbstractString || throw(ArgumentError(
            "GPT-2 BPE merge $rank must be a string",
        ))
        pair_values = split(String(raw_merge), ' ')
        length(pair_values) == 2 || throw(ArgumentError(
            "GPT-2 BPE merge $rank must contain exactly two symbols",
        ))
        pair = (pair_values[1], pair_values[2])
        haskey(merge_ranks, pair) && throw(ArgumentError("duplicate GPT-2 BPE merge"))
        haskey(vocabulary, pair[1]) || throw(ArgumentError("unknown left BPE symbol"))
        haskey(vocabulary, pair[2]) || throw(ArgumentError("unknown right BPE symbol"))
        haskey(vocabulary, pair[1] * pair[2]) ||
            throw(ArgumentError("merged GPT-2 BPE symbol is absent from vocabulary"))
        merge_ranks[pair] = rank
    end

    endoftext = get(vocabulary, "<|endoftext|>", nothing)
    endoftext === nothing && throw(ArgumentError("GPT-2 vocabulary lacks `<|endoftext|>`"))
    for (token, id) in vocabulary
        id == endoftext && continue
        all(character -> haskey(char_to_byte, character), token) || throw(ArgumentError(
            "GPT-2 BPE token $(repr(token)) contains a symbol outside the byte alphabet",
        ))
    end
    return vocabulary, merge_ranks, endoftext
end

function _gpt2_validate_added_tokens(tokenizer_json, endoftext_id::Int)
    added = _hf_required(tokenizer_json, "added_tokens", "tokenizer.json")
    added isa JSON3.Array && length(added) == 1 || throw(ArgumentError(
        "GPT-2 tokenizer must declare exactly one added token",
    ))
    token = only(added)
    token isa JSON3.Object || throw(ArgumentError("GPT-2 added token must be an object"))
    _gpt2_exact_keys(
        token,
        ["id", "special", "content", "single_word", "lstrip", "rstrip", "normalized"],
        "GPT-2 added token",
    )
    _hf_exact_value(token, "id", endoftext_id - 1, "GPT-2 added token")
    _hf_exact_value(token, "content", "<|endoftext|>", "GPT-2 added token")
    _hf_exact_bool(token, "special", true, "GPT-2 added token")
    _hf_exact_bool(token, "single_word", false, "GPT-2 added token")
    _hf_exact_bool(token, "lstrip", false, "GPT-2 added token")
    _hf_exact_bool(token, "rstrip", false, "GPT-2 added token")
    _hf_exact_bool(token, "normalized", true, "GPT-2 added token")
    return nothing
end

function _gpt2_generation_config(raw, vocabulary_size::Int, endoftext_id::Int)
    config = _hf_json(raw, "generation_config.json")
    _gpt2_exact_keys(
        config,
        ["bos_token_id", "eos_token_id", "transformers_version", "_from_model_config"],
        "generation_config.json",
    )
    _hf_exact_bool(config, "_from_model_config", true, "generation_config.json")
    bos = _hf_generation_id(config["bos_token_id"], vocabulary_size, "bos_token_id")
    eos = _hf_generation_id(config["eos_token_id"], vocabulary_size, "eos_token_id")
    bos == endoftext_id && eos == endoftext_id || throw(ArgumentError(
        "GPT-2 BOS/EOS must both name `<|endoftext|>`",
    ))
    version = config["transformers_version"]
    version isa AbstractString && !isempty(version) || throw(ArgumentError(
        "transformers_version must be a non-empty string",
    ))
    return HFGPT2GenerationConfig(
        bos,
        [eos],
        eos,
        false,
        1.0f0,
        50,
        1.0f0,
        String(version),
    )
end

function _gpt2_tokenizer_from_json(
    raw_tokenizer::AbstractString,
    raw_config::AbstractString,
    raw_generation::AbstractString;
    revision::AbstractString="",
)
    tokenizer_json = _hf_json(raw_tokenizer, "tokenizer.json")
    _gpt2_exact_keys(
        tokenizer_json,
        [
            "version",
            "truncation",
            "padding",
            "added_tokens",
            "normalizer",
            "pre_tokenizer",
            "post_processor",
            "decoder",
            "model",
        ],
        "tokenizer.json",
    )
    _hf_exact_value(tokenizer_json, "version", "1.0", "tokenizer.json")
    _hf_exact_value(tokenizer_json, "truncation", nothing, "tokenizer.json")
    _hf_exact_value(tokenizer_json, "padding", nothing, "tokenizer.json")
    _hf_exact_value(tokenizer_json, "normalizer", nothing, "tokenizer.json")
    _gpt2_exact_byte_level(tokenizer_json["pre_tokenizer"], false, true, "pre_tokenizer")
    _gpt2_exact_byte_level(tokenizer_json["post_processor"], true, false, "post_processor")
    _gpt2_exact_byte_level(tokenizer_json["decoder"], true, true, "decoder")

    alphabet = hf_byte_unicode_alphabet()
    vocabulary, merge_ranks, endoftext_id = _gpt2_parse_model(
        tokenizer_json["model"],
        alphabet.char_to_byte,
    )
    _gpt2_validate_added_tokens(tokenizer_json, endoftext_id)

    config = _hf_json(raw_config, "tokenizer_config.json")
    _gpt2_exact_keys(config, ["model_max_length"], "tokenizer_config.json")
    model_max_length = config["model_max_length"]
    model_max_length isa Integer && model_max_length > 0 || throw(ArgumentError(
        "model_max_length must be a positive integer",
    ))
    generation = _gpt2_generation_config(
        raw_generation,
        length(vocabulary),
        endoftext_id,
    )

    id_to_token = Vector{String}(undef, length(vocabulary))
    token_bytes = [UInt8[] for _ in 1:length(vocabulary)]
    for (token, id) in vocabulary
        id_to_token[id] = token
        token_bytes[id] = id == endoftext_id ?
            Vector{UInt8}(codeunits(token)) :
            UInt8[alphabet.char_to_byte[character] for character in token]
    end
    return HFGPT2Tokenizer(
        vocabulary,
        id_to_token,
        token_bytes,
        merge_ranks,
        Regex(_GPT2_TOKENIZER_REGEX),
        endoftext_id,
        [endoftext_id],
        Set([endoftext_id]),
        generation,
        Int(model_max_length),
        String(revision),
        String(raw_tokenizer),
        String(raw_config),
        String(raw_generation),
        _sha256_hex(raw_tokenizer),
        _sha256_hex(raw_config),
        _sha256_hex(raw_generation),
    )
end

"""
    load_hf_gpt2_tokenizer(model_dir; revision="")

Strictly load the frozen GPT-2 tokenizer files from a local model directory.
No network access or fallback tokenizer implementation is used.
"""
function load_hf_gpt2_tokenizer(
    model_dir::AbstractString;
    revision::AbstractString="",
)
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    paths = (
        tokenizer=joinpath(model_dir, "tokenizer.json"),
        config=joinpath(model_dir, "tokenizer_config.json"),
        generation=joinpath(model_dir, "generation_config.json"),
    )
    for path in values(paths)
        isfile(path) || throw(ArgumentError("required GPT-2 tokenizer file is absent: $path"))
    end
    return _gpt2_tokenizer_from_json(
        read(paths.tokenizer, String),
        read(paths.config, String),
        read(paths.generation, String);
        revision,
    )
end

_normalization_mode(::HFGPT2Tokenizer) = :none
vocab_size(tokenizer::HFGPT2Tokenizer) = length(tokenizer.id_to_token)
Base.length(tokenizer::HFGPT2Tokenizer) = vocab_size(tokenizer)

function hf_generation_config(tokenizer::HFGPT2Tokenizer)
    generation = tokenizer.generation
    return (;
        bos_id=generation.bos_id,
        eos_ids=copy(generation.eos_ids),
        pad_id=generation.pad_id,
        do_sample=generation.do_sample,
        temperature=generation.temperature,
        top_k=generation.top_k,
        top_p=generation.top_p,
        transformers_version=generation.transformers_version,
    )
end

function special_token_id(tokenizer::HFGPT2Tokenizer, name)
    Symbol(name) in (:bos, :eos, :unk) && return tokenizer.endoftext_id
    Symbol(name) === :pad && return nothing
    return nothing
end

function _gpt2_plain_segments(tokenizer::HFGPT2Tokenizer, input::AbstractString)
    text = String(input)
    isempty(text) && return Tuple{Bool,String}[]
    output = Tuple{Bool,String}[]
    cursor = firstindex(text)
    terminal = ncodeunits(text) + 1
    special = "<|endoftext|>"
    while cursor < terminal
        range = findnext(special, text, cursor)
        if range === nothing
            push!(output, (false, String(SubString(text, cursor))))
            break
        end
        first(range) > cursor && push!(output, (
            false,
            String(SubString(text, cursor, prevind(text, first(range)))),
        ))
        push!(output, (true, special))
        cursor = nextind(text, last(range))
    end
    return output
end

function _gpt2_pretoken_records(tokenizer::HFGPT2Tokenizer, text::String)
    isempty(text) && return NamedTuple[]
    matches = collect(eachmatch(tokenizer.pretokenizer_regex, text))
    join((match.match for match in matches)) == text || throw(ArgumentError(
        "GPT-2 regex did not cover the complete input",
    ))
    return [begin
        character_start = match.offset == firstindex(text) ? 0 :
            length(SubString(text, firstindex(text), prevind(text, match.offset)))
        (
            text=String(match.match),
            symbols=_hf_byte_symbols(String(match.match)),
            character_start,
            character_stop=character_start + length(match.match),
            byte_start=match.offset - 1,
            byte_stop=match.offset - 1 + ncodeunits(match.match),
        )
    end for match in matches]
end

"""Expose GPT-2 regex pieces and byte-level symbols for parity tests."""
function hf_gpt2_pretokenize(tokenizer::HFGPT2Tokenizer, text::AbstractString)
    return Tuple(_gpt2_pretoken_records(tokenizer, String(text)))
end

function _gpt2_bpe_tokens(tokenizer::HFGPT2Tokenizer, piece::String)
    symbols = string.(collect(_hf_byte_symbols(piece)))
    length(symbols) <= 1 && return symbols
    while true
        best_rank = typemax(Int)
        best_pair = nothing
        for index in 1:(length(symbols) - 1)
            pair = (symbols[index], symbols[index + 1])
            rank = get(tokenizer.merge_ranks, pair, typemax(Int))
            if rank < best_rank
                best_rank = rank
                best_pair = pair
            end
        end
        best_pair === nothing && break
        output = String[]
        index = 1
        while index <= length(symbols)
            if index < length(symbols) &&
               symbols[index] == best_pair[1] &&
               symbols[index + 1] == best_pair[2]
                push!(output, best_pair[1] * best_pair[2])
                index += 2
            else
                push!(output, symbols[index])
                index += 1
            end
        end
        symbols = output
    end
    return symbols
end

function encode(
    tokenizer::HFGPT2Tokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)
    add_special_tokens isa Bool || throw(ArgumentError("add_special_tokens must be Bool"))
    ids = Int[]
    for (is_special, segment) in _gpt2_plain_segments(tokenizer, text)
        if is_special
            push!(ids, tokenizer.endoftext_id)
            continue
        end
        for record in _gpt2_pretoken_records(tokenizer, segment)
            for token in _gpt2_bpe_tokens(tokenizer, record.text)
                id = get(tokenizer.vocabulary, token, nothing)
                id === nothing && throw(ArgumentError(
                    "GPT-2 BPE output token $(repr(token)) is absent from vocabulary",
                ))
                push!(ids, id)
            end
        end
    end
    return ids
end

function decode_bytes(
    tokenizer::HFGPT2Tokenizer,
    ids;
    skip_special_tokens::Bool=false,
)
    output = UInt8[]
    for raw_id in ids
        raw_id isa Integer || throw(ArgumentError("token id must be an integer"))
        id = Int(raw_id)
        1 <= id <= vocab_size(tokenizer) || throw(ArgumentError(
            "token id $id is outside the GPT-2 vocabulary",
        ))
        skip_special_tokens && id == tokenizer.endoftext_id && continue
        append!(output, tokenizer.token_bytes[id])
    end
    return output
end

function decode(
    tokenizer::HFGPT2Tokenizer,
    ids;
    errors::Symbol=:strict,
    skip_special_tokens::Bool=false,
)
    return _decode_utf8(decode_bytes(tokenizer, ids; skip_special_tokens); errors)
end

function token_byte_length(tokenizer::HFGPT2Tokenizer, id::Integer)
    1 <= id <= vocab_size(tokenizer) || throw(ArgumentError("token id is outside vocabulary"))
    return Int(id) == tokenizer.endoftext_id ? 0 : length(tokenizer.token_bytes[Int(id)])
end

function tokenizer_config(tokenizer::HFGPT2Tokenizer)
    return (;
        type=:hf_gpt2_bpe,
        id_base=1,
        normalization=:none,
        vocabulary_size=vocab_size(tokenizer),
        merge_count=length(tokenizer.merge_ranks),
        eos_ids=copy(tokenizer.eos_ids),
        model_max_length=tokenizer.model_max_length,
        revision=tokenizer.revision,
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        tokenizer_config_sha256=tokenizer.tokenizer_config_sha256,
        generation_config_sha256=tokenizer.generation_config_sha256,
    )
end

function tokenizer_fingerprint(tokenizer::HFGPT2Tokenizer)
    output = IOBuffer()
    println(output, "schema=", TOKENIZER_ARTIFACT_VERSION)
    println(output, "type=hf_gpt2_bpe")
    println(output, "id_base=1")
    println(output, "revision=", tokenizer.revision)
    println(output, "tokenizer=", tokenizer.tokenizer_sha256)
    println(output, "tokenizer_config=", tokenizer.tokenizer_config_sha256)
    println(output, "generation_config=", tokenizer.generation_config_sha256)
    return bytes2hex(sha256(take!(output)))
end

function _tokenizer_artifact(tokenizer::HFGPT2Tokenizer)
    return Dict{String,Any}(
        "schema_version" => TOKENIZER_ARTIFACT_VERSION,
        "type" => "hf_gpt2_bpe",
        "id_base" => 1,
        "revision" => tokenizer.revision,
        "tokenizer_json_hex" => bytes2hex(codeunits(tokenizer.raw_tokenizer_json)),
        "tokenizer_config_json_hex" =>
            bytes2hex(codeunits(tokenizer.raw_tokenizer_config_json)),
        "generation_config_json_hex" =>
            bytes2hex(codeunits(tokenizer.raw_generation_config_json)),
        "fingerprint" => tokenizer_fingerprint(tokenizer),
    )
end

_model_tokenizer_vocab_compatible(model_vocab_size::Int, tokenizer::HFGPT2Tokenizer) =
    model_vocab_size == vocab_size(tokenizer)
