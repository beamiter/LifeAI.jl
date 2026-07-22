using JSON3

const _QWEN3_TOKENIZER_REGEX = raw"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
const _QWEN3_CHAT_TEMPLATE_SHA256 =
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
const _QWEN3_TEST_CHAT_TEMPLATE_SHA256 =
    "05b2396f106741b64a90f891dfe124f20bf76150f2d8523fa52d4967a32e1893"

struct HFAddedToken
    id::Int
    content::String
    single_word::Bool
    lstrip::Bool
    rstrip::Bool
    normalized::Bool
    special::Bool
end

"""Validated sampling and stop-token settings from Qwen3 `generation_config.json`."""
struct HFQwen3GenerationConfig
    bos_id::Int
    eos_ids::Vector{Int}
    pad_id::Int
    do_sample::Bool
    temperature::Float32
    top_k::Int
    top_p::Float32
    transformers_version::String
end

"""A strict, imported HuggingFace Qwen3 byte-level BPE tokenizer."""
struct HFQwen3Tokenizer <: AbstractTokenizer
    vocabulary::Dict{String,Int}
    id_to_token::Vector{String}
    token_bytes::Vector{Vector{UInt8}}
    model_vocabulary_size::Int
    merge_ranks::Dict{Tuple{String,String},Int}
    pretokenizer_pattern::String
    pretokenizer_regex::Regex
    added_tokens::Vector{HFAddedToken}
    added_by_content::Dict{String,HFAddedToken}
    special_ids::Set{Int}
    bos_id::Union{Nothing,Int}
    eos_id::Union{Nothing,Int}
    eos_ids::Vector{Int}
    pad_id::Union{Nothing,Int}
    generation::HFQwen3GenerationConfig
    model_max_length::Int
    chat_template::String
    revision::String
    raw_tokenizer_json::String
    raw_tokenizer_config_json::String
    raw_generation_config_json::String
    tokenizer_sha256::String
    tokenizer_config_sha256::String
    generation_config_sha256::String
end

function _hf_json(raw::AbstractString, label::AbstractString)
    value = try
        JSON3.read(raw)
    catch err
        throw(ArgumentError("invalid JSON in $label: $(sprint(showerror, err))"))
    end
    value isa JSON3.Object || throw(ArgumentError("$label root must be an object"))
    return value
end

function _hf_required(object, name::AbstractString, label::AbstractString)
    haskey(object, name) || throw(ArgumentError("missing `$name` in $label"))
    return object[name]
end

function _hf_exact_bool(object, name::AbstractString, expected::Bool, label::AbstractString)
    value = _hf_required(object, name, label)
    value isa Bool || throw(ArgumentError("`$name` must be boolean in $label"))
    value == expected || throw(ArgumentError(
        "unsupported `$name=$(repr(value))` in $label; expected $expected",
    ))
    return value
end


function _hf_required_bool(object, name::AbstractString, label::AbstractString)
    value = _hf_required(object, name, label)
    value isa Bool || throw(ArgumentError("`$name` must be boolean in $label"))
    return value
end

function _hf_exact_value(object, name::AbstractString, expected, label::AbstractString)
    value = _hf_required(object, name, label)
    value == expected || throw(ArgumentError(
        "unsupported `$name=$(repr(value))` in $label; expected $(repr(expected))",
    ))
    return value
end

"""
    hf_byte_unicode_alphabet()

Return the reversible GPT-2/Qwen mapping between all 256 byte values and the
Unicode symbols stored in the byte-level BPE vocabulary.
"""
function hf_byte_unicode_alphabet()
    byte_values = vcat(
        collect(Int('!'):Int('~')),
        collect(Int('¡'):Int('¬')),
        collect(Int('®'):Int('ÿ')),
    )
    codepoints = copy(byte_values)
    included = Set(byte_values)
    extra = 0
    for byte in 0:255
        if !(byte in included)
            push!(byte_values, byte)
            push!(codepoints, 256 + extra)
            extra += 1
        end
    end
    byte_to_char = Dict(UInt8(byte) => Char(codepoint) for (byte, codepoint) in zip(byte_values, codepoints))
    char_to_byte = Dict(character => byte for (byte, character) in byte_to_char)
    length(byte_to_char) == 256 || error("internal byte-unicode alphabet is incomplete")
    length(char_to_byte) == 256 || error("internal byte-unicode alphabet is not one-to-one")
    return (; byte_to_char, char_to_byte)
end

function _hf_byte_flags(object, label::AbstractString)
    _hf_exact_value(object, "type", "ByteLevel", label)
    _hf_exact_bool(object, "add_prefix_space", false, label)
    _hf_exact_bool(object, "trim_offsets", false, label)
    _hf_exact_bool(object, "use_regex", false, label)
    return nothing
end

function _hf_validate_pipeline(tokenizer_json)
    _hf_exact_value(tokenizer_json, "version", "1.0", "tokenizer.json")
    _hf_exact_value(tokenizer_json, "truncation", nothing, "tokenizer.json")
    _hf_exact_value(tokenizer_json, "padding", nothing, "tokenizer.json")

    normalizer = _hf_required(tokenizer_json, "normalizer", "tokenizer.json")
    normalizer isa JSON3.Object || throw(ArgumentError("normalizer must be an object"))
    _hf_exact_value(normalizer, "type", "NFC", "tokenizer.json normalizer")

    pretokenizer = _hf_required(tokenizer_json, "pre_tokenizer", "tokenizer.json")
    pretokenizer isa JSON3.Object || throw(ArgumentError("pre_tokenizer must be an object"))
    _hf_exact_value(pretokenizer, "type", "Sequence", "tokenizer.json pre_tokenizer")
    components = _hf_required(pretokenizer, "pretokenizers", "tokenizer.json pre_tokenizer")
    components isa JSON3.Array && length(components) == 2 || throw(ArgumentError(
        "Qwen3 pre_tokenizer must contain Split followed by ByteLevel",
    ))

    split_component = components[1]
    split_component isa JSON3.Object || throw(ArgumentError("Split pre-tokenizer must be an object"))
    _hf_exact_value(split_component, "type", "Split", "tokenizer.json Split")
    _hf_exact_value(split_component, "behavior", "Isolated", "tokenizer.json Split")
    _hf_exact_bool(split_component, "invert", false, "tokenizer.json Split")
    pattern_object = _hf_required(split_component, "pattern", "tokenizer.json Split")
    pattern_object isa JSON3.Object || throw(ArgumentError("Split pattern must be an object"))
    pattern = _hf_required(pattern_object, "Regex", "tokenizer.json Split pattern")
    pattern isa AbstractString || throw(ArgumentError("Split Regex must be a string"))
    String(pattern) == _QWEN3_TOKENIZER_REGEX || throw(ArgumentError(
        "unsupported Qwen3 pre-tokenizer regex",
    ))

    byte_component = components[2]
    byte_component isa JSON3.Object || throw(ArgumentError("ByteLevel pre-tokenizer must be an object"))
    _hf_byte_flags(byte_component, "tokenizer.json ByteLevel pre-tokenizer")

    post_processor = _hf_required(tokenizer_json, "post_processor", "tokenizer.json")
    post_processor isa JSON3.Object || throw(ArgumentError("post_processor must be an object"))
    _hf_byte_flags(post_processor, "tokenizer.json post_processor")

    decoder = _hf_required(tokenizer_json, "decoder", "tokenizer.json")
    decoder isa JSON3.Object || throw(ArgumentError("decoder must be an object"))
    _hf_byte_flags(decoder, "tokenizer.json decoder")
    return String(pattern)
end

function _hf_parse_model(tokenizer_json, char_to_byte)
    model = _hf_required(tokenizer_json, "model", "tokenizer.json")
    model isa JSON3.Object || throw(ArgumentError("tokenizer model must be an object"))
    _hf_exact_value(model, "type", "BPE", "tokenizer.json model")
    _hf_exact_value(model, "dropout", nothing, "tokenizer.json model")
    _hf_exact_value(model, "unk_token", nothing, "tokenizer.json model")
    _hf_exact_value(model, "continuing_subword_prefix", "", "tokenizer.json model")
    _hf_exact_value(model, "end_of_word_suffix", "", "tokenizer.json model")
    _hf_exact_bool(model, "fuse_unk", false, "tokenizer.json model")
    _hf_exact_bool(model, "byte_fallback", false, "tokenizer.json model")
    _hf_exact_bool(model, "ignore_merges", false, "tokenizer.json model")

    raw_vocabulary = _hf_required(model, "vocab", "tokenizer.json model")
    raw_vocabulary isa JSON3.Object || throw(ArgumentError("BPE vocab must be an object"))
    vocabulary = Dict{String,Int}()
    seen_ids = Set{Int}()
    for raw_token in keys(raw_vocabulary)
        token = String(raw_token)
        isempty(token) && throw(ArgumentError("BPE token must not be empty"))
        raw_id = raw_vocabulary[raw_token]
        raw_id isa Integer || throw(ArgumentError("BPE token id must be an integer"))
        id = Int(raw_id)
        id >= 0 || throw(ArgumentError("BPE token ids must be non-negative"))
        id in seen_ids && throw(ArgumentError("duplicate BPE token id $id"))
        vocabulary[token] = id + 1
        push!(seen_ids, id)
        all(character -> haskey(char_to_byte, character), token) || throw(ArgumentError(
            "BPE token $(repr(token)) contains a symbol outside the byte alphabet",
        ))
    end
    model_vocabulary_size = length(vocabulary)
    seen_ids == Set(0:(model_vocabulary_size - 1)) || throw(ArgumentError(
        "BPE vocabulary ids must be contiguous from zero",
    ))

    raw_merges = _hf_required(model, "merges", "tokenizer.json model")
    raw_merges isa JSON3.Array || throw(ArgumentError("BPE merges must be an array"))
    merge_ranks = Dict{Tuple{String,String},Int}()
    for (rank, raw_pair) in enumerate(raw_merges)
        raw_pair isa JSON3.Array && length(raw_pair) == 2 || throw(ArgumentError(
            "BPE merge $rank must be a two-element array",
        ))
        raw_pair[1] isa AbstractString && raw_pair[2] isa AbstractString || throw(ArgumentError(
            "BPE merge $rank must contain strings",
        ))
        pair = (String(raw_pair[1]), String(raw_pair[2]))
        haskey(merge_ranks, pair) && throw(ArgumentError("duplicate BPE merge pair $pair"))
        haskey(vocabulary, pair[1]) || throw(ArgumentError("unknown left token in BPE merge $rank"))
        haskey(vocabulary, pair[2]) || throw(ArgumentError("unknown right token in BPE merge $rank"))
        haskey(vocabulary, pair[1] * pair[2]) || throw(ArgumentError(
            "BPE merge $rank has no concatenated vocabulary token",
        ))
        merge_ranks[pair] = rank
    end
    return vocabulary, model_vocabulary_size, merge_ranks
end

function _hf_parse_added_tokens(tokenizer_json, model_vocabulary_size::Int, vocabulary)
    raw_added = _hf_required(tokenizer_json, "added_tokens", "tokenizer.json")
    raw_added isa JSON3.Array || throw(ArgumentError("added_tokens must be an array"))
    added_tokens = HFAddedToken[]
    seen_contents = Set{String}()
    for (offset, raw_token) in enumerate(raw_added)
        raw_token isa JSON3.Object || throw(ArgumentError("added token must be an object"))
        raw_id = _hf_required(raw_token, "id", "tokenizer.json added token")
        raw_id isa Integer || throw(ArgumentError("added token id must be an integer"))
        expected_id = model_vocabulary_size + offset - 1
        Int(raw_id) == expected_id || throw(ArgumentError(
            "added token ids must be contiguous after the BPE vocabulary; expected $expected_id",
        ))
        content = _hf_required(raw_token, "content", "tokenizer.json added token")
        content isa AbstractString || throw(ArgumentError("added token content must be a string"))
        content_string = String(content)
        isempty(content_string) && throw(ArgumentError("added token content must not be empty"))
        content_string in seen_contents && throw(ArgumentError("duplicate added token content"))
        haskey(vocabulary, content_string) && throw(ArgumentError(
            "added token duplicates a BPE vocabulary token",
        ))
        token = HFAddedToken(
            expected_id + 1,
            content_string,
            _hf_required_bool(raw_token, "single_word", "tokenizer.json added token"),
            _hf_required_bool(raw_token, "lstrip", "tokenizer.json added token"),
            _hf_required_bool(raw_token, "rstrip", "tokenizer.json added token"),
            _hf_required_bool(raw_token, "normalized", "tokenizer.json added token"),
            _hf_required_bool(raw_token, "special", "tokenizer.json added token"),
        )
        !token.single_word && !token.lstrip && !token.rstrip && !token.normalized ||
            throw(ArgumentError(
                "Week 08 only supports Qwen3 added tokens with single_word/lstrip/rstrip/normalized=false",
            ))
        push!(added_tokens, token)
        push!(seen_contents, content_string)
    end
    return added_tokens
end

function _hf_config_token_id(value, added_by_content, name::AbstractString)
    value === nothing && return nothing
    value isa AbstractString || throw(ArgumentError("`$name` must be a token string or null"))
    token = get(added_by_content, String(value), nothing)
    token === nothing && throw(ArgumentError("`$name` does not name an added token"))
    return token.id
end

function _hf_validate_tokenizer_config(
    config,
    added_tokens,
    added_by_content,
    model_vocabulary_size::Int,
)
    _hf_exact_value(config, "tokenizer_class", "Qwen2Tokenizer", "tokenizer_config.json")
    _hf_exact_bool(config, "add_bos_token", false, "tokenizer_config.json")
    _hf_exact_bool(config, "add_prefix_space", false, "tokenizer_config.json")
    _hf_exact_bool(config, "clean_up_tokenization_spaces", false, "tokenizer_config.json")
    _hf_exact_bool(config, "split_special_tokens", false, "tokenizer_config.json")
    _hf_exact_value(config, "errors", "replace", "tokenizer_config.json")
    _hf_exact_value(config, "unk_token", nothing, "tokenizer_config.json")

    decoder = _hf_required(config, "added_tokens_decoder", "tokenizer_config.json")
    decoder isa JSON3.Object || throw(ArgumentError("added_tokens_decoder must be an object"))
    length(decoder) == length(added_tokens) || throw(ArgumentError(
        "added_tokens_decoder does not cover every tokenizer added token",
    ))
    for token in added_tokens
        key = string(token.id - 1)
        haskey(decoder, key) || throw(ArgumentError("added_tokens_decoder is missing id $key"))
        entry = decoder[key]
        entry isa JSON3.Object || throw(ArgumentError("added_tokens_decoder entry must be an object"))
        _hf_exact_value(entry, "content", token.content, "added_tokens_decoder[$key]")
        _hf_exact_bool(entry, "single_word", token.single_word, "added_tokens_decoder[$key]")
        _hf_exact_bool(entry, "lstrip", token.lstrip, "added_tokens_decoder[$key]")
        _hf_exact_bool(entry, "rstrip", token.rstrip, "added_tokens_decoder[$key]")
        _hf_exact_bool(entry, "normalized", token.normalized, "added_tokens_decoder[$key]")
        _hf_exact_bool(entry, "special", token.special, "added_tokens_decoder[$key]")
    end

    model_max_length = _hf_required(config, "model_max_length", "tokenizer_config.json")
    model_max_length isa Integer && model_max_length > 0 || throw(ArgumentError(
        "model_max_length must be a positive integer",
    ))
    chat_template = _hf_required(config, "chat_template", "tokenizer_config.json")
    chat_template isa AbstractString && !isempty(chat_template) || throw(ArgumentError(
        "chat_template must be a non-empty string",
    ))
    template_hash = _sha256_hex(chat_template)
    is_official_template = template_hash == _QWEN3_CHAT_TEMPLATE_SHA256
    is_tiny_test_fixture = model_vocabulary_size == 258 &&
        template_hash == _QWEN3_TEST_CHAT_TEMPLATE_SHA256
    is_official_template || is_tiny_test_fixture || throw(ArgumentError(
        "unsupported Qwen3 chat_template revision",
    ))
    bos_id = _hf_config_token_id(_hf_required(config, "bos_token", "tokenizer_config.json"), added_by_content, "bos_token")
    eos_id = _hf_config_token_id(_hf_required(config, "eos_token", "tokenizer_config.json"), added_by_content, "eos_token")
    pad_id = _hf_config_token_id(_hf_required(config, "pad_token", "tokenizer_config.json"), added_by_content, "pad_token")
    raw_additional = _hf_required(config, "additional_special_tokens", "tokenizer_config.json")
    raw_additional isa JSON3.Array && all(value -> value isa AbstractString, raw_additional) ||
        throw(ArgumentError("additional_special_tokens must be an array of strings"))
    additional = Set(String.(collect(raw_additional)))
    length(additional) == length(raw_additional) || throw(ArgumentError(
        "additional_special_tokens contains duplicates",
    ))
    expected_additional = Set(
        token.content for token in added_tokens if token.special && token.id != pad_id
    )
    additional == expected_additional || throw(ArgumentError(
        "additional_special_tokens conflicts with tokenizer added tokens",
    ))
    return bos_id, eos_id, pad_id, Int(model_max_length), String(chat_template)
end

function _hf_generation_id(value, total_vocabulary::Int, name::AbstractString)
    value isa Integer || throw(ArgumentError("`$name` must be an integer"))
    id = Int(value)
    0 <= id < total_vocabulary || throw(ArgumentError("`$name` is outside tokenizer vocabulary"))
    return id + 1
end

function _hf_validate_generation_config(config, total_vocabulary::Int)
    allowed_fields = Set([
        "bos_token_id",
        "do_sample",
        "eos_token_id",
        "pad_token_id",
        "temperature",
        "top_k",
        "top_p",
        "transformers_version",
    ])
    unknown_fields = setdiff(Set(String.(collect(keys(config)))), allowed_fields)
    isempty(unknown_fields) || throw(ArgumentError(
        "unsupported generation_config.json fields: $(join(sort!(collect(unknown_fields)), ", "))",
    ))
    bos_id = _hf_generation_id(_hf_required(config, "bos_token_id", "generation_config.json"), total_vocabulary, "bos_token_id")
    pad_id = _hf_generation_id(_hf_required(config, "pad_token_id", "generation_config.json"), total_vocabulary, "pad_token_id")
    raw_eos = _hf_required(config, "eos_token_id", "generation_config.json")
    values = raw_eos isa Integer ? [raw_eos] : raw_eos
    values isa AbstractVector && !isempty(values) || throw(ArgumentError(
        "eos_token_id must be an integer or non-empty array",
    ))
    eos_ids = [_hf_generation_id(value, total_vocabulary, "eos_token_id") for value in values]
    length(unique(eos_ids)) == length(eos_ids) || throw(ArgumentError("duplicate eos_token_id"))
    do_sample = _hf_required(config, "do_sample", "generation_config.json")
    do_sample isa Bool || throw(ArgumentError("do_sample must be boolean"))
    temperature = _hf_required(config, "temperature", "generation_config.json")
    temperature isa Real && isfinite(temperature) && temperature > 0 || throw(ArgumentError(
        "temperature must be a finite positive number",
    ))
    top_k = _hf_required(config, "top_k", "generation_config.json")
    top_k isa Integer && top_k > 0 || throw(ArgumentError("top_k must be a positive integer"))
    top_p = _hf_required(config, "top_p", "generation_config.json")
    top_p isa Real && isfinite(top_p) && 0 < top_p <= 1 || throw(ArgumentError(
        "top_p must be in (0, 1]",
    ))
    transformers_version = _hf_required(
        config,
        "transformers_version",
        "generation_config.json",
    )
    transformers_version isa AbstractString && !isempty(transformers_version) || throw(
        ArgumentError("transformers_version must be a non-empty string"),
    )
    return HFQwen3GenerationConfig(
        bos_id,
        eos_ids,
        pad_id,
        do_sample,
        Float32(temperature),
        Int(top_k),
        Float32(top_p),
        String(transformers_version),
    )
end

function _hf_qwen3_tokenizer_from_json(
    raw_tokenizer_json::AbstractString,
    raw_tokenizer_config_json::AbstractString,
    raw_generation_config_json::AbstractString;
    revision::AbstractString="",
)
    tokenizer_json = _hf_json(raw_tokenizer_json, "tokenizer.json")
    tokenizer_config_json = _hf_json(raw_tokenizer_config_json, "tokenizer_config.json")
    generation_config_json = _hf_json(raw_generation_config_json, "generation_config.json")
    pattern = _hf_validate_pipeline(tokenizer_json)
    alphabet = hf_byte_unicode_alphabet()
    vocabulary, model_vocabulary_size, merge_ranks = _hf_parse_model(
        tokenizer_json,
        alphabet.char_to_byte,
    )
    added_tokens = _hf_parse_added_tokens(tokenizer_json, model_vocabulary_size, vocabulary)
    total_vocabulary = model_vocabulary_size + length(added_tokens)
    id_to_token = Vector{String}(undef, total_vocabulary)
    token_bytes = [UInt8[] for _ in 1:total_vocabulary]
    for (token, id) in vocabulary
        id_to_token[id] = token
        token_bytes[id] = UInt8[alphabet.char_to_byte[character] for character in token]
    end
    added_by_content = Dict{String,HFAddedToken}()
    special_ids = Set{Int}()
    for token in added_tokens
        id_to_token[token.id] = token.content
        token_bytes[token.id] = Vector{UInt8}(codeunits(token.content))
        added_by_content[token.content] = token
        token.special && push!(special_ids, token.id)
    end
    tokenizer_bos, tokenizer_eos, tokenizer_pad, model_max_length, chat_template =
        _hf_validate_tokenizer_config(
            tokenizer_config_json,
            added_tokens,
            added_by_content,
            model_vocabulary_size,
        )
    generation = _hf_validate_generation_config(
        generation_config_json,
        total_vocabulary,
    )
    tokenizer_bos === nothing || tokenizer_bos == generation.bos_id || throw(ArgumentError(
        "tokenizer and generation BOS ids conflict",
    ))
    tokenizer_pad == generation.pad_id || throw(ArgumentError(
        "tokenizer and generation PAD ids conflict",
    ))
    tokenizer_eos in generation.eos_ids || throw(ArgumentError(
        "tokenizer EOS id is absent from generation eos_token_id",
    ))
    compiled_pattern = try
        Regex(pattern)
    catch err
        throw(ArgumentError("Qwen3 pre-tokenizer regex cannot be compiled: $(sprint(showerror, err))"))
    end
    return HFQwen3Tokenizer(
        vocabulary,
        id_to_token,
        token_bytes,
        model_vocabulary_size,
        merge_ranks,
        pattern,
        compiled_pattern,
        added_tokens,
        added_by_content,
        special_ids,
        generation.bos_id,
        tokenizer_eos,
        generation.eos_ids,
        generation.pad_id,
        generation,
        model_max_length,
        chat_template,
        String(revision),
        String(raw_tokenizer_json),
        String(raw_tokenizer_config_json),
        String(raw_generation_config_json),
        _sha256_hex(raw_tokenizer_json),
        _sha256_hex(raw_tokenizer_config_json),
        _sha256_hex(raw_generation_config_json),
    )
end

"""
    load_hf_qwen3_tokenizer(model_dir; revision="")

Strictly load `tokenizer.json`, `tokenizer_config.json`, and
`generation_config.json` from a local Qwen3 model directory. No downloads are
performed.
"""
function load_hf_qwen3_tokenizer(
    model_dir::AbstractString;
    revision::AbstractString="",
)
    isdir(model_dir) || throw(ArgumentError("model directory does not exist: $model_dir"))
    paths = (
        tokenizer=joinpath(model_dir, "tokenizer.json"),
        tokenizer_config=joinpath(model_dir, "tokenizer_config.json"),
        generation_config=joinpath(model_dir, "generation_config.json"),
    )
    for path in values(paths)
        isfile(path) || throw(ArgumentError("required Qwen3 tokenizer file does not exist: $path"))
    end
    return _hf_qwen3_tokenizer_from_json(
        read(paths.tokenizer, String),
        read(paths.tokenizer_config, String),
        read(paths.generation_config, String);
        revision,
    )
end

_normalization_mode(::HFQwen3Tokenizer) = :nfc
vocab_size(tokenizer::HFQwen3Tokenizer) = length(tokenizer.id_to_token)
Base.length(tokenizer::HFQwen3Tokenizer) = vocab_size(tokenizer)

"""
    hf_generation_config(tokenizer)

Return the validated Qwen3 generation settings with LifeAI's public 1-based
token ids. Mutable vectors are copied so callers cannot alter the tokenizer.
"""
function hf_generation_config(tokenizer::HFQwen3Tokenizer)
    config = tokenizer.generation
    return (;
        bos_id=config.bos_id,
        eos_ids=copy(config.eos_ids),
        pad_id=config.pad_id,
        do_sample=config.do_sample,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        transformers_version=config.transformers_version,
    )
end

function special_token_id(tokenizer::HFQwen3Tokenizer, name)
    symbol = Symbol(name)
    symbol === :bos && return tokenizer.bos_id
    symbol === :eos && return tokenizer.eos_id
    symbol === :pad && return tokenizer.pad_id
    symbol === :unk && return nothing
    return nothing
end

function _hf_next_added_match(text::String, start::Int, added_tokens)
    best_token = nothing
    best_range = nothing
    for token in added_tokens
        range = findnext(token.content, text, start)
        range === nothing && continue
        if best_range === nothing || first(range) < first(best_range) ||
           (first(range) == first(best_range) &&
            (ncodeunits(token.content) > ncodeunits(best_token.content) ||
             (ncodeunits(token.content) == ncodeunits(best_token.content) && token.id < best_token.id)))
            best_token = token
            best_range = range
        end
    end
    return best_token, best_range
end

function _hf_added_segments(tokenizer::HFQwen3Tokenizer, input::AbstractString)
    text = String(input)
    isempty(text) && return Tuple{Bool,String,Union{Nothing,HFAddedToken}}[]
    segments = Tuple{Bool,String,Union{Nothing,HFAddedToken}}[]
    cursor = firstindex(text)
    terminal = ncodeunits(text) + 1
    while cursor < terminal
        token, range = _hf_next_added_match(text, cursor, tokenizer.added_tokens)
        if range === nothing
            push!(segments, (false, String(SubString(text, cursor)), nothing))
            break
        end
        match_start = first(range)
        if cursor < match_start
            push!(segments, (
                false,
                String(SubString(text, cursor, prevind(text, match_start))),
                nothing,
            ))
        end
        push!(segments, (true, token.content, token))
        cursor = nextind(text, last(range))
    end
    return segments
end

function _hf_pretoken_records(tokenizer::HFQwen3Tokenizer, text::String)
    isempty(text) && return NamedTuple[]
    matches = collect(eachmatch(tokenizer.pretokenizer_regex, text))
    join((match.match for match in matches)) == text || throw(ArgumentError(
        "Qwen3 pre-tokenizer regex did not cover the complete normalized input",
    ))
    records = NamedTuple[]
    for match in matches
        byte_start = match.offset - 1
        byte_stop = byte_start + ncodeunits(match.match)
        character_start = match.offset == firstindex(text) ? 0 :
            length(SubString(text, firstindex(text), prevind(text, match.offset)))
        character_stop = character_start + length(match.match)
        push!(records, (;
            text=String(match.match),
            symbols=_hf_byte_symbols(String(match.match)),
            character_start,
            character_stop,
            byte_start,
            byte_stop,
        ))
    end
    return records
end

"""
    hf_qwen3_pretokenize(tokenizer, text)

Normalize `text` with NFC and expose the exact regex/ByteLevel pre-tokenizer
pieces. Character offsets match HuggingFace's zero-based offsets; byte offsets
make Julia UTF-8 indexing explicit.
"""
function hf_qwen3_pretokenize(tokenizer::HFQwen3Tokenizer, text::AbstractString)
    normalized = normalize_text(text, :nfc)
    return (; normalized, pieces=Tuple(_hf_pretoken_records(tokenizer, normalized)))
end

function _hf_byte_symbols(text::String)
    byte_to_char = hf_byte_unicode_alphabet().byte_to_char
    return String([byte_to_char[byte] for byte in codeunits(text)])
end

function _hf_bpe_tokens(tokenizer::HFQwen3Tokenizer, piece::String)
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
        sizehint!(output, length(symbols))
        index = 1
        while index <= length(symbols)
            if index < length(symbols) &&
               symbols[index] == best_pair[1] && symbols[index + 1] == best_pair[2]
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
    tokenizer::HFQwen3Tokenizer,
    text::AbstractString;
    add_special_tokens::Bool=false,
)
    # Qwen3's ByteLevel post-processor adds no boundary tokens, so this flag is
    # intentionally accepted but produces the same ids in both modes.
    add_special_tokens isa Bool || throw(ArgumentError("add_special_tokens must be Bool"))
    ids = Int[]
    for (is_added, segment, added_token) in _hf_added_segments(tokenizer, text)
        if is_added
            push!(ids, added_token.id)
            continue
        end
        normalized = normalize_text(segment, :nfc)
        for record in _hf_pretoken_records(tokenizer, normalized)
            for token in _hf_bpe_tokens(tokenizer, record.text)
                id = get(tokenizer.vocabulary, token, nothing)
                id === nothing && throw(ArgumentError(
                    "BPE output token $(repr(token)) is absent from the imported vocabulary",
                ))
                push!(ids, id)
            end
        end
    end
    return ids
end

function decode_bytes(
    tokenizer::HFQwen3Tokenizer,
    ids;
    skip_special_tokens::Bool=false,
)
    output = UInt8[]
    for raw_id in ids
        raw_id isa Integer || throw(ArgumentError("token id $(repr(raw_id)) is not an integer"))
        id = Int(raw_id)
        1 <= id <= vocab_size(tokenizer) || throw(ArgumentError(
            "token id $id is outside the imported tokenizer vocabulary",
        ))
        skip_special_tokens && id in tokenizer.special_ids && continue
        append!(output, tokenizer.token_bytes[id])
    end
    return output
end

function decode(
    tokenizer::HFQwen3Tokenizer,
    ids;
    errors::Symbol=:strict,
    skip_special_tokens::Bool=false,
)
    return _decode_utf8(decode_bytes(tokenizer, ids; skip_special_tokens); errors)
end

function token_byte_length(tokenizer::HFQwen3Tokenizer, id::Integer)
    1 <= id <= vocab_size(tokenizer) || throw(ArgumentError("token id is outside vocabulary"))
    return Int(id) in tokenizer.special_ids ? 0 : length(tokenizer.token_bytes[Int(id)])
end

function tokenizer_config(tokenizer::HFQwen3Tokenizer)
    return (;
        type=:hf_qwen3_bpe,
        id_base=1,
        normalization=:nfc,
        vocabulary_size=vocab_size(tokenizer),
        model_vocabulary_size=tokenizer.model_vocabulary_size,
        merge_count=length(tokenizer.merge_ranks),
        added_token_count=length(tokenizer.added_tokens),
        eos_ids=copy(tokenizer.eos_ids),
        do_sample=tokenizer.generation.do_sample,
        temperature=tokenizer.generation.temperature,
        top_k=tokenizer.generation.top_k,
        top_p=tokenizer.generation.top_p,
        transformers_version=tokenizer.generation.transformers_version,
        model_max_length=tokenizer.model_max_length,
        revision=tokenizer.revision,
        tokenizer_sha256=tokenizer.tokenizer_sha256,
        tokenizer_config_sha256=tokenizer.tokenizer_config_sha256,
        generation_config_sha256=tokenizer.generation_config_sha256,
    )
end

function tokenizer_fingerprint(tokenizer::HFQwen3Tokenizer)
    output = IOBuffer()
    println(output, "schema=", TOKENIZER_ARTIFACT_VERSION)
    println(output, "type=hf_qwen3_bpe")
    println(output, "id_base=1")
    println(output, "revision=", tokenizer.revision)
    println(output, "tokenizer=", tokenizer.tokenizer_sha256)
    println(output, "tokenizer_config=", tokenizer.tokenizer_config_sha256)
    println(output, "generation_config=", tokenizer.generation_config_sha256)
    return bytes2hex(sha256(take!(output)))
end

function _tokenizer_artifact(tokenizer::HFQwen3Tokenizer)
    return Dict{String,Any}(
        "schema_version" => TOKENIZER_ARTIFACT_VERSION,
        "type" => "hf_qwen3_bpe",
        "id_base" => 1,
        "revision" => tokenizer.revision,
        "tokenizer_json_hex" => bytes2hex(codeunits(tokenizer.raw_tokenizer_json)),
        "tokenizer_config_json_hex" => bytes2hex(codeunits(tokenizer.raw_tokenizer_config_json)),
        "generation_config_json_hex" => bytes2hex(codeunits(tokenizer.raw_generation_config_json)),
        "fingerprint" => tokenizer_fingerprint(tokenizer),
    )
end

function _hf_message_value(message, name::Symbol; default=nothing, required::Bool=false)
    value = if message isa NamedTuple
        hasproperty(message, name) ? getproperty(message, name) : default
    elseif message isa AbstractDict
        haskey(message, name) ? message[name] : get(message, String(name), default)
    else
        throw(ArgumentError("chat messages must be NamedTuples or dictionaries"))
    end
    required && value === nothing && throw(ArgumentError("chat message is missing `$name`"))
    return value
end

function _strip_newlines(input::AbstractString; left::Bool=true, right::Bool=true)
    text = String(input)
    first_position = firstindex(text)
    terminal = ncodeunits(text) + 1
    if left
        while first_position < terminal && text[first_position] == '\n'
            first_position = nextind(text, first_position)
        end
    end
    last_position = isempty(text) ? 0 : prevind(text, terminal)
    if right
        while last_position >= first_position && text[last_position] == '\n'
            last_position == firstindex(text) && return ""
            last_position = prevind(text, last_position)
        end
    end
    first_position >= terminal || last_position < first_position ? "" :
        String(SubString(text, first_position, last_position))
end

function _hf_assistant_content(message, raw_content::String)
    reasoning_value = _hf_message_value(message, :reasoning_content; default=nothing)
    reasoning_value === nothing || reasoning_value isa AbstractString || throw(ArgumentError(
        "assistant reasoning_content must be a string when present",
    ))
    reasoning = reasoning_value === nothing ? "" : String(reasoning_value)
    content = raw_content
    if reasoning_value === nothing && occursin("</think>", content)
        before = first(split(content, "</think>"))
        reasoning = last(split(before, "<think>"))
        content = last(split(content, "</think>"))
        reasoning = _strip_newlines(reasoning)
        content = _strip_newlines(content; right=false)
    end
    return reasoning, content
end

"""
    apply_qwen3_chat_template(tokenizer, messages; kwargs...)

Render the no-tools Qwen3 system/user/assistant chat-template subset. Inputs
outside that explicitly supported subset are rejected.
"""
function apply_qwen3_chat_template(
    tokenizer::HFQwen3Tokenizer,
    messages;
    add_generation_prompt::Bool=true,
    enable_thinking::Bool=true,
)
    message_list = collect(messages)
    isempty(message_list) && throw(ArgumentError("chat messages must not be empty"))
    roles = String[]
    contents = String[]
    for message in message_list
        role = _hf_message_value(message, :role; required=true)
        content = _hf_message_value(message, :content; required=true)
        role isa AbstractString || throw(ArgumentError("chat role must be a string"))
        content isa AbstractString || throw(ArgumentError("chat content must be a string"))
        role_string = String(role)
        role_string in ("system", "user", "assistant") || throw(ArgumentError(
            "unsupported Qwen3 chat role $(repr(role_string)); tools are outside Week 08 scope",
        ))
        tool_calls = _hf_message_value(message, :tool_calls; default=nothing)
        (tool_calls === nothing || isempty(tool_calls)) || throw(ArgumentError(
            "Qwen3 tool calls are outside Week 08 scope",
        ))
        push!(roles, role_string)
        push!(contents, String(content))
    end

    last_user = findlast(==("user"), roles)
    last_user === nothing && (last_user = 0)
    output = IOBuffer()
    if roles[1] == "system"
        print(output, "<|im_start|>system\n", contents[1], "<|im_end|>\n")
    end
    for index in eachindex(message_list)
        role = roles[index]
        content = contents[index]
        index == 1 && role == "system" && continue
        if role == "user" || role == "system"
            print(output, "<|im_start|>", role, "\n", content, "<|im_end|>\n")
        else
            reasoning, visible_content = _hf_assistant_content(message_list[index], content)
            if index > last_user && (index == length(message_list) || !isempty(reasoning))
                print(
                    output,
                    "<|im_start|>assistant\n<think>\n",
                    _strip_newlines(reasoning),
                    "\n</think>\n\n",
                    _strip_newlines(visible_content; right=false),
                    "<|im_end|>\n",
                )
            else
                print(output, "<|im_start|>assistant\n", content, "<|im_end|>\n")
            end
        end
    end
    if add_generation_prompt
        print(output, "<|im_start|>assistant\n")
        enable_thinking || print(output, "<think>\n\n</think>\n\n")
    end
    return String(take!(output))
end

_model_tokenizer_vocab_compatible(model_vocab_size::Int, tokenizer::HFQwen3Tokenizer) =
    model_vocab_size >= vocab_size(tokenizer)
