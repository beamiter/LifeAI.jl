using JSON3

const _QWEN3_TOKENIZER_REGEX = raw"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
const _QWEN3_CHAT_TEMPLATE_SHA256 =
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
const _QWEN3_EMBEDDING_CHAT_TEMPLATE_SHA256 =
    "87a2728cb8dc9fe424d624542f6060ec05a1d285ebbec578bb078900e33396b5"
const _QWEN3_VL_CHAT_TEMPLATE_SHA256 =
    "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"
const _QWEN3_TEST_CHAT_TEMPLATE_SHA256 =
    "07bdce04691aa17c70cd6bebfaeae06e324440e5d24fbeeb4f2628c4c997a3ea"

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
    profile::Symbol
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

function _hf_template_sequence_entry(entry, expected::AbstractString, label)
    entry isa JSON3.Object || throw(ArgumentError("$label must be an object"))
    length(entry) == 1 && haskey(entry, "Sequence") || throw(ArgumentError(
        "$label must contain exactly one Sequence entry",
    ))
    sequence = entry["Sequence"]
    sequence isa JSON3.Object || throw(ArgumentError(
        "$label Sequence must be an object",
    ))
    _hf_exact_value(sequence, "id", String(expected), label)
    _hf_exact_value(sequence, "type_id", 0, label)
    return nothing
end

function _hf_template_special_entry(entry, label)
    entry isa JSON3.Object || throw(ArgumentError("$label must be an object"))
    length(entry) == 1 && haskey(entry, "SpecialToken") || throw(ArgumentError(
        "$label must contain exactly one SpecialToken entry",
    ))
    special = entry["SpecialToken"]
    special isa JSON3.Object || throw(ArgumentError(
        "$label SpecialToken must be an object",
    ))
    _hf_exact_value(special, "id", "<|endoftext|>", label)
    _hf_exact_value(special, "type_id", 0, label)
    return nothing
end

function _hf_validate_embedding_post_processor(post_processor)
    label = "tokenizer.json embedding post_processor"
    _hf_exact_value(post_processor, "type", "Sequence", label)
    processors = _hf_required(post_processor, "processors", label)
    processors isa JSON3.Array && length(processors) == 2 || throw(ArgumentError(
        "embedding post_processor must contain ByteLevel and TemplateProcessing",
    ))
    byte_processor = processors[1]
    byte_processor isa JSON3.Object || throw(ArgumentError(
        "embedding ByteLevel post-processor must be an object",
    ))
    _hf_byte_flags(byte_processor, "$label ByteLevel")

    template = processors[2]
    template isa JSON3.Object || throw(ArgumentError(
        "embedding TemplateProcessing post-processor must be an object",
    ))
    _hf_exact_value(template, "type", "TemplateProcessing", label)
    single = _hf_required(template, "single", label)
    single isa JSON3.Array && length(single) == 2 || throw(ArgumentError(
        "embedding single template must be A followed by <|endoftext|>",
    ))
    _hf_template_sequence_entry(single[1], "A", "$label single[1]")
    _hf_template_special_entry(single[2], "$label single[2]")
    pair = _hf_required(template, "pair", label)
    pair isa JSON3.Array && length(pair) == 3 || throw(ArgumentError(
        "embedding pair template must be A, B, then <|endoftext|>",
    ))
    _hf_template_sequence_entry(pair[1], "A", "$label pair[1]")
    _hf_template_sequence_entry(pair[2], "B", "$label pair[2]")
    _hf_template_special_entry(pair[3], "$label pair[3]")

    special_tokens = _hf_required(template, "special_tokens", label)
    special_tokens isa JSON3.Object &&
        Set(String.(collect(keys(special_tokens)))) == Set(["<|endoftext|>"]) ||
        throw(ArgumentError(
            "embedding template must define only <|endoftext|>",
        ))
    endoftext = special_tokens["<|endoftext|>"]
    endoftext isa JSON3.Object || throw(ArgumentError(
        "embedding <|endoftext|> template metadata must be an object",
    ))
    _hf_exact_value(endoftext, "id", "<|endoftext|>", label)
    ids = _hf_required(endoftext, "ids", label)
    ids isa JSON3.Array && length(ids) == 1 && ids[1] isa Integer ||
        throw(ArgumentError(
            "embedding <|endoftext|> must define exactly one integer id",
        ))
    tokens = _hf_required(endoftext, "tokens", label)
    tokens isa JSON3.Array && String.(collect(tokens)) == ["<|endoftext|>"] ||
        throw(ArgumentError(
            "embedding template token payload must be <|endoftext|>",
        ))
    return Int(ids[1]) + 1
end

function _hf_validate_pipeline(tokenizer_json, profile::Symbol)
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
    boundary_id = if profile !== :embedding
        _hf_byte_flags(post_processor, "tokenizer.json post_processor")
        nothing
    else
        _hf_validate_embedding_post_processor(post_processor)
    end

    decoder = _hf_required(tokenizer_json, "decoder", "tokenizer.json")
    decoder isa JSON3.Object || throw(ArgumentError("decoder must be an object"))
    _hf_byte_flags(decoder, "tokenizer.json decoder")
    return (; pattern=String(pattern), boundary_id)
end

function _hf_parse_model(tokenizer_json, char_to_byte, profile::Symbol)
    model = _hf_required(tokenizer_json, "model", "tokenizer.json")
    model isa JSON3.Object || throw(ArgumentError("tokenizer model must be an object"))
    _hf_exact_value(model, "type", "BPE", "tokenizer.json model")
    _hf_exact_value(model, "dropout", nothing, "tokenizer.json model")
    _hf_exact_value(model, "unk_token", nothing, "tokenizer.json model")
    _hf_exact_value(model, "continuing_subword_prefix", "", "tokenizer.json model")
    _hf_exact_value(model, "end_of_word_suffix", "", "tokenizer.json model")
    _hf_exact_bool(model, "fuse_unk", false, "tokenizer.json model")
    _hf_exact_bool(model, "byte_fallback", false, "tokenizer.json model")
    if profile === :qwen3_vl_generation
        haskey(model, "ignore_merges") && throw(ArgumentError(
            "Qwen3-VL tokenizer model must omit `ignore_merges`",
        ))
    else
        _hf_exact_bool(model, "ignore_merges", false, "tokenizer.json model")
    end

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
        pair = if profile === :qwen3_vl_generation
            raw_pair isa AbstractString || throw(ArgumentError(
                "Qwen3-VL BPE merge $rank must be a string pair",
            ))
            pieces = split(String(raw_pair), ' '; keepempty=true)
            length(pieces) == 2 && all(!isempty, pieces) || throw(ArgumentError(
                "Qwen3-VL BPE merge $rank must contain two tokens separated by one space",
            ))
            (pieces[1], pieces[2])
        else
            raw_pair isa JSON3.Array && length(raw_pair) == 2 || throw(ArgumentError(
                "BPE merge $rank must be a two-element array",
            ))
            raw_pair[1] isa AbstractString && raw_pair[2] isa AbstractString || throw(ArgumentError(
                "BPE merge $rank must contain strings",
            ))
            (String(raw_pair[1]), String(raw_pair[2]))
        end
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
    profile::Symbol,
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
    is_official_template = if profile === :generation
        template_hash == _QWEN3_CHAT_TEMPLATE_SHA256
    elseif profile === :qwen3_vl_generation
        template_hash == _QWEN3_VL_CHAT_TEMPLATE_SHA256
    else
        template_hash == _QWEN3_EMBEDDING_CHAT_TEMPLATE_SHA256
    end
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

function _hf_validate_generation_config(
    config,
    total_vocabulary::Int;
    qwen3_vl::Bool=false,
)
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
    qwen3_vl && push!(allowed_fields, "repetition_penalty")
    unknown_fields = setdiff(Set(String.(collect(keys(config)))), allowed_fields)
    isempty(unknown_fields) || throw(ArgumentError(
        "unsupported generation_config.json fields: $(join(sort!(collect(unknown_fields)), ", "))",
    ))
    if qwen3_vl
        repetition_penalty = _hf_required(
            config,
            "repetition_penalty",
            "generation_config.json",
        )
        repetition_penalty isa Real && !(repetition_penalty isa Bool) &&
            Float64(repetition_penalty) == 1.0 || throw(ArgumentError(
                "Qwen3-VL repetition_penalty must be exactly 1.0",
            ))
    end
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

function _hf_validate_embedding_generation_config(
    config,
    total_vocabulary::Int,
    pad_id::Int,
)
    allowed_fields = Set([
        "bos_token_id",
        "eos_token_id",
        "max_new_tokens",
        "transformers_version",
    ])
    unknown_fields = setdiff(Set(String.(collect(keys(config)))), allowed_fields)
    isempty(unknown_fields) || throw(ArgumentError(
        "unsupported embedding generation_config.json fields: " *
        "$(join(sort!(collect(unknown_fields)), ", "))",
    ))
    bos_id = _hf_generation_id(
        _hf_required(config, "bos_token_id", "generation_config.json"),
        total_vocabulary,
        "bos_token_id",
    )
    eos_id = _hf_generation_id(
        _hf_required(config, "eos_token_id", "generation_config.json"),
        total_vocabulary,
        "eos_token_id",
    )
    max_new_tokens = _hf_required(
        config,
        "max_new_tokens",
        "generation_config.json",
    )
    max_new_tokens isa Integer && max_new_tokens > 0 || throw(ArgumentError(
        "max_new_tokens must be a positive integer",
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
        [eos_id],
        pad_id,
        false,
        1.0f0,
        total_vocabulary,
        1.0f0,
        String(transformers_version),
    )
end

function _hf_qwen3_tokenizer_from_json(
    raw_tokenizer_json::AbstractString,
    raw_tokenizer_config_json::AbstractString,
    raw_generation_config_json::AbstractString;
    revision::AbstractString="",
    profile::Symbol=:generation,
)
    profile in (:generation, :embedding, :qwen3_vl_generation) || throw(ArgumentError(
        "Qwen3 tokenizer profile must be :generation, :embedding, or " *
        ":qwen3_vl_generation",
    ))
    tokenizer_json = _hf_json(raw_tokenizer_json, "tokenizer.json")
    tokenizer_config_json = _hf_json(raw_tokenizer_config_json, "tokenizer_config.json")
    generation_config_json = _hf_json(raw_generation_config_json, "generation_config.json")
    pipeline = _hf_validate_pipeline(tokenizer_json, profile)
    pattern = pipeline.pattern
    alphabet = hf_byte_unicode_alphabet()
    vocabulary, model_vocabulary_size, merge_ranks = _hf_parse_model(
        tokenizer_json,
        alphabet.char_to_byte,
        profile,
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
            profile,
        )
    tokenizer_pad === nothing && throw(ArgumentError(
        "Qwen3 tokenizer must define a padding token",
    ))
    pipeline.boundary_id === nothing ||
        pipeline.boundary_id == tokenizer_pad || throw(ArgumentError(
            "embedding post-processor token id must match tokenizer pad token",
        ))
    generation = if profile !== :embedding
        value = _hf_validate_generation_config(
            generation_config_json,
            total_vocabulary,
            qwen3_vl=profile === :qwen3_vl_generation,
        )
        tokenizer_bos === nothing || tokenizer_bos == value.bos_id || throw(ArgumentError(
            "tokenizer and generation BOS ids conflict",
        ))
        tokenizer_pad == value.pad_id || throw(ArgumentError(
            "tokenizer and generation PAD ids conflict",
        ))
        tokenizer_eos in value.eos_ids || throw(ArgumentError(
            "tokenizer EOS id is absent from generation eos_token_id",
        ))
        value
    else
        _hf_validate_embedding_generation_config(
            generation_config_json,
            total_vocabulary,
            tokenizer_pad,
        )
    end
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
        profile,
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
    load_hf_qwen3_vl_tokenizer(model_dir; revision="")

Strictly load the tokenizer profile shipped with Qwen3-VL.  This profile is
separate from ordinary Qwen3 generation because its added vision tokens,
chat template, tokenizer model fields, and generation config are different.
No files are downloaded.
"""
function load_hf_qwen3_vl_tokenizer(
    model_dir::AbstractString;
    revision::AbstractString="",
)
    isdir(model_dir) || throw(ArgumentError(
        "model directory does not exist: $model_dir",
    ))
    paths = (
        tokenizer=joinpath(model_dir, "tokenizer.json"),
        tokenizer_config=joinpath(model_dir, "tokenizer_config.json"),
        generation_config=joinpath(model_dir, "generation_config.json"),
    )
    for path in values(paths)
        isfile(path) || throw(ArgumentError(
            "required Qwen3-VL tokenizer file does not exist: $path",
        ))
    end
    return _hf_qwen3_tokenizer_from_json(
        read(paths.tokenizer, String),
        read(paths.tokenizer_config, String),
        read(paths.generation_config, String);
        revision,
        profile=:qwen3_vl_generation,
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

"""
    load_hf_qwen3_embedding_tokenizer(model_dir; revision="")

Strictly load the tokenizer shipped with a Qwen3-Embedding checkpoint. Its
small `generation_config.json` and frozen chat template intentionally differ
from text-generation checkpoints, so the two profiles cannot be silently
interchanged.
"""
function load_hf_qwen3_embedding_tokenizer(
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
        isfile(path) || throw(ArgumentError(
            "required Qwen3 embedding tokenizer file does not exist: $path",
        ))
    end
    return _hf_qwen3_tokenizer_from_json(
        read(paths.tokenizer, String),
        read(paths.tokenizer_config, String),
        read(paths.generation_config, String);
        revision,
        profile=:embedding,
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
    tokenizer.profile !== :embedding || throw(ArgumentError(
        "embedding tokenizer does not define text-generation sampling settings",
    ))
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
    if add_special_tokens && tokenizer.profile === :embedding
        push!(ids, tokenizer.pad_id)
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
        profile=tokenizer.profile,
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
    tokenizer.profile !== :generation &&
        println(output, "profile=", tokenizer.profile)
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
        "profile" => String(tokenizer.profile),
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

"""
    OrderedJSONObject(entries)

JSON object that keeps its keys in insertion order. `Dict` cannot be rendered byte
stably and `JSON3.Object` loses the JSON number form, so tool payloads travel in this
type instead.
"""
struct _OrderedJSONMissing end
const _ORDERED_JSON_MISSING = _OrderedJSONMissing()

struct OrderedJSONObject <: AbstractDict{String,Any}
    entries::Vector{Pair{String,Any}}
end

OrderedJSONObject() = OrderedJSONObject(Pair{String,Any}[])

Base.length(object::OrderedJSONObject) = length(object.entries)
Base.iterate(object::OrderedJSONObject, state...) = iterate(object.entries, state...)
Base.pairs(object::OrderedJSONObject) = object.entries
Base.keys(object::OrderedJSONObject) = String[first(entry) for entry in object.entries]
Base.values(object::OrderedJSONObject) = Any[last(entry) for entry in object.entries]

_ordered_json_key(key) = key isa Symbol ? String(key) : String(key)

function Base.get(object::OrderedJSONObject, key, default)
    name = _ordered_json_key(key)
    for entry in object.entries
        first(entry) == name && return last(entry)
    end
    return default
end

Base.haskey(object::OrderedJSONObject, key) =
    get(object, key, _ORDERED_JSON_MISSING) !== _ORDERED_JSON_MISSING

function Base.getindex(object::OrderedJSONObject, key)
    value = get(object, key, _ORDERED_JSON_MISSING)
    value === _ORDERED_JSON_MISSING && throw(KeyError(key))
    return value
end

"""
    _python_float_repr(value)

CPython `repr(float)`: the shortest round-tripping decimal, written in fixed notation
when the decimal point lands in `-3:16` and otherwise as `e±NN` with at least two
exponent digits. Julia and CPython both emit shortest round-trip digits; only the
placement rules differ, so this reformats rather than recomputes.
"""
function _python_float_repr(value::AbstractFloat)
    number = Float64(value)
    isfinite(number) || throw(ArgumentError(
        "non-finite floats are unsupported in Qwen3 tool payloads",
    ))
    sign = signbit(number) ? "-" : ""
    number == 0 && return sign * "0.0"

    text = string(abs(number))
    exponent_position = findfirst('e', text)
    mantissa, exponent = exponent_position === nothing ? (text, 0) :
        (text[1:exponent_position - 1], parse(Int, text[exponent_position + 1:end]))
    point_position = findfirst('.', mantissa)
    integer_part, fraction_part = point_position === nothing ? (mantissa, "") :
        (mantissa[1:point_position - 1], mantissa[point_position + 1:end])

    digits = integer_part * fraction_part
    decimal_point = length(integer_part) + exponent
    while length(digits) > 1 && first(digits) == '0'
        digits = digits[2:end]
        decimal_point -= 1
    end
    while length(digits) > 1 && last(digits) == '0'
        digits = digits[1:end - 1]
    end

    if decimal_point <= -4 || decimal_point > 16
        power = decimal_point - 1
        head = digits[1:1] * (length(digits) > 1 ? "." * digits[2:end] : "")
        return sign * head * "e" * (power < 0 ? "-" : "+") * lpad(abs(power), 2, '0')
    elseif decimal_point <= 0
        return sign * "0." * "0"^(-decimal_point) * digits
    elseif decimal_point >= length(digits)
        return sign * digits * "0"^(decimal_point - length(digits)) * ".0"
    end
    return sign * digits[1:decimal_point] * "." * digits[decimal_point + 1:end]
end

"""
    _python_json(io, value)

Serialize `value` byte for byte like CPython `json.dumps(value, ensure_ascii=false)`,
which is the `tojson` filter Transformers installs when it compiles a chat template:
`", "` and `": "` separators, insertion order, no key sorting and no ASCII escaping.

Only order-preserving containers are accepted, and a `JSON3.Object` is rejected because
`JSON3.read` narrows whole-valued floats (`1.0`, `1e2`) to `Int64`; rendering those as
`1` and `100` would silently diverge from the reference prompt. Parse tool payloads
with [`parse_qwen3_json`](@ref) instead.
"""
function _python_json(io::IO, value)
    if value === nothing
        print(io, "null")
    elseif value isa Bool
        print(io, value ? "true" : "false")
    elseif value isa Integer
        print(io, string(value))
    elseif value isa AbstractFloat
        print(io, _python_float_repr(value))
    elseif value isa AbstractString || value isa Symbol
        _python_json_string(io, String(value))
    elseif value isa NamedTuple
        _python_json_object(io, zip(keys(value), values(value)))
    elseif value isa OrderedJSONObject
        _python_json_object(io, ((first(entry), last(entry)) for entry in value.entries))
    elseif value isa JSON3.Object
        throw(ArgumentError(
            "JSON3.read narrows whole-valued floats (1.0 becomes 1) and would silently " *
            "change the rendered prompt; parse tool payloads with parse_qwen3_json",
        ))
    elseif (value isa AbstractVector || value isa Tuple) &&
           !isempty(value) && all(entry -> entry isa Pair, value)
        # An empty container stays an array: `[]` is a valid schema value and an
        # empty object is expressible as `NamedTuple()` or `OrderedJSONObject()`.
        _python_json_object(io, ((first(entry), last(entry)) for entry in value))
    elseif value isa AbstractVector || value isa Tuple
        print(io, "[")
        for (position, entry) in enumerate(value)
            position == 1 || print(io, ", ")
            _python_json(io, entry)
        end
        print(io, "]")
    elseif value isa AbstractDict
        throw(ArgumentError(
            "unordered Dict cannot be rendered byte-stably; use a NamedTuple, a Vector of Pairs or an OrderedJSONObject",
        ))
    else
        throw(ArgumentError("unsupported JSON value of type $(typeof(value))"))
    end
    return io
end

function _python_json_object(io::IO, entries)
    print(io, "{")
    first_entry = true
    for (key, value) in entries
        key isa AbstractString || key isa Symbol || throw(ArgumentError(
            "JSON object keys must be strings or symbols, got $(typeof(key))",
        ))
        first_entry || print(io, ", ")
        first_entry = false
        _python_json_string(io, String(key))
        print(io, ": ")
        _python_json(io, value)
    end
    print(io, "}")
    return io
end

function _python_json_string(io::IO, text::AbstractString)
    print(io, '"')
    for character in text
        if character == '"'
            print(io, "\\\"")
        elseif character == '\\'
            print(io, "\\\\")
        elseif character == '\n'
            print(io, "\\n")
        elseif character == '\r'
            print(io, "\\r")
        elseif character == '\t'
            print(io, "\\t")
        elseif character == '\b'
            print(io, "\\b")
        elseif character == '\f'
            print(io, "\\f")
        elseif character < '\x20'
            print(io, "\\u", string(UInt16(character); base=16, pad=4))
        else
            print(io, character)
        end
    end
    print(io, '"')
    return io
end

_python_json_text(value) = String(take!(_python_json(IOBuffer(), value)))

mutable struct _JSONReader
    text::String
    index::Int
end

"""
    parse_qwen3_json(text)

Parse JSON while preserving the distinction CPython preserves and `JSON3.read` does
not: a number written with a fraction or an exponent is a float, everything else is an
integer of arbitrary precision. Objects become [`OrderedJSONObject`](@ref) so key order
survives, arrays become `Vector{Any}`.

This is the parser tool declarations and model-emitted `<tool_call>` arguments go
through, because both are re-serialized into a prompt that must match the reference
byte for byte.
"""
function parse_qwen3_json(text::AbstractString)
    reader = _JSONReader(String(text), 1)
    value = _json_value!(reader)
    _json_skip_space!(reader)
    reader.index > ncodeunits(reader.text) || throw(ArgumentError(
        "trailing content after JSON value at byte $(reader.index)",
    ))
    return value
end

_json_at_end(reader::_JSONReader) = reader.index > ncodeunits(reader.text)

function _json_skip_space!(reader::_JSONReader)
    while !_json_at_end(reader)
        character = reader.text[reader.index]
        character in (' ', '\t', '\n', '\r') || break
        reader.index = nextind(reader.text, reader.index)
    end
    return reader
end

function _json_expect!(reader::_JSONReader, character::Char)
    _json_skip_space!(reader)
    (!_json_at_end(reader) && reader.text[reader.index] == character) || throw(ArgumentError(
        "expected $(repr(character)) at byte $(reader.index)",
    ))
    reader.index = nextind(reader.text, reader.index)
    return reader
end

function _json_value!(reader::_JSONReader)
    _json_skip_space!(reader)
    _json_at_end(reader) && throw(ArgumentError("unexpected end of JSON input"))
    character = reader.text[reader.index]
    character == '{' && return _json_object!(reader)
    character == '[' && return _json_array!(reader)
    character == '"' && return _json_string!(reader)
    for (literal, value) in (("true", true), ("false", false), ("null", nothing))
        stop = reader.index + ncodeunits(literal) - 1
        if stop <= ncodeunits(reader.text) &&
           SubString(reader.text, reader.index, stop) == literal
            reader.index = stop + 1
            return value
        end
    end
    (character == '-' || isdigit(character)) && return _json_number!(reader)
    throw(ArgumentError("unexpected character $(repr(character)) at byte $(reader.index)"))
end

function _json_object!(reader::_JSONReader)
    _json_expect!(reader, '{')
    entries = Pair{String,Any}[]
    _json_skip_space!(reader)
    if !_json_at_end(reader) && reader.text[reader.index] == '}'
        reader.index = nextind(reader.text, reader.index)
        return OrderedJSONObject(entries)
    end
    while true
        _json_skip_space!(reader)
        key = _json_string!(reader)
        _json_expect!(reader, ':')
        push!(entries, key => _json_value!(reader))
        _json_skip_space!(reader)
        _json_at_end(reader) && throw(ArgumentError("unterminated JSON object"))
        separator = reader.text[reader.index]
        reader.index = nextind(reader.text, reader.index)
        separator == ',' && continue
        separator == '}' && break
        throw(ArgumentError("expected ',' or '}' at byte $(reader.index - 1)"))
    end
    return OrderedJSONObject(entries)
end

function _json_array!(reader::_JSONReader)
    _json_expect!(reader, '[')
    values = Any[]
    _json_skip_space!(reader)
    if !_json_at_end(reader) && reader.text[reader.index] == ']'
        reader.index = nextind(reader.text, reader.index)
        return values
    end
    while true
        push!(values, _json_value!(reader))
        _json_skip_space!(reader)
        _json_at_end(reader) && throw(ArgumentError("unterminated JSON array"))
        separator = reader.text[reader.index]
        reader.index = nextind(reader.text, reader.index)
        separator == ',' && continue
        separator == ']' && break
        throw(ArgumentError("expected ',' or ']' at byte $(reader.index - 1)"))
    end
    return values
end

function _json_string!(reader::_JSONReader)
    _json_expect!(reader, '"')
    output = IOBuffer()
    while true
        _json_at_end(reader) && throw(ArgumentError("unterminated JSON string"))
        character = reader.text[reader.index]
        reader.index = nextind(reader.text, reader.index)
        character == '"' && break
        if character != '\\'
            character < '\x20' && throw(ArgumentError(
                "unescaped control character in JSON string at byte $(reader.index - 1)",
            ))
            print(output, character)
            continue
        end
        _json_at_end(reader) && throw(ArgumentError("unterminated JSON escape"))
        escape = reader.text[reader.index]
        reader.index = nextind(reader.text, reader.index)
        if escape == 'u'
            code = _json_hex4!(reader)
            if 0xD800 <= code <= 0xDBFF
                (reader.index + 1 <= ncodeunits(reader.text) &&
                 reader.text[reader.index] == '\\' &&
                 reader.text[nextind(reader.text, reader.index)] == 'u') || throw(ArgumentError(
                    "lone high surrogate in JSON string",
                ))
                reader.index = nextind(reader.text, nextind(reader.text, reader.index))
                low = _json_hex4!(reader)
                0xDC00 <= low <= 0xDFFF || throw(ArgumentError("invalid low surrogate"))
                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00)
            end
            print(output, Char(code))
        else
            replacement = escape == '"' ? '"' : escape == '\\' ? '\\' :
                escape == '/' ? '/' : escape == 'b' ? '\b' : escape == 'f' ? '\f' :
                escape == 'n' ? '\n' : escape == 'r' ? '\r' : escape == 't' ? '\t' :
                throw(ArgumentError("invalid JSON escape $(repr(escape))"))
            print(output, replacement)
        end
    end
    return String(take!(output))
end

function _json_hex4!(reader::_JSONReader)
    stop = reader.index + 3
    stop <= ncodeunits(reader.text) || throw(ArgumentError("truncated \\u escape"))
    digits = SubString(reader.text, reader.index, stop)
    reader.index = stop + 1
    value = tryparse(UInt32, digits; base=16)
    value === nothing && throw(ArgumentError("invalid \\u escape $(repr(String(digits)))"))
    return value
end

function _json_number!(reader::_JSONReader)
    start = reader.index
    is_float = false
    while !_json_at_end(reader)
        character = reader.text[reader.index]
        if isdigit(character) || character == '-' || character == '+'
            reader.index = nextind(reader.text, reader.index)
        elseif character == '.' || character == 'e' || character == 'E'
            is_float = true
            reader.index = nextind(reader.text, reader.index)
        else
            break
        end
    end
    lexeme = String(SubString(reader.text, start, prevind(reader.text, reader.index)))
    if is_float
        parsed = tryparse(Float64, lexeme)
        parsed === nothing && throw(ArgumentError("invalid JSON number $(repr(lexeme))"))
        return parsed
    end
    small = tryparse(Int, lexeme)
    small === nothing || return small
    large = tryparse(BigInt, lexeme)
    large === nothing && throw(ArgumentError("invalid JSON number $(repr(lexeme))"))
    return large
end

"""Normalized view of one chat message, mirroring the official template's guards."""
struct _Qwen3ChatEntry
    role::String
    content::String
    content_is_string::Bool
    reasoning::Union{Nothing,String}
    tool_calls::Vector{Any}
end

function _qwen3_chat_entry(message)
    role = _hf_message_value(message, :role; required=true)
    role isa AbstractString || throw(ArgumentError("chat role must be a string"))
    role_string = String(role)
    role_string in ("system", "user", "assistant", "tool") || throw(ArgumentError(
        "unsupported Qwen3 chat role $(repr(role_string))",
    ))
    raw_content = _hf_message_value(message, :content; default=nothing)
    content_is_string = raw_content isa AbstractString
    content_is_string || raw_content === nothing || throw(ArgumentError(
        "chat content must be a string when present",
    ))
    content = content_is_string ? String(raw_content) : ""
    raw_reasoning = _hf_message_value(message, :reasoning_content; default=nothing)
    raw_reasoning === nothing || raw_reasoning isa AbstractString || throw(ArgumentError(
        "assistant reasoning_content must be a string when present",
    ))
    reasoning = raw_reasoning === nothing ? nothing : String(raw_reasoning)
    raw_calls = _hf_message_value(message, :tool_calls; default=nothing)
    tool_calls = raw_calls === nothing ? Any[] : collect(Any, raw_calls)
    isempty(tool_calls) || role_string == "assistant" || throw(ArgumentError(
        "tool_calls are only supported on assistant messages",
    ))
    return _Qwen3ChatEntry(role_string, content, content_is_string, reasoning, tool_calls)
end

"""Split an assistant turn into `(reasoning, visible content)` like the official template."""
function _qwen3_assistant_split(entry::_Qwen3ChatEntry)
    entry.reasoning === nothing || return entry.reasoning, entry.content
    content = entry.content
    occursin("</think>", content) || return "", content
    before = first(split(content, "</think>"))
    reasoning = _strip_newlines(last(split(before, "<think>")))
    visible = _strip_newlines(last(split(content, "</think>")); right=false)
    return reasoning, visible
end

"""
Index of the newest real user turn, mirroring `ns.last_query_index`. A user message
that is itself a `<tool_response>` block is a tool result rather than a query, and
when no real query exists the index stays past the end so no think block is synthesised.
"""
function _qwen3_last_query_index(entries::Vector{_Qwen3ChatEntry})
    for index in length(entries):-1:1
        entry = entries[index]
        entry.role == "user" && entry.content_is_string || continue
        startswith(entry.content, "<tool_response>") &&
            endswith(entry.content, "</tool_response>") && continue
        return index
    end
    return length(entries)
end

function _qwen3_tool_call_fields(call)
    inner = _hf_message_value(call, :function; default=nothing)
    target = inner === nothing ? call : inner
    name = _hf_message_value(target, :name; required=true)
    name isa AbstractString || throw(ArgumentError("tool call name must be a string"))
    arguments = _hf_message_value(target, :arguments; default=nothing)
    arguments === nothing && throw(ArgumentError("tool call is missing `arguments`"))
    return String(name), arguments
end

const _QWEN3_TOOL_HEADER = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>"
const _QWEN3_TOOL_FOOTER = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n"

"""
    apply_qwen3_chat_template(tokenizer, messages; tools=nothing,
                              add_generation_prompt=true, enable_thinking=true)

Render the official Qwen3 chat template: system/user/assistant turns, the `# Tools`
header, assistant `<tool_call>` blocks and `tool`-role `<tool_response>` turns. The
output is byte-for-byte identical to the frozen HuggingFace Jinja template.

`tools` accepts order-preserving JSON values only (`NamedTuple`, `Vector` of `Pair`s or
a parsed `JSON3.Object`) because the official `tojson` filter emits keys in insertion
order. Passing tools, `tool` messages or assistant `tool_calls` requires the official
Qwen3 `chat_template` revision and fails closed against any other template.
"""
function apply_qwen3_chat_template(
    tokenizer::HFQwen3Tokenizer,
    messages;
    tools=nothing,
    add_generation_prompt::Bool=true,
    enable_thinking::Bool=true,
)
    message_list = collect(messages)
    isempty(message_list) && throw(ArgumentError("chat messages must not be empty"))
    entries = _Qwen3ChatEntry[_qwen3_chat_entry(message) for message in message_list]
    # A bare NamedTuple or object would iterate over its *values* and render one bogus
    # declaration per field, so a single tool has to be wrapped explicitly.
    (tools isa NamedTuple || tools isa AbstractDict) && throw(ArgumentError(
        "tools must be a list of tool declarations; wrap a single tool in a vector",
    ))
    tool_list = tools === nothing ? Any[] : collect(Any, tools)
    uses_tool_protocol = !isempty(tool_list) ||
        any(entry -> entry.role == "tool" || !isempty(entry.tool_calls), entries)
    uses_tool_protocol && _sha256_hex(tokenizer.chat_template) != _QWEN3_CHAT_TEMPLATE_SHA256 &&
        throw(ArgumentError(
            "Qwen3 tools, tool messages and tool_calls require the official Qwen3 chat_template revision",
        ))

    total = length(entries)
    last_query = _qwen3_last_query_index(entries)
    output = IOBuffer()
    leading_system = entries[1].role == "system"
    if !isempty(tool_list)
        print(output, "<|im_start|>system\n")
        leading_system && print(output, entries[1].content, "\n\n")
        print(output, _QWEN3_TOOL_HEADER)
        for tool in tool_list
            print(output, "\n")
            _python_json(output, tool)
        end
        print(output, _QWEN3_TOOL_FOOTER)
    elseif leading_system
        print(output, "<|im_start|>system\n", entries[1].content, "<|im_end|>\n")
    end

    for index in 1:total
        entry = entries[index]
        role = entry.role
        if role == "user" || (role == "system" && index != 1)
            print(output, "<|im_start|>", role, "\n", entry.content, "<|im_end|>\n")
        elseif role == "assistant"
            reasoning, content = _qwen3_assistant_split(entry)
            if index > last_query && (index == total || !isempty(reasoning))
                print(
                    output,
                    "<|im_start|>assistant\n<think>\n",
                    _strip_newlines(reasoning),
                    "\n</think>\n\n",
                    _strip_newlines(content; right=false),
                )
            else
                print(output, "<|im_start|>assistant\n", content)
            end
            for (position, call) in enumerate(entry.tool_calls)
                (position > 1 || !isempty(content)) && print(output, "\n")
                name, arguments = _qwen3_tool_call_fields(call)
                print(output, "<tool_call>\n{\"name\": \"", name, "\", \"arguments\": ")
                if arguments isa AbstractString
                    print(output, arguments)
                else
                    _python_json(output, arguments)
                end
                print(output, "}\n</tool_call>")
            end
            print(output, "<|im_end|>\n")
        elseif role == "tool"
            (index == 1 || entries[index - 1].role != "tool") &&
                print(output, "<|im_start|>user")
            print(output, "\n<tool_response>\n", entry.content, "\n</tool_response>")
            (index == total || entries[index + 1].role != "tool") &&
                print(output, "<|im_end|>\n")
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
