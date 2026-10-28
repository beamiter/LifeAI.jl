"""Native byte-level BPE tokenizer for the released Flint tokenizer.json (one-based LifeAI IDs)."""
struct HFFlintTokenizer <: AbstractTokenizer
    vocabulary::Dict{String,Int}
    id_to_token::Vector{String}
    token_bytes::Vector{Vector{UInt8}}
    merge_ranks::Dict{Tuple{String,String},Int}
    added_tokens::Vector{HFAddedToken}
    special_ids::Set{Int}
    patterns::Vector{Regex}
    bos_id::Int
    eos_ids::Vector{Int}
    pad_id::Int
    chat_template::String
end

vocab_size(tokenizer::HFFlintTokenizer) = length(tokenizer.id_to_token)

# Oniguruma accepts \uXXXX in regexes; translate those escapes to literal
# characters for PCRE while preserving all other escapes exactly as published.
_flint_regex(pattern) = Regex(replace(String(pattern), r"(?<!\\)\\u([0-9A-Fa-f]{4})" =>
    text -> string(Char(parse(UInt32, text[3:end]; base=16)))))

function load_hf_flint_tokenizer(path::AbstractString)
    object = _hf_json(read(joinpath(path, "tokenizer.json"), String), "Flint tokenizer.json")
    get(object, "normalizer", nothing) === nothing || throw(ArgumentError("Flint tokenizer normalizer is unsupported"))
    get(object, "post_processor", nothing) === nothing || throw(ArgumentError("Flint tokenizer post_processor is unsupported"))
    get(object, "padding", nothing) === nothing || throw(ArgumentError("Flint tokenizer padding is unsupported"))
    get(object, "truncation", nothing) === nothing || throw(ArgumentError("Flint tokenizer truncation is unsupported"))
    model = object["model"]
    get(model, "type", nothing) == "BPE" || throw(ArgumentError("Flint requires a BPE tokenizer"))
    for key in ("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix")
        get(model, key, nothing) in (nothing, "") || throw(ArgumentError("unsupported Flint BPE $key"))
    end
    for key in ("fuse_unk", "byte_fallback", "ignore_merges")
        get(model, key, false) === false || throw(ArgumentError("unsupported Flint BPE $key"))
    end
    alphabet = hf_byte_unicode_alphabet().char_to_byte
    vocabulary = Dict{String,Int}()
    for (token, raw_id) in pairs(model["vocab"])
        id = _hf_one_based_token_id(raw_id, "BPE token id")
        vocabulary[String(token)] = id
    end
    count = length(vocabulary)
    Set(values(vocabulary)) == Set(1:count) || throw(ArgumentError("Flint BPE vocabulary must be contiguous and unique"))
    all(ch -> haskey(vocabulary, string(ch)), keys(alphabet)) || throw(ArgumentError("Flint vocabulary must cover all bytes"))
    merges = Dict{Tuple{String,String},Int}()
    for (rank, pair) in enumerate(model["merges"])
        pair isa AbstractVector && length(pair) == 2 || throw(ArgumentError("Flint merge must contain two strings"))
        key = (String(pair[1]), String(pair[2]))
        haskey(merges, key) && throw(ArgumentError("duplicate Flint BPE merge"))
        all(t -> haskey(vocabulary, t), (key..., key[1] * key[2])) || throw(ArgumentError("Flint BPE merge token absent from vocabulary"))
        merges[key] = rank
    end
    added = HFAddedToken[]
    specials = Set{Int}()
    for entry in object["added_tokens"]
        token = HFAddedToken(_hf_one_based_token_id(entry["id"], "added token ID"),
            entry["content"], entry["single_word"], entry["lstrip"], entry["rstrip"], entry["normalized"], entry["special"])
        haskey(vocabulary, token.content) && throw(ArgumentError("Flint added token overlaps BPE vocabulary"))
        token.id in values(vocabulary) && throw(ArgumentError("duplicate Flint added token ID"))
        vocabulary[token.content] = token.id
        push!(added, token)
        token.special && push!(specials, token.id)
    end
    total = length(vocabulary)
    Set(values(vocabulary)) == Set(1:total) || throw(ArgumentError("Flint added token IDs must be contiguous"))
    tokens = Vector{String}(undef, total)
    bytes = Vector{Vector{UInt8}}(undef, total)
    added_ids = Set(t.id for t in added)
    for (token, id) in vocabulary
        tokens[id] = token
        bytes[id] = id in added_ids ? collect(codeunits(token)) : UInt8[alphabet[ch] for ch in token]
    end
    pretokenizer = object["pre_tokenizer"]
    pretokenizer["type"] == "Sequence" || throw(ArgumentError("Flint requires a sequence pre-tokenizer"))
    components = pretokenizer["pretokenizers"]
    !isempty(components) || throw(ArgumentError("empty Flint pre-tokenizer"))
    patterns = Regex[]
    for component in components[1:end-1]
        component["type"] == "Split" && component["behavior"] == "Isolated" && component["invert"] === false ||
            throw(ArgumentError("unsupported Flint Split pre-tokenizer"))
        push!(patterns, _flint_regex(component["pattern"]["Regex"]))
    end
    bytelevel = components[end]
    bytelevel["type"] == "ByteLevel" && bytelevel["add_prefix_space"] === false && bytelevel["use_regex"] === false ||
        throw(ArgumentError("unsupported Flint ByteLevel pre-tokenizer"))
    object["decoder"]["type"] == "ByteLevel" || throw(ArgumentError("unsupported Flint tokenizer decoder"))
    config = _hf_json(read(joinpath(path, "tokenizer_config.json"), String), "Flint tokenizer_config.json")
    get(config, "add_bos_token", false) === false && get(config, "add_eos_token", false) === false ||
        throw(ArgumentError("Flint tokenizer must not automatically add BOS/EOS"))
    bos = vocabulary[String(config["bos_token"])]
    eos = vocabulary[String(config["eos_token"])]
    generation_file = joinpath(path, "generation_config.json")
    generation = isfile(generation_file) ? _hf_json(read(generation_file, String), "Flint generation config") : nothing
    eos_ids = if generation !== nothing && haskey(generation, "eos_token_id")
        ids = generation["eos_token_id"]
        [_hf_one_based_token_id(i, "eos_token_id") for i in (ids isa Integer ? [ids] : ids)]
    else
        [eos]
    end
    pad = generation !== nothing && haskey(generation, "pad_token_id") ?
        _hf_one_based_token_id(generation["pad_token_id"], "pad_token_id") : eos
    all(i -> 1 <= i <= total, [bos; pad; eos_ids]) || throw(ArgumentError("Flint tokenizer special IDs out of range"))
    template_file = joinpath(path, "chat_template.jinja")
    template = isfile(template_file) ? read(template_file, String) : String(get(config, "chat_template", ""))
    return HFFlintTokenizer(vocabulary, tokens, bytes, merges, added, specials, patterns, bos, eos_ids, pad, template)
end

function _flint_split(pattern::Regex, text::String)
    isempty(text) && return String[]
    pieces = String[]
    cursor = firstindex(text)
    for match in eachmatch(pattern, text)
        isempty(match.match) && throw(ArgumentError("Flint split regex must not match empty text"))
        match.offset > cursor && push!(pieces, String(SubString(text, cursor, prevind(text, match.offset))))
        push!(pieces, String(match.match))
        cursor = match.offset + ncodeunits(match.match)
    end
    cursor <= ncodeunits(text) && push!(pieces, String(SubString(text, cursor)))
    return pieces
end

function _flint_bpe(tokenizer::HFFlintTokenizer, piece::String)
    symbols = string.(collect(_hf_byte_symbols(piece)))
    while length(symbols) > 1
        best_rank = typemax(Int)
        best = nothing
        for index in 1:length(symbols)-1
            pair = (symbols[index], symbols[index+1])
            rank = get(tokenizer.merge_ranks, pair, typemax(Int))
            if rank < best_rank
                best, best_rank = pair, rank
            end
        end
        best === nothing && break
        merged = String[]
        i = 1
        while i <= length(symbols)
            if i < length(symbols) && (symbols[i], symbols[i+1]) == best
                push!(merged, symbols[i] * symbols[i+1])
                i += 2
            else
                push!(merged, symbols[i])
                i += 1
            end
        end
        symbols = merged
    end
    return symbols
end

function _flint_encode_text!(ids, tokenizer, text)
    pieces = [String(text)]
    for pattern in tokenizer.patterns
        pieces = reduce(vcat, (_flint_split(pattern, piece) for piece in pieces); init=String[])
    end
    for piece in pieces, token in _flint_bpe(tokenizer, piece)
        push!(ids, tokenizer.vocabulary[token])
    end
    return ids
end

function encode(tokenizer::HFFlintTokenizer, input::AbstractString; add_special_tokens::Bool=false)
    text = String(input)
    ids = Int[]
    cursor = firstindex(text)
    while cursor <= ncodeunits(text)
        token, range = _hf_next_added_match(text, cursor, tokenizer.added_tokens)
        if range === nothing
            _flint_encode_text!(ids, tokenizer, SubString(text, cursor))
            break
        end
        first(range) > cursor && _flint_encode_text!(ids, tokenizer, SubString(text, cursor, prevind(text, first(range))))
        push!(ids, token.id)
        cursor = nextind(text, last(range))
    end
    return ids
end

function decode_bytes(tokenizer::HFFlintTokenizer, ids; skip_special_tokens::Bool=false)
    bytes = UInt8[]
    for raw_id in ids
        id = _hf_strict_host_int(raw_id, "Flint token ID")
        1 <= id <= vocab_size(tokenizer) || throw(ArgumentError("Flint token ID out of range"))
        skip_special_tokens && id in tokenizer.special_ids && continue
        append!(bytes, tokenizer.token_bytes[id])
    end
    return bytes
end

decode(tokenizer::HFFlintTokenizer, ids; errors::Symbol=:strict, skip_special_tokens::Bool=false) =
    _decode_utf8(decode_bytes(tokenizer, ids; skip_special_tokens); errors)

function special_token_id(tokenizer::HFFlintTokenizer, name)
    name === :bos && return tokenizer.bos_id
    name === :eos && return first(tokenizer.eos_ids)
    name === :pad && return tokenizer.pad_id
    return get(tokenizer.vocabulary, String(name), nothing)
end

function _flint_chat_text(text)
    text isa AbstractString || throw(ArgumentError("Flint chat content must be a string"))
    result = replace(String(text), "\0" => "", "<think>" => "", "</think>" => "")
    for token in ("<|begin_of_text|>", "<|end_of_text|>", "<|system|>", "<|developer|>", "<|user|>", "<|assistant|>", "<|observation|>")
        result = replace(result, token => replace(token, "<" => "&lt;", ">" => "&gt;"))
    end
    return result
end

"""Apply Flint's basic text chat template. Supports system/developer/user/assistant messages and reasoning history; tool schemas/calls require an external renderer."""
function apply_flint_chat_template(tokenizer::HFFlintTokenizer, messages;
    add_generation_prompt::Bool=true, tokenize::Bool=false, tools=nothing)
    (tools === nothing || isempty(tools)) || throw(ArgumentError("Flint tool schema rendering is not implemented"))
    io = IOBuffer()
    for message in messages
        role = String(_hf_message_value(message, :role; required=true))
        role in ("system", "developer", "user", "assistant") || throw(ArgumentError("unsupported Flint chat role $role"))
        calls = _hf_message_value(message, :tool_calls; default=nothing)
        (calls === nothing || isempty(calls)) || throw(ArgumentError("Flint tool call rendering is not implemented"))
        content = _hf_message_value(message, :content; default="")
        content === nothing && (content = "")
        print(io, "<|", role, "|>")
        if role == "assistant"
            reasoning = _hf_message_value(message, :reasoning_content; default="")
            reasoning === nothing && (reasoning = "")
            reasoning isa AbstractString || throw(ArgumentError("Flint reasoning_content must be a string"))
            isempty(reasoning) || print(io, "<think>", reasoning, "</think>")
            content isa AbstractString || throw(ArgumentError("Flint assistant content must be a string"))
            print(io, content)
        else
            print(io, _flint_chat_text(content))
        end
    end
    add_generation_prompt && print(io, "<|assistant|>")
    text = String(take!(io))
    return tokenize ? encode(tokenizer, text) : text
end
