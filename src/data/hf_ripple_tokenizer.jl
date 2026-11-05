"""Ripple byte BPE tokenizer. Wraps the shared byte/merge engine with Ripple metadata and splitting rules."""
struct HFRippleTokenizer <: AbstractTokenizer
    bpe::HFFlintTokenizer
    pattern::Regex
    template_kind::Symbol
end

vocab_size(t::HFRippleTokenizer) = vocab_size(t.bpe)
special_token_id(t::HFRippleTokenizer, name) = special_token_id(t.bpe, name)
decode_bytes(t::HFRippleTokenizer, ids; kwargs...) = decode_bytes(t.bpe, ids; kwargs...)
decode(t::HFRippleTokenizer, ids; kwargs...) = decode(t.bpe, ids; kwargs...)

const _RIPPLE_CHAT_TEMPLATE_SHA256 = "877c2dbf365243dda14738ca250b31398059b35ffd91d4ed0ec68be45ca6ebd3"
const _RIPPLE_SIMPLE_TEMPLATE_SHA256 = "5958bf5c3ae31d278e5e65f37ea7119df04b5d56564d8fd802a8c6692567db17"

function load_hf_ripple_tokenizer(path::AbstractString)
    object = _hf_json(read(joinpath(path, "tokenizer.json"), String), "Ripple tokenizer.json")
    _hf_reject_duplicate_fields(object, "Ripple tokenizer.json")
    for key in ("normalizer", "post_processor", "padding", "truncation")
        get(object, key, nothing) === nothing || throw(ArgumentError("unsupported Ripple tokenizer $key"))
    end
    model = object["model"]
    get(model, "type", nothing) == "BPE" || throw(ArgumentError("Ripple requires BPE"))
    for key in ("dropout", "unk_token", "continuing_subword_prefix", "end_of_word_suffix")
        get(model, key, nothing) in (nothing, "") || throw(ArgumentError("unsupported Ripple BPE $key"))
    end
    for key in ("fuse_unk", "byte_fallback", "ignore_merges")
        get(model, key, false) === false || throw(ArgumentError("unsupported Ripple BPE $key"))
    end
    vocabulary = Dict{String,Int}(String(token) => _hf_one_based_token_id(id, "Ripple token ID") for (token, id) in pairs(model["vocab"]))
    length(vocabulary) > 0 && Set(values(vocabulary)) == Set(1:length(vocabulary)) ||
        throw(ArgumentError("Ripple BPE vocabulary must have contiguous unique IDs"))
    alphabet = hf_byte_unicode_alphabet().char_to_byte
    all(ch -> haskey(vocabulary, string(ch)), keys(alphabet)) || throw(ArgumentError("Ripple vocabulary must cover every byte"))
    added = HFAddedToken[]
    specials = Set{Int}()
    added_ids = Set{Int}()
    added_contents = Set{String}()
    for entry in object["added_tokens"]
        token = HFAddedToken(_hf_one_based_token_id(entry["id"], "Ripple added ID"),
            entry["content"], entry["single_word"], entry["lstrip"], entry["rstrip"], entry["normalized"], entry["special"])
        token.id in added_ids && throw(ArgumentError("duplicate Ripple added token ID"))
        token.content in added_contents && throw(ArgumentError("duplicate Ripple added token"))
        if haskey(vocabulary, token.content)
            vocabulary[token.content] == token.id || throw(ArgumentError("Ripple added token ID conflicts with BPE vocabulary"))
        else
            token.id in values(vocabulary) && throw(ArgumentError("Ripple added token ID already in use"))
            vocabulary[token.content] = token.id
        end
        push!(added, token)
        push!(added_ids, token.id)
        push!(added_contents, token.content)
        token.special && push!(specials, token.id)
    end
    Set(values(vocabulary)) == Set(1:length(vocabulary)) || throw(ArgumentError("Ripple vocabulary IDs must be contiguous"))
    tokens = Vector{String}(undef, length(vocabulary))
    bytes = Vector{Vector{UInt8}}(undef, length(vocabulary))
    for (token, id) in vocabulary
        tokens[id] = token
        bytes[id] = id in added_ids ? collect(codeunits(token)) : UInt8[alphabet[ch] for ch in token]
    end
    ranks = Dict{Tuple{String,String},Int}()
    for (rank, pair) in enumerate(model["merges"])
        pair isa AbstractVector && length(pair) == 2 || throw(ArgumentError("Ripple merge must contain two strings"))
        key = (String(pair[1]), String(pair[2]))
        haskey(ranks, key) && throw(ArgumentError("duplicate Ripple merge"))
        all(t -> haskey(vocabulary, t), (key..., key[1]*key[2])) || throw(ArgumentError("Ripple merge token absent from vocabulary"))
        ranks[key] = rank
    end
    pre = object["pre_tokenizer"]
    pre["type"] == "Sequence" && length(pre["pretokenizers"]) == 2 || throw(ArgumentError("unsupported Ripple pre-tokenizer"))
    split, bytelevel = pre["pretokenizers"]
    split["type"] == "Split" && split["behavior"] == "Removed" && split["invert"] === true ||
        throw(ArgumentError("Ripple requires inverted Removed splitting"))
    bytelevel["type"] == "ByteLevel" && bytelevel["add_prefix_space"] === false && bytelevel["use_regex"] === false ||
        throw(ArgumentError("unsupported Ripple byte pre-tokenizer"))
    object["decoder"]["type"] == "ByteLevel" || throw(ArgumentError("unsupported Ripple decoder"))
    pattern = _flint_regex(split["pattern"]["Regex"])
    config = _hf_json(read(joinpath(path, "tokenizer_config.json"), String), "Ripple tokenizer config")
    get(config, "add_bos_token", false) === false && get(config, "add_eos_token", false) === false ||
        throw(ArgumentError("Ripple automatic BOS/EOS is unsupported"))
    bos = vocabulary[String(config["bos_token"])]
    eos = vocabulary[String(config["eos_token"])]
    pad = vocabulary[String(config["pad_token"])]
    model_file = joinpath(path, "config.json")
    metadata = isfile(model_file) ? _hf_json(read(model_file, String), "Ripple config") : nothing
    eos_ids = metadata !== nothing && haskey(metadata, "eos_token_id") ?
        [_hf_one_based_token_id(id, "Ripple EOS") for id in (metadata["eos_token_id"] isa Integer ? [metadata["eos_token_id"]] : metadata["eos_token_id"])] : [eos]
    !isempty(eos_ids) && all(id -> 1 <= id <= length(tokens), eos_ids) || throw(ArgumentError("invalid Ripple EOS IDs"))
    template_file = joinpath(path, "chat_template.jinja")
    template = isfile(template_file) ? read(template_file, String) : String(get(config, "chat_template", ""))
    digest = bytes2hex(sha256(template))
    kind = digest == _RIPPLE_CHAT_TEMPLATE_SHA256 ? :tools : digest == _RIPPLE_SIMPLE_TEMPLATE_SHA256 ? :simple : :unknown
    bpe = HFFlintTokenizer(vocabulary, tokens, bytes, ranks, added, specials, Regex[], bos, eos_ids, pad, template)
    return HFRippleTokenizer(bpe, pattern, kind)
end

function _ripple_encode_text!(ids, tokenizer, text)
    # Removed + invert=true retains only regex matches, unlike Flint's Isolated
    # split, which also retains unmatched gaps. Do not normalize Unicode.
    for match in eachmatch(tokenizer.pattern, String(text))
        isempty(match.match) && throw(ArgumentError("Ripple split pattern must not match empty text"))
        for token in _flint_bpe(tokenizer.bpe, String(match.match))
            push!(ids, tokenizer.bpe.vocabulary[token])
        end
    end
    return ids
end

function encode(t::HFRippleTokenizer, input::AbstractString; add_special_tokens::Bool=false)
    text = String(input)
    ids = Int[]
    cursor = firstindex(text)
    while cursor <= ncodeunits(text)
        token, range = _hf_next_added_match(text, cursor, t.bpe.added_tokens)
        if range === nothing
            _ripple_encode_text!(ids, t, SubString(text, cursor))
            break
        end
        first(range) > cursor && _ripple_encode_text!(ids, t, SubString(text, cursor, prevind(text, first(range))))
        push!(ids, token.id)
        cursor = nextind(text, last(range))
    end
    return ids
end

# Transformers installs CPython json.dumps(ensure_ascii=false) as tojson.
# Reuse the shared serializer: callers can preserve key order and numeric types
# with NamedTuple / OrderedJSONObject or parse_qwen3_json for JSON payloads.
_ripple_tojson(value) = _python_json_text(value)

_ripple_message(m, key, default=nothing) = _hf_message_value(m, key; default)
function _ripple_content(message)
    content = _ripple_message(message, :content, "")
    content === nothing && return ""
    content isa AbstractString || throw(ArgumentError("Ripple content must be a string"))
    return String(content)
end

"""Render the published Ripple chat template, including tool calls/responses and reasoning history."""
function apply_ripple_chat_template(t::HFRippleTokenizer, messages; add_generation_prompt::Bool=true,
        tokenize::Bool=false, tools=nothing, clear_thinking::Bool=false)
    t.template_kind === :unknown && throw(ArgumentError("unrecognized Ripple chat template"))
    messages = collect(messages)
    (tools isa NamedTuple || tools isa AbstractDict) && throw(ArgumentError("Ripple tools must be a list of declarations"))
    io = IOBuffer()
    if t.template_kind === :simple
        (tools === nothing || isempty(tools)) || throw(ArgumentError("this Ripple template does not support tools"))
        for m in messages
            print(io, "<|im_start|>", _ripple_message(m, :role), '\n', _ripple_content(m), "<|im_end|>\n")
        end
    else
        first_system = !isempty(messages) && _ripple_message(first(messages), :role) == "system"
        if tools !== nothing && !isempty(tools)
            print(io, "<|im_start|>system\n")
            first_system && print(io, _ripple_content(first(messages)), "\n\n")
            print(io, "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools> XML tags:\n<tools>")
            for tool in tools
                print(io, '\n', _ripple_tojson(tool))
            end
            print(io, "\n</tools>\n\nFor each function call, return exactly one valid JSON object with function name and arguments inside XML tool-call tags. The arguments field must be a JSON object, not a string, and strings must use valid JSON escaping:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": {\"command\": \"pwd\"}}\n</tool_call><|im_end|>\n")
        elseif first_system
            print(io, "<|im_start|>system\n", _ripple_content(first(messages)), "<|im_end|>\n")
        end
        for (i, m) in enumerate(messages)
            role = _ripple_message(m, :role)
            role in ("system", "user", "assistant", "tool") || throw(ArgumentError("unsupported Ripple role $role"))
            content = _ripple_content(m)
            reasoning = _ripple_message(m, :reasoning_content, "")
            if role == "assistant" && reasoning !== nothing && !isempty(reasoning) && !clear_thinking && !startswith(lstrip(content), "<think>")
                reasoning isa AbstractString || throw(ArgumentError("Ripple reasoning_content must be a string"))
                content = "<think>\n" * reasoning * "\n</think>\n" * content
            end
            calls = _ripple_message(m, :tool_calls)
            if role == "user" || (role == "system" && i != 1) || (role == "assistant" && (calls === nothing || isempty(calls)))
                print(io, "<|im_start|>", role, '\n', content, "<|im_end|>\n")
            elseif role == "assistant"
                print(io, "<|im_start|>assistant")
                isempty(content) || print(io, '\n', content)
                for call in calls
                    call = _ripple_message(call, :function, call)
                    print(io, "\n<tool_call>\n{\"name\": ", _ripple_tojson(_ripple_message(call, :name)),
                        ", \"arguments\": ", _ripple_tojson(_ripple_message(call, :arguments)), "}\n</tool_call>")
                end
                print(io, "<|im_end|>\n")
            elseif role == "tool"
                (i == 1 || _ripple_message(messages[i-1], :role) != "tool") && print(io, "<|im_start|>user")
                print(io, "\n<tool_response>\n", content, "\n</tool_response>")
                (i == length(messages) || _ripple_message(messages[i+1], :role) != "tool") && print(io, "<|im_end|>\n")
            end
        end
    end
    add_generation_prompt && print(io, "<|im_start|>assistant\n")
    text = String(take!(io))
    return tokenize ? encode(t, text) : text
end
