using JSON3
using LifeAI: hf_byte_unicode_alphabet

const WEEK08_FIXTURE_REGEX = raw"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

function week08_tokenizer_payloads()
    alphabet = hf_byte_unicode_alphabet().byte_to_char
    vocabulary = Dict(string(alphabet[UInt8(byte)]) => byte for byte in 0:255)
    vocabulary["hi"] = 256
    vocabulary["hi!"] = 257
    merges = Any[Any["h", "i"], Any["hi", "!"]]
    added_specs = [
        (258, "<|endoftext|>", true),
        (259, "<|im_start|>", true),
        (260, "<|im_end|>", true),
        (261, "<think>", false),
        (262, "</think>", false),
    ]
    added_tokens = Any[
        Dict(
            "id" => id,
            "content" => content,
            "single_word" => false,
            "lstrip" => false,
            "rstrip" => false,
            "normalized" => false,
            "special" => special,
        ) for (id, content, special) in added_specs
    ]
    tokenizer = Dict{String,Any}(
        "version" => "1.0",
        "truncation" => nothing,
        "padding" => nothing,
        "added_tokens" => added_tokens,
        "normalizer" => Dict("type" => "NFC"),
        "pre_tokenizer" => Dict(
            "type" => "Sequence",
            "pretokenizers" => Any[
                Dict(
                    "type" => "Split",
                    "pattern" => Dict("Regex" => WEEK08_FIXTURE_REGEX),
                    "behavior" => "Isolated",
                    "invert" => false,
                ),
                Dict(
                    "type" => "ByteLevel",
                    "add_prefix_space" => false,
                    "trim_offsets" => false,
                    "use_regex" => false,
                ),
            ],
        ),
        "post_processor" => Dict(
            "type" => "ByteLevel",
            "add_prefix_space" => false,
            "trim_offsets" => false,
            "use_regex" => false,
        ),
        "decoder" => Dict(
            "type" => "ByteLevel",
            "add_prefix_space" => false,
            "trim_offsets" => false,
            "use_regex" => false,
        ),
        "model" => Dict(
            "type" => "BPE",
            "dropout" => nothing,
            "unk_token" => nothing,
            "continuing_subword_prefix" => "",
            "end_of_word_suffix" => "",
            "fuse_unk" => false,
            "byte_fallback" => false,
            "ignore_merges" => false,
            "vocab" => vocabulary,
            "merges" => merges,
        ),
    )
    added_decoder = Dict{String,Any}()
    for (id, content, special) in added_specs
        added_decoder[string(id)] = Dict(
            "content" => content,
            "single_word" => false,
            "lstrip" => false,
            "rstrip" => false,
            "normalized" => false,
            "special" => special,
        )
    end
    tokenizer_config = Dict{String,Any}(
        "add_bos_token" => false,
        "add_prefix_space" => false,
        "added_tokens_decoder" => added_decoder,
        "additional_special_tokens" => ["<|im_start|>", "<|im_end|>"],
        "bos_token" => nothing,
        "chat_template" => "week08-fixture-template",
        "clean_up_tokenization_spaces" => false,
        "eos_token" => "<|im_end|>",
        "errors" => "replace",
        "model_max_length" => 128,
        "pad_token" => "<|endoftext|>",
        "split_special_tokens" => false,
        "tokenizer_class" => "Qwen2Tokenizer",
        "unk_token" => nothing,
    )
    generation_config = Dict{String,Any}(
        "bos_token_id" => 258,
        "eos_token_id" => [260, 258],
        "pad_token_id" => 258,
    )
    return (; tokenizer, tokenizer_config, generation_config)
end

function write_week08_tokenizer_fixture(directory; payloads=week08_tokenizer_payloads())
    mkpath(directory)
    for (name, payload) in pairs(payloads)
        filename = name === :tokenizer ? "tokenizer.json" :
            name === :tokenizer_config ? "tokenizer_config.json" :
            "generation_config.json"
        open(joinpath(directory, filename), "w") do io
            JSON3.write(io, payload)
        end
    end
    return directory
end
