using Test
using JSON3
using SHA: sha256
using LifeAI
using LifeAI:
    apply_qwen3_chat_template,
    encode,
    load_hf_qwen3_tokenizer,
    parse_qwen3_json

if !isdefined(@__MODULE__, :write_qwen3_tokenizer_fixture)
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tokenizer_fixture.jl"))
end

const CHAPTER36_FIXTURES = joinpath(@__DIR__, "fixtures")
const CHAPTER36_OFFICIAL_TEMPLATE_SHA256 =
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"

chapter36_sha256(text::AbstractString) = bytes2hex(sha256(codeunits(text)))

"""Tokenizer fixture that carries the official Qwen3 template with a tiny vocabulary.

The rendering path is pure string work, so the frozen template text is enough to
check byte-level parity offline; token-id parity against the real 151,669-entry
vocabulary stays in the opt-in real-asset test.
"""
function chapter36_official_template_tokenizer(directory, template::AbstractString)
    payloads = qwen3_tokenizer_fixture_payloads()
    payloads.tokenizer_config["chat_template"] = template
    return load_hf_qwen3_tokenizer(write_qwen3_tokenizer_fixture(directory; payloads))
end

@testset "Chapter 36 — official template asset is frozen" begin
    template = read(joinpath(CHAPTER36_FIXTURES, "official_chat_template.jinja"), String)
    @test chapter36_sha256(template) == CHAPTER36_OFFICIAL_TEMPLATE_SHA256

    cases_text = read(joinpath(CHAPTER36_FIXTURES, "chat_template_cases.json"), String)
    reference = JSON3.read(read(joinpath(CHAPTER36_FIXTURES, "chat_template_reference.json"), String))
    @test String(reference.template_sha256) == CHAPTER36_OFFICIAL_TEMPLATE_SHA256
    # The reference is only meaningful for the exact case list it was generated from.
    @test String(reference.cases_sha256) == chapter36_sha256(cases_text)

    # parse_qwen3_json rather than JSON3.read: JSON3 narrows whole-valued floats
    # (1.0 -> 1), which is exactly what the float cases below exist to catch.
    cases = parse_qwen3_json(cases_text)
    @test String(cases["template_sha256"]) == CHAPTER36_OFFICIAL_TEMPLATE_SHA256
    @test length(cases["cases"]) == length(reference.renders)
    @test length(unique(String(case["name"]) for case in cases["cases"])) == length(cases["cases"])
end

@testset "Chapter 36 — HuggingFace Jinja parity" begin
    template = read(joinpath(CHAPTER36_FIXTURES, "official_chat_template.jinja"), String)
    cases = parse_qwen3_json(read(joinpath(CHAPTER36_FIXTURES, "chat_template_cases.json"), String))
    reference = JSON3.read(read(joinpath(CHAPTER36_FIXTURES, "chat_template_reference.json"), String))
    mktempdir() do directory
        tokenizer = chapter36_official_template_tokenizer(directory, template)
        for case in cases["cases"]
            name = String(case["name"])
            expected = String(reference.renders[Symbol(name)])
            rendered = apply_qwen3_chat_template(
                tokenizer,
                case["messages"];
                tools=case["tools"],
                add_generation_prompt=case["add_generation_prompt"],
                enable_thinking=case["enable_thinking"],
            )
            @test rendered == expected
        end
    end
end

@testset "Chapter 36 — Chapter 08 divergences stay fixed" begin
    template = read(joinpath(CHAPTER36_FIXTURES, "official_chat_template.jinja"), String)
    mktempdir() do directory
        tokenizer = chapter36_official_template_tokenizer(directory, template)
        # A historical assistant turn keeps only its visible content: the previous
        # implementation replayed the raw `<think>` block back into the prompt.
        @test apply_qwen3_chat_template(
            tokenizer,
            [
                (role="user", content="U"),
                (role="assistant", content="<think>\nR\n</think>\n\nA"),
                (role="user", content="V"),
            ];
            add_generation_prompt=false,
        ) == "<|im_start|>user\nU<|im_end|>\n" *
             "<|im_start|>assistant\nA<|im_end|>\n" *
             "<|im_start|>user\nV<|im_end|>\n"

        # With no user turn at all `last_query_index` stays past the end, so no
        # think block is synthesised. The previous implementation used index 0.
        @test apply_qwen3_chat_template(
            tokenizer,
            [(role="system", content="S"), (role="assistant", content="A")];
            add_generation_prompt=false,
        ) == "<|im_start|>system\nS<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n"

        # A user turn that is itself a tool response is not a query.
        @test apply_qwen3_chat_template(
            tokenizer,
            [
                (role="user", content="Q"),
                (role="assistant", content="mid"),
                (role="user", content="<tool_response>\n3\n</tool_response>"),
                (role="assistant", content="final"),
            ];
            add_generation_prompt=false,
        ) == "<|im_start|>user\nQ<|im_end|>\n" *
             "<|im_start|>assistant\nmid<|im_end|>\n" *
             "<|im_start|>user\n<tool_response>\n3\n</tool_response><|im_end|>\n" *
             "<|im_start|>assistant\n<think>\n\n</think>\n\nfinal<|im_end|>\n"
    end
end

@testset "Chapter 36 — tool protocol fails closed off the official template" begin
    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_qwen3_tokenizer_fixture(directory))
        tools = [(; type="function", var"function"=(; name="add", description="", parameters=(; type="object")))]
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="user", content="hi")];
            tools,
        )
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="tool", content="3")],
        )
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="assistant", content="", tool_calls=[(; name="add", arguments=(; a=1))])],
        )
        # An empty tool list changes no byte, so it stays allowed.
        @test apply_qwen3_chat_template(
            tokenizer,
            [(role="user", content="hi")];
            tools=[],
            add_generation_prompt=false,
        ) == "<|im_start|>user\nhi<|im_end|>\n"
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="banana", content="hi")],
        )
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="user", content="hi", tool_calls=[(; name="add", arguments=(; a=1))])],
        )
    end
end

@testset "Chapter 36 — CPython-compatible JSON rendering" begin
    render = LifeAI._python_json_text
    @test render((; a=1, b="x")) == "{\"a\": 1, \"b\": \"x\"}"
    @test render(NamedTuple()) == "{}"
    @test render(Any[]) == "[]"
    @test render(String[]) == "[]"
    @test render(["a" => 1, "b" => 2]) == "{\"a\": 1, \"b\": 2}"
    @test render((; flag=true, opt=nothing)) == "{\"flag\": true, \"opt\": null}"
    @test render([1, 2, 3]) == "[1, 2, 3]"
    # ensure_ascii=False keeps non-ASCII literal.
    @test render((; description="读取笔记")) == "{\"description\": \"读取笔记\"}"
    @test render("line\nbreak\ttab\"quote\\slash") == "\"line\\nbreak\\ttab\\\"quote\\\\slash\""
    @test render("\x01\x1f") == "\"\\u0001\\u001f\""
    # Insertion order is preserved through a parsed JSON object.
    @test render(parse_qwen3_json("{\"z\": 1, \"a\": {\"y\": [], \"x\": 2}}")) ==
          "{\"z\": 1, \"a\": {\"y\": [], \"x\": 2}}"
    @test render(parse_qwen3_json("{}")) == "{}"
    @test render(parse_qwen3_json("[]")) == "[]"
    # Byte-stable output is impossible for an unordered Dict.
    @test_throws ArgumentError render(Dict("a" => 1, "b" => 2))
    # JSON3 narrows whole-valued floats, so it is refused outright rather than
    # silently rendering `1` where CPython renders `1.0`.
    @test JSON3.read("{\"a\": 1.0}").a === 1
    @test_throws ArgumentError render(JSON3.read("{\"a\": 1.0}"))
end

@testset "Chapter 36 — CPython float repr" begin
    render = LifeAI._python_json_text
    # Left column is the JSON source, right column is CPython repr of the parsed
    # float. Regenerating fixtures/chat_template_reference.json re-derives these
    # from the oracle; they are pinned here so a refactor cannot drift silently.
    for (source, expected) in [
        ("1.0", "1.0"),
        ("0.5", "0.5"),
        ("100.0", "100.0"),
        ("2.0", "2.0"),
        ("-0.0", "-0.0"),
        ("0.0001", "0.0001"),
        ("1e-5", "1e-05"),
        ("1e-7", "1e-07"),
        ("1e15", "1000000000000000.0"),
        ("1e16", "1e+16"),
        ("1.5", "1.5"),
        ("1.2345678901234567e20", "1.2345678901234567e+20"),
        ("3.141592653589793", "3.141592653589793"),
        ("-2.5e-13", "-2.5e-13"),
    ]
        @test render(parse_qwen3_json(source)) == expected
    end
    # Integers keep their exact digits at any width, matching CPython's arbitrary
    # precision ints; JSON3 would have turned this one into 1.0e23.
    @test render(parse_qwen3_json("99999999999999999999999")) == "99999999999999999999999"
    @test parse_qwen3_json("99999999999999999999999") isa BigInt
    @test parse_qwen3_json("1.0") isa Float64
    @test parse_qwen3_json("1") isa Int
    @test_throws ArgumentError render(NaN)
    @test_throws ArgumentError render(Inf)
end

@testset "Chapter 36 — JSON reader" begin
    @test parse_qwen3_json("{\"a\": [1, {\"b\": null}], \"c\": true}")["c"] === true
    @test parse_qwen3_json("\"\\u4e2d\\u6587\"") == "中文"
    @test parse_qwen3_json("\"\\ud83d\\ude00\"") == "😀"
    @test parse_qwen3_json("  [ 1 , 2 ]  ") == Any[1, 2]
    @test collect(keys(parse_qwen3_json("{\"z\": 1, \"a\": 2}"))) == ["z", "a"]
    for malformed in ("{", "[1,]", "{\"a\" 1}", "tru", "1 2", "\"unterminated", "{\"a\": \"\x01\"}")
        @test_throws ArgumentError parse_qwen3_json(malformed)
    end
end

@testset "Chapter 36 — a single tool must be wrapped in a list" begin
    template = read(joinpath(CHAPTER36_FIXTURES, "official_chat_template.jinja"), String)
    mktempdir() do directory
        tokenizer = chapter36_official_template_tokenizer(directory, template)
        single = (; type="function", var"function"=(; name="add", description="", parameters=(; type="object")))
        # Iterating a bare NamedTuple would walk its values and emit one bogus
        # declaration per field instead of one tool.
        @test_throws ArgumentError apply_qwen3_chat_template(
            tokenizer,
            [(role="user", content="hi")];
            tools=single,
        )
        @test occursin(
            "\"name\": \"add\"",
            apply_qwen3_chat_template(
                tokenizer,
                [(role="user", content="hi")];
                tools=[single],
            ),
        )
    end
end

# Opt-in: the offline parity above compares strings against the frozen reference with
# a tiny vocabulary. Only the real 151,669-entry tokenizer can show that the identical
# string also encodes to the identical token ids.
if haskey(ENV, "LIFEAI_QWEN3_4B_MODEL_DIR")
    @testset "Chapter 36 — real Qwen3-4B tokenizer parity" begin
        tokenizer = load_hf_qwen3_tokenizer(ENV["LIFEAI_QWEN3_4B_MODEL_DIR"])
        @test chapter36_sha256(tokenizer.chat_template) == CHAPTER36_OFFICIAL_TEMPLATE_SHA256

        cases = parse_qwen3_json(read(joinpath(CHAPTER36_FIXTURES, "chat_template_cases.json"), String))
        reference = JSON3.read(read(joinpath(CHAPTER36_FIXTURES, "chat_template_reference.json"), String))
        for case in cases["cases"]
            expected = String(reference.renders[Symbol(String(case["name"]))])
            rendered = apply_qwen3_chat_template(
                tokenizer,
                case["messages"];
                tools=case["tools"],
                add_generation_prompt=case["add_generation_prompt"],
                enable_thinking=case["enable_thinking"],
            )
            @test rendered == expected
            @test encode(tokenizer, rendered; add_special_tokens=false) ==
                  encode(tokenizer, expected; add_special_tokens=false)
        end

        # The tool protocol markers must survive as single tokens, otherwise the
        # loop would have to parse them out of split byte-level pieces.
        for marker in ("<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")
            @test length(encode(tokenizer, marker; add_special_tokens=false)) == 1
        end
    end
end
