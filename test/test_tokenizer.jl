using Test
using LifeAI: Tokenizer, fit_tokenizer, encode, decode, vocab_size

@testset "character tokenizer round trip" begin
    text = "你好，世界。\n你好，GPT！"
    tokenizer = fit_tokenizer(text)

    ids = encode(tokenizer, text)

    @test tokenizer isa Tokenizer
    @test eltype(ids) == Int
    @test length(ids) == length(text)
    @test all(id -> 1 <= id <= vocab_size(tokenizer), ids)
    @test decode(tokenizer, ids) == text
    @test length(tokenizer) == vocab_size(tokenizer)
    @test '你' in tokenizer
    @test !('未' in tokenizer)
end

@testset "character tokenizer vocabulary is deterministic" begin
    tokenizer_a = fit_tokenizer("cabca")
    tokenizer_b = fit_tokenizer("abc")

    @test tokenizer_a.id_to_token == ['a', 'b', 'c']
    @test tokenizer_a.id_to_token == tokenizer_b.id_to_token
    @test tokenizer_a.token_to_id == tokenizer_b.token_to_id
end

@testset "character tokenizer unknown handling" begin
    tokenizer_strict = fit_tokenizer("abc")

    @test_throws ArgumentError encode(tokenizer_strict, "abd")

    tokenizer_with_unk = fit_tokenizer("abc"; add_unk=true)

    ids = encode(tokenizer_with_unk, "abd")

    @test tokenizer_with_unk.unk_id == 1
    @test ids[end] == tokenizer_with_unk.unk_id
    @test decode(tokenizer_with_unk, ids) == "ab�"
end

@testset "character tokenizer invalid inputs" begin
    @test_throws AssertionError fit_tokenizer("")

    tokenizer = fit_tokenizer("abc")

    @test_throws ArgumentError decode(tokenizer, [0])
    @test_throws ArgumentError decode(tokenizer, [vocab_size(tokenizer) + 1])
    @test_throws ArgumentError decode(tokenizer, [1.5])
end
