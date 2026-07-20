using Test
using Random
using LifeAI: ByteTokenizer, decode, decode_bytes, encode, fit_byte_bpe

@testset "Week 05 deterministic random valid UTF-8 round trips" begin
    rng = Xoshiro(20260723)
    alphabet = collect("abcXYZ 生命感观察记忆反馈行动🐕🤖✨éΩЖあ한e\u0301\n")
    texts = String[]
    for _ in 1:64
        sample_length = rand(rng, 0:96)
        push!(texts, join(rand(rng, alphabet, sample_length)))
    end

    byte = ByteTokenizer()
    bpe = fit_byte_bpe(
        vcat(texts, [repeat("生命感来自观察、记忆、反馈和行动。", 32)]);
        vocab_size=288,
        min_frequency=2,
    )
    for tokenizer in (byte, bpe)
        for text in texts
            ids = encode(tokenizer, text)
            @test decode_bytes(tokenizer, ids) == Vector{UInt8}(codeunits(text))
            @test decode(tokenizer, ids) == text
        end
    end
end
