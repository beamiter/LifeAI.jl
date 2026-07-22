using Test
using Random: Xoshiro
using Lux
using LifeAI:
    GPTModel,
    XLAKVDecoder,
    decode,
    encode,
    generate_xla_cached!,
    load_hf_qwen3_tokenizer,
    vocab_size

isdefined(@__MODULE__, :week08_tokenizer_payloads) || include("week08_fixture.jl")

@testset "Week 08 host tokenizer to XLA static generation" begin
    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_week08_tokenizer_fixture(directory))
        model = GPTModel(
            vocab_size(tokenizer),
            8,
            2,
            1;
            max_seq_len=8,
            use_rope=true,
            norm_type=:rmsnorm,
            mlp_type=:swiglu,
            tie_embeddings=true,
        )
        parameters, states = Lux.setup(Xoshiro(810), model)
        decoder = XLAKVDecoder(model, parameters, states; xla_backend="cpu")
        prompt_ids = encode(tokenizer, "hi")
        generated, _ = generate_xla_cached!(
            decoder,
            prompt_ids;
            max_new_tokens=2,
            temperature=0,
        )
        @test prompt_ids == [257]
        @test length(generated) == 3
        @test decode(tokenizer, generated; errors=:replace) isa String
    end
end
