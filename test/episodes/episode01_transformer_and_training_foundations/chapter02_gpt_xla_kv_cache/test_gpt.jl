using Test
using Random
using Lux
using LifeAI: GPTModel

@testset "GPTModel forward" begin
    rng = MersenneTwister(20260710)

    vocab_size = 128
    d_model = 64
    num_heads = 4
    num_layers = 2
    seq_len = 16
    batch_size = 3

    model = GPTModel(
        vocab_size,
        d_model,
        num_heads,
        num_layers;
        max_seq_len=32,
        use_rope=true,
        use_bias=false,
    )

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)

    tokens = rand(rng, 1:vocab_size, seq_len, batch_size)
    logits, st_new = model(tokens, ps, st)

    @test size(logits) == (vocab_size, seq_len, batch_size)
    @test eltype(logits) <: AbstractFloat
    @test all(isfinite, logits)

    @test model.vocab_size == vocab_size
    @test model.d_model == d_model
    @test model.num_heads == num_heads
    @test model.num_layers == num_layers
    @test model.max_seq_len == 32
    @test model.use_rope == true

    @test haskey(st_new, :token_embedding)
    @test haskey(st_new, :blocks)
    @test haskey(st_new, :final_norm)
    @test haskey(st_new, :lm_head)

    @test length(keys(st_new.blocks)) == num_layers
end

@testset "GPTModel supports different batch sizes" begin
    rng = MersenneTwister(20260711)

    model = GPTModel(64, 32, 4, 2; max_seq_len=16)
    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)

    for batch_size in (1, 2, 5)
        tokens = rand(rng, 1:64, 8, batch_size)
        logits, _ = model(tokens, ps, st)

        @test size(logits) == (64, 8, batch_size)
        @test all(isfinite, logits)
    end
end

@testset "GPTModel without RoPE" begin
    rng = MersenneTwister(20260712)

    model = GPTModel(
        32,
        24,
        3,
        1;
        use_rope=false,
        max_seq_len=12,
    )

    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)

    tokens = rand(rng, 1:32, 10, 2)
    logits, _ = model(tokens, ps, st)

    @test size(logits) == (32, 10, 2)
    @test all(isfinite, logits)
    @test model.use_rope == false
end

@testset "GPTModel input checks" begin
    rng = MersenneTwister(20260713)

    model = GPTModel(32, 32, 4, 2; max_seq_len=8)
    ps = Lux.initialparameters(rng, model)
    st = Lux.initialstates(rng, model)

    tokens_too_long = rand(rng, 1:32, 9, 2)
    tokens_bad_rank = rand(rng, 1:32, 8)
    tokens_bad_low = fill(0, 8, 2)
    tokens_bad_high = fill(33, 8, 2)
    tokens_float = Float32.(rand(rng, 1:32, 8, 2))

    @test_throws AssertionError model(tokens_too_long, ps, st)
    @test_throws AssertionError model(tokens_bad_rank, ps, st)
    @test_throws AssertionError model(tokens_bad_low, ps, st)
    @test_throws AssertionError model(tokens_bad_high, ps, st)
    @test_throws AssertionError model(tokens_float, ps, st)
end

@testset "GPTModel constructor checks" begin
    @test_throws AssertionError GPTModel(0, 32, 4, 2)
    @test_throws AssertionError GPTModel(32, 0, 4, 2)
    @test_throws AssertionError GPTModel(32, 32, 0, 2)
    @test_throws AssertionError GPTModel(32, 32, 4, 0)
    @test_throws AssertionError GPTModel(32, 32, 4, 2; max_seq_len=0)
end
