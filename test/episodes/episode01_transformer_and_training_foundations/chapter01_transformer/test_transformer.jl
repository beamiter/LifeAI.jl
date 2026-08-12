using Test
using Random
using Lux
using LifeAI: TransformerBlock

@testset "TransformerBlock forward" begin
    rng = MersenneTwister(20260706)

    d_model = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2

    block = TransformerBlock(
        d_model,
        num_heads;
        is_causal=true,
        use_bias=false,
        use_rope=false,
    )

    ps = Lux.initialparameters(rng, block)
    st = Lux.initialstates(rng, block)

    x = randn(rng, Float32, d_model, seq_len, batch_size)

    y, st_new = block(x, ps, st)

    @test size(y) == size(x)
    @test eltype(y) == eltype(x)
    @test all(isfinite, y)

    @test block.d_model == d_model
    @test block.num_heads == num_heads
    @test block.mlp_hidden_dim == 4 * d_model
    @test block.is_causal == true
    @test block.use_rope == false

    @test haskey(st_new, :norm1)
    @test haskey(st_new, :attn)
    @test haskey(st_new, :norm2)
    @test haskey(st_new, :mlp)

    @test haskey(st_new.attn, :q_proj)
    @test haskey(st_new.attn, :k_proj)
    @test haskey(st_new.attn, :v_proj)
    @test haskey(st_new.attn, :o_proj)
end

@testset "TransformerBlock forward with custom MLP hidden dim" begin
    rng = MersenneTwister(20260707)

    d_model = 24
    num_heads = 3
    mlp_hidden_dim = 48
    seq_len = 5
    batch_size = 2

    block = TransformerBlock(
        d_model,
        num_heads;
        mlp_hidden_dim,
        is_causal=false,
        use_bias=true,
        use_rope=false,
    )

    ps = Lux.initialparameters(rng, block)
    st = Lux.initialstates(rng, block)

    x = randn(rng, Float32, d_model, seq_len, batch_size)
    y, st_new = block(x, ps, st)

    @test size(y) == size(x)
    @test all(isfinite, y)
    @test block.mlp_hidden_dim == mlp_hidden_dim
    @test block.is_causal == false

    @test haskey(st_new, :norm1)
    @test haskey(st_new, :attn)
    @test haskey(st_new, :norm2)
    @test haskey(st_new, :mlp)
end

@testset "TransformerBlock forward with RoPE" begin
    rng = MersenneTwister(20260708)

    d_model = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2

    block = TransformerBlock(
        d_model,
        num_heads;
        is_causal=true,
        use_bias=false,
        use_rope=true,
        max_seq_len=16,
        rope_theta=10000.0,
    )

    ps = Lux.initialparameters(rng, block)
    st = Lux.initialstates(rng, block)

    x = randn(rng, Float32, d_model, seq_len, batch_size)
    y, st_new = block(x, ps, st)

    @test size(y) == size(x)
    @test eltype(y) == eltype(x)
    @test all(isfinite, y)

    @test block.use_rope == true
    @test block.attn.use_rope == true
    @test block.attn.rope !== nothing
    @test block.attn.rope.head_dim == d_model ÷ num_heads
    @test block.attn.rope.max_seq_len == 16

    @test haskey(st_new, :norm1)
    @test haskey(st_new, :attn)
    @test haskey(st_new, :norm2)
    @test haskey(st_new, :mlp)
end

@testset "TransformerBlock with RoPE max_seq_len check" begin
    rng = MersenneTwister(20260709)

    d_model = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2

    block = TransformerBlock(
        d_model,
        num_heads;
        use_rope=true,
        max_seq_len=4,
    )

    ps = Lux.initialparameters(rng, block)
    st = Lux.initialstates(rng, block)

    x = randn(rng, Float32, d_model, seq_len, batch_size)

    @test_throws AssertionError block(x, ps, st)
end

@testset "TransformerBlock input shape checks" begin
    rng = MersenneTwister(20260710)

    d_model = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2

    block = TransformerBlock(d_model, num_heads)
    ps = Lux.initialparameters(rng, block)
    st = Lux.initialstates(rng, block)

    x_bad_rank = randn(rng, Float32, d_model, seq_len)
    x_bad_d_model = randn(rng, Float32, d_model + 1, seq_len, batch_size)

    @test_throws AssertionError block(x_bad_rank, ps, st)
    @test_throws AssertionError block(x_bad_d_model, ps, st)
end

@testset "TransformerBlock constructor checks" begin
    @test_throws AssertionError TransformerBlock(0, 4)
    @test_throws AssertionError TransformerBlock(32, 0)
    @test_throws AssertionError TransformerBlock(32, 4; mlp_ratio=0)
    @test_throws AssertionError TransformerBlock(32, 4; mlp_hidden_dim=0)

    # RoPE rotates pairs of dimensions, so each head dimension must be even.
    @test_throws AssertionError TransformerBlock(30, 6; use_rope=true)
end
