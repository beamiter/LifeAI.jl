@testset "MultiHeadAttention forward" begin
    rng = MersenneTwister(42)

    d_in = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2

    mha = MultiHeadAttention(
        d_in,
        num_heads;
        is_causal=true,
        use_bias=false,
    )

    ps = Lux.initialparameters(rng, mha)
    st = Lux.initialstates(rng, mha)

    x = randn(rng, Float32, d_in, seq_len, batch_size)

    y, st_new = mha(x, ps, st)

    @test size(y) == size(x)
    @test all(isfinite, y)

    # state 结构应该仍然包含这几个子层
    @test haskey(st_new, :q_proj)
    @test haskey(st_new, :k_proj)
    @test haskey(st_new, :v_proj)
    @test haskey(st_new, :o_proj)
end

@testset "MultiHeadAttention forward with RoPE" begin
    rng = MersenneTwister(43)

    d_in = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2

    mha = MultiHeadAttention(
        d_in,
        num_heads;
        is_causal=true,
        use_bias=false,
        use_rope=true,
        max_seq_len=16,
        rope_theta=10000.0,
    )

    ps = Lux.initialparameters(rng, mha)
    st = Lux.initialstates(rng, mha)

    x = randn(rng, Float32, d_in, seq_len, batch_size)

    y, st_new = mha(x, ps, st)

    @test size(y) == size(x)
    @test all(isfinite, y)

    @test mha.use_rope == true
    @test mha.rope !== nothing
    @test mha.rope.head_dim == d_in ÷ num_heads

    @test haskey(st_new, :q_proj)
    @test haskey(st_new, :k_proj)
    @test haskey(st_new, :v_proj)
    @test haskey(st_new, :o_proj)
end

@testset "MultiHeadAttention with RoPE max_seq_len check" begin
    rng = MersenneTwister(44)

    d_in = 32
    num_heads = 4
    seq_len = 6
    batch_size = 2

    mha = MultiHeadAttention(
        d_in,
        num_heads;
        use_rope=true,
        max_seq_len=4,
    )

    ps = Lux.initialparameters(rng, mha)
    st = Lux.initialstates(rng, mha)

    x = randn(rng, Float32, d_in, seq_len, batch_size)

    @test_throws AssertionError mha(x, ps, st)
end