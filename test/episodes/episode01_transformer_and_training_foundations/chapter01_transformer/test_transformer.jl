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
    @test_throws ArgumentError TransformerBlock(32, 4; mlp_ratio=0)
    @test_throws ArgumentError TransformerBlock(32, 4; mlp_hidden_dim=0)

    # RoPE rotates pairs of dimensions, so each head dimension must be even.
    @test_throws AssertionError TransformerBlock(30, 6; use_rope=true)

    wide_hidden = TransformerBlock(
        4,
        1;
        mlp_hidden_dim=Int128(8),
    )
    wide_ratio = TransformerBlock(4, 1; mlp_ratio=big(3))
    rational_ratio = TransformerBlock(4, 1; mlp_ratio=5 // 2)
    @test wide_hidden.mlp_hidden_dim === 8
    @test wide_ratio.mlp_hidden_dim === 12
    @test rational_ratio.mlp_hidden_dim === 10

    for invalid_ratio in (true, 0, NaN, Inf)
        error = try
            TransformerBlock(4, 1; mlp_ratio=invalid_ratio)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("mlp_ratio", sprint(showerror, error))
    end

    for invalid_hidden in (true, 0, big(typemax(Int)) + 1)
        error = try
            TransformerBlock(4, 1; mlp_hidden_dim=invalid_hidden)
            nothing
        catch caught
            caught
        end
        @test error isa ArgumentError
        @test occursin("mlp_hidden_dim", sprint(showerror, error))
    end

    wrapped_ratio = typemax(Int) ÷ 2 + 2
    @test_throws ArgumentError TransformerBlock(
        4,
        1;
        mlp_ratio=wrapped_ratio,
    )
    @test_throws ArgumentError TransformerBlock(
        2,
        1;
        mlp_hidden_dim=typemax(Int),
    )
    @test_throws ArgumentError TransformerBlock(
        1,
        1;
        head_dim=typemax(Int) ÷ 6,
        mlp_hidden_dim=typemax(Int) ÷ 5,
        use_rope=false,
    )

    gelu_block = TransformerBlock(4, 1; mlp_hidden_dim=8)
    swiglu_block = TransformerBlock(
        4,
        1;
        mlp_hidden_dim=8,
        mlp_type=:swiglu,
    )
    moe_block = TransformerBlock(
        4,
        1;
        mlp_hidden_dim=8,
        mlp_type=:qwen3_moe,
        num_experts=3,
        experts_per_token=2,
    )
    @test Lux.parameterlength(gelu_block) == 144
    @test Lux.parameterlength(swiglu_block) == 176
    @test Lux.parameterlength(moe_block) == 380
end
