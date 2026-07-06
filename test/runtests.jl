using Test
using Random
using Lux
using LifeAI: MultiHeadAttention, manual_scaled_dot_product_attention, batched_scaled_dot_product_attention,
RoPE, apply_rope, TransformerBlock

@testset "Attention" begin
    include("test_manual_attention.jl")
    include("test_mha.jl")
end

@testset "RoPE" begin
    include("test_rope.jl")
end

@testset "TransformerBlock" begin
    include("test_transformer.jl")
end
