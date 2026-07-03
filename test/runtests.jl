using Test
using Random
using Lux
using LifeAI: MultiHeadAttention, manual_scaled_dot_product_attention, batched_scaled_dot_product_attention

@testset "Attention" begin
    include("test_manual_attention.jl")
    include("test_mha.jl")
end
