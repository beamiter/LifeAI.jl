using Test
using Random
using Lux
using LifeAI: MultiHeadAttention, manual_scaled_dot_product_attention, batched_scaled_dot_product_attention,
    RoPE, apply_rope, TransformerBlock, GPTModel

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

@testset "GPT" begin
    include("test_gpt.jl")
end

@testset "Tokenizer" begin
    include("test_tokenizer.jl")
end

@testset "DatasetLoader" begin
    include("test_dataset.jl")
end

@testset "GPT training and generation" begin
    include("test_train_gpt.jl")
end
