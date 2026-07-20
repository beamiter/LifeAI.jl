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

@testset "GPT KV cache" begin
    include("test_kv_cache.jl")
end

@testset "Week 03 reproducible experiments" begin
    include("test_week03.jl")
end

@testset "Week 04 modern GPT components" begin
    include("test_week04.jl")
end

@testset "Week 05 tokenizers and Chinese data pipeline" begin
    include("test_week05.jl")
    include("test_week05_matrix.jl")
end

if lowercase(get(ENV, "LIFEAI_TEST_XLA", "false")) in ("1", "true", "yes")
    @testset "Reactant/XLA KV cache" begin
        include("test_xla_kv_cache.jl")
        include("test_week05_xla.jl")
    end
end
