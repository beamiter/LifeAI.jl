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
    include("test_week05_random_utf8.jl")
end

@testset "Week 06 GQA and QK-Norm" begin
    include("test_week06.jl")
end

@testset "Week 07 HuggingFace weight loading" begin
    include("test_week07.jl")
end

@testset "Week 08 HuggingFace tokenizer and text parity" begin
    include("test_week08.jl")
end

@testset "Week 09 Qwen3 sampling and inference fidelity" begin
    include("test_week09.jl")
end

@testset "Week 10 GPT-2 architecture and HuggingFace parity" begin
    include("test_week10.jl")
end

@testset "Week 11 Qwen3 dense family completion" begin
    include("test_week11.jl")
end

@testset "Week 12 Qwen3 dense real-weight parity" begin
    include("test_week12.jl")
end

@testset "Week 13 Qwen3 streamed large-weight parity" begin
    include("test_week13.jl")
end

@testset "Week 14 Qwen3 native BF16 compute" begin
    include("test_week14.jl")
end

@testset "Week 15 Qwen3 BF16 CUDA/XLA acceleration" begin
    include("test_week15.jl")
end

@testset "Week 16 Qwen3 XLA decode and quantization" begin
    include("test_week16.jl")
end

@testset "Week 17 calibrated INT4 and quantization plans" begin
    include("test_week17.jl")
end

@testset "Week 18 activation-aware INT4 calibration" begin
    include("test_week18.jl")
end

@testset "Week 19 Qwen3 4090D local deployment" begin
    include("test_week19.jl")
end

@testset "Week 20 Qwen3 XLA single-residency deployment" begin
    include("test_week20.jl")
end

@testset "Week 21 Qwen3 XLA resident HTTP service" begin
    include("test_week21.jl")
end

@testset "Week 22 Qwen3 embedding and semantic memory" begin
    include("test_week22.jl")
end

@testset "Week 23 Qwen3 XLA device-resident sampling" begin
    include("test_week23.jl")
end

if lowercase(get(ENV, "LIFEAI_TEST_XLA", "false")) in ("1", "true", "yes")
    @testset "Reactant/XLA KV cache" begin
        include("test_xla_kv_cache.jl")
        include("test_week05_xla.jl")
        include("test_week06_xla.jl")
        include("test_week07_xla.jl")
        include("test_week08_xla.jl")
        include("test_week10_xla.jl")
        include("test_week11_xla.jl")
        include("test_week20_xla.jl")
        include("test_week23_xla.jl")
    end
end
