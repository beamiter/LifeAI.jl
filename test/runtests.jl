using Test
using Random
using Lux
using LifeAI: MultiHeadAttention, manual_scaled_dot_product_attention, batched_scaled_dot_product_attention,
    RoPE, apply_rope, TransformerBlock, GPTModel

include("repository_test_assets.jl")

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

@testset "Reproducible training and evaluation" begin
    include("test_reproducible_training.jl")
end

@testset "Modern GPT components" begin
    include("test_modern_gpt_components.jl")
end

@testset "Tokenizers and Chinese data pipeline" begin
    include("test_tokenizer_data_pipeline.jl")
    include("test_tokenizer_integration_matrix.jl")
    include("test_tokenizer_random_utf8.jl")
end

@testset "Qwen3 GQA and QK-Norm" begin
    include("test_qwen3_gqa_qknorm.jl")
end

@testset "Qwen3 HuggingFace weight loading" begin
    include("test_qwen3_hf_weight_loading.jl")
end

@testset "Qwen3 tokenizer and text generation" begin
    include("test_qwen3_tokenizer_text_generation.jl")
end

@testset "Qwen3 sampling and inference fidelity" begin
    include("test_qwen3_sampling_inference.jl")
end

@testset "GPT-2 architecture and HuggingFace parity" begin
    include("test_gpt2_hf_parity.jl")
end

@testset "Qwen3 dense family contract" begin
    include("test_qwen3_dense_family.jl")
end

@testset "Qwen3 dense real-weight parity" begin
    include("test_qwen3_dense_real_weights.jl")
end

@testset "Qwen3 streamed large-weight parity" begin
    include("test_qwen3_streamed_large_weights.jl")
end

@testset "Qwen3 native BF16 compute" begin
    include("test_qwen3_bf16_compute.jl")
end

@testset "Qwen3 BF16 CUDA/XLA acceleration" begin
    include("test_qwen3_bf16_acceleration.jl")
end

@testset "Qwen3 XLA decode and quantization" begin
    include("test_qwen3_xla_decode_quantization.jl")
end

@testset "Qwen3 calibrated INT4 and quantization plans" begin
    include("test_qwen3_calibrated_int4.jl")
end

@testset "Qwen3 activation-aware INT4 calibration" begin
    include("test_qwen3_activation_aware_int4.jl")
end

@testset "Qwen3 CUDA local deployment" begin
    include("test_qwen3_cuda_deployment.jl")
end

@testset "Qwen3 XLA single-residency deployment" begin
    include("test_qwen3_xla_deployment.jl")
end

@testset "Qwen3 XLA resident HTTP service" begin
    include("test_qwen3_resident_http_service.jl")
end

@testset "Qwen3 embedding and semantic memory" begin
    include("test_qwen3_embedding_memory.jl")
end

@testset "Qwen3 XLA device-resident sampling" begin
    include("test_qwen3_device_sampling.jl")
end

@testset "Qwen3 MoE architecture" begin
    include("qwen3_moe_router_test.jl")
    include("qwen3_moe_expert_mixture_test.jl")
    include("qwen3_moe_weight_loading_test.jl")
    include("qwen3_moe_cached_decode_test.jl")
end

if lowercase(get(ENV, "LIFEAI_TEST_XLA", "false")) in ("1", "true", "yes")
    @testset "Reactant/XLA KV cache" begin
        include("test_xla_kv_cache.jl")
        include("test_tokenizer_xla.jl")
        include("test_qwen3_gqa_qknorm_xla.jl")
        include("test_qwen3_hf_weight_loading_xla.jl")
        include("test_qwen3_tokenizer_text_generation_xla.jl")
        include("test_gpt2_hf_parity_xla.jl")
        include("test_qwen3_dense_family_xla.jl")
        include("test_qwen3_xla_deployment_xla.jl")
        include("test_qwen3_device_sampling_xla.jl")
    end
end
