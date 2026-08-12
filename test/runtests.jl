using Test
using Random
using Lux
using LifeAI: MultiHeadAttention, manual_scaled_dot_product_attention, batched_scaled_dot_product_attention,
    RoPE, apply_rope, TransformerBlock, GPTModel

const LIFEAI_TEST_ROOT = @__DIR__
const LIFEAI_REPO_ROOT = dirname(LIFEAI_TEST_ROOT)
const LIFEAI_TEST_SUPPORT = joinpath(LIFEAI_TEST_ROOT, "support")
const LIFEAI_TEST_EPISODES = joinpath(LIFEAI_TEST_ROOT, "episodes")

include(joinpath(LIFEAI_TEST_SUPPORT, "repository_test_assets.jl"))

chapter_test(episode, chapter, filename) =
    joinpath(LIFEAI_TEST_EPISODES, episode, chapter, filename)

@testset "Episode 01 — Transformer and training foundations" begin
    @testset "Chapter 01 — Transformer foundations" begin
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter01_transformer", "test_manual_attention.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter01_transformer", "test_mha.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter01_transformer", "test_rope.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter01_transformer", "test_transformer.jl"))
    end

    @testset "Chapter 02 — GPT, XLA, and KV cache" begin
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter02_gpt_xla_kv_cache", "test_gpt.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter02_gpt_xla_kv_cache", "test_tokenizer.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter02_gpt_xla_kv_cache", "test_dataset.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter02_gpt_xla_kv_cache", "test_train_gpt.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter02_gpt_xla_kv_cache", "test_kv_cache.jl"))
    end

    @testset "Chapter 03 — Reproducible training" begin
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter03_reproducible_training", "test_reproducible_training.jl"))
    end

    @testset "Chapter 04 — Model modernization" begin
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter04_model_modernization", "test_modern_gpt_components.jl"))
    end

    @testset "Chapter 05 — Tokenizer and data pipeline" begin
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter05_tokenizer_data_pipeline", "test_tokenizer_data_pipeline.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter05_tokenizer_data_pipeline", "test_tokenizer_integration_matrix.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter05_tokenizer_data_pipeline", "test_tokenizer_random_utf8.jl"))
    end
end

@testset "Episode 02 — Qwen3 end-to-end parity" begin
    @testset "Chapter 06 — GQA and QK-Norm" begin
        include(chapter_test("episode02_qwen3_end_to_end_parity", "chapter06_gqa_qwen3_parity", "test_qwen3_gqa_qknorm.jl"))
    end

    @testset "Chapter 07 — HuggingFace weight loading" begin
        include(chapter_test("episode02_qwen3_end_to_end_parity", "chapter07_hf_weight_loading", "test_qwen3_hf_weight_loading.jl"))
    end

    @testset "Chapter 08 — Tokenizer and text parity" begin
        include(chapter_test("episode02_qwen3_end_to_end_parity", "chapter08_hf_tokenizer_text_parity", "test_qwen3_tokenizer_text_generation.jl"))
    end

    @testset "Chapter 09 — Sampling and inference" begin
        include(chapter_test("episode02_qwen3_end_to_end_parity", "chapter09_qwen3_sampling_performance", "test_qwen3_sampling_inference.jl"))
    end
end

@testset "Episode 03 — Model family and large weights" begin
    @testset "Chapter 10 — GPT-2 parity" begin
        include(chapter_test("episode03_model_family_and_large_weights", "chapter10_gpt2_hf_parity", "test_gpt2_hf_parity.jl"))
    end

    @testset "Chapter 11 — Qwen3 dense family" begin
        include(chapter_test("episode03_model_family_and_large_weights", "chapter11_qwen3_dense_family", "test_qwen3_dense_family.jl"))
    end

    @testset "Chapter 12 — Dense real weights" begin
        include(chapter_test("episode03_model_family_and_large_weights", "chapter12_qwen3_dense_real_weights", "test_qwen3_dense_real_weights.jl"))
    end

    @testset "Chapter 13 — Streamed large weights" begin
        include(chapter_test("episode03_model_family_and_large_weights", "chapter13_qwen3_streamed_large_weights", "test_qwen3_streamed_large_weights.jl"))
    end
end

@testset "Episode 04 — Efficient inference and quantization" begin
    @testset "Chapter 14 — Native BF16 compute" begin
        include(chapter_test("episode04_efficient_inference_and_quantization", "chapter14_qwen3_bf16_compute", "test_qwen3_bf16_compute.jl"))
    end

    @testset "Chapter 15 — BF16 acceleration" begin
        include(chapter_test("episode04_efficient_inference_and_quantization", "chapter15_qwen3_bf16_accel", "test_qwen3_bf16_acceleration.jl"))
    end

    @testset "Chapter 16 — XLA decode and quantization" begin
        include(chapter_test("episode04_efficient_inference_and_quantization", "chapter16_qwen3_xla_decode_quant", "test_qwen3_xla_decode_quantization.jl"))
    end

    @testset "Chapter 17 — Calibrated INT4" begin
        include(chapter_test("episode04_efficient_inference_and_quantization", "chapter17_qwen3_calibrated_int4", "test_qwen3_calibrated_int4.jl"))
    end

    @testset "Chapter 18 — Activation calibration" begin
        include(chapter_test("episode04_efficient_inference_and_quantization", "chapter18_qwen3_activation_calibration", "test_qwen3_activation_aware_int4.jl"))
    end
end

@testset "Episode 05 — Deployment, memory, and sampling" begin
    @testset "Chapter 19 — CUDA deployment" begin
        include(chapter_test("episode05_deployment_memory_and_sampling", "chapter19_qwen3_8b_4090d_deployment", "test_qwen3_cuda_deployment.jl"))
    end

    @testset "Chapter 20 — XLA deployment" begin
        include(chapter_test("episode05_deployment_memory_and_sampling", "chapter20_qwen3_8b_xla_deployment", "test_qwen3_xla_deployment.jl"))
    end

    @testset "Chapter 21 — Resident HTTP service" begin
        include(chapter_test("episode05_deployment_memory_and_sampling", "chapter21_qwen3_xla_resident_service", "test_qwen3_resident_http_service.jl"))
    end

    @testset "Chapter 22 — Embedding and semantic memory" begin
        include(chapter_test("episode05_deployment_memory_and_sampling", "chapter22_qwen3_embedding_memory", "test_qwen3_embedding_memory.jl"))
    end

    @testset "Chapter 23 — Device-resident sampling" begin
        include(chapter_test("episode05_deployment_memory_and_sampling", "chapter23_qwen3_xla_device_sampling", "test_qwen3_device_sampling.jl"))
    end
end

@testset "Episode 06 — Qwen3 MoE and model expansion" begin
    @testset "Chapter 24 — Qwen3 MoE architecture" begin
        chapter = "chapter24_qwen3_moe_architecture"
        episode = "episode06_qwen3_moe_and_model_expansion"
        include(chapter_test(episode, chapter, "qwen3_moe_router_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_expert_mixture_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_weight_loading_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_cached_decode_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_transformers_parity_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_real_checkpoint_contract_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_real_parity_contract_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_sparse_dispatch_test.jl"))
        include(chapter_test(episode, chapter, "qwen3_moe_device_sparse_dispatch_test.jl"))
    end
end

if lowercase(get(ENV, "LIFEAI_TEST_XLA", "false")) in ("1", "true", "yes")
    @testset "Reactant/XLA episode extensions" begin
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter02_gpt_xla_kv_cache", "test_xla_kv_cache.jl"))
        include(chapter_test("episode01_transformer_and_training_foundations", "chapter05_tokenizer_data_pipeline", "test_tokenizer_xla.jl"))
        include(chapter_test("episode02_qwen3_end_to_end_parity", "chapter06_gqa_qwen3_parity", "test_qwen3_gqa_qknorm_xla.jl"))
        include(chapter_test("episode02_qwen3_end_to_end_parity", "chapter07_hf_weight_loading", "test_qwen3_hf_weight_loading_xla.jl"))
        include(chapter_test("episode02_qwen3_end_to_end_parity", "chapter08_hf_tokenizer_text_parity", "test_qwen3_tokenizer_text_generation_xla.jl"))
        include(chapter_test("episode03_model_family_and_large_weights", "chapter10_gpt2_hf_parity", "test_gpt2_hf_parity_xla.jl"))
        include(chapter_test("episode03_model_family_and_large_weights", "chapter11_qwen3_dense_family", "test_qwen3_dense_family_xla.jl"))
        include(chapter_test("episode05_deployment_memory_and_sampling", "chapter20_qwen3_8b_xla_deployment", "test_qwen3_xla_deployment_xla.jl"))
        include(chapter_test("episode05_deployment_memory_and_sampling", "chapter23_qwen3_xla_device_sampling", "test_qwen3_device_sampling_xla.jl"))
        include(chapter_test("episode06_qwen3_moe_and_model_expansion", "chapter24_qwen3_moe_architecture", "qwen3_moe_device_sparse_dispatch_xla_test.jl"))
    end
end

if lowercase(get(ENV, "LIFEAI_TEST_CUDA", "false")) in ("1", "true", "yes")
    @testset "CUDA episode extensions" begin
        include(chapter_test("episode06_qwen3_moe_and_model_expansion", "chapter24_qwen3_moe_architecture", "qwen3_moe_device_sparse_dispatch_cuda_test.jl"))
    end
end
