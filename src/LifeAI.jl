module LifeAI

export MultiHeadAttention
export manual_scaled_dot_product_attention
export batched_scaled_dot_product_attention
export repeat_kv
export RMSNormLayer, SwiGLU, TransformerBlock
export RoPE, apply_rope
export SamplingSchedule
export AbstractTokenizer, Tokenizer, ByteTokenizer, ByteBPETokenizer
export fit_tokenizer, fit_byte_bpe, encode, decode, decode_bytes, vocab_size
export normalize_text, special_token_id, token_byte_length, encoded_byte_length
export tokenizer_config, tokenizer_fingerprint, tokenizer_statistics
export TOKENIZER_ARTIFACT_VERSION, save_tokenizer, load_tokenizer
export DatasetLoader, DocumentDatasetLoader, num_samples, num_batches, target_byte_count
export split_token_stream, split_text_stream, train_validation_loaders
export TextDocument, load_text_documents, split_documents, build_document_dataset
export DATASET_ARTIFACT_VERSION, save_dataset_artifact, load_dataset_artifact
export GPTModel, TiedOutputProjection, gpt_config
export TrainerGPT, init_train_state, next_token_loss, next_token_nll_sum
export global_gradient_norm, clip_global_gradient_norm
export train_step!, train_gpt!
export evaluate_gpt, bits_per_byte
export CHECKPOINT_FORMAT_VERSION, save_checkpoint, load_checkpoint, resume_gpt!
export generate
export LayerKVCache, GPTKVCache, init_kv_cache, prefill, decode_step, generate_cached
export StaticLayerKVCache, StaticGPTKVCache, init_static_kv_cache
export XLAKVDecoder, xla_prefill!, xla_decode_step!, generate_xla_cached!
export kv_cache_correctness, benchmark_kv_cache, benchmark_xla_kv_cache
export benchmark_xla_cache_modes

include("core/rope.jl")
include("core/attention.jl")
include("core/normalization.jl")
include("core/mlp.jl")
include("core/transformer.jl")
include("core/sampling.jl")
include("data/tokenizer.jl")
include("data/dataset.jl")
include("data/data_pipeline.jl")
include("models/output_projection.jl")
include("models/gpt.jl")
include("train/train_gpt.jl")
include("train/evaluation.jl")
include("train/checkpoint.jl")
include("generation/text_generation.jl")
include("generation/kv_cache.jl")
include("generation/xla_kv_cache.jl")
include("generation/kv_benchmark.jl")
include("generation/xla_cache_modes_benchmark.jl")

end # module LifeAI
