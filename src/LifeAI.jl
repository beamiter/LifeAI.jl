module LifeAI

export MultiHeadAttention
export manual_scaled_dot_product_attention
export batched_scaled_dot_product_attention
export TransformerBlock
export RoPE, apply_rope
export SamplingSchedule
export Tokenizer, fit_tokenizer, encode, decode, vocab_size
export DatasetLoader, num_samples, num_batches
export split_token_stream, split_text_stream, train_validation_loaders
export GPTModel, gpt_config
export TrainerGPT, init_train_state, next_token_loss, next_token_nll_sum
export global_gradient_norm, clip_global_gradient_norm
export train_step!, train_gpt!
export evaluate_gpt
export CHECKPOINT_FORMAT_VERSION, save_checkpoint, load_checkpoint, resume_gpt!
export generate
export LayerKVCache, GPTKVCache, init_kv_cache, prefill, decode_step, generate_cached
export StaticLayerKVCache, StaticGPTKVCache, init_static_kv_cache
export XLAKVDecoder, xla_prefill!, xla_decode_step!, generate_xla_cached!
export kv_cache_correctness, benchmark_kv_cache, benchmark_xla_kv_cache
export benchmark_xla_cache_modes

include("core/rope.jl")
include("core/attention.jl")
include("core/transformer.jl")
include("core/sampling.jl")
include("data/tokenizer.jl")
include("data/dataset.jl")
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
