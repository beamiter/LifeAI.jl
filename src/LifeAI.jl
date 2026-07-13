module LifeAI

export MultiHeadAttention
export manual_scaled_dot_product_attention
export batched_scaled_dot_product_attention
export TransformerBlock
export RoPE, apply_rope
export SamplingSchedule
export Tokenizer, fit_tokenizer, encode, decode, vocab_size
export DatasetLoader, num_samples, num_batches
export GPTModel
export TrainerGPT, init_train_state, next_token_loss, train_step!, train_gpt!
export generate
export LayerKVCache, GPTKVCache, init_kv_cache, prefill, decode_step, generate_cached

include("core/rope.jl")
include("core/attention.jl")
include("core/transformer.jl")
include("core/sampling.jl")
include("data/tokenizer.jl")
include("data/dataset.jl")
include("models/gpt.jl")
include("train/train_gpt.jl")
include("generation/text_generation.jl")
include("generation/kv_cache.jl")

end # module LifeAI
