# Public API
#
# Keep exports separate from module assembly so `LifeAI.jl` describes the
# dependency order of the implementation. Grouping by responsibility also
# makes additions to the public surface deliberate and reviewable.

# Core model components
export MultiHeadAttention
export manual_scaled_dot_product_attention
export batched_scaled_dot_product_attention
export repeat_kv
export RMSNormLayer, SwiGLU, TransformerBlock, gelu_new
export RoPE, apply_rope
export SamplingSchedule

# Tokenization and datasets
export AbstractTokenizer, Tokenizer, ByteTokenizer, ByteBPETokenizer
export HFQwen3Tokenizer, HFQwen3GenerationConfig
export HFGPT2Tokenizer, HFGPT2GenerationConfig
export hf_byte_unicode_alphabet, load_hf_qwen3_tokenizer, hf_generation_config
export hf_qwen3_pretokenize, apply_qwen3_chat_template
export load_hf_gpt2_tokenizer, hf_gpt2_pretokenize
export fit_tokenizer, fit_byte_bpe, encode, decode, decode_bytes, vocab_size
export normalize_text, special_token_id, token_byte_length, encoded_byte_length
export tokenizer_config, tokenizer_fingerprint, tokenizer_statistics
export TOKENIZER_ARTIFACT_VERSION, save_tokenizer, load_tokenizer
export DatasetLoader, DocumentDatasetLoader, num_samples, num_batches, target_byte_count
export split_token_stream, split_text_stream, train_validation_loaders
export TextDocument, load_text_documents, split_documents, build_document_dataset
export DATASET_ARTIFACT_VERSION, save_dataset_artifact, load_dataset_artifact

# Models and HuggingFace interoperability
export GPTModel, TiedOutputProjection, gpt_config
export Qwen3DenseSpec, qwen3_dense_specs, qwen3_dense_spec
export qwen3_dense_parameter_count
export load_hf_qwen3_config, load_safetensors, load_hf_qwen3_parameters
export load_hf_qwen3_model, hf_token_ids, hf_qwen3_forward_trace
export HFSafetensorsReader, open_safetensors_reader, read_safetensors_tensor
export stream_hf_qwen3_forward
export hf_qwen3_bf16_forward
export hf_qwen3_bf16_accel_forward
export Int8ChannelWeight, Int4GroupWeight, quantize_bf16_parameters
export LinearQuantizationSpec, QuantizationPlan, quantization_spec
export quantized_parameter_bytes, estimate_qwen3_quantized_bytes
export load_hf_qwen3_quantized
export load_hf_qwen3_bundle, generate_hf_text
export load_hf_gpt2_config, load_hf_gpt2_parameters, load_hf_gpt2_model
export load_hf_gpt2_bundle, hf_gpt2_forward_trace

# Training and persistence
export TrainerGPT, init_train_state, next_token_loss, next_token_nll_sum
export global_gradient_norm, clip_global_gradient_norm
export train_step!, train_gpt!
export evaluate_gpt, bits_per_byte
export CHECKPOINT_FORMAT_VERSION, save_checkpoint, load_checkpoint, resume_gpt!

# Generation and KV caches
export generate
export LayerKVCache, GPTKVCache, init_kv_cache, prefill, decode_step, generate_cached
export StaticLayerKVCache, StaticGPTKVCache, init_static_kv_cache
export XLAKVDecoder, xla_prefill!, xla_decode_step!, generate_xla_cached!
export kv_cache_correctness, benchmark_kv_cache, benchmark_xla_kv_cache
export benchmark_xla_cache_modes
