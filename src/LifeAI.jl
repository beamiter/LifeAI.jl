module LifeAI

# Foundational model and data layers.
include("core/rope.jl")
include("core/attention.jl")
include("core/normalization.jl")
include("core/mlp.jl")
include("core/transformer.jl")
include("core/sampling.jl")
include("data/tokenizer.jl")
include("data/hf_tokenizer.jl")
include("data/hf_gpt2_tokenizer.jl")
include("data/dataset.jl")
include("data/data_pipeline.jl")
include("models/output_projection.jl")
include("models/gpt.jl")

# Shared inference policies and host-side input boundaries.
include("generation/sampling.jl")
include("generation/xla_sampling.jl")
include("generation/token_inputs.jl")

# External formats and training workflows.
include("io/huggingface.jl")
include("io/hf_qwen3_embedding.jl")
include("io/hf_gpt2.jl")
include("train/train_gpt.jl")
include("train/evaluation.jl")
include("train/checkpoint.jl")

# Eager and cached generation.
include("generation/text_generation.jl")
include("generation/kv_cache.jl")

# Large-weight and accelerator-specific inference implementations.
include("io/hf_streaming.jl")
include("io/hf_qwen3_vl.jl")
include("data/hf_qwen3_vl_processor.jl")
include("models/bf16_inference.jl")
include("models/bf16_accel.jl")
include("models/qwen3_vl_vision.jl")
include("generation/qwen3_embedding.jl")
include("models/bf16_xla.jl")
include("models/quantized.jl")
include("generation/xla_kv_cache.jl")
include("generation/hf_text_generation.jl")
include("generation/qwen3_deployment.jl")
include("generation/qwen3_moe_offload.jl")
include("generation/qwen3_xla_deployment.jl")
include("generation/qwen3_xla_service.jl")
include("generation/kv_benchmark.jl")
include("generation/xla_cache_modes_benchmark.jl")

# Tool declarations and the minimal stateful agent loop.
include("agent/tools.jl")
include("agent/memory.jl")
include("agent/tool_loop.jl")
include("agent/environment.jl")
include("agent/environment_memory.jl")

# Task-level quality evaluation.
include("eval/task_quality.jl")
include("eval/memory_quality.jl")
include("eval/environment_quality.jl")

include("api.jl")

end # module LifeAI
