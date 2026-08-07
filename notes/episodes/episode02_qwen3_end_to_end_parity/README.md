# Episode 02 — Qwen3 端到端对齐

> 状态：Closed
>
> 收录章节：Chapter 06–09

这一卷从结构、权重、tokenizer、生成策略和真实后端性能五个层面，完成 Qwen3-0.6B 的端到端复现闭环。

## 章节目录

1. [`Chapter 06 — GQA, QK-Norm and Qwen3 Structural Parity`](chapter06_gqa_qwen3_parity.md)
2. [`Chapter 07 — HuggingFace Weight Loading and Qwen3 Logits Parity`](chapter07_hf_weight_loading.md)
3. [`Chapter 08 — HuggingFace Qwen3 Tokenizer and Text-to-Text Parity`](chapter08_hf_tokenizer_text_parity.md)
4. [`Chapter 09 — Qwen3 Sampling Fidelity and Real Inference Performance`](chapter09_qwen3_sampling_performance.md)

## 本卷结果

Qwen3-0.6B 获得逐层 logits、KV Cache、text-to-text greedy、官方采样语义和长位置 RoPE 的真实 reference parity，并建立 CPU/CUDA/XLA 性能基线。
