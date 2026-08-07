# Episode 03 — 模型家族与大权重验证

> 状态：Closed
>
> 收录章节：Chapter 10–13

这一卷用 GPT-2 验证第二种经典架构，再把 Qwen3 从单一尺寸扩展到完整 dense family，并解决大权重内存边界。

## 章节目录

1. [`Chapter 10 — GPT-2 Architecture, HuggingFace Weights and Text Parity`](chapter10_gpt2_hf_parity.md)
2. [`Chapter 11 — Qwen3 Dense Family Completion`](chapter11_qwen3_dense_family.md)
3. [`Chapter 12 — Qwen3 Dense Family Real-Weight Parity`](chapter12_qwen3_dense_real_weights.md)
4. [`Chapter 13 — Qwen3 Streamed Loading and 8B/14B/32B Real-Weight Parity`](chapter13_qwen3_streamed_large_weights.md)

## 本卷结果

GPT-2 124M 完成官方 checkpoint 推理对齐；Qwen3 dense 0.6B–32B 六个尺寸完成真实权重验证，其中 8B–32B 通过逐层流式加载把峰值内存压到 8.9 GiB 以内。
