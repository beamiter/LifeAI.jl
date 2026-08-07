# Episode 01 — Transformer 与训练基础

> 状态：Closed
>
> 收录章节：Chapter 01–05

这一卷建立 LifeAI.jl 最早的模型、训练、评估与文本数据基础，从 Attention / RoPE 逐步推进到可恢复训练、现代 GPT 组件和版本化 tokenizer。

## 章节目录

1. [`Chapter 01 — Project Skeleton and Transformer Foundations`](chapter01_transformer.md)
2. [`Chapter 02 — Minimal GPT, XLA Training, and KV-Cached Decoding`](chapter02_gpt_xla_kv_cache.md)
3. [`Chapter 03 — Reproducible Training and Evaluation`](chapter03_reproducible_training.md)
4. [`Chapter 04 — Modern GPT Building Blocks`](chapter04_model_modernization.md)
5. [`Chapter 05 — Versioned Tokenizers and Chinese Data Pipeline`](chapter05_tokenizer_data_pipeline.md)

## 本卷结果

形成了从文本数据、模型前向、训练、validation、checkpoint 到生成和 KV Cache 的最小闭环，并为后续 HuggingFace 模型对齐提供了可复用测试基线。
