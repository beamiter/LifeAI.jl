# Episode 06 — Qwen3 MoE 与模型架构扩展

> 状态：Open
>
> 收录章节：Chapter 24–

这一卷继续推进模型本体能力：从已经闭环的 Qwen3 Dense 扩展到稀疏 MoE，随后再评估 reranker 与视觉语言模型。每种新结构都以官方配置和权重命名为契约，以可独立复现的数值 parity 为完成依据，不把仅能构造 topology 写成真实 checkpoint 已可用。

## 章节目录

1. [`Chapter 24 — Qwen3 MoE 架构支持`](chapter24_qwen3_moe_architecture.md)（Open）

## 预期能力变化

- **模型基本组件**：具备 Qwen3 top-k router、稀疏 expert SwiGLU 和 MoE decoder block。
- **训练与推理**：先建立 Float32 correctness oracle，再推进 sparse dispatch 与 CUDA/XLA。
- **HuggingFace 互操作**：严格解析 `qwen3_moe` config，映射官方 expert 权重并完成逐层/logits/cache parity。
- **工程与测试**：测试文件按 router、expert mixture、权重加载和 cached decode 的实际内容命名。

## Episode Close 条件

- 至少一种官方 Qwen3 MoE checkpoint 完成可复现的真实权重 parity。
- MoE prefill/decode 不再计算全部未选中的 expert，并有性能与内存证据。
- 后续 Qwen 模型方向已经根据已验证的架构缺口确定。

## 本卷回顾

- **形成的能力闭环**：进行中。
- **关键实验与指标**：进行中。
- **失败、偏差与未解决问题**：进行中。
- **重要架构决策**：correctness oracle 与生产 sparse dispatch 分层实现。
- **最重要的认知变化**：进行中。
- **进入下一卷的问题**：进行中。
