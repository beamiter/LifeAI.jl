# Episode 06 — Qwen3 MoE 与模型架构扩展

> 状态：Open
>
> 收录章节：Chapter 24–

这一卷继续推进模型本体能力：从已经闭环的 Qwen3 Dense 扩展到稀疏 MoE，随后再评估 reranker 与视觉语言模型。每种新结构都以官方配置和权重命名为契约，以可独立复现的数值 parity 为完成依据，不把仅能构造 topology 写成真实 checkpoint 已可用。

## 章节目录

1. [`Chapter 24 — Qwen3 MoE 架构支持`](chapter24_qwen3_moe_architecture.md)（Closed）

## 预期能力变化

- **模型基本组件**：具备 Qwen3 top-k router、稀疏 expert SwiGLU 和 MoE decoder block。
- **训练与推理**：Float32 correctness oracle、Float32/native BF16 active-expert streaming、CPU sparse dispatch、compact Reactant/XLA CPU 与低 workspace CUDA indexed kernels 已完成；CUDA 大型 prefill 会在设备端稳定按 expert 分桶，在官方投影宽度上 32/64 token 相对 token-major indexed 加速约 `1.48× / 3.28×`。下一步把已验证的真实权重 streamer 接入 GPU resident/offload session。
- **HuggingFace 互操作**：严格解析 `qwen3_moe` config，映射官方 expert 权重；tiny 与官方 30B-A3B 均完成逐层/logits/cache parity，真实 BF16 路由边界差异单独量化。
- **工程与测试**：测试文件按 router、expert mixture、权重加载和 cached decode 的实际内容命名。

## Episode Close 条件

- 至少一种官方 Qwen3 MoE checkpoint 完成可复现的真实权重 parity。
- MoE prefill/decode 不再计算全部未选中的 expert，并有性能与内存证据。
- 后续 Qwen 模型方向已经根据已验证的架构缺口确定。

## 本卷回顾

- **形成的能力闭环**：Chapter 24 已完成原始 Qwen3 MoE 从 config/checkpoint 契约、独立 parity 到 CPU/XLA/CUDA sparse dispatch 的闭环；Episode 仍为后续模型扩展保持 Open。
- **关键实验与指标**：30B Float32 top-8 槽位 `1,152 / 1,152`，BF16 槽位重合 `95.92%`，两种口径 prompt/decode argmax 均一致；BF16 streamer 峰值 RSS `3.50 GiB`。
- **失败、偏差与未解决问题**：BF16 跨框架 GEMM 累加顺序会在第 8/9 expert 边界换路由；完整 40K context 和 GPU resident/offload 尚未实跑。
- **重要架构决策**：correctness oracle、active-expert streamer 与生产 sparse dispatch 分层实现；Float32 严格路由和 BF16 容差证据不混写。
- **最重要的认知变化**：BF16 支持必须落到具体生命周期和算子口径，只有 expert kernel 支持不能替代真实 checkpoint 全链路 parity。
- **进入下一卷的问题**：如何把 BF16 streamer、GPU grouped dispatch、KV cache 与 40K context 组合为可部署的 30B-A3B session。
