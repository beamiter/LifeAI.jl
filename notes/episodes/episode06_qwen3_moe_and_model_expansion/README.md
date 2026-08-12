# Episode 06 — Qwen3 MoE 与模型架构扩展

> 状态：Open
>
> 收录章节：Chapter 24–26

这一卷继续推进模型本体能力：从已经闭环的 Qwen3 Dense 扩展到稀疏 MoE，随后再评估 reranker 与视觉语言模型。每种新结构都以官方配置和权重命名为契约，以可独立复现的数值 parity 为完成依据，不把仅能构造 topology 写成真实 checkpoint 已可用。

## 章节目录

1. [`Chapter 24 — Qwen3 MoE 架构支持`](chapter24_qwen3_moe_architecture.md)（Closed）
2. [`Chapter 25 — Qwen3-30B-A3B GPU resident/offload session`](chapter25_qwen3_moe_gpu_offload.md)（Closed）
3. [`Chapter 26 — Qwen3 MoE active-expert device cache`](chapter26_qwen3_moe_expert_cache.md)（Closed）

## 预期能力变化

- **模型基本组件**：具备 Qwen3 top-k router、稀疏 expert SwiGLU 和 MoE decoder block。
- **训练与推理**：Float32 correctness oracle、Float32/native BF16 active-expert streaming、CPU/XLA/CUDA sparse dispatch，以及真实 30B-A3B GPU resident/offload session 已完成。RTX 4090 D 实际分配完整 40,960-token BF16 KV；32-token grouped WMMA 的端到端 prefill/decode 相对 scalar production 为 `1.092× / 1.335×`。可选的 8 GiB device expert LRU 在冻结短请求上消除全部重复 read/upload，端到端加速 `1.722×`。
- **HuggingFace 互操作**：严格解析 `qwen3_moe` config，映射官方 expert 权重；tiny 与官方 30B-A3B 均完成逐层/logits/cache parity，真实 BF16 路由边界差异单独量化。
- **工程与测试**：测试按 Chapter 24 架构/parity、Chapter 25 offload 容量/路由/result contract 与 Chapter 26 缓存生命周期/I/O/result contract 分层。

## Episode Close 条件

- 至少一种官方 Qwen3 MoE checkpoint 完成可复现的真实权重 parity。
- MoE prefill/decode 不再计算全部未选中的 expert，并有性能与内存证据。
- 后续 Qwen 模型方向已经根据已验证的架构缺口确定。

## 本卷回顾

- **形成的能力闭环**：Chapter 24 已完成 config/checkpoint/parity/sparse dispatch，Chapter 25 把真实 BF16 streamer、40K static KV 和 grouped WMMA 接成 RTX 4090 D offload session，Chapter 26 再加入显式预算的跨请求 device expert LRU；Episode 仍为后续模型扩展保持 Open。
- **关键实验与指标**：30B Float32 top-8 槽位 `1,152 / 1,152`，BF16 槽位重合 `95.92%`；offload 硬下限 `7.166 GiB`，2-token reference prompt/decode argmax 均一致；8 GiB cache 覆盖冻结请求的 892 个 expert-layer entries，重复请求 read/upload 为零，request 加速 `1.722×`。
- **失败、偏差与未解决问题**：BF16 跨框架 GEMM 累加顺序会在第 8/9 expert 边界换路由；2-token grouped 慢于 scalar，40K cache 已分配但 full-window prefill/长序列质量尚未实跑；miss path 仍同步且没有 pinned-memory/异步预取。
- **重要架构决策**：correctness oracle、active-expert streamer 与生产 sparse dispatch 分层实现；expert cache 默认关闭、按实际 tensor bytes 预算并在 request reset 后保留；Float32 严格路由和 BF16 容差证据不混写。
- **最重要的认知变化**：BF16 支持必须落到具体生命周期和算子口径，只有 expert kernel 支持不能替代真实 checkpoint 全链路 parity。
- **进入下一章的问题**：如何用长自然文本/多轮 trace 选择 cache budget，并用异步预取和 pinned-memory 双缓冲隐藏 miss path 的 expert I/O。
