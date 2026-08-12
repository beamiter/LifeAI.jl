# Episode 06 — Qwen3 MoE 与模型架构扩展

> 状态：Open
>
> 收录章节：Chapter 24–29

这一卷继续推进模型本体能力：从已经闭环的 Qwen3 Dense 扩展到稀疏 MoE，随后再评估 reranker 与视觉语言模型。每种新结构都以官方配置和权重命名为契约，以可独立复现的数值 parity 为完成依据，不把仅能构造 topology 写成真实 checkpoint 已可用。

## 章节目录

1. [`Chapter 24 — Qwen3 MoE 架构支持`](chapter24_qwen3_moe_architecture.md)（Closed）
2. [`Chapter 25 — Qwen3-30B-A3B GPU resident/offload session`](chapter25_qwen3_moe_gpu_offload.md)（Closed）
3. [`Chapter 26 — Qwen3 MoE active-expert device cache`](chapter26_qwen3_moe_expert_cache.md)（Closed）
4. [`Chapter 27 — Qwen3 MoE layer-balanced expert cache`](chapter27_qwen3_moe_layer_balanced_cache.md)（Closed）
5. [`Chapter 28 — Qwen3 MoE scattered expert cache dispatch`](chapter28_qwen3_moe_scattered_cache.md)（Closed）
6. [`Chapter 29 — Qwen3 MoE scattered dispatch state reuse`](chapter29_qwen3_moe_scattered_reuse.md)（Closed）

## 预期能力变化

- **模型基本组件**：具备 Qwen3 top-k router、稀疏 expert SwiGLU 和 MoE decoder block。
- **训练与推理**：Float32 correctness oracle、Float32/native BF16 active-expert streaming、CPU/XLA/CUDA sparse dispatch，以及真实 30B-A3B GPU resident/offload session 已完成。RTX 4090 D 实际分配完整 40,960-token BF16 KV；32-token grouped WMMA 的端到端 prefill/decode 相对 scalar production 为 `1.092× / 1.335×`。device expert cache 支持 global/layer-balanced LRU；后者在自然文本 trace 的同 8 GiB 预算下少读 `18.71%`、加速 `1.082×`。scalar CUDA cache hit 可用 scattered pointer-table dispatch 直接读取 BF16 expert matrices，冻结短请求相对 materialized baseline 加速 `82.89×`；pointer plans/workspaces 现在有界跨请求复用。
- **HuggingFace 互操作**：严格解析 `qwen3_moe` config，映射官方 expert 权重；tiny 与官方 30B-A3B 均完成逐层/logits/cache parity，真实 BF16 路由边界差异单独量化。
- **工程与测试**：测试按 Chapter 24 架构/parity、Chapter 25 offload session、Chapter 26 device LRU、Chapter 27 scan-resistant 策略/自然文本预算曲线、Chapter 28 scattered dispatch/GC 与 Chapter 29 pointer/workspace reuse 生命周期分层。

## Episode Close 条件

- 至少一种官方 Qwen3 MoE checkpoint 完成可复现的真实权重 parity。
- MoE prefill/decode 不再计算全部未选中的 expert，并有性能与内存证据。
- 后续 Qwen 模型方向已经根据已验证的架构缺口确定。

## 本卷回顾

- **形成的能力闭环**：Chapter 24 已完成 config/checkpoint/parity/sparse dispatch，Chapter 25 形成 RTX 4090 D offload session，Chapter 26 加入 byte-budgeted device LRU，Chapter 27 按模型逐层访问拓扑解决全局 LRU scan thrashing，Chapter 28 消除命中路径的 active tensor materialization，Chapter 29 再以 generation-safe 的有界状态复用 pointer plans/workspaces；Episode 仍为后续模型扩展保持 Open。
- **关键实验与指标**：30B Float32 top-8 槽位 `1,152 / 1,152`，BF16 槽位重合 `95.92%`；offload 硬下限 `7.166 GiB`；自然文本 A→B→A 上 layer-balanced 8 GiB 相对 global 8 GiB 少读 `18.71%`、加速 `1.082×`；2-token 全命中请求用 scattered + gc8 消除 `10.551 GB` materialization，从 `12.477 s` 降到 `0.151 s`（`82.89×`）。
- **失败、偏差与未解决问题**：BF16 跨框架路由边界仍有差异；40K full-window/长序列质量尚未实跑；scattered 当前只支持 scalar CUDA BF16 cache，grouped WMMA 与同步 miss path 尚未优化。复用后完全关闭 forced GC 的 500 次请求没有 live leak，但 allocator pool 仍扩张 `885.9 MB` 且 max `758 ms`，因此没有改变 gc8 推荐。
- **重要架构决策**：correctness oracle、streamer 与 production dispatch 分层；cache 默认关闭且默认策略保持 global 兼容，真实 sequential-scan workload 显式使用 layer-balanced；scattered 与放宽 GC 均保持 opt-in；reuse 自动启用但按每层 4 plans/全 session 4 shapes 有界，entry generation 而非裸地址定义 identity。
- **最重要的认知变化**：BF16 支持和 cache hit 都必须落到具体生命周期与数据移动；只有 dtype/kernel/cache 状态正确，不能替代对 D2D copy、临时分配和 GC 的真实测量。
- **进入下一章的问题**：如何用 pinned-memory、异步 H2D 与 layer-ahead prefetch 隐藏自然文本 cache miss；同时保留严格的 safetensors 生命周期、容量门禁和 parity。
