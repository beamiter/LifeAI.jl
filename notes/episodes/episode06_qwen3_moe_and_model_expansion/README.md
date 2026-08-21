# Episode 06 — Qwen3 MoE 与模型架构扩展

> 状态：Closed
>
> 收录章节：Chapter 24–35、41

这一卷继续推进模型本体能力：从已经闭环的 Qwen3 Dense 扩展到稀疏 MoE，随后再评估 reranker 与视觉语言模型。每种新结构都以官方配置和权重命名为契约，以可独立复现的数值 parity 为完成依据，不把仅能构造 topology 写成真实 checkpoint 已可用。

## 章节目录

1. [`Chapter 24 — Qwen3 MoE 架构支持`](chapter24_qwen3_moe_architecture.md)（Closed）
2. [`Chapter 25 — Qwen3-30B-A3B GPU resident/offload session`](chapter25_qwen3_moe_gpu_offload.md)（Closed）
3. [`Chapter 26 — Qwen3 MoE active-expert device cache`](chapter26_qwen3_moe_expert_cache.md)（Closed）
4. [`Chapter 27 — Qwen3 MoE layer-balanced expert cache`](chapter27_qwen3_moe_layer_balanced_cache.md)（Closed）
5. [`Chapter 28 — Qwen3 MoE scattered expert cache dispatch`](chapter28_qwen3_moe_scattered_cache.md)（Closed）
6. [`Chapter 29 — Qwen3 MoE scattered dispatch state reuse`](chapter29_qwen3_moe_scattered_reuse.md)（Closed）
7. [`Chapter 30 — Qwen3 MoE bounded async miss pipeline`](chapter30_qwen3_moe_async_miss_pipeline.md)（Closed）
8. [`Chapter 31 — Qwen3 MoE storage-aware read-worker sweep`](chapter31_qwen3_moe_read_worker_sweep.md)（Closed）
9. [`Chapter 32 — Qwen3 MoE adjacent safetensors read experiment`](chapter32_qwen3_moe_coalesced_reads.md)（Closed）
10. [`Chapter 33 — Qwen3 MoE safetensors decode copy elision`](chapter33_qwen3_moe_decode_copy_elision.md)（Closed）
11. [`Chapter 34 — Qwen3 MoE bounded safetensors read-buffer reuse`](chapter34_qwen3_moe_read_buffer_reuse.md)（Closed）
12. [`Chapter 35 — Qwen3 MoE bounded host projection-buffer reuse`](chapter35_qwen3_moe_host_buffer_reuse.md)（Closed）
13. [`Chapter 41 — Qwen3 MoE grouped scattered expert-cache dispatch`](chapter41_qwen3_moe_grouped_scattered.md)（Closed）

## 预期能力变化

- **模型基本组件**：具备 Qwen3 top-k router、稀疏 expert SwiGLU 和 MoE decoder block。
- **训练与推理**：Float32 correctness oracle、Float32/native BF16 active-expert streaming、CPU/XLA/CUDA sparse dispatch，以及真实 30B-A3B GPU resident/offload session 已完成。RTX 4090 D 实际分配完整 40,960-token BF16 KV；device expert cache 支持 global/layer-balanced LRU，后者在自然文本 trace 的同 8 GiB 预算下少读 `18.71%`、加速 `1.082×`。scalar 与 grouped BF16 WMMA 都可用 generation-safe scattered pointer tables 直接读取分散 cache matrices；Chapter 41 的 grouped 路径相对相同数值契约的 materialized 路径，在 2/32-token request 分别加速 `10.745× / 6.103×`，消除 `10.560 / 16.930 GB` active concat。自然文本 cold/revisit miss path 在本机 8-worker 下相对 1-worker 分别加速 `2.840× / 2.566×`。
- **HuggingFace 互操作**：严格解析 `qwen3_moe` config，映射官方 expert 权重；tiny 与官方 30B-A3B 均完成逐层/logits/cache parity，真实 BF16 路由边界差异单独量化。
- **工程与测试**：测试按 Chapter 24 架构/parity、Chapter 25 offload session、Chapter 26 device LRU、Chapter 27 scan-resistant 策略/自然文本预算曲线、Chapter 28 scalar scattered dispatch/GC、Chapter 29 pointer/workspace reuse、Chapter 30 bounded miss pipeline、Chapter 31 storage-aware worker sweep、Chapter 32 adjacent-read 实验、Chapter 33 decode copy elision、Chapter 34/35 bounded raw/final host-buffer reuse，以及 Chapter 41 grouped pointer-WMMA/lifecycle/真实结果契约分层。

### 历史 benchmark provenance

Chapter 25–35 冻结报告中的 `source_sha256` 表示**产生该次 timing 时的源码字节**，
不要求后来演进的工作树继续保持相同 hash。每份报告各自的
`original_report_commit`（即首次加入该 measured report 的提交）及其源码快照由
[`chapter25_35_source_snapshot.json`](../../../benchmark_results/qwen3_moe_historical_provenance/chapter25_35_source_snapshot.json)
集中冻结。此次 provenance 修复把各 summary 的 `source_sha256` 元数据恢复为其
`original_report_commit` 已记录的原值，并恢复 Chapter 33 的
`baseline.source_report_sha256` 这一处 Chapter 32 报告引用；后者是唯一位于
`source_sha256` 外的修复字段。timing、traffic、memory、correctness 及其他 metric
均未改动。测试将旧报告 hash 与对应的 per-report 快照对照；有完整 Git 历史时
还会从原提交重算，无完整历史的源码包仍可验证登记信息。当前源码路径必须存在，但
文件可以继续演进。没有重跑原 workload 时，禁止把新源码 hash 写回旧 timing/report；
新实现必须生成自己的新章节报告与 provenance。

## Episode Close 条件

- 至少一种官方 Qwen3 MoE checkpoint 完成可复现的真实权重 parity。
- MoE prefill/decode 不再计算全部未选中的 expert，并有性能与内存证据。
- 后续 Qwen 模型方向已经根据已验证的架构缺口确定。

## 本卷回顾

- **形成的能力闭环**：Chapter 24 已完成 config/checkpoint/parity/sparse dispatch，Chapter 25 形成 RTX 4090 D offload session，Chapter 26–31 依次加入 device LRU、scan-resistant policy、scalar scattered state 与 bounded miss overlap，Chapter 32–35 通过负/正实验收敛 storage/host allocation；Chapter 41 最后让 grouped BF16 WMMA 直接消费分散 cache pointers，在不改变 grouped logits、routes 或 traffic 的前提下取消 active 3D materialization。三个 Episode Close 条件全部完成。
- **关键实验与指标**：30B Float32 top-8 槽位 `1,152 / 1,152`，BF16 槽位重合 `95.92%`；offload 硬下限 `7.166 GiB`；自然文本 A→B→A 上 layer-balanced 8 GiB 相对 global 8 GiB 少读 `18.71%`、加速 `1.082×`；English32 miss path 以 96 MiB raw+final staging pool 将 Julia allocation 降至约 `0.288 GB`；Chapter 41 grouped-scattered 在 2/32-token 上相对 materialized-grouped 加速 `10.745× / 6.103×`，前者 outputs/routes 和每次 I/O/cache traffic 全部 exact。
- **失败、偏差与未解决问题**：BF16 跨框架路由边界仍有差异；40K full-window/长序列质量尚未实跑。完全关闭 forced GC 的 500 次请求没有 live leak，但 allocator pool 仍扩张 `885.9 MB` 且 max `758 ms`。pinned async 比 overlapped pageable 慢 `1.07%`；相邻 tensor coalescing虽减少 `64.8%` read syscall，cold/revisit 却分别慢约 `23.9% / 33.6%`。grouped retained workspace 是 4-shape 数量上限而非独立 byte budget，session 仍限定单请求/default stream。这些边界都没有被包装成已完成能力。
- **重要架构决策**：correctness oracle、streamer 与 production dispatch 分层；cache 默认关闭且默认策略保持 global 兼容，真实 sequential-scan workload 显式使用 layer-balanced；scattered 与放宽 GC 均保持 opt-in；reuse 自动启用但按每层 4 plans/全 session 4 shapes 有界，entry generation 而非裸地址定义 identity。启用正 device-cache budget 的 Ampere+ BF16 配置建议显式选 grouped scattered，但零配置 streaming 默认不变。
- **最重要的认知变化**：BF16 支持和 cache hit 都必须落到具体生命周期与数据移动；“同算法”还不足以声称可替换，只有同一 grouped 舍入口径下的 bitwise logits/routes 与逐次 traffic equality 才能把 pointer path 升为正式实现。并行 read 的收益也不能自动归功于 pinned H2D。
- **进入下一卷的问题**：模型线恢复时，下一 Qwen 架构确定为 Qwen3-VL，以补齐 vision encoder/projector/image-token layout；reranker 归入记忆检索线。当前先按用户优先级回到 Episode 08，完成环境事件记忆写回与跨 adapter safety semantics。
