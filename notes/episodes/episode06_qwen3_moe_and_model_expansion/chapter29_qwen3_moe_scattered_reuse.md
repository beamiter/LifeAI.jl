# Chapter 29 — Qwen3 MoE scattered dispatch state reuse

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Closed

## Open：核心问题

Chapter 28 消除了 cache hit 的 `10.551 GB` active tensor materialization，但
scattered dispatch 仍在每层上传 pointer table，并重新分配 hidden、routed output
和 combined output workspace。完全关闭 forced GC 的 100 次实验损失了
`0.9375 GiB` allocator free。怎样在不让 pointer plan 持有已淘汰 expert、也不
引入悬空地址复用的前提下，有界复用这些状态？复用后 `gc=0` 是否足够稳定，能
取代每 8 层 GC 的建议？

## 预期结果

本章 Close 时，应当可以展示或验证：

1. 同一 active expert generation 的重复 hit 不再上传或构建 pointer table；
2. 相同 route shape 的 scattered hidden/output workspace 跨层、跨请求复用；
3. pointer/workspace cache 都有明确上限，cache clear/reconfigure 会使其失效；
4. expert 被淘汰、重新载入或 CUDA allocator 重用裸地址时，不会命中旧 plan；
5. 官方 30B-A3B 至少运行 100 次 `gc=8` 和 500 次 `gc=0`，区分 allocator
   pool 高水位与真正的 live allocation 泄漏；
6. 如果复用没有消除 `gc=0` 的尾延迟或显存压力，如实保留负面结果。

## 计划与结果

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| reuse state | 模型 / 工程 | pointer plans + shape workspaces | build/reuse counters | 已完成 |
| stale pointer 防护 | 工程 | entry generation signature | clear/rebuild exact | 已完成 |
| 有界生命周期 | 工程 | 4 plans/layer + 4 shapes | byte stats 与 clear | 已完成 |
| tiny CUDA 长重复 | 模型 / 工程 | 20-hit exact loop | 零 build/allocation | 已完成 |
| 真实长期 sweep | 模型 / 工程 | gc8 100 + gc0 500 | timing/free/reclaim | 已完成 |

## Close 条件

- tiny repeated request 的 pointer builds/uploads 与 workspace allocations 全为零；
- 复用前后 logits 逐位一致，clear 后 state bytes 归零并能正确重建；
- 真实重复请求全部 exact、零 expert I/O，并且每次都是 96/96 纯复用；
- pointer plans 不持有 expert tensor owners，容量有上限；
- `gc=0` 是否可用于长期运行由实测 allocator、reclaim 和尾延迟共同决定。

## 学习重点

- **裸地址不能独自充当 cache identity**：CUDA allocator 可能把已释放地址分配
  给别的 expert。每次从磁盘读取的 expert entry 现在取得单调 generation；plan
  以有序 expert ids + generations 匹配，只在新 generation 上读取矩阵地址并
  上传 device pointer table。
- **安全复用不能悄悄延长 expert 生命周期**：pointer plan 只保存 ids、generation
  与 raw pointers，不保存 `CuArray` owner。LRU 淘汰仍能释放 expert；旧 plan 最多
  每层保留 4 个，并且不会因地址偶然相同而命中新 entry。
- **workspace 可以按 route shape 共享**：单个 session 的 `d_model`、hidden dim
  和 top-k 固定，因此以 `(num_tokens, pair_count)` 为 key 即可复用三块数组。
  shape cache 最多 4 组，默认 stream 的顺序执行保证下一层覆盖前，上一层 residual
  已消费 output。session 本来就不是并发请求容器。
- **真实状态很小**：冻结 workload 的 96 层调用只需 `26,832` pointer bytes 和
  `294,912` workspace bytes。fill 创建 96 个 plans、2 个 shape workspaces；hit
  为 96 plan reuses、96 workspace reuses，build/upload/allocation 全为零。
- **复用不是 GC 的完全替代**：`gc=0` 500 次没有 live leak——显式 reclaim 后
  free memory 比开始多 `65,536,000` bytes；但运行中 allocator pool 仍扩张
  `885,915,648` bytes，p95 `283 ms`、max `758 ms`。其来源是 route、attention、
  residual 等尚未复用的临时数组，而不是 scattered state。
- **负面性能结果同样重要**：本章不宣称 latency speedup。pointer payload 本来
  只有 26 KB，且 `gc=8` 会回收其他路径的临时对象；复用的主要价值是确定性
  生命周期、零重复上传和定位剩余内存池压力。

## 风险与取舍

- reuse 随 CUDA scattered dispatch 自动启用，没有新增用户配置；portable
  fallback 的 reuse counters/bytes 保持零，原有 materialized 行为不变。
- 每层 4 个 pointer plans 足够容纳冻结 prefill/decode generations，但自然文本
  路由可能滚动淘汰旧 plan；这是有界 cache，不承诺任意 trace 零重建。
- workspace cache 最多 4 个 route shapes。超过上限时按 LRU 删除；CUDA.jl
  allocator/stream 负责安全延迟回收底层 storage。
- `gc=8` 的 100 次 free checkpoints 从 `8,751,153,152` 到
  `8,838,184,960` bytes，没有下降；但这仍不是 arbitrary-long 服务的形式保证。
- `gc=0` 比 `gc=8` 的重复中位快 `1.211×`，但显存池高水位和更差尾延迟使它
  继续不适合作为长期默认。本章没有改 Chapter 28 的推荐。
- 真实实验仍是 2-token prompt + 1-token decode 的全命中 workload；自然文本
  cache-miss trace、异步 I/O、pinned host buffer 与 prefetch 尚未覆盖。

## 实验与过程记录

### 2026-08-12：tiny reuse 与失效

- tiny 2 层首次 scattered prefill 构建 2 plans、上传 192 bytes；只分配 1 个
  workspace，第二层复用。重复 prefill 构建/上传/分配均为零，2 plans 和同一
  workspace 全复用，logits exact。
- 连续 20 次 repeated prefill 均 exact，且每次 pointer builds 与 workspace
  allocations 为零。
- `clear_hf_qwen3_moe_expert_cache!` 同时清除 expert cache 和 opaque scattered
  state，pointer/workspace bytes 归零；下一次请求重新构建并保持 exact。
- portable fallback 不创建 CUDA state，新增统计全部为零。

### 2026-08-12：真实 30B-A3B 100/500 次稳定性

- RTX 4090 D、官方 revision `ad44e777…85d39`、native BF16、40,960-token static
  KV capacity、8 GiB global LRU、scattered scalar CUDA。两档各自 clear 后 fill，
  再测 hit 与长重复；每 1/10 窗口记录 GPU free。
- `gc=8` hit 为 `0.170 s`；100 次 median/p95/max 为
  `0.180 / 0.201 / 0.483 s`。所有请求均 exact、零 I/O、96 pointer reuses、
  96 workspace reuses；free checkpoints 没有下降，explicit reclaim 后比开始
  多 `97,583,104` bytes。
- `gc=0` hit 为 `0.136 s`；500 次 median/p95/max 为
  `0.149 / 0.283 / 0.758 s`。同样全部 exact、零 I/O、纯复用；allocator free
  从 `8,870,035,456` 降到 `7,984,119,808` bytes，但 reclaim 后恢复为
  `8,935,571,456` bytes。这排除 live leak，同时确认没有 forced GC 时的 pool
  高水位与尾延迟仍存在。
- 两档 fill/hit 与跨配置 logits 逐位一致；相对 Transformers BF16 reference，
  prefill/decode max-abs 为 `0.3125 / 0.25`，argmax 均一致。原始报告位于
  `/tmp/qwen3_moe_cuda_scattered_reuse_final.json`，仓库冻结精简摘要与源码哈希。

## Close 回顾

- **完成了什么**：为 CUDA scattered dispatch 加入有界 pointer-plan/workspace
  reuse，以 entry generation 保证地址安全，并把 build/reuse/live bytes 暴露到
  request stats。
- **验证证据**：默认 Chapter 29 专项 `70 / 70`、CUDA 专项 `90 / 90`；真实
  100 + 500 次均 exact、零 I/O、零 hit build/upload/allocation。
- **没有完成及原因**：没有让 `gc=0` 成为长期建议，也没有获得可归因的 latency
  加速；实测把剩余压力定位到 scattered dispatch 之外的临时数组。
- **最重要的认知变化**：allocator free 下降不等同于 live leak，必须在同步后
  reclaim 并结合时间序列判断；同时，“可复用”必须先解决身份与所有权，而不能
  只把裸指针保存更久。
- **是否满足 Close 条件**：是。
- **带到下一章的问题**：优先优化 cache miss 的同步磁盘读取/H2D，还是先复用
  router/attention/residual 临时数组来压低 `gc=0` 的 pool 高水位？自然文本
  miss-heavy trace 表明前者对真实端到端收益更大，下一章应进入 pinned-memory、
  layer-ahead async prefetch 的可行性与 correctness 边界。
