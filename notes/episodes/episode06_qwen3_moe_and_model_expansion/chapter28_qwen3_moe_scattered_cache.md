# Chapter 28 — Qwen3 MoE scattered expert cache dispatch

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Closed

## Open：核心问题

Chapter 27 让 device cache 在自然文本顺序扫描中保住更多 expert，但命中路径仍
把每个 expert 的三个独立 BF16 matrix 用 device `cat` 拼成 active-local 3D
tensor，并在每层强制触发 GC。缓存已经避免磁盘读取和 H2D 上传，为什么重复短
请求仍需十余秒？能否让 CUDA kernel 直接消费 cache 持有的分散矩阵，同时给出
不会用无界显存换取最低单次延迟的 GC 策略？

## 预期结果

本章 Close 时，应当可以展示或验证：

1. cache hit 不再为每层 materialize 三个 active expert tensors；
2. CUDA indexed 与 bucketed sparse dispatch 都能通过小型 device pointer table
   直接读取缓存中的 BF16 matrices；
3. materialized/scattered 输出逐位一致，并保持 Transformers BF16 reference
   的 argmax parity；
4. 强制 GC cadence 可配置，真实 30B-A3B 对比每层 GC、每 8 层 GC 和关闭强制
   GC 的 latency/显存边界；
5. 兼容默认值、accelerator/type 限制与下一瓶颈均明确记录。

## 计划与结果

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 命中路径审计 | 模型 / 工程 | materialization/GC 计数 | 真实 30B bytes/calls | 已完成 |
| scattered dispatch | 模型 / 工程 | CUDA pointer-table kernels | indexed/bucketed tiny exact | 已完成 |
| lifecycle 配置 | 工程 | dispatch 与 GC interval API | load/runtime fail-closed | 已完成 |
| 真实 checkpoint 对照 | 模型 / 工程 | 4 配置 benchmark | fill/hit/parity/latency | 已完成 |
| 长重复边界 | 工程 | 100-hit `gc=0` trace | exact、I/O、显存、尾延迟 | 已完成 |

## Close 条件

- CUDA tiny materialized/scattered 的 fill 与 hit logits 逐位一致；
- 真实 30B hit 的 active materialization bytes 从非零降到零，expert read/upload
  继续为零；
- 推荐配置相对 materialized baseline 获得数量级加速，且 measured GPU free
  memory 不出现持续下降；
- 禁用 GC 如果更快但不稳定，必须作为负面证据而不是默认推荐；
- 默认 API 保持兼容，grouped/scattered 与非 CUDA 边界 fail closed 或明确回退。

## 学习重点

- **LifeAI 早已支持 BF16，瓶颈不是 dtype**：缓存中的 expert weights 和新 kernel
  都使用 BF16；旧命中路径慢在每层把分散矩阵重新拷贝成连续 3D tensor，以及
  随后的 host GC，而不是退回 Float32 权重。
- **cache hit 仍可能包含巨量 device-to-device copy**：冻结请求的 48 层
  prefill + 48 层 decode 共 materialize 96 次，累计复制
  `10,550,771,712` bytes。磁盘与 H2D 计数为零并不等于数据路径已零开销。
- **地址表足以表达分散权重**：每层只上传 gate/up/down 三组 device pointers，
  整个 hit request 共 `26,832` bytes；相对被消除的 active materialization 是
  `393,216×` 更小。indexed 与 bucketed kernel 都按 local expert id 解引用同一
  表，不再改变 cache 的所有权或布局。
- **GC cadence 是吞吐与生命周期的显式参数**：scattered 仍会产生 route/workspace
  临时数组。每层 GC 共 96 次；每 8 层只做 12 次，使 hit request 再加速
  `1.599×`。完全关闭强制 GC 的单次 hit 更快，但 100 次后 allocator free 少了
  `0.9375 GiB`，并出现 `728.8 ms` 尾延迟，因此不能作为长期默认。
- **命中计算终于暴露出真正的下一层开销**：稳定配置的 2-token + 1-decode
  hit 已从 `12.477 s` 降到 `0.151 s`。下一步不再是继续调 cache policy，而是
  复用 pointer table/route workspace，并优化 miss path 的同步读取与上传。

## 风险与取舍

- `expert_cache_dispatch=:materialized` 和 `expert_gc_interval_layers=1` 保持默认，
  避免改变已有 CPU、非 CUDA accelerator、无 cache 与 grouped WMMA 行为。
- `:scattered` 当前要求正的 device cache budget、`grouped_experts=false`，CUDA
  specialization 只接受形状一致的 BF16 `CuMatrix` expert entries；portable
  路径会安全回退到 materialized dispatch。
- raw device pointer 的有效期由 session cache entries 持有；clear/reconfigure
  只在请求边界使用。pointer tables 目前每层重新构造和上传，尚未复用。
- `82.89×` 是 8 GiB global cache、冻结 2-token prompt + 1-token decode 的纯命中
  request 加速，不代表 cache miss、32-token 自然文本或任意生成长度的端到端
  加速。Chapter 27 的自然文本预算建议仍独立成立，两种策略可以组合配置。
- `gc=8` 的稳定判断来自本轮配置起止 free memory 一致；它不是 arbitrary-long
  workload 的数学保证。更长服务 trace 仍应观测 allocator 和尾延迟。
- 本章没有实现 grouped WMMA scattered dispatch、pointer-table/workspace reuse、
  pinned host memory、异步 H2D 或 layer-ahead prefetch。

## 实验与过程记录

### 2026-08-12：tiny CUDA pointer-table correctness

- generic session 继续支持 materialized fallback，并新增 dispatch、GC interval、
  active materialization、pointer upload、scattered dispatch 和 forced-GC 统计。
- CUDA tiny fixture 的 materialized/scattered fill 与重复 hit 逐位一致；scattered
  hit 为 8 cache hits、零 read/upload、零 active materialization、2 次 direct
  dispatch 和 192 pointer bytes，`gc=0` 时零 forced GC。
- scattered + grouped、零 cache budget、未知 dispatch 与负 GC interval 均拒绝；
  runtime reconfigure 会先清空 cache，再原子切换这些配置。

### 2026-08-12：真实 30B-A3B dispatch/GC sweep

- RTX 4090 D、官方 revision `ad44e777…85d39`、native BF16、40,960-token static
  KV capacity、8 GiB global LRU；workload 为 2-token prompt + 1-token decode，
  每档先 fill 再测全命中请求。
- materialized + 每层 GC 的 hit 为 `9.750 / 2.727 / 12.477 s`
  （prefill/decode/request），96 次 active materialization 共 `10.551 GB`，96 次
  forced GC。expert read/upload 已是零。
- scattered + 每层 GC 降到 `0.120 / 0.121 / 0.241 s`；materialization 为零，
  只上传 `26,832` pointer bytes，request 加速 `51.86×`。
- scattered + 每 8 层 GC 为 `0.078 / 0.073 / 0.151 s`，只做 12 次 forced GC；
  相对 scattered gc1 再快 `1.599×`，相对 materialized request 快 `82.89×`。
  该配置的 measured GPU free 从 `17,535,401,984` 到 cache-filled 后的
  `9,102,753,792` bytes，与同配置 fill/hit 结束值一致。
- `gc=0` 单次 hit 为 `0.141 s`；100 次重复中位 `0.109 s`、最快 `0.105 s`，
  但最慢 `0.729 s`，allocator free 从 `9,069,199,360` 降到
  `8,062,566,400` bytes。100 次均 logits exact 且零 expert I/O，这证明退化来自
  临时分配/回收，而不是 cache miss。
- 所有配置 fill/hit 和跨配置 logits 逐位一致；相对 Transformers BF16 reference，
  prefill/decode max-abs 为 `0.3125 / 0.25`，两者 argmax 一致。原始报告位于
  `/tmp/qwen3_moe_cuda_scattered_cache_repeat100.json`，仓库冻结精简摘要与三份
  源码 SHA256。

## Close 回顾

- **完成了什么**：让 CUDA sparse kernels 直接读取 cache 持有的 BF16 expert
  matrices，消除命中路径的 active tensor 拼接，并把 forced-GC cadence 变成
  可测、可重配的 session 参数。
- **验证证据**：默认 Chapter 28 专项 `73 / 73`、CUDA 专项 `13 / 13`；真实
  30B hit materialization 从 `10.551 GB` 降到零，推荐 `scattered + gc8` 的
  request 从 `12.477 s` 降到 `0.151 s`（`82.89×`），correctness 全保持。
- **没有完成及原因**：没有复用 pointer tables/workspaces，也没有优化 miss I/O；
  本章先关闭已经量化清楚且独立的命中计算瓶颈。
- **最重要的认知变化**：BF16 与 cache hit 都只是数据格式/状态事实，不自动
  意味着快速；必须继续核对一次请求内部实际发生的 D2D copy、分配和 GC。
- **是否满足 Close 条件**：是。
- **带到下一章的问题**：怎样让每层 pointer table 和 sparse workspace 跨请求
  复用，并在更长自然文本生成 trace 上验证稳定性，再用 pinned-memory/异步预取
  隐藏不可避免的 cache miss？
