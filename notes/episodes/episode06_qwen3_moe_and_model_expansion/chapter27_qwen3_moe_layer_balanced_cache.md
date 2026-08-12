# Chapter 27 — Qwen3 MoE layer-balanced expert cache

> 所属 Episode：Episode 06 — Qwen3 MoE 与模型架构扩展
>
> 状态：Closed

## Open：核心问题

Chapter 26 证明 8 GiB device LRU 可以完整容纳一个 2-token 冻结请求，但真实
自然文本会在 48 层依次扫描数千个 expert-layer entries。当工作集大于缓存时，
全局 LRU 是否会在下一次请求到达早期层之前把这些层的 entries 全部淘汰？怎样
在相同显存预算下保留跨请求路由局部性，并找到真实 workload 的容量拐点？

## 预期结果

本章 Close 时，应当可以展示或验证：

1. tiny fixture 可确定性复现全局 LRU 的 sequential-scan thrashing；
2. 新策略按 decoder layer 公平分配 expert slot，并继续遵守总 byte budget；
3. runtime 可以在不重载 resident tree、40K KV 或 compiled kernels 的情况下
   清空并切换缓存预算/策略；
4. 官方 30B-A3B 用固定中英文自然文本 A→B→A trace 对比 global 8 GiB 与
   layer-balanced 4/6/8 GiB 的命中、淘汰、I/O、显存和端到端时间；
5. 容量建议同时考虑 latency 与 GPU headroom，不把最高命中率直接等同于最佳
   部署配置。

## 计划与结果

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| scan-thrashing oracle | 模型 / 工程 | tiny global/balanced 对照 | 重复请求 hit/eviction | 已完成 |
| layer-balanced LRU | 模型 / 工程 | `:layer_balanced_lru` | 分层 slot 与总 byte 门禁 | 已完成 |
| runtime reconfigure | 工程 | `configure_hf_qwen3_moe_expert_cache!` | 不重载 resident session | 已完成 |
| 真实自然文本 sweep | 模型 / 工程 | benchmark 与冻结摘要 | 4/6/8 GiB A→B→A | 已完成 |
| 容量决策 | 工程 | workload-bound recommendation | 时间、I/O、free memory | 已完成 |

## Close 条件

- 相同 tiny cache budget 下，layer-balanced 的重复请求命中严格多于 global，
  logits 完全一致；
- 真实 30B trace 中，layer-balanced 8 GiB 相对 global 8 GiB 同时提高命中率、
  减少 read/upload 并获得正的端到端加速；
- 4/6/8 GiB 容量增加时，命中率单调增加、expert I/O 单调下降；
- 推荐预算明确绑定冻结 workload，并保留 arbitrary workload 不保证的边界；
- 如果 I/O 降幅没有等比例转化为 latency，明确记录新的主导瓶颈。

## 学习重点

- **MoE layer scan 不是普通随机访问**：一次 request 固定从 layer 1 走到 48。
  全局 LRU 在 910-slot cache 上扫描约 3,000 次 expert-layer accesses，会在下个
  request 再到 layer 1 前淘汰早期层，即使两次 prompt 完全相同也会 thrash。
- **公平保留比全局“最新”更符合模型拓扑**：layer-balanced 将总 slot capacity
  用 `divrem` 分给每层，再在层内按 LRU 淘汰。4/6/8 GiB 对应每层至少
  `9 / 14 / 18` 个 entries，能让所有层都保留一部分历史路由。
- **更高 hit rate 不必然线性加速**：8 GiB balanced 相对 global 少读/少传
  `18.71%`，端到端只快 `8.24%`。命中后仍需把独立 expert tensors 拼成
  active-local 3D tensor，当前每层还有 device `cat`、临时 workspace 和 GC。
- **预算选择是多目标问题**：8 GiB 命中最高，但最终只余 `1.88 GB` GPU free；
  4 GiB 的 trace 时间只慢 `1.48%`，却余 `7.44 GB`，因此冻结 workload 的
  推荐值选择 4 GiB。

## 风险与取舍

- `:global_lru` 仍是默认策略，保证 Chapter 26 API/行为兼容；真实 workload
  可显式选择 `:layer_balanced_lru`。
- layer-balanced 假设各层单个 expert payload 大小相同；当前 Qwen3-30B-A3B
  每项固定 `9,437,184` bytes。全局 byte 门禁仍作为最终容量保护。
- 自然文本 workload 是固定 32-token 英文、32-token 中文，再回访同一英文，
  每次追加一次 greedy decode；它比 2-token case 真实，但不是完整多轮聊天分布。
- 同一 session 依序跑配置可避免重复 load/compile，但 allocator 历史仍可能给
  timing 带来小幅顺序偏差；结论同时依赖精确 hit/I/O，而不只依赖秒数。
- 本章没有优化 device concat/GC，也没有实现 pinned host buffer、异步 H2D 或
  layer-ahead prefetch。

## 实验与过程记录

### 2026-08-12：tiny scan-thrashing oracle

- tiny fixture 为 2 层，每次两 token prompt 在每层激活 4 experts；缓存预算
  只有 4 entries。
- global LRU 第一次请求 miss 8、evict 4；第二次仍 hit 0、miss 8、evict 8，
  证明顺序扫描完全冲掉缓存。
- 同 session runtime 切换为 layer-balanced 后，每层保留 2 entries；第二次请求
  hit 4、miss 4、evict 0，read/upload 减半，logits 与 global 逐位相同。

### 2026-08-12：真实 30B 自然文本 cache sweep

- RTX 4090 D、官方 revision `ad44e777…85d39`、40,960-token BF16 static KV、
  scalar production dispatch。固定 workload 为
  `English32 → Chinese32 → English32`，每条 prompt 后 greedy decode 1 token；
  文本、token ids 与 tokenizer 均以 SHA256 冻结。
- global 8 GiB 的 910 slots 在 trace 上只有 `435 / 9,194 = 4.73%` 命中，读取
  与上传 `82,660,294,656` bytes，耗时 `553.94 s`；English revisit 命中率
  只有 `5.11%`，验证全局 sequential-scan thrashing。
- layer-balanced 4/6/8 GiB 的 trace 命中率为
  `12.07% / 17.81% / 22.56%`，I/O 为
  `76.29 / 71.32 / 67.19 GB`，耗时 `519.35 / 512.83 / 511.76 s`；容量
  增加时命中/I/O 单调改善。
- balanced 8 GiB 相对 global 8 GiB 命中率增加 `17.83` 个百分点，少读/少传
  `18.71%`，端到端加速 `1.082×`。English revisit 命中率达到 `34.66%`。
- balanced 4 GiB 相对 global 8 GiB 已少读 `7.71%`、加速 `1.067×`；它比
  balanced 8 GiB 只慢 `1.48%`，最终 free memory 却为
  `7,440,629,760` vs `1,883,635,712` bytes，因此选为该 trace 的推荐预算。
- 四档 English revisit 的 prefill/decode logits 和 greedy decode token 都与各自
  首次 English 请求逐位一致。原始报告保存在本机
  `/tmp/qwen3_moe_cuda_layer_balanced_cache.json`；仓库冻结精简摘要与源码哈希。

## Close 回顾

- **完成了什么**：实现 scan-resistant layer-balanced LRU 与运行时 cache
  reconfigure，并用真实 30B 中英文 A→B→A trace 给出 4/6/8 GiB 容量曲线。
- **验证证据**：tiny 策略测试 `29 / 29`、真实结果 contract `36 / 36`、默认
  全套 `6,108 / 6,108`；8 GiB 同预算相对 global 少读 `18.71%`、加速
  `1.082×`，所有 replay exact。
- **没有完成及原因**：没有把 I/O 降幅完全转成 latency；device active tensor
  concat、临时分配与 GC 已成为更突出瓶颈，需独立修改数据布局与 kernel 接口。
- **最重要的认知变化**：单卡 MoE 缓存不能只选一个通用 LRU；模型的逐层访问
  顺序本身就是缓存策略输入。真实部署预算也不能只追命中率，必须同时保留运行
  workspace 和长上下文余量。
- **是否满足 Close 条件**：是。
- **带到下一章的问题**：怎样让 sparse dispatch 直接消费分散的 cached expert
  tensors，或复用固定 active workspace，从而消除每层 device `cat` 与强制 GC，
  再把 miss path 接到 pinned-memory/异步预取？
