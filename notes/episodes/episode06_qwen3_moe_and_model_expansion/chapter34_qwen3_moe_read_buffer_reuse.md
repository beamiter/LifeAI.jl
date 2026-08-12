# Chapter 34 — Qwen3 MoE bounded safetensors read-buffer reuse

> 状态：Closed
> 日期：2026-08-12
> 真实资产：`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
> 设备：NVIDIA GeForce RTX 4090 D，本地 NVMe，Julia 8 threads

## 本章问题

Chapter 33 已移除多维 BF16 decode 的线性中间 copy，但 English32 请求仍分配
约 `57.646 GB`。其中每个 expert projection 的 `read(io, 3 MiB)` 都新建 raw
byte vector；3,039 个 active-expert miss、每个 3 个 projection，逻辑 payload
合计 `28.680 GB`。本章验证能否按 reader 并发度有界复用这层 raw storage，且不
改变 Chapter 31/32 选出的逐 projection I/O/decode overlap。

## 实现

新增的 `_SafetensorsReadBufferPool` 预分配 `Channel{Vector{UInt8}}`：

- `:overlapped` tensor reads 的池大小等于配置的 `read_workers`；
- `:sequential` tensor reads 只保留 1 个 buffer；
- cache budget 为 0、`:shared_open` 或 `:coalesced` 时不建池；
- 每个 active expert 借用 1 次，依次读取/解码 gate、up、down，三张最终 owning
  matrix 都产生后立即归还；异常路径也在 `finally` 中归还；
- pool 在 request allocation 计量之外建立，真实 8-worker 配置常驻
  `8 × 3,145,728 = 25,165,824` bytes。

`read_safetensors_tensor` 可把 payload 读入调用者提供的 `Vector{UInt8}`。Chapter
33 已保证多维结果由 `permutedims` 独立拥有数据，vector/scalar 也显式 copy，
所以 buffer 一经 decode 完成即可安全覆写。最终 host matrix 仍不复用；这避免
更改 H2D ownership，尤其没有暗中放宽 pinned asynchronous upload 的生命周期。

默认开启 `expert_read_buffer_reuse=true`，但保留 loader 与
`configure_hf_qwen3_moe_expert_cache!` 上的显式开关，便于同一 resident session
做严格 A/B，也能退回旧分配行为。

## 测量方法

同一进程、同一 resident session 交错运行：

1. repetition 1：unpooled → pooled；
2. repetition 2：pooled → unpooled；
3. repetition 3：unpooled → pooled。

每个配置先清 device expert cache，对 16 个 shard 执行
`POSIX_FADV_DONTNEED` 后跑 cold，再清 device cache 跑紧随其后的 page-cache
revisit。工作负载保持 32-token English + 1 greedy decode、40,960 context、4 GiB
layer-balanced cache、scattered dispatch、gc8、tensor mode、8-worker pageable
overlap。每次记录 `/proc/self/io`、Julia allocation/GC、pool 借用计数和完整
prefill/decode logits。

复现命令：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_read_buffer_reuse.jl \
  MODEL_DIR \
  benchmark_results/qwen3_moe_cuda_read_buffer_reuse/summary.json
```

## 真实结果

| phase | unpooled allocation | pooled allocation | reduction | unpooled latency | pooled latency | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | `57.647 GB` | `28.967 GB` | `49.75%` | `17.079 s` | `14.052 s` | `1.215×` |
| revisit | `57.647 GB` | `28.967 GB` | `49.75%` | `15.576 s` | `11.653 s` | `1.337×` |

cold/revisit 分别少分配 `28,679,613,984 / 28,679,996,392` bytes；logical raw
payload 为 `28,679,602,176` bytes，误差分别只有约 `12 KiB / 385 KiB`。这是
“每份 raw payload allocation 恰好消失一次”的直接机制证据，而 24 MiB pool
本身相对请求流量有界且在计时前建立。

每次请求仍为 `71 hits / 3,039 misses / 313 evictions`，读取和上传
`28,679,602,176` bytes；pooled 恰好 `3,039` 次 buffer borrow。六个 cold 的
storage reads 均约 `26.236 GB`，六个 revisit 均为零。所有配置、phase、重复的
prefill logits、decode logits 和 greedy token 逐位一致。

同进程交错 A/B 中 pooled 的三轮 cold 为 `14.231 / 14.052 / 13.868 s`，revisit
为 `11.653 / 11.854 / 11.292 s`，方向稳定。不过这里只冻结当前 NVMe、checkpoint
layout 与 English32 trace，不把 `1.215× / 1.337×` 泛化成任意 storage/workload
的保证。

## 决策

- cache-backed `:tensor` expert reads 默认启用有界 raw buffer pool；
- pool 数量由有效 reader worker 配置限定，sequential 退化为单 buffer；
- 保留显式禁用开关及无池 read modes，配置切换会清 cache 并重建正确大小的池；
- 只复用 decode 前的 raw bytes，不复用最终 host matrix，不改变 H2D/pinned
  ownership；
- 下一章可在明确同步边界后研究按 pipeline slot 复用最终 host projection
  matrices；当前剩余约 `28.967 GB` 分配说明这一层是主要候选。

## 验证

- Chapter 34 portable + result contract：`225 / 225`；
- raw buffer 重复覆写后旧结果不变，异常路径归还、计数 reset、pool 参数失败闭合；
- overlapped/sequential/unpooled/shared-open/coalesced/zero-budget tiny 路径逐位一致；
- 3 cold + 3 revisit × 2 modes 的真实请求全部 exact；
- 默认全套与 CUDA 专项复核见 `notes/current_status.md`；
- 冻结报告包含交错顺序、logical/storage I/O、Julia allocation/GC、pool resident/
  borrow 统计、source paths 与 source SHA256。
