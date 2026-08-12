# Chapter 33 — Qwen3 MoE safetensors decode copy elision

> 状态：Closed
> 日期：2026-08-12
> 真实资产：`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
> 设备：NVIDIA GeForce RTX 4090 D，本地 NVMe，Julia 8 threads

## 本章问题

Chapter 32 证明放大 raw read 粒度会破坏 8-worker 的 I/O/decode overlap，但也
测得一个 English32 请求产生约 `86.3 GB` Julia allocation。每个 3 MiB BF16
projection 的旧 decode 生命周期是：

1. `read` 分配 raw byte vector；
2. `copy(reinterpret(BFloat16, raw))` 分配线性 BF16 vector；
3. `permutedims` 将 safetensors row-major shape 转成 Julia 语义矩阵，再分配最终
   matrix。

对二维权重而言，第 2 步只在第 3 步立刻被复制和释放。本章验证能否移除这个中间
副本，同时保持最终矩阵 ownership、逐位语义和现有 tensor 级并发。

## 实现

`_decode_safetensors_values` 对多维 BF16/BF16 与 F32/F32 直接把 raw buffer 的
reinterpret view 交给 `_semantic_array`；后者的 `permutedims` 仍生成拥有独立
存储的最终数组。零维与一维结果继续显式 `copy`，不会返回依赖临时 raw buffer 的
view。BF16/F32 dtype 转换路径本来就必须产生转换结果，保持不变。

这个改动没有新增 session 模式，也没有复用可能仍参与 H2D 的最终矩阵；它只移除
确定冗余、生命周期完全包含在一次 decode 内的线性中间数组。因此默认读取仍是
Chapter 32 选出的 `expert_read_mode=:tensor`，I/O 粒度和 8-worker 调度不变。

单个真实 expert projection 的分配从 `9,438,424` 降到 `6,292,688` bytes，
恰好少 `3,145,736` bytes，接近一个 projection 的 BF16 payload
`3,145,728` bytes。

## 测量方法

复用 Chapter 32 的 32-token English + greedy decode、4 GiB layer-balanced
cache、scattered dispatch、gc8、tensor reads 与 8 workers。新路径运行 3 个
`POSIX_FADV_DONTNEED` cold / post-cold revisit pair；每轮记录 `/proc/self/io`、
Julia allocation/GC 与完整 logits。

旧路径来自紧邻本章、相同硬件/参数/三重复方法的 Chapter 32 tensor 冻结报告，并
将其报告 SHA256 纳入新结果。它不是同进程 A/B，因此微小 latency 变化只作为观察；
allocation 差额还必须独立满足“约等于一次 logical BF16 payload copy”的机制校验。

复现命令：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_decode_copy_elision.jl \
  MODEL_DIR \
  benchmark_results/qwen3_moe_cuda_decode_copy_elision/summary.json
```

## 真实结果

| phase | old allocation | direct allocation | reduction | old latency | direct latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| cold | `86.326 GB` | `57.646 GB` | `33.22%` | `17.122 s` | `16.540 s` |
| revisit | `86.326 GB` | `57.646 GB` | `33.22%` | `14.589 s` | `14.882 s` |

cold/revisit 分别少分配 `28,679,752,184 / 28,679,731,464` bytes；本 workload
的 logical BF16 expert payload 是 `28,679,602,176` bytes，两者只差约
`150 / 129 KiB`。这证明减少量来自每个 projection 恰好消失的一份 payload
副本，而不是 GC 时机或路由变化。

三轮均为 `71 hits / 3,039 misses / 313 evictions`，读取和上传
`28,679,602,176` bytes；cold storage reads 约 `26.236 GB`，revisit 为零。
所有 prefill logits、decode logits 与 greedy token 逐位一致。

latency 的相邻实验对照为 cold 快 `3.5%`、revisit 慢 `2.0%`。由于不是同进程
交错 A/B，且方向不一致，本章不声称通用 speedup；确定结论是 exact 前提下将
allocator traffic 降低三分之一。

## 决策

- 多维同 dtype safetensors decode 默认移除线性中间 copy；
- 零维/一维继续返回 owning storage，最终多维语义矩阵仍独立拥有数据；
- 不复用最终 host matrix，不改变 H2D 生命周期和 pinned upload 语义；
- 不将本次小幅 latency 变化泛化到其他 checkpoint、storage 或 workload；
- 下一章可研究按 worker 有界复用 raw read buffer，目标是再移除剩余约一份
  payload 的 allocator traffic，同时保持 projection 粒度的 I/O/decode overlap。

## 验证

- Chapter 33 portable + result contract：`108 / 108`；
- 3 个 cold + 3 个 revisit 真实请求全部 exact；
- 默认全套与 CUDA 专项复核见 `notes/current_status.md`；
- 冻结报告包含 Chapter 32 baseline SHA256、logical/storage I/O、Julia
  allocation/GC、source paths 与 source SHA256。
