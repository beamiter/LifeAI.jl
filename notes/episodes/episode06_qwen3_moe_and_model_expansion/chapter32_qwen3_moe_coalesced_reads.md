# Chapter 32 — Qwen3 MoE adjacent safetensors read experiment

> 状态：Closed
> 日期：2026-08-12
> 真实资产：`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
> 设备：NVIDIA GeForce RTX 4090 D，本地 NVMe，Julia 8 threads

## 本章问题

Chapter 31 已把 bounded miss pipeline 扩到 8 workers，但一次 32-token 请求仍有
`3,039` 次 expert miss；旧路径为每个 expert 分别打开并读取 gate/up/down 三个
projection，进程约发生 `9,400` 次 read syscall。本章回答：Qwen3-30B-A3B
checkpoint 中相邻 projection 能否合并为一次读，并同时降低 I/O/allocator 成本与
端到端 latency。

## 实现与安全边界

新增 `read_safetensors_tensors` 批量接口。它按 shard 分组，只合并字节范围严格
相邻的 tensor；不跨 gap，不为减少 syscall 读取未请求 payload。offload session
增加三个可选 `expert_read_mode`：

- `:tensor`：每个 projection 独立 open/read/decode，保持原路径和默认值；
- `:shared_open`：同一 expert 共用一次 shard open，仍逐 projection
  read/decode；
- `:coalesced`：将相邻的 down/gate/up 一次读入，再分别 decode/transpose。

真实 checkpoint 的抽样与全量运行确认每个 expert 的布局都是
`down → gate → up`，每个 projection `3,145,728` bytes，三个 projection
严格相邻、合计 `9,437,184` bytes。这个观察只冻结当前 revision，不泛化到任意
checkpoint；批量 reader 本身仍逐 location 验证相邻关系。

## 测量方法

复用 Chapter 31 的 32-token English prompt + 一次 greedy decode、4 GiB
layer-balanced cache、scattered dispatch、gc8、8 read workers。三种模式各运行
3 个 cold/revisit pair，并用不同顺序交错：

1. tensor → shared_open → coalesced；
2. coalesced → shared_open → tensor；
3. tensor → shared_open → coalesced。

每个 cold 前对 16 个 checkpoint shards 执行 `POSIX_FADV_DONTNEED`，并用
`/proc/self/io` 记录 storage bytes、read syscalls；紧随其后的 revisit 只清 device
cache。每轮还记录 `Base.gc_num()` 的 allocated bytes 与 GC time。复现命令：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_coalesced_reads.jl \
  MODEL_DIR \
  benchmark_results/qwen3_moe_cuda_coalesced_reads/summary.json
```

## 真实结果

| mode | cold median | cold speedup | cold read syscalls | revisit median | revisit speedup | revisit read syscalls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tensor | `17.122 s` | `1.000×` | `9,413` | `14.589 s` | `1.000×` | `9,368` |
| shared_open | `17.147 s` | `0.999×` | `9,391` | `14.586 s` | `1.000×` | `9,361` |
| coalesced | `21.221 s` | `0.807×` | `3,304` | `19.488 s` | `0.749×` | `3,298` |

coalesced 的 read syscall 分别减少 `64.90% / 64.80%`，但 cold 慢约 `23.9%`，
revisit 慢约 `33.6%`；分配量也没有下降，反而增加约 `0.0022%`。shared-open
保留原来的 read/decode 交错，端到端与 tensor 的差异仅 `-0.14% / +0.02%`，在
本次波动中没有可归因收益。

三种模式每轮均为 `71 hits / 3,039 misses / 313 evictions`，logical expert
read/upload 都是 `28,679,602,176` bytes。cold 实际 storage reads 约
`26.236 GB`，revisit 为零；所有 prefill logits、decode logits 与 greedy token
逐位一致。

## 原因与决策

一次 9.4 MB read 虽减少 syscall，却必须等整块读取完成后才能依次完成三次 BF16
decode/transpose；旧 tensor 路径和 shared-open 路径可以在 8 个 reader tasks
之间交错较小的 I/O 与 decode 工作。结果表明这里的瓶颈不是 syscall 数本身，粗粒度
合并破坏了已有 pipeline overlap，额外大 buffer 也没有减少最终 tensor 分配。

- 默认继续使用 `expert_read_mode=:tensor`；不把 syscall reduction 包装为性能收益；
- `:shared_open` 与 `:coalesced` 保留为显式 opt-in 和后续 I/O 实验基线；
- 公共 batch reader 保留，因为它提供严格相邻 coalescing、跨 shard 分组和两种 dtype
  的独立能力，但调用方仍需用端到端 workload 验证是否合适；
- 下一章不再继续扩大 raw read 粒度；优先研究可复用 host decode/transpose buffer，
  或保持 projection 粒度的 read/decode pipeline，避免扩大临时对象生命周期。

## 验证

- Chapter 32 portable + result contract：`190 / 190`；
- 三种模式、cold/revisit 共 18 次真实请求全部 exact；
- 默认全套与 CUDA 专项复核见 `notes/current_status.md`；
- 冻结报告包含运行顺序、checkpoint layout、logical/storage I/O、read syscall、
  Julia allocation/GC、source paths 与 source SHA256。
