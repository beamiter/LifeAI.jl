# Chapter 31 — Qwen3 MoE storage-aware read-worker sweep

> 状态：Closed
> 日期：2026-08-12
> 真实资产：`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
> 设备：NVIDIA GeForce RTX 4090 D，本地 NVMe，Julia 8 threads

## 本章问题

Chapter 30 在 2-token、warm page cache 条件下证明 4-worker read pipeline
相对 sequential 可加速 `1.969×`，但它没有回答三个部署问题：

1. 32-token 自然文本的大量 miss 是否仍能扩展；
2. 真正发生 storage reads 时，并行度是否同方向；
3. `1/2/4/8` workers 的拐点在哪里，默认 worker 上限应是多少。

## 测量方法

固定使用 Chapter 27 的 32-token English prompt，追加一次 greedy decode；session
为 40,960-token capacity、4 GiB layer-balanced LRU、scattered dispatch、gc8。
每次请求前都清空 device expert cache，并执行 `GC.gc(true) + CUDA.reclaim()`。

对 16 个 checkpoint 分片调用 Linux
`posix_fadvise(POSIX_FADV_DONTNEED)` 后运行 cold phase；随后不再次丢页，清空
device cache后立即运行 post-cold revisit。每个请求前后读取 `/proc/self/io`：

- 两轮所有 cold phase 的 `read_bytes` 均为约 `26.236 GB`；
- 两轮所有 revisit 的 `read_bytes` 均为 `0`；
- 两类请求的 LifeAI logical expert reads 都是 `28.680 GB`。

因此 cold/revisit 的区别有实际 block-device I/O 证据。`fadvise` 只是内核 hint，
报告仍保留 `cold_guaranteed=false`，不把它写成跨系统的强保证。第一轮按
`1→2→4→8`，第二轮反向 `8→4→2→1`，减弱顺序、温度和 allocator 历史偏差。

复现命令：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_read_worker_sweep.jl \
  MODEL_DIR \
  benchmark_results/qwen3_moe_cuda_read_worker_sweep/summary.json
```

## 真实结果

| workers | cold median | cold speedup | revisit median | revisit speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `48.181 s` | `1.000×` | `37.349 s` | `1.000×` |
| 2 | `30.614 s` | `1.574×` | `23.316 s` | `1.602×` |
| 4 | `20.737 s` | `2.323×` | `17.015 s` | `2.195×` |
| 8 | `16.962 s` | `2.840×` | `14.553 s` | `2.566×` |

每次请求的路由/cache traffic 完全相同：`71 hits / 3,039 misses / 313
evictions`，读取并上传 `28,679,602,176` logical bytes。8 个配置阶段的
prefill logits、decode logits 与 greedy token 全部逐位一致。

随着 workers 增加，sum-of-task host-read time 上升而 wall time 下降：8-worker
cold 的任务累计约 `107—114 s`，miss-stage wall 只有 `16.1—17.1 s`。这说明
指标必须区分并发任务累计时间和关键路径，不能把前者当作请求 latency。

## 决策

- overlapped pipeline 仍是 opt-in；库默认 `expert_miss_pipeline=:sequential`，
  不改变既有 CPU/单线程/小 workload 行为；
- 当用户启用 overlap 且未显式指定 worker 时，默认值改为
  `min(8, Threads.nthreads())`。单线程仍为 1，最多为 8；
- `expert_read_workers` 继续支持显式覆盖，便于慢盘、共享存储和服务并发限流；
- 8 workers 只绑定本机 NVMe、当前 BF16 decode 和 32-token trace，不泛化到
  arbitrary storage/thread count，也不意味着更多 workers 一定继续加速；
- 下一轮应减少每个 tensor 的 open/seek/read/transpose 分配，或将同 shard 的
  expert reads 合并，避免仅靠提高 task 并发继续堆叠 CPU/内存带宽压力。

## 验证

- Chapter 31 portable + result contract：`112 / 112`；
- 两轮 16 个真实请求全部 exact；
- 冻结报告包含 worker order、logical I/O、process rchar/read syscalls、storage
  read bytes、source paths 与 source SHA256。
