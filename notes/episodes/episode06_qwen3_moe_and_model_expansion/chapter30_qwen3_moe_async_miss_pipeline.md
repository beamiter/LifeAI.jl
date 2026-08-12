# Chapter 30 — Qwen3 MoE bounded async miss pipeline

> 状态：Closed
> 日期：2026-08-12
> 真实资产：`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
> 设备：NVIDIA GeForce RTX 4090 D，CUDA capability 8.9

## 本章问题

Chapter 29 已把冻结 cache-hit 请求压到约 `0.15 s`，但 cold device-cache
请求仍逐 expert 执行 safetensors read、BF16 decode 和 H2D。目标是在不改变
router、cache policy、LRU 顺序和 logits 的前提下隐藏 miss I/O。

最初设想中的 layer-ahead prefetch 不成立：第 `L+1` 层 router 依赖第 `L` 层
输出，因此在当前层计算完成前并不知道下一层 active experts。本章只并行化
**当前层 router 已完成以后**的独立 expert miss，不猜测未来路由。

## 实现

`HFQwen3MoEOffloadSession` 新增三项 opt-in 配置：

- `expert_miss_pipeline=:overlapped`：以 `Threads.@spawn` 并行读取当前层 miss；
- `expert_read_workers=4`：限制同时在途的 host read；
- `expert_pinned_upload=true`：CUDA 上用独立 stream 上传 pinned BF16 matrices。

读取任务可以乱序完成，但主线程始终按排序后的 active-expert 顺序 `fetch`、上传
和写入 cache。因此 entry generation、LRU clock、容量门禁及逐层 dispatch 顺序
与 sequential 路径一致。CUDA pinned 路径按 worker window 同步 transfer stream
并显式 unpin，最多保留一个有界窗口；进入 expert kernel 前一定完成本层上传。
异常路径也会等待已启动 reader，并同步/解绑已注册 host arrays。

默认仍是 `:sequential` 且 pinned upload 关闭；非 CUDA device 请求 pinned upload
会 fail closed。请求统计新增 host-read 累计时间、miss-stage wall time、上传等待、
read jobs、并行层数与 pinned bytes。

## 真实实验

命令：

```bash
julia --threads=4 --project=. \
  scripts/benchmark_qwen3_moe_cuda_async_miss_pipeline.jl \
  MODEL_DIR \
  MODEL_DIR/lifeai-references/chapter24-real-parity/bfloat16 \
  benchmark_results/qwen3_moe_cuda_async_miss_pipeline/summary.json
```

session 固定 40,960-token capacity、8 GiB global cache、scattered dispatch、
每 8 层 GC。三种路径各先完整热身一次，再按固定顺序交错测量三轮；每轮前清空
device expert cache。操作系统 page cache 不清空，所以这是明确的 warm-page-cache
miss-path 实验，不是冷盘吞吐实验。

| 路径 | request median | miss-stage median | 相对 sequential | 结论 |
| --- | ---: | ---: | ---: | --- |
| sequential pageable | `10.677 s` | `10.406 s` | `1.000×` | 基线 |
| 4-worker overlapped pageable | `5.423 s` | `5.177 s` | `1.969×` | 最快 |
| 4-worker overlapped pinned | `5.481 s` | `5.207 s` | `1.948×` | 未胜过 pageable |

每次请求都读取并上传 `8,417,968,128` bytes，traffic 为 `226 hits / 892
misses / 0 evictions`，并执行 96 次 pointer-plan build 与 12 次 forced GC。
overlapped 路径在 95 个层调用中形成多 read 并行；pinned 路径确实注册并上传
全部 `8,417,968,128` bytes，但 transfer-stream wait 为 `0.249—0.360 s`，
最终比 overlapped pageable 慢约 `1.07%`。

三种配置的 warmup 与全部 9 次 measured prefill/decode logits 均逐位一致；相对
Transformers BF16 reference，prefill/decode max-abs 为 `0.3125 / 0.25`，
argmax 均一致。

## 决策与边界

- miss-heavy、本机 warm-page-cache workload 推荐显式使用
  `miss_pipeline=:overlapped, read_workers=4, pinned_upload=false`；
- 不改变库默认值：小请求、单线程进程、慢盘、其他 device 和其他并发度没有被
  这份实验覆盖；
- pinned async 实现保留为 opt-in 能力，但本机数据不支持把它设为推荐；
- 没有实现或宣称 layer-ahead prefetch，也不宣称 `1.969×` 可泛化到冷盘；
- 下一轮应测自然文本 layer-balanced cache trace，并 sweep 1/2/4/8 workers，
  区分 page-cache decode、磁盘队列深度、BF16 transpose 和 H2D 各自占比。

## 验证

- Chapter 30 portable：`86 / 86`；
- Chapter 30 CUDA：`11 / 11`；
- 真实报告冻结 schema、流量、latency、exact parity、决策与三个 source SHA256。
