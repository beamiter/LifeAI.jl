# Chapter 35 — Qwen3 MoE bounded host projection-buffer reuse

> 状态：Closed
> 日期：2026-08-12
> 真实资产：`Qwen/Qwen3-30B-A3B@ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`
> 设备：NVIDIA GeForce RTX 4090 D，本地 NVMe，Julia 8 threads

## 本章问题

Chapter 34 已用 24 MiB raw pool 把 English32 请求分配从 `57.647 GB` 降到
`28.967 GB`。剩余量几乎等于 3,039 个 active-expert miss 的最终 gate/up/down
BF16 host matrices：每个 expert `9,437,184` bytes，合计 `28,679,602,176`
bytes。本章验证能否按 reader pipeline slot 有界复用这些矩阵，同时严格守住
pageable、pinned 与 CPU identity 三种不同的 ownership 边界。

## 生命周期边界

不能仅因 `to_device` 已返回就对所有 backend 复用 host matrix：

- CPU 默认 `to_device=identity`，device cache 实际直接持有同一 host array；复用
  会静默篡改已缓存 expert，因此永不建 final pool；
- CUDA pinned path 在专用 stream 完成前仍由 DMA 读取原数组；本章自动禁用 final
  pool，保持 Chapter 30 的 pin/synchronize/finalize 生命周期；
- CUDA pageable `Array → CuArray` 由 CUDA.jl 先把 unpinned source stage 到驱动
  持有的存储，调用返回后源数组可被回收。本章另以 3 MiB BF16 buffer 做 64 次
  “`CUDA.cu` 返回即覆写、再 synchronize”实测，全部 device payload exact。

因此 `_qwen3_moe_host_buffer_reuse_supported` 默认返回 false，仅 CUDA extension
对 `CuArray` prototype 声明支持，而且 `pinned_upload=true` 仍强制无池。

## 实现

`_decode_safetensors_values!` 与 `_read_safetensors_tensor!` 将 row-major
safetensors payload 直接 transpose/copy 到调用者提供的最终 matrix，不再创建
拥有新 payload 的 `permutedims` 结果。256×256 BF16 热路径只产生 `608 bytes`
小对象分配，不分配 128 KiB matrix payload；BF16/F32 四种 source/target 组合与
既有 decode 逐位一致。

`_Qwen3MoEHostStagingPool` 每个 slot 持有 gate/up/down 三张 BF16 matrix：

- overlapped pool 数量等于 reader workers，sequential 退化为 1；
- 真实 8-worker pool 常驻 `75,497,472` bytes（72 MiB）；
- reader task 借用 slot 并 in-place decode，raw pool 仍在 decode 后立即归还；
- pageable upload 返回后归还 matrix lease；double-return 与 outstanding reset
  fail closed；
- 任一 read/upload 失败时，当前 lease 在 `finally` 归还，尚未消费的 reader task
  由 cleanup fetch 后归还。注入 expert upload failure 后 borrow/return 配对且同一
  session 可恢复 exact。

loader 与 runtime configure 新增 `expert_host_buffer_reuse` /
`host_buffer_reuse`，默认 true；这是“允许 backend 在满足契约时启用”，不是强迫
所有 backend 建池。stats 同时暴露 configured flag、实际 count/bytes 与
borrow/return。

## 测量方法

复用 Chapter 34 的 32-token English + 1 greedy decode、40,960 context、4 GiB
layer-balanced cache、scattered dispatch、gc8、tensor mode、8-worker pageable
overlap。raw reuse 两边都开启，只交错切换 final host pool：

1. repetition 1：unpooled → pooled；
2. repetition 2：pooled → unpooled；
3. repetition 3：unpooled → pooled。

每个配置先清 device expert cache，对 16 个 shard 执行
`POSIX_FADV_DONTNEED` 后跑 cold，再清 device cache 跑紧随其后的 page-cache
revisit。pool 在 request allocation 计量前建立；每次记录 `/proc/self/io`、Julia
allocation/GC、两层 pool 统计与完整 prefill/decode logits。

复现命令：

```bash
julia --threads=8 --project=. \
  scripts/benchmark_qwen3_moe_cuda_host_buffer_reuse.jl \
  MODEL_DIR \
  benchmark_results/qwen3_moe_cuda_host_buffer_reuse/summary.json
```

## 真实结果

| phase | unpooled allocation | pooled allocation | reduction | unpooled latency | pooled latency | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cold | `28.967 GB` | `0.288 GB` | `99.01%` | `13.770 s` | `10.471 s` | `1.315×` |
| revisit | `28.967 GB` | `0.288 GB` | `99.01%` | `11.358 s` | `8.763 s` | `1.296×` |

cold/revisit 分别少分配 `28,679,018,608 / 28,679,325,400` bytes，与 logical
final matrix payload `28,679,602,176` bytes 只差约 `570 / 270 KiB`。这再次给出
“每份 final payload allocation 恰好消失一次”的机制证据。raw + final pool 合计
`25,165,824 + 75,497,472 = 100,663,296` bytes（96 MiB），相对每请求 28.68 GB
allocator traffic 有界。

pooled 的三轮 cold 为 `10.471 / 10.432 / 11.600 s`，revisit 为
`8.441 / 8.763 / 9.208 s`。每次请求仍为 `71 hits / 3,039 misses / 313
evictions`，读取和上传 `28,679,602,176` bytes；raw 与 final 都恰好 3,039 次
borrow，final 也恰好 3,039 次 return。六个 cold storage reads 均约
`26.236 GB`，六个 revisit 均为零。所有 logits/token 逐位一致。

## 决策

- final host projection reuse 默认允许，但只在 backend 显式声明 pageable source
  可安全复用时实际启用；
- CPU identity、非 tensor mode、zero cache 与 pinned async 全部自动无 final
  pool，保持原 ownership；
- 继续保留显式关闭开关，真实报告使用同一 resident session 做交错 A/B；
- 将 `99.01%` allocation reduction 与当前 trace 的 `1.315× / 1.296×` 分开：
  前者有 payload 机制证据，后者不泛化到任意 storage/workload；
- 当前请求只剩约 `288 MB` Julia allocation。下一章应先 profile 其组成，不在
  没有证据时继续预分配；也可转向更高价值的 grouped scattered dispatch 缺口。

## 验证

- Chapter 35 portable + result contract：`293 / 293`；
- Chapter 35 CUDA ownership/failure recovery：`29 / 29`；
- 3 cold + 3 revisit × 2 modes 的真实请求全部 exact；
- 默认全套与 CUDA 专项复核见 `notes/current_status.md`；
- 冻结报告包含交错顺序、logical/storage I/O、Julia allocation/GC、raw/final pool
  resident/borrow/return、source paths 与 source SHA256。
