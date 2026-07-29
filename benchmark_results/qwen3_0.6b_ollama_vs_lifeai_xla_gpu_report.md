# Qwen3-0.6B：LifeAI XLA GPU 优化与 Ollama CUDA 对比

记录日期：2026-07-30

GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB，驱动 570.153.02

LifeAI：Julia 1.12.6，Reactant 0.2.275，BF16 XLA GPU

模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`

## 结论

相同的 18-token cache-busted prompt、batch 1、greedy 固定生成 32
token、5 次稳态测量下，LifeAI 优化已把主要差距收敛到同一性能带：

- Prefill 从 **5.230 ms** 降至 **4.955 ms**，延迟降低 **5.3%**，
  吞吐从 3,441.60 提升到 **3,632.34 tok/s**。
- 首 token 后的 31 步从 **149.818 ms** 降至 **70.026 ms**，
  总延迟降低 **53.3%**，诊断吞吐从 206.92 提升到
  **442.69 tok/s（2.139×）**。
- 完整 LifeAI 请求从 **156.110 ms** 降至 **75.660 ms**，
  输出吞吐从 204.98 提升到 **422.95 tok/s（2.063×）**。
- 同工作量 aggregate compute 指标为 LifeAI **425.88 tok/s**、
  Ollama **433.44 tok/s**；LifeAI 中位数仅低 **1.74%**。本轮已达到
  near-parity，但没有把一次较快样本当成“稳定超越”。
- 完整 32-token 结果可用时间为 LifeAI **75.660 ms**、Ollama
  **202.131 ms**。LifeAI 低 **62.6% / 2.672×**，但这是
  in-process bulk 返回对 localhost HTTP streaming 的产品路径比较。

Prefill 仍是 Ollama 较快：Ollama 原生 prompt eval 为 **4.522 ms /
3,980.54 tok/s**，LifeAI 延迟高 9.6%、吞吐低 8.7%。绝对差为
0.433 ms。

## 可比较指标

| 指标（稳态中位数） | LifeAI 优化前 | LifeAI 优化后 | Ollama | 说明 |
|---|---:|---:|---:|---|
| Prefill | 5.230 ms | **4.955 ms** | 4.522 ms | 有条件的原生 compute 诊断 |
| Prefill 吞吐 | 3,441.60 | **3,632.34** | 3,980.54 tok/s | 有条件的原生 compute 诊断 |
| 32-token aggregate compute | 206.39 | **425.88** | 433.44 tok/s | `32 / (prefill + generation)` |
| 完整结果 wall | 156.110 ms | **75.660 ms** | 202.131 ms | LifeAI in-process；Ollama HTTP |
| 完整结果吞吐 | 204.98 | **422.95** | 158.31 tok/s | 按各自真实调用路径 |

Aggregate compute 是本报告判断“是否对齐”的主计算指标。LifeAI 按每个
请求的 `32 / (prefill + bulk decode)` 后取中位数；Ollama 按每个请求的
`32 / (prompt_eval_duration + eval_duration)` 后取中位数。它消除了
旧报告把 LifeAI 31 个 post-first steps 与 Ollama `eval_count=32`
直接相除的 off-by-one 口径问题。

## 不能直接比较的 streaming / bulk 指标

| 指标 | LifeAI 优化后 | Ollama |
|---|---:|---:|
| Post-first 执行 | 70.026 ms / 31 steps | 69.106 ms / 31 chunk intervals |
| 诊断吞吐 | 442.69 tok/s | 448.59 tok/s |
| Streaming ITL 中位数 | N/A | 2.192 ms |
| Streaming ITL p90 | N/A | 2.290 ms |
| 中间 token 对调用方可见 | 否 | 是 |

LifeAI 的优化路径先在 benchmark 内部取得首 token，然后用一次 PJRT
调用中的 StableHLO `while` 完成余下 31 步，期间没有中间 token
D2H，最后一次性返回 32-token 向量。因此 442.69 对 448.59
（差 1.31%）只能说明生成核心已进入同一性能带，不能声称 LifeAI
streaming 已超过 Ollama。原单步 API 仍保留给 streaming 调用方，但
本报告没有把它混入 bulk 正式数字。

同理，LifeAI 的 **5.420 ms** 是 benchmark 内部 first-token-ready，
token 没有 yield 给调用方；Ollama 的 **132.872 ms** 是客户端收到
首个 HTTP chunk。两者不计算倍率。

## 优化内容

1. Prefill 仅对最后一个 prompt hidden state 做 151,936 词表投影，
   不再计算随后会被丢弃的其余 logits。
2. 31 个 post-first decode step 合并到一次 PJRT 调用中的 StableHLO
   `while`；取消逐 token PJRT dispatch、同步和 D2H。
3. 每层 Q/K/V 永久打包为一个投影，gate/up 永久打包为一个投影。
   每层 decode dense 从 7 次降到 4 次，28 层从 196 次降到 112 次，
   再加 LM-head。
4. Decode executable 只接收实际使用的 compact 参数树；入口参数从
   429 个降到 289 个（降低 32.6%），不会再次复制未改动的 device
   权重。
5. GQA 直接以 KV head 为 batch、query group 为 row 计算，取消 K/V
   从 8 heads 显式扩张到 16 heads。
6. 固定工作负载的 cache 长度从 50 收紧为实际需要的 49。

## HLO 硬门

独立的最终 HLO dump 验证：

- optimized HLO 只有 **1 个 while**，`known_trip_count = 31`；
- thunk 序列只有 **1 个 `kWhile`**，其 body 是一个
  `kCommandBuffer`；
- ENTRY 参数编号为 `0:288`，即 **289 个**，与 compact 参数树一致；
- buffer assignment 有 60 个 `maybe-live-out` allocation，包括 56
  个 K/V cache、generated/token/position 状态，确认 mutation
  走入口/输出复用路径；
- WHILE command buffer 本身没有启用；这里只声明“一次 PJRT 调用含
  StableHLO while”，不把它误写成完全由 GPU 控制循环。

取证文件 SHA-256：

- optimized HLO：
  `b60c9fb8721138656a8258b5c18057d9802a9c5368723355c252e3b09ea568c3`
- buffer assignment：
  `acab8d287ea9b2c275e9a0d7dd7a1bdaaa09ab9bb72fb97c1136772430b76f10`
- final thunk sequence：
  `da72185cb208e50cde7fc2dd0006fa428379ffade19d79ea597baf7074398fd9`

## 参数内存与冷启动代价

| 项目 | 数值 |
|---|---:|
| 原 BF16 参数 | 1,192,099,840 bytes（1.110 GiB） |
| 永久 packed 投影额外占用 | 898,367,488 bytes（0.837 GiB） |
| 原参数 + packed 投影 | 1.947 GiB |
| Compact decode 逻辑参数 | 227 tensors / 1.400 GiB |
| 最终 load + pack + transfer + compile | 102.532 s |

Compact 树复用原参数的 device buffers；额外显存来自 QKV、gate/up 和
预转置 tied LM-head 的永久 packing。XLA 日志中的 17.63 GiB 是 BFC
allocator reservation，不是参数本身的实际大小。优化换取了约
0.837 GiB 额外 packed 权重与 1.751 s pack 时间，冷启动仍明显慢于
Ollama 的预转换 GGUF runner。

## 正确性与公平性

- 两端都是 Qwen3-0.6B 同一 source revision。LifeAI Safetensors
  SHA-256：
  `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`；
  Ollama BF16 GGUF SHA-256：
  `65a16246f5814dc0587acadcf0328186b17febf6dcaeb1b13efa9243b551d38e`。
- 两端均为 `temperature=0`、no-thinking、固定 32 token；Ollama
  `raw=true`，避免二次 chat template。
- 1 cold、1 warmup、5 measured，五个 measured prompt 均为 18
  token，首 token 不同，避免跨请求 prefix/KV cache 复用。
- 所有已执行 case 的完成文本一致，SHA-256：
  `7c6522b3c3913af0ec09641d8a885105fd6ab3e399f2c423a421ff6ab4571e34`。
- CPU 单元测试只用于数值正确性，不作为性能数据。Grouped GQA 对
  删除前 expanded-head oracle bitwise 一致；last-token-only 与完整
  logits 最后一列一致；packed/unpacked 单步、多步 token 和 KV cache
  均 bitwise 一致。
- A/B 中关闭 Triton GEMM、关闭 cuBLASLt、关闭 Triton-any 或追加
  WHILE command buffer 都没有稳定收益，因此正式结果使用默认 XLA
  自动调优组合。

## 适用范围

结果只覆盖短 prompt、短输出、batch 1、Qwen3-0.6B BF16、RTX 4090 D。
LifeAI fast path 当前是 bulk completion；不能外推为长上下文、并发、
batching 或 streaming 性能。若目标是稳定超过 Ollama 中位数，下一步
需要更侵入的 LM-head GEMV+argmax 融合或专用 decode GEMV kernel，
而不是继续堆叠没有实测收益的 XLA flags。

## 原始产物

- [LifeAI 优化后正式 JSON](qwen3_0.6b_xla_gpu_e2e_cachebusted_optimized.json)
- [LifeAI 优化前基线 JSON](qwen3_0.6b_xla_gpu_e2e_cachebusted.json)
- [Ollama CUDA 正式 JSON](qwen3_0.6b_ollama_bf16_gpu_e2e_cachebusted.json)
- [LifeAI benchmark 脚本](../scripts/benchmark_qwen3_xla_e2e.jl)
- [Ollama benchmark 脚本](../scripts/benchmark_qwen3_e2e_ollama.py)
- [Cache-busted 输入](../benchmark_inputs/qwen3_e2e_cachebusters.json)
