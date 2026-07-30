# Qwen3-0.6B：LifeAI XLA GPU 优化与 Ollama CUDA 对比

记录日期：2026-07-30

GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB，驱动 570.153.02

LifeAI：Julia 1.12.6，Reactant 0.2.275，BF16 XLA GPU

模型 revision：`c1899de289a04d12100db370d81485cdf75e47ca`

## 结论

相同的 18-token cache-busted prompt、batch 1、greedy 固定生成 32
token、5 次稳态测量下，LifeAI 已进入 Ollama 的同一性能带。本轮新增
packed prefill 后做了两次独立正式运行：

- 严格配对 A/B 中，packed prefill 的中位收益为 **0.188 ms**，
  95% bootstrap CI 为 **[0.145, 0.281] ms**，24 个 block 中
  **20 个**胜出。这是本轮可以归因于代码变化的稳定收益。
- 正式 repeat A 为 prefill **4.622 ms**、post-first decode
  **67.902 ms / 456.54 tok/s**、aggregate compute
  **441.48 tok/s**。
- 正式 repeat B 为 prefill **7.198 ms**、post-first decode
  **69.286 ms / 447.42 tok/s**、aggregate compute
  **419.66 tok/s**。
- Ollama 为 prefill **4.522 ms**、post-first interval
  **69.106 ms / 448.59 tok/s**、aggregate compute
  **433.44 tok/s**。

因此 decode 已对齐：两次 LifeAI 正式运行相对 Ollama 为
**-0.26% 到 +1.77%**。Aggregate compute 有一次高 **1.86%**，另一次
低 **3.18%**；可以说“已出现超越”，但不能说“稳定超越”。两次运行的
主要分歧来自 prefill（4.622 vs 7.198 ms），而不是生成 token、源码或
工作负载变化。报告保留两份原始 JSON，不用较快一次覆盖复现波动。

完整 32-token 结果可用时间为 LifeAI **72.875–77.545 ms**、Ollama
**202.131 ms**。LifeAI 产品路径低 **61.6%–63.9% / 2.61–2.77×**，
但这是 in-process bulk 返回对 localhost HTTP streaming 的路径比较，
不能解释为 streaming 核心快同样倍率。

## 可比较指标

| 指标（稳态中位数） | LifeAI 上一主线 | Packed repeat A | Packed repeat B | Ollama |
|---|---:|---:|---:|---:|
| Prefill | 4.955 ms | **4.622 ms** | 7.198 ms | 4.522 ms |
| Prefill 吞吐 | 3,632.34 | **3,894.21** | 2,500.82 | 3,980.54 tok/s |
| Post-first 31 步 | 70.026 ms | **67.902 ms** | 69.286 ms | 69.106 ms |
| Post-first 诊断吞吐 | 442.69 | **456.54** | 447.42 | 448.59 tok/s |
| 32-token aggregate compute | 425.88 | **441.48** | 419.66 | 433.44 tok/s |
| 完整结果 wall | 75.660 ms | **72.875 ms** | 77.545 ms | 202.131 ms |
| 完整结果吞吐 | 422.95 | **439.11** | 412.66 | 158.31 tok/s |

Aggregate compute 是本报告判断“是否对齐”的主计算指标。LifeAI 按每个
请求的 `32 / (prefill + bulk decode)` 后取中位数；Ollama 按每个请求的
`32 / (prompt_eval_duration + eval_duration)` 后取中位数。它消除了
旧报告把 LifeAI 31 个 post-first steps 与 Ollama `eval_count=32`
直接相除的 off-by-one 口径问题。

## 不能直接比较的 streaming / bulk 指标

| 指标 | LifeAI packed 两次正式运行 | Ollama |
|---|---:|---:|
| Post-first 执行 | 67.902–69.286 ms / 31 steps | 69.106 ms / 31 chunk intervals |
| 诊断吞吐 | 447.42–456.54 tok/s | 448.59 tok/s |
| Streaming ITL 中位数 | N/A | 2.192 ms |
| Streaming ITL p90 | N/A | 2.290 ms |
| 中间 token 对调用方可见 | 否 | 是 |

LifeAI 的优化路径先在 benchmark 内部取得首 token，然后用一次 PJRT
调用中的 StableHLO `while` 完成余下 31 步，期间没有中间 token
D2H，最后一次性返回 32-token 向量。因此 447.42–456.54 对 448.59
只能说明生成核心已进入同一性能带，不能声称 LifeAI streaming 已超过
Ollama。原单步 API 仍保留给 streaming 调用方，但本报告没有把它混入
bulk 正式数字。

同理，LifeAI 的 **4.943–8.322 ms** 是 benchmark 内部
first-token-ready，token 没有 yield 给调用方；Ollama 的
**132.872 ms** 是客户端收到首个 HTTP chunk。两者不计算倍率。

## 优化内容

1. Prefill 仅对最后一个 prompt hidden state 做 151,936 词表投影，
   不再计算随后会被丢弃的其余 logits。
2. 31 个 post-first decode step 合并到一次 PJRT 调用中的 StableHLO
   `while`；取消逐 token PJRT dispatch、同步和 D2H。
3. 每层 Q/K/V 永久打包为一个投影，gate/up 永久打包为一个投影。
   现在 prefill 与 decode 都复用它们；每层 dense 从 7 次降到 4 次，
   28 层从 196 次降到 112 次，再加 LM-head。Prefill 不新增 packed
   权重或显存，只复用 decode 已有的 device buffer。
4. Prefill 与 decode executable 都接收实际使用的 compact 参数树；
   decode 入口参数从 429 个降到 289 个（降低 32.6%），不会再次复制
   未改动的 device 权重。
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
| 最终 load + pack + transfer + compile | 99.967–100.098 s |

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
- CPU 单元测试只用于数值正确性，不作为性能数据。完整测试通过；
  Week 16 为 **168/168**。Grouped GQA 对删除前 expanded-head oracle
  bitwise 一致；last-token-only 与完整 logits 最后一列一致。
- Packed prefill 的 CPU logits/KV 与 unpacked 路径 bitwise 一致；
  XLA GPU 因合并 GEMM 的归约顺序允许 BF16 数值非 bitwise，但严格
  A/B 的五个 workload 首 token 与完整 32-token completion 均一致，
  两次正式运行也与上一主线全部 case 的 token 完全一致。
- A/B 中关闭 Triton GEMM、关闭 cuBLASLt、关闭 Triton-any 或追加
  WHILE command buffer 都没有稳定收益，因此正式结果使用默认 XLA
  自动调优组合。

## 适用范围

结果只覆盖短 prompt、短输出、batch 1、Qwen3-0.6B BF16、RTX 4090 D。
LifeAI fast path 当前是 bulk completion；不能外推为长上下文、并发、
batching 或 streaming 性能。

本轮还对潜在 decode 优化设置了止损门：

- fused LM-head GEMV+argmax 的 full-vocab head-only 单步仅快 **0.45%**，
  且两个 custom call 会退出 command buffer，拒绝；
- Q/K 合并虽使 kernel 数从 716 降到 688，但 GPU 归约轨迹变化导致
  token parity 失败，拒绝；
- 静态展开 8-token decode 慢约 **30%**，拒绝；
- 独立 embedding/LM-head 布局、cuBLASLt/Triton/auto-layout flags
  均无严格配对的稳定收益。

因此当前收尾点是：保留有显著严格 A/B 收益且语义通过的 packed
prefill，不把复杂度更高但收益不足或正确性失败的实验带入主线。

## 原始产物

- [LifeAI packed prefill 正式 repeat A](qwen3_0.6b_xla_gpu_e2e_cachebusted_packed_prefill_repeat_a.json)
- [LifeAI packed prefill 正式 repeat B](qwen3_0.6b_xla_gpu_e2e_cachebusted_packed_prefill_repeat_b.json)
- [LifeAI 上一主线正式 JSON](qwen3_0.6b_xla_gpu_e2e_cachebusted_optimized.json)
- [LifeAI 优化前基线 JSON](qwen3_0.6b_xla_gpu_e2e_cachebusted.json)
- [Ollama CUDA 正式 JSON](qwen3_0.6b_ollama_bf16_gpu_e2e_cachebusted.json)
- [LifeAI benchmark 脚本](../scripts/benchmark_qwen3_xla_e2e.jl)
- [Ollama benchmark 脚本](../scripts/benchmark_qwen3_e2e_ollama.py)
- [Cache-busted 输入](../benchmark_inputs/qwen3_e2e_cachebusters.json)
