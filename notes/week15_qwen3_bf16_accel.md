# Week 15 — Qwen3 BF16 CUDA / XLA Accelerated Inference

> 状态：Open
>
> 开启记录：2026-07-26
>
> 依赖基线：[`Week 14 — Qwen3 Native BF16 Mixed-Precision Compute`](week14_qwen3_bf16_compute.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Week 14 的 BF16 推理路径全部在 CPU 上验证，8B 一次前向
> 要 287 秒——语义正确但不是可用的推理形态。本周把同一套 BF16 混合
> 精度契约搬上 GPU（CUDA eager 与 Reactant/XLA 编译），此后推理验证
> 与生成的主战场从 CPU 转向 CUDA / XLA。

## 核心问题

> 同一套 HF BF16 混合精度语义，能否在 RTX 5080 上以原生 BF16 张量核
> 计算（CUDA eager 与 XLA 编译两种形态）复现 CPU 路径的 token 级
> 行为，并把生成速度从"每 token 十秒级"拉到实用量级？

## 已通过的能力闸门（Open 前实测）

1. **CUDA BF16 原生支持**：RTX 5080（sm_120）上 `CuArray{BFloat16}`
   的 broadcast、CUBLAS gemm（F32 累加）、NNlib `batched_mul`、
   `argmax` 全部可用——原生路径可行，无需在 GPU 上做 F32 仿真。
2. **Reactant XLA BF16**：BF16 RArray 可编译执行；逐元素升降精度需用
   可 trace 的写法（`1.0f0 .* x` 升 F32、`BFloat16.(x)` 舍回，均已
   验证），裸 `Float32.(traced_bf16)` 缺方法。

## 设计决策

- Week 14 的 CPU BF16 路径含逐 head 标量循环，GPU 不可用。新增一套
  **设备通用的向量化 BF16 前向**：只用 broadcast、`batched_mul`、
  gather 等设备友好原语，同一份代码服务 CPU 数组、CuArray 与
  Reactant tracing；混合精度契约与 Week 14 逐条相同。
- GPU 上线性层直接用 CUBLAS BF16 gemm（硬件 F32 累加），与 Week 14
  "BF16 存储 + F32 累加 + BF16 舍入"语义一致；不同的是求和顺序——
  沿用 Week 13/14 的方法论：数值以 BF16 量级容差衡量，正确性以
  token 级行为（argmax、greedy 序列）为准。
- Week 14 CPU 路径及其冻结证据零改动；向量化路径与 CPU 循环路径在
  合成模型上先互证，再上真实权重。
- reference 复用 Week 14 已导出的 `week14-bf16-parity`（HF CPU BF16
  逐层 + 16 步 greedy），不重复导出。

## 本周资源边界（先冻结再执行）

- RTX 5080 VRAM 16.3 GiB：BF16 树 0.6B 1.11 / 1.7B 3.21 / 4B
  7.49 GiB 可驻留；**8B 15.26 GiB 超出 VRAM**（含 CUDA context 与
  workspace 无余量），本周不做 8B GPU 驻留或 CPU-GPU 混合。
- XLA 编译以固定形状为前提；本周编译目标为 0.6B（可选 4B）。

## 实现范围

- 新增向量化 BF16 前向/decode/greedy（设备通用），embedding gather、
  RoPE、GQA attention（F32 softmax）、SwiGLU、logits 全部无标量循环。
- CUDA eager：0.6B / 1.7B / 4B BF16 树上载 GPU，逐层 parity（vs
  Week 14 HF BF16 reference）、16 步 greedy 一致性、dynamic cache
  greedy decode 吞吐（tok/s）与 VRAM 实测。
- Reactant XLA BF16：固定形状 prefill 全前向编译执行，logits parity
  与 argmax/greedy 首 token 检查、compile/steady 耗时记录；固定形状
  单 token decode 为 stretch 目标。
- `test/test_week15.jl`：离线合成模型上"向量化 vs Week 14 循环路径"
  对照进默认套件；CUDA/XLA 真实验证经环境变量 opt-in。
- 吞吐与 Week 14 CPU 数值（0.6B 34.7 s / 4B 182.6 s 每 18 pass）
  同表对照，记录加速倍数。

## 验证分层

| 证据层 | 目标状态 |
| --- | --- |
| 向量化 BF16 路径 vs Week 14 CPU 循环路径 | 合成模型默认离线对照 |
| CUDA BF16 逐层 / logits / decode parity | 0.6B/1.7B/4B 在 BF16 量级容差内 |
| CUDA BF16 greedy 16 步 vs HF BF16 | 完全一致，或逐 token 一致率量测后冻结 |
| CUDA BF16 decode 吞吐 / VRAM | 实测记录并对照 CPU |
| XLA BF16 编译 prefill parity | 0.6B 通过并记录编译/执行耗时 |
| 既有 CPU BF16 / F32 / 流式 / XLA F32 路径 | 零改动，无回归 |

## Close 条件

- 向量化 BF16 路径在合成模型上与 Week 14 CPU 路径对照通过并进默认
  套件；真实 0.6B/1.7B/4B 的 CUDA parity 与 greedy 结果冻结进
  fixture（若 greedy 出现近平局翻转，逐 token 一致率与差异位置必须
  量化记录而不是降低标准掩盖）。
- CUDA BF16 dynamic cache greedy decode 吞吐与 VRAM 实测记录在案，
  与 CPU BF16 的耗时对照成表。
- 0.6B XLA BF16 编译 prefill 与 reference 对齐，编译与 steady 耗时
  记录在案。
- 默认全套与既有 opt-in（分进程协议）无回归；Week 14 CPU 路径证据
  不变。
- 文档明确：8B+ GPU 驻留、XLA BF16 完整生成闭环、量化仍未实现。

## 非目标

- 不做 8B/14B/32B 的 GPU 驻留、多卡或 CPU-GPU 权重切分。
- 不做 BF16 训练/反向、FP8/INT8/INT4 量化。
- 不做 sampling replay、chat 端到端或长上下文 GPU 优化。
- 不改动 Week 14 CPU BF16 路径与其冻结容差/greedy 证据。

## 过程记录

### 2026-07-26：Open

- Week 14 保持 Closed；Week 15 承接 BF16 的 CUDA/XLA 落地。
- 两道能力闸门（CUDA BF16 原语、Reactant BF16 编译与转换写法）在
  Open 前实测通过，结论记入本页。
- 资源边界冻结：GPU 驻留上限 4B（7.49 GiB / 16.3 GiB VRAM）；8B 出界。
