# Chapter 15 — Qwen3 BF16 CUDA / XLA Accelerated Inference

> 状态：Closed
>
> 开启记录：2026-07-26
>
> 关闭记录：2026-07-26
>
> 依赖基线：[`Chapter 14 — Qwen3 Native BF16 Mixed-Precision Compute`](../../episodes/episode04_efficient_inference_and_quantization/chapter14_qwen3_bf16_compute.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Chapter 14 的 BF16 推理路径全部在 CPU 上验证，8B 一次前向
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

- Chapter 14 的 CPU BF16 路径含逐 head 标量循环，GPU 不可用。新增一套
  **设备通用的向量化 BF16 前向**：只用 broadcast、`batched_mul`、
  gather 等设备友好原语，同一份代码服务 CPU 数组、CuArray 与
  Reactant tracing；混合精度契约与 Chapter 14 逐条相同。
- GPU 上线性层直接用 CUBLAS BF16 gemm（硬件 F32 累加），与 Chapter 14
  "BF16 存储 + F32 累加 + BF16 舍入"语义一致；不同的是求和顺序——
  沿用 Chapter 13/14 的方法论：数值以 BF16 量级容差衡量，正确性以
  token 级行为（argmax、greedy 序列）为准。
- Chapter 14 CPU 路径及其冻结证据零改动；向量化路径与 CPU 循环路径在
  合成模型上先互证，再上真实权重。
- reference 复用 Chapter 14 已导出的 `week14-bf16-parity`（HF CPU BF16
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
  Chapter 14 HF BF16 reference）、16 步 greedy 一致性、dynamic cache
  greedy decode 吞吐（tok/s）与 VRAM 实测。
- Reactant XLA BF16：固定形状 prefill 全前向编译执行，logits parity
  与 argmax/greedy 首 token 检查、compile/steady 耗时记录；固定形状
  单 token decode 为 stretch 目标。
- `test/episodes/episode04_efficient_inference_and_quantization/chapter15_qwen3_bf16_accel/test_qwen3_bf16_acceleration.jl`：离线合成模型上"向量化 vs Chapter 14 循环路径"
  对照进默认套件；CUDA/XLA 真实验证经环境变量 opt-in。
- 吞吐与 Chapter 14 CPU 数值（0.6B 34.7 s / 4B 182.6 s 每 18 pass）
  同表对照，记录加速倍数。

## 验证分层

| 证据层 | 最终状态 |
| --- | --- |
| 向量化 BF16 路径 vs Chapter 14 CPU 循环路径 | **逐位相同**（合成 tied/untied + 真实 0.6B，默认离线覆盖） |
| CUDA BF16 逐层 / logits / decode parity | 0.6B/1.7B/4B 全部在 BF16 量级容差内，argmax 零失配 |
| CUDA BF16 greedy 16 步 vs HF BF16 | 三尺寸全部完全一致（无近平局翻转） |
| CUDA BF16 decode 吞吐 / VRAM | 15.3 / 14.1 / 8.1 tok/s，VRAM ≤ 12.1 GiB，对照 CPU 33—92× |
| XLA BF16 编译 prefill parity | 0.6B 通过：编译 44.8 s、steady 1.36 ms、argmax/greedy 首 token 一致 |
| 既有 CPU BF16 / F32 / 流式 / XLA F32 路径 | 零改动，默认全套无回归 |

## Close 条件

- 向量化 BF16 路径在合成模型上与 Chapter 14 CPU 路径对照通过并进默认
  套件；真实 0.6B/1.7B/4B 的 CUDA parity 与 greedy 结果冻结进
  fixture（若 greedy 出现近平局翻转，逐 token 一致率与差异位置必须
  量化记录而不是降低标准掩盖）。
- CUDA BF16 dynamic cache greedy decode 吞吐与 VRAM 实测记录在案，
  与 CPU BF16 的耗时对照成表。
- 0.6B XLA BF16 编译 prefill 与 reference 对齐，编译与 steady 耗时
  记录在案。
- 默认全套与既有 opt-in（分进程协议）无回归；Chapter 14 CPU 路径证据
  不变。
- 文档明确：8B+ GPU 驻留、XLA BF16 完整生成闭环、量化仍未实现。

## 非目标

- 不做 8B/14B/32B 的 GPU 驻留、多卡或 CPU-GPU 权重切分。
- 不做 BF16 训练/反向、FP8/INT8/INT4 量化。
- 不做 sampling replay、chat 端到端或长上下文 GPU 优化。
- 不改动 Chapter 14 CPU BF16 路径与其冻结容差/greedy 证据。

## 过程记录

### 2026-07-26：Open

- Chapter 14 保持 Closed；Chapter 15 承接 BF16 的 CUDA/XLA 落地。
- 两道能力闸门（CUDA BF16 原语、Reactant BF16 编译与转换写法）在
  Open 前实测通过，结论记入本页。
- 资源边界冻结：GPU 驻留上限 4B（7.49 GiB / 16.3 GiB VRAM）；8B 出界。

### 2026-07-26：实现与全量验证

- 新增 `src/models/bf16_accel.jl`：设备通用向量化 BF16 前向
  `hf_qwen3_bf16_accel_forward`，只用 broadcast / `batched_mul` /
  gather / 归约；两处 CPU 专用分派保住契约——线性层走 Chapter 14 分块
  F32 kernel，batched matmul 在 CPU 上显式 F32 gemm + BF16 舍入
  （通用 fallback 会以 BF16 累加，实测曾导致 greedy 漂移）。
- **CPU 上向量化路径与 Chapter 14 循环路径逐位相同**（trace、decode、
  greedy 全部 `==`，真实 0.6B 与合成 tied/untied 模型均验证）——
  向量化重写没有引入任何数值变化。
- 两个环境坑记录：(1) cuDNN 库在 LifeAI 之后加载会 `libcudnn_cnn.so`
  初始化失败，脚本必须先 `using LuxCUDA` 再 `using LifeAI`；测试内
  改用纯 `CUDA.cu` 移树（CUDA 是硬依赖，无需 cuDNN）。(2) Reactant
  traced 数组 eltype 是 `TracedRNumber{BFloat16}`，签名不能写死
  `AbstractArray{BFloat16}`。
- **CUDA eager 三尺寸全部通过**（RTX 5080，vs Chapter 14 HF BF16
  reference，argmax 零失配，16 步 greedy 全部完全一致）：

  | variant | logits max/mean | GPU 树 | VRAM 峰值 | warm 16-token | tok/s | vs CPU BF16 |
  | --- | --- | ---: | ---: | ---: | ---: | ---: |
  | 0.6B | 0.688 / 0.075 | 1.12 GiB | 8.3 GiB | 1.04 s | 15.3 | ≈ 33× |
  | 1.7B | 0.844 / 0.057 | 3.23 GiB | 8.7 GiB | 1.13 s | 14.1 | ≈ 69× |
  | 4B | 0.266 / 0.037 | 7.50 GiB | 12.1 GiB | 1.98 s | 8.1 | ≈ 92× |

- **Reactant XLA BF16 编译通过**（0.6B prefill 全前向）：编译 44.8 s，
  首执行 1.72 s，**steady 1.36 ms**；logits max-abs 0.578（mean
  0.061），argmax 与 greedy 首 token 与 HF 一致，两次执行逐位相同。
- 全部结果冻结进 `test/episodes/episode04_efficient_inference_and_quantization/chapter15_qwen3_bf16_accel/fixtures/qwen3_bf16_acceleration/assets.json`
  （容差沿用 Chapter 14：logits/decode 2.0、blocks scaled 5e-2）。

### 2026-07-26：验证与 Close

- 默认离线全套 `4792 / 4792` 通过（Chapter 15 离线专项 `63 / 63`：
  batched matmul 契约、向量化 vs 循环逐位一致、CUDA/XLA 资产
  contract）；Chapter 05—14 计数与各自 Close 时一致，无回归。
- CUDA opt-in（三尺寸同进程 `Pkg.test`）：Chapter 15 专项 `173 / 173`
  通过。第一次运行暴露 world-age 坑：在测试函数内 `Base.require`
  动态加载 CUDA 后，函数体内后续泛型调用仍在旧 world——修复为文件
  顶层按需 `using CUDA`，积分函数在其后定义。
- XLA BF16 由独立脚本验证（`scripts/verify_qwen3_bf16_xla.jl`），
  结果冻结进 fixture 并由离线 contract 校验。

## Close 回顾

- **完成了什么**：BF16 推理落地 GPU——设备通用向量化路径（CPU 上与
  Chapter 14 循环路径逐位相同）在 RTX 5080 上以原生 BF16 张量核运行；
  0.6B/1.7B/4B CUDA parity 与 16 步 greedy 全部与 HF BF16 一致；
  Reactant XLA BF16 编译 prefill 通过。推理验证的主战场从 CPU 移到
  CUDA/XLA。
- **验证证据**：CUDA 三尺寸 argmax 零失配、greedy 完全一致、吞吐
  15.3/14.1/8.1 tok/s（CPU 的 33/69/92 倍）、VRAM ≤ 12.1 GiB；XLA
  steady 1.36 ms/prefill 且两次执行逐位相同；默认全套 `4792 / 4792`、
  CUDA opt-in `173 / 173`。
- **没有完成及原因**：8B GPU 驻留（15.26 GiB > 16.3 GiB VRAM 无余量）
  与 XLA 完整生成闭环（固定形状 decode 编译）留待后续；量化、
  sampling replay、长上下文 GPU 优化按计划未做。
- **最重要的认知变化**：其一，同一套混合精度契约可以"一份实现、三种
  执行形态"（CPU 仿真、CUDA eager、XLA 编译），前提是只用设备友好
  原语并显式控制每一次舍入；其二，通用 fallback 是数值契约的隐形
  杀手——CPU batched matmul 默默用 BF16 累加直接改变 greedy 输出，
  分派边界必须显式钉死；其三，加载顺序（cuDNN）、traced eltype、
  world age 这类工程细节在 GPU 栈里比数值本身更容易咬人，全部记入
  可复现命令。
- **是否满足 Close 条件**：是。逐位对照、CUDA parity/greedy/吞吐、
  XLA prefill、默认与 opt-in 回归、文档边界均已落实。
