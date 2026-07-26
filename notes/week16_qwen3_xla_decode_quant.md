# Week 16 — Qwen3 XLA BF16 Compiled Decode 与 INT8/INT4 量化

> 状态：Open
>
> 开启记录：2026-07-26
>
> 依赖基线：[`Week 15 — Qwen3 BF16 CUDA / XLA Accelerated Inference`](week15_qwen3_bf16_accel.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Week 15 的 CUDA eager BF16 生成吞吐 8—15 tok/s，瓶颈在
> 逐步 kernel launch 与 dynamic cache 的 `cat` 重分配；同时 8B BF16
> （15.26 GiB）仍进不了 16.3 GiB VRAM。本周两条线：XLA 编译
> static-cache decode 把吞吐拉一个量级；INT8/INT4 权重量化打开 8B+
> 的 GPU 驻留。

## 核心问题

> 1. BF16 混合精度契约能否在固定形状 static KV cache 下编译为单个
>    XLA decode 可执行文件，逐 token 行为与 eager 路径一致，且 steady
>    吞吐达到 eager 的 10 倍量级？
> 2. RTN 权重量化（INT8 per-channel / INT4 group-128）能否把 8B 乃至
>    14B 装进 16.3 GiB VRAM，并把量化引入的 token 级行为漂移量化清楚
>    而不是掩盖？

## 设计决策

- **XLA decode 镜像既有 F32 static 模式**：traced position、
  `cos_cache[:, position]` 动态取位、`cache[..., write_position, :] =`
  动态写入、`key_position .<= valid_length` 有效前缀掩码——Week 02—09
  已验证的编译形态，换成 Week 14/15 的 BF16 算子语义。
- **量化是权重-only 的 RTN**：INT8 对称 per-output-channel；INT4 对称
  group-128、两值一字节打包。embedding 保持 BF16（gather 位精确），
  全部线性层（含 lm_head）量化。计算路径：GPU 上按层解包/反量化为
  BF16 后走既有 `_bf16a_linear`——量化只改变权重驻留格式，不改变
  计算契约。
- **验证基准分层**：量化模型先与自家 BF16 路径对照（隔离量化误差），
  再与 HF BF16 reference 对照 token 级行为；INT4 若出现 greedy 漂移，
  记录逐 token 一致率与首个分歧位置，不降低标准掩盖。
- 14B 缺 BF16 reference：用 Week 13 的 accelerate disk offload 以
  `--compute-dtype bfloat16` 导出（14B BF16 29.5 GiB 超 RAM，offload
  可行已验证）。

## 本周资源边界（先冻结再执行）

| 配置 | 权重驻留 | 16.3 GiB VRAM | 计划 |
| --- | ---: | --- | --- |
| 0.6B BF16 XLA static decode | 1.1 GiB + XLA 缓冲 | 可行 | 必做 |
| 4B BF16 XLA static decode | 7.5 GiB + XLA 缓冲 | 边缘 | stretch |
| 8B INT8 | ≈ 8.2 GiB + BF16 embedding | 可行 | 必做 |
| 14B INT4 g128 | ≈ 7.4 GiB + scales + BF16 embedding | 可行 | 必做（行为漂移如实记录） |
| 32B INT4 | ≈ 16.4 GiB | 出界 | 不做 |

## 实现范围

- BF16 static KV cache 与 traced decode step（单 token）：prefill 编译
  填充 cache，decode 编译复用同一 executable；0.6B 与 eager CUDA 的
  greedy 序列逐 token 对照，warm/steady 吞吐与编译成本记录。
- `quantize` API：BF16 树 → INT8/INT4 量化树（打包 + scales）；
  量化树可整体 `cu()` 上载；`_bf16a_linear` 对量化权重的 dequant
  分派；量化/反量化 round-trip 与打包正确性离线测试。
- 8B INT8 GPU 驻留：VRAM 实测；vs 自家 CPU BF16 logits 与 vs HF BF16
  reference（week14-bf16-parity）的 parity/greedy；吞吐记录。
- 14B BF16 offload reference 导出（层输出 + 16 步 greedy）；14B INT4
  GPU 驻留：logits 偏差、argmax/greedy 一致率、吞吐、VRAM。
- `test/test_week16.jl`：量化数值语义与 round-trip 离线测试 + 合成
  模型量化推理对照；GPU/XLA 真实验证 opt-in。

## 验证分层

| 证据层 | 目标状态 |
| --- | --- |
| XLA BF16 decode vs eager greedy 序列 | 0.6B 逐 token 一致 |
| XLA decode steady 吞吐 | 记录并对照 eager（目标 ≥10×） |
| 量化 round-trip / 打包语义 | 默认离线覆盖 |
| 8B INT8 GPU：VRAM / parity / greedy | 驻留实测；token 级行为与 BF16 对照冻结 |
| 14B INT4 GPU：VRAM / parity / greedy 一致率 | 驻留实测；漂移如实量化 |
| 既有全部路径 | 零改动，无回归 |

## Close 条件

- 0.6B XLA BF16 compiled decode 与 eager CUDA greedy 逐 token 一致，
  steady 吞吐实测并对照 eager；编译成本记录。
- INT8/INT4 量化树的构造、打包、反量化有离线数值测试；量化只作用于
  线性层权重且计算契约不变有明确陈述。
- 8B INT8 在 GPU 驻留并完成 16 步 greedy：VRAM、吞吐、与 HF BF16
  reference 的 logits 偏差和 greedy 一致性全部冻结。
- 14B INT4 在 GPU 驻留并完成 16 步 greedy：同上；若 greedy 与 BF16
  reference 不完全一致，逐 token 一致率与首个分歧位置记录在案。
- 默认全套与既有 opt-in 无回归；文档明确 32B、量化训练、GPTQ/AWQ
  等边界。

## 非目标

- 不做 GPTQ/AWQ/校准式量化、FP8、KV cache 量化或激活量化。
- 不做 32B GPU 驻留、多卡或权重切分。
- 不做 XLA 端到端 chat/sampling 闭环；compiled greedy decode 是本周
  XLA 证据边界。
- 不改动既有 CPU BF16 / F32 / 流式路径的数值行为。

## 过程记录

### 2026-07-26：Open

- Week 15 保持 Closed；Week 16 承接 XLA decode 编译与量化两条线。
- F32 XLA static 模式（traced position/动态写/前缀掩码）确认可镜像；
  量化方案冻结为 RTN INT8 per-channel 与 INT4 group-128。
- 资源边界冻结如上表；32B 出界。
