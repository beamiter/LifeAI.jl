# Episode 04 — 高效推理与量化

> 状态：Closed
>
> 收录章节：Chapter 14–18

这一卷把正确但昂贵的 Float32 路径推进到 native BF16、CUDA/XLA 编译解码和可预算混合精度量化，并如实记录校准量化的负结果。

## 章节目录

1. [`Chapter 14 — Qwen3 Native BF16 Mixed-Precision Compute`](chapter14_qwen3_bf16_compute.md)
2. [`Chapter 15 — Qwen3 BF16 CUDA / XLA Accelerated Inference`](chapter15_qwen3_bf16_accel.md)
3. [`Chapter 16 — Qwen3 XLA BF16 Compiled Decode 与 INT8/INT4 量化`](chapter16_qwen3_xla_decode_quant.md)
4. [`Chapter 17 — Qwen3 Reconstruction-Calibrated INT4 与预算化混合精度`](chapter17_qwen3_calibrated_int4.md)
5. [`Chapter 18 — Qwen3 Activation-Aware INT4 校准`](chapter18_qwen3_activation_calibration.md)

## 本卷结果

0.6B–8B 获得 HF BF16 行为一致性，XLA greedy decode 达到 246 tok/s；8B INT8 与 14B INT4 首次 GPU 驻留，同时明确了 reconstruction / activation-weighted clipping 不能直接代理生成 fidelity。
