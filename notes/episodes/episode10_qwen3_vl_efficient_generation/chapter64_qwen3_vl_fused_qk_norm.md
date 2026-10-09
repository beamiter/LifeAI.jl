# Chapter 64 — Qwen3-VL decode QK RMSNorm 合成 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

`sum!(abs2, ...)` 的归约留在原来的单 block 路径上。均值除以 `head_dim`
之后，归一化和 scale 合成一次 kernel：

`bf16(f32(bf16(f32(x) / sqrt(mean + eps))) * f32(scale))`

CPU 仍走原来的广播。query 16 head 与 key 8 head 各 4 组随机输入与 CPU
路径逐位一致，复用调用是 `0` GPU bytes。

联合关闭采样见 Chapter 67。
