# Chapter 66 — Qwen3-VL decode hidden residual 合成 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

attention 输出投影和 MLP down 之后的残差原先是三次隐藏维广播：
`bf16(f32(residual) + f32(projected))`。GPU 路径收成一次 kernel。CPU
仍走广播。GEMM 本身没改。

长度 `2048`、投影宽度 `128` 的 3 组随机输入与 CPU 路径逐位一致，复用
调用是 `0` GPU bytes。

联合关闭采样见 Chapter 67。
