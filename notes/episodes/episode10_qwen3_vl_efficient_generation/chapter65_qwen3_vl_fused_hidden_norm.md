# Chapter 65 — Qwen3-VL decode hidden RMSNorm 元素合成 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

长度 `2048` 的 `sum!` 仍复用 Chapter 60 的 partial，归约顺序不变。均值
算完之后，隐藏维的归一化和 scale 合成一次 kernel，舍入顺序与 Chapter 64
相同。CPU 仍走广播。

长度 `2048` 的 4 组随机输入与 CPU 路径逐位一致，复用调用是 `0` GPU bytes。

联合关闭采样见 Chapter 67。
