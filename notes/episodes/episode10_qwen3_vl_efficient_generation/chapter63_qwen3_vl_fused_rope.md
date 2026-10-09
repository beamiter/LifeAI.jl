# Chapter 63 — Qwen3-VL decode RoPE 合成 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

单 token RoPE 原先是十几条 GPU 广播。每条广播不占 GPU 字节，但会造一批
Julia 启动对象。GPU 路径改成一次 kernel，按原顺序做 Float32 乘、BF16
舍入、再加。CPU 仍走原来的广播。

RTX 4090 D 上，query 16 head 和 key 8 head 各 4 组随机输入与 CPU 路径逐位
一致。复用调用是 `0` GPU bytes。一次 RoPE 的 CPU 分配降到 `176` bytes、
`7` 次。

联合关闭采样见 Chapter 67。128/256 仍在 step 70 分叉。
