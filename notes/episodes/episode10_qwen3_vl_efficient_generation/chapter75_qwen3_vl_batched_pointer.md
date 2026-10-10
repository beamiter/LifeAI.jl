# Chapter 75 — Qwen3-VL decode per-head GEMM 的指针偏移

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 74 之后，一次 attention 调用是 `3,888` bytes。剩下的两块 per-head
GEMM 各是 `768` bytes。每一头原来先做 `view`，再取那块矩阵的地址。

GPU 路径不再造 view。第 `index` 头的地址是基址加上
`(index - 1) * stride(3) * sizeof(BFloat16)`，lda 仍是 `stride(2)`。
`cublasGemmEx` 的 compute type、算法、alpha 和 beta 不变。某一边 batch 为 1
时仍重复那一头。CPU 仍走原来的 per-head 循环。

RTX 4090 D 上，sequence `1 / 80 / 331` 的 score 和 context，以及左边 batch
为 1 的广播，都与逐头 `view` 路径是 0 个不一致。一次 8 头 GEMM 从 `768`
bytes 降到 `128`。一次 attention 调用降到 `2,336` bytes，sequence `80` 和
`331` 相同。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_batched_pointer.json`，
SHA256 为
`8cfa7b7f67eb1f0c241b751d6a047a68c997e1e25db81a7d58143eca6485beae`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 是 `581,600`、`581,648`、
`581,600`。256-token 稳态一步从 Chapter 74 的 `617,440` 降到 `581,600`。
第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70 分叉：
LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段变成 QK norm。Episode 10 保持 Open。最终
logits 没有升成 Float32。
