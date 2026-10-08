# Chapter 57 — Qwen3-VL decode token embedding 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## 结果

单 token 的 `gather(embedding, tokens)` 在 RTX 4090 D 上是 `4,105` bytes、`3`
次分配。输出是 `(2048, 1, 1)` BF16，`4,096` bytes。把这一列写进预分配 buffer
与 `gather` 逐位一致，复用调用是 `0` GPU bytes。

驻留 scratch 是 `4,096` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_token_embedding.json`，SHA256 为
`dfaa168759d4beddc9e8db405cbcdf7a3a0896a83831a3f2df6b0335fd575742`。
position `107 / 203 / 331` 的最后一步都是 `6,328` bytes、`1,349` 次分配。
`token_embedding` 是 `0`。相对 Chapter 56 的 `10,433 / 1,352`，正好少了
`4,105` bytes 和 `3` 次分配。前四枚是 `[1987, 2169, 375, 265]`。128/256 仍在
step 70 分叉，margin `0.02797699`。没有新的线性头。

最大阶段回到 `attention` 的 `3,584` bytes：每层两次 `batched_mul!` 各 `64`
bytes、`16` 次分配。Episode 10 保持 Open。最终 logits 没有升成 Float32。
