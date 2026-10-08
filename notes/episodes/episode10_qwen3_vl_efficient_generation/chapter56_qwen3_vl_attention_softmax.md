# Chapter 56 — Qwen3-VL decode attention softmax 归约

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## 结果

复用后的 attention 每层仍是 `256` bytes、`34` 次分配。拆开之后是四笔：

| 调用 | bytes | 次数 |
| --- | ---: | ---: |
| score `batched_mul!` | 64 | 16 |
| `maximum` | 64 | 1 |
| softmax 里的 `sum` | 64 | 1 |
| context `batched_mul!` | 64 | 16 |

`maximum!` 和 `sum!` 写进已有的 `(groups, 1, kv_heads)` buffer，在 sequence `331`
的前缀 view 上是 `0` GPU bytes，并且与 `maximum` / `sum` 的 `dims=2` 结果逐位
一致。两次 `batched_mul!` 的 `64` bytes 还在。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attention_softmax.json`，SHA256 为
`3dda0a60100a6dbd73b0931b858b34e51205899daf941acf065ea16e9019ff49`。
position `107 / 203 / 331` 的最后一步都是 `10,433` bytes、`1,352` 次分配。
`attention` 从 `7,168 / 952` 降到 `3,584 / 896`。`3,584 = 28 × 128`，正好是
每层两次 `batched_mul!`。相对 Chapter 55 的 `14,017 / 1,408`，正好少了
`3,584` bytes 和 `56` 次分配。前四枚是 `[1987, 2169, 375, 265]`。128/256 仍在
step 70 分叉，margin `0.02797699`。没有新的线性头。

最大阶段现在是 `token_embedding`，`4,105` bytes、`3` 次分配、`1` 次调用。
attention 的两笔 `64` byte GEMM 临时区排在其后。Episode 10 保持 Open。最终
logits 没有升成 Float32。
