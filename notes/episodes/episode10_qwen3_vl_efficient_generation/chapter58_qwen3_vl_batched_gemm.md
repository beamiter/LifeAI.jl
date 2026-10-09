# Chapter 58 — Qwen3-VL decode attention batched GEMM 的 64 bytes

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## Open：核心问题

Chapter 57 之后，最大阶段是 `attention` 的 `3,584` bytes。它是每层两次
`batched_mul!`，每次 `64` bytes、`16` 次分配。`16 = 8` 个 KV head × 每次小
GEMM 的 `2` 次分配，`64 = 8 × 8`。BF16 没有走 cuBLAS 的 strided batched GEMM，
而是对每个 head 调一次 `mul!`。

## 没有采纳的替换

`CUDA.CUBLAS.gemmStridedBatchedEx!` 把一次调用降到 `8` bytes、`2` 次分配。
大多数 sequence length 上，它和 `NNlib.batched_mul!` 的完整 attention 输出逐位
一致。sequence 等于 capacity `331` 时不是：`5,296` 个 score 里有 `1` 个差
`0.0625`。

把这条 GEMM 接进 decode 后，一次 repeat 的剖面在
`.../chapter47-long-oracle/qwen3_vl_allocation_strided_gemm.json`，SHA256 为
`b8542c41ada07f774c728e0f72457b6627098822759f1c55b2ed87caa7c50561`。
`attention` 从 `3,584 / 896` 降到 `448 / 112`，整步是 `3,192` bytes。但
128/256 的自由生成分叉从 step 70 回到了 attention scratch 之前的 step 79：
LifeAI zero-based `281`，HF `2518`，margin `0.04928589`。这不是 Chapter 57
那条 BF16 路径。

因此代码仍使用原来的 per-head `batched_mul!`。`64` bytes 留下，是因为换掉它
会改变 BF16 数值。

## Close 回顾

- **完成了什么**：核对了 `64` bytes 的来源，并证明一次 strided GEMM 能把分配
  降到 `8` bytes，但不能保持与当前 BF16 attention 逐位一致。
- **验证证据**：sequence `1` 到 `330` 的完整 attention 输出与 `NNlib` 路径逐位
  一致；`331` 有一个元素差 `0.0625`。接进模型后的剖面 SHA 见上。该 GEMM 没有
  留在代码里。
- **带到下一问的问题**：代码回到 per-head `batched_mul!` 之后，最大阶段仍是
  `attention` 的 `3,584` bytes。它不能靠这次 strided GEMM 去掉。每次普通
  `mul!` 留下的 `8` bytes 是另一笔更小的账。

Episode 10 保持 Open。最终 logits 没有升成 Float32。
