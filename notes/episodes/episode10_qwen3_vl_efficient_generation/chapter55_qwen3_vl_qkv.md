# Chapter 55 — Qwen3-VL decode QKV projection 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## 结果

每层 Q/K/V 是三次 BF16 GEMM。输出分别是 `(128, 16, 1, 1)`、`(128, 8, 1, 1)`、
`(128, 8, 1, 1)`，合计 `8,192` bytes。三次 `mul!` 再留下 `24` bytes。
`28 × (8,192 + 24) = 230,048`，与复用前的阶段一致。

28 层共用这三块输出。QK-norm 和 attention 在下一层开始前已经读完它们。
GPU `mul!` 与分配型 GEMM 逐位一致；CPU 走 chunked Float32。

驻留 scratch 是 `8,192` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qkv.json`，SHA256 为
`5c27f76184dca24a0eb0ce48dec26bf92809baad5a92407affcd461c0aaf0995`。
position `107 / 203 / 331` 的最后一步都是 `14,017` bytes、`1,408` 次分配。
`qkv_projection` 从 `230,048 / 252` 降到 `672 / 168`。`672 = 28 × 3 × 8`。
相对 Chapter 54 的 `243,393 / 1,492`，正好少了 `229,376` bytes 和 `84` 次分配。
前四枚是 `[1987, 2169, 375, 265]`。128/256 仍在 step 70 分叉，margin
`0.02797699`。没有新的线性头。

下一问是 `attention` 剩下的 `7,168` bytes（每层 `256` bytes、`34` 次分配）。
Episode 10 保持 Open。最终 logits 没有升成 Float32。
