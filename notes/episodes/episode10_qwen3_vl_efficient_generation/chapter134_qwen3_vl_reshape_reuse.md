# Chapter 134 — Qwen3-VL 复用 reshape 矩阵

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

QKV、tied logits、query 和 attention 投影的 reshape 原来在每个新
scratch 上各建一张矩阵。现在改写已经建好的那张的指针和形状。
计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_reshape_reuse.json`，
SHA256 为
`92cce38e05d6123d6f55f4a520cbc4c7df36ce3be53339690296fafa904cb17a`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `7,408` 降到 `6,768`。下降都是
`640` bytes。256-token 第一次 decode 的 GPU 字节仍是 `2,018,180`，
分配次数仍是 `2`。前四枚是 `[1987, 2169, 375, 265]`。32-token 没有
分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF
`64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
