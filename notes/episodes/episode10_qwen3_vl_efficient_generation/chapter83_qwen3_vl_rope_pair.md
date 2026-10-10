# Chapter 83 — Qwen3-VL decode Q/K RoPE 的同一次启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 82 之后，每层 RoPE 对 Q 和 K 各启动一次，公式相同，合起来是
`352` bytes。

一次启动覆盖 query heads 和 KV heads。每个元素的乘加和 BF16 舍入与原来的
kernel 相同。3 维和 2 维的 cos/sin 表上，Q 与 K 的输出都是 0 个不一致。
CPU 仍走两次单张量路径。合并后的调用是 `208` bytes。

一层少 `144` bytes，28 层是 `4,032`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_rope_pair.json`，
SHA256 为
`39f8c6ab8f0320a24c27c7f5af400ea98c45322fe91282be5893fbc96ca80fad`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `384,464`。
256-token 稳态一步从 Chapter 82 的 `388,496` 降到 `384,464`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
