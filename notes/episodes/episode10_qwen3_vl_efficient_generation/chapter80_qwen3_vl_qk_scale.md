# Chapter 80 — Qwen3-VL decode QK scale 的 reshape

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 79 之后，Q 或 K 的归一化 kernel 是 `272` bytes。其中 `96` bytes 是把
长度 `128` 的 BF16 scale 收成 `(head_dim, 1, 1, 1)`。权重在一次生成里不变。

Q 和 K 各留一份具体类型的 reshape。已经是 4 维的 scale 不经过这份缓存。
和每次现做的 reshape 相比，query 与 key 的归一化输出都是 0 个不一致。
带缓存的 kernel 是 `176` bytes。一次 Q 或 K 归一化从 `560` 降到 `464`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qk_scale.json`，
SHA256 为
`9105965aca4d2c296b5f41b34f4820af00a60ee277153fb38db3dc0ed129fb50`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `415,744`。
256-token 稳态一步从 Chapter 79 的 `415,936` 降到 `415,744`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
