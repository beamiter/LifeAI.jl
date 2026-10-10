# Chapter 79 — Qwen3-VL decode QK abs2 的具体引用

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 78 之后，Q 和 K 的 RMSNorm 各走一次 `sum!(abs2)`。生产路径上的
输入是 head 前缀 view。计划放在 `Dict{Any,Any}` 里，每次取出都会装箱。
query 与 key 各是 `304` bytes。一次 Q 或 K 归一化是 `720` bytes。

Q 和 K 的单 block 计划改存进具体 `Ref`。head 数或归约长度对不上时仍走原来的
字典，hidden 长度 `2048` 的多 block 归约也留在那里。数值和 `sum!(abs2)` 相同。

query 与 key 的 `sum!(abs2)` 都降到 `144` bytes。一次 Q 或 K 归一化降到
`560` bytes。一层少 `320` bytes，28 层是 `8,960`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qk_abs2.json`，
SHA256 为
`10da9cdb7e18588ce9431f3a8dcc9b719a9062ea12d53cb107304f1d4396fcc2`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `415,936`。
256-token 稳态一步从 Chapter 78 的 `424,896` 降到 `415,936`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
