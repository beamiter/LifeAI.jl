# Chapter 120 — Qwen3-VL attention scratch 计字节不再装箱

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

核对 attention scratch 字节时，循环遍历那组不同类型的数组，每个元素
都会装箱。改成直接把 13 个字段的 `sizeof` 相加。公式没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attention_bytes.json`，
SHA256 为
`16645b1138ca6369ca3a0b386ad53422680ae645765a2cc1addbf27d76abebcf`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `43,424` 降到 `41,664`。128/256 从
`37,664` 降到 `35,904`。下降都是 `1,760` bytes。第一次 decode 的 GPU
分配仍是 `2,018,180` bytes、`56` 次。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
