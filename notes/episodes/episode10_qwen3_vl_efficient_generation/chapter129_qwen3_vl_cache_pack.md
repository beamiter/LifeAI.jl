# Chapter 129 — Qwen3-VL 参数 pack 随 cache 创建

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

blocks 那份 tuple 原来在第一次 decode 里拷进 pack。现在建 static
cache 时拷一次，第一次 decode 直接用它。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_cache_pack.json`，
SHA256 为
`137d69319e472b2faaf15ca5e89b097ebf87d09142d668a64fec64ba12b677c5`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `27,472` 降到 `24,784`。128/256 从
`21,712` 降到 `19,024`。下降都是 `2,688` bytes。256-token 第一次
decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `21`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
