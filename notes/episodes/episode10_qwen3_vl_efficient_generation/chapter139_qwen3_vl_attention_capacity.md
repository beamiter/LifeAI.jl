# Chapter 139 — Qwen3-VL attention capacity 原地更新

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

`capacity` 改成可写。后来的请求不再新建 attention scratch，只改这个
字段。数组本身已经按这次容量改过指针和形状。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attention_capacity.json`，
SHA256 为
`d3597accee924dba321e9252a6f9411fbd6838152971d5a9c28302ba1a3fca42`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `352` 降到 `192`。下降都是
`160` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin
`0.02797699`。

第一次 decode 仍有 `192` bytes。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
