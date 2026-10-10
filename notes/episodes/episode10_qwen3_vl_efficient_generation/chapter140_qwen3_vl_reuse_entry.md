# Chapter 140 — Qwen3-VL 复用入口不再装箱 spec

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-10
>
> 关闭：2026-10-10

## 结果

复用路径从大函数里拆出来。text spec 改由已经捉住的 pack 在具体函数里读取，
动态调用不再装箱整个 spec。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_reuse_entry.json`，
SHA256 为
`65ce9f34941b20837d9aeae43badbf13e7bc03366a1ba0c1a26ce8303bda0bf4`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32/128/256 的第一次 decode CPU 都从 `192` 降到 `48`。下降都是
`144` bytes。第一次 decode 的 GPU 字节仍是 `0`，分配次数仍是 `0`。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin
`0.02797699`。

第一次 decode 仍有 `48` bytes。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
