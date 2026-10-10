# Chapter 109 — Qwen3-VL 长生成计时不再捕获参数

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

长生成函数直接用 `CUDA.@timed` 包住 decode。这个计时会同步执行表达式，
大函数里的那次调用把整份参数元组捕获进闭包。真实参数元组是 `2,968`
bytes，测到的一步因此多出 `3,072` bytes。

计时改到一个小函数里。参数、token 和 cache 先作为参数传进去，再计时
同一次 decode。decode 计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_time_decode.json`，
SHA256 为
`d82e5e953538c6d1631e09ac9bfe887325439f4055831627371c1378dbfa74aa`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `7,216`、
`7,216` 和 `7,200`。256-token 稳态一步从 Chapter 108 的 `10,272`
降到 `7,200`。整步下降是 `3,072` bytes。第一次 decode 的 GPU 分配仍是
`2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based
`21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
