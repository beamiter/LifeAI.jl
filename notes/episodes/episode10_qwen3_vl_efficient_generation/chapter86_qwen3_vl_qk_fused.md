# Chapter 86 — Qwen3-VL decode QK RMSNorm 的同一次启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 85 之后，Q 和 K 的 RMSNorm 各做一遍升精度、`sum!(abs2)` 和
归一化。两边共用 scratch 的 head 前缀，所以 K 必须等同一个 head 的 Q
归一化写完。

一次启动按这个顺序做完。长度 `128` 的 `abs2` 归约与 `sum!(abs2)` 逐位
一致。Q、K 的输出和最终 `mean_square` 与两条单张量路径都是 0 个不一致。
CPU 仍走两条单张量路径。

一层少 `432` bytes，28 层是 `12,096`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qk_fused.json`，
SHA256 为
`1f6e990c7b7e060edd280755d60ff385279370538e3f81a038e71eeaa78d9f3c`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `365,648`。
256-token 稳态一步从 Chapter 85 的 `377,744` 降到 `365,648`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段是 `mlp_gate_up_projection`。Episode 10
保持 Open。最终 logits 没有升成 Float32。
