# Chapter 100 — Qwen3-VL decode QK RMSNorm 的直接启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 99 之后，QK RMSNorm 的一次启动是 `224` bytes。scale 的 reshape
只留下最先看到的两份，其余每次现做。

参数留在具体盒子里，直接调用 `cuLaunchKernel`。每层换上当前的 Q、K 和
scale。scale 现在按 objectid 留住最多 96 份。四组 scale、两组输入的
query、key 和 `mean_square` 与两条单张量路径都是 0 个不一致。编译后的
一次调用是 `0` bytes。

28 层的启动少 `224` bytes，合计 `6,272`。原先没留下的 54 份 scale
reshape 各是 `96` bytes，合计 `5,184`。两笔加起来是 `11,456`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_qk_raw.json`，
SHA256 为
`cc01142c0d717ac61f2e769e90e88ff6f0a6c41a34311fdd33833c0e8ef1e84d`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `211,776`。
256-token 稳态一步从 Chapter 99 的 `223,232` 降到 `211,776`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
