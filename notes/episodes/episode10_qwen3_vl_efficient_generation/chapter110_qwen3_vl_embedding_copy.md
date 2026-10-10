# Chapter 110 — Qwen3-VL decode token embedding 直接拷贝

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

单 token embedding 原来用广播把 embedding 的一列写进 scratch。
这一列是连续的。现在用一次 kernel 按元素抄过去。CPU 数组仍走原来的广播。

抄写结果和广播的那一列逐位一致。换一个 token 也一致。单独调用是
`0` bytes。整步下降是 `1,856` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_embedding_copy.json`，
SHA256 为
`ddd117e0443de28401032c581f353bfcdf8f983894b3a2a42ec2d01a5bb3655b`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `5,360`、
`5,360` 和 `5,344`。256-token 稳态一步从 Chapter 109 的 `7,200`
降到 `5,344`。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
