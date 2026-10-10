# Chapter 127 — Qwen3-VL ready 不再拷贝 blocks

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

ready 原来把 pack 里的 blocks 再拷进自己。现在 ready 只留着那个
pack。计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_ready_pack.json`，
SHA256 为
`e5f49acc926b4b205186f93ed9f42c8e3c21583ffc5f1b4c01085e6f6594ed73`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
32-token 的第一次 decode CPU 从 `37,024` 降到 `31,616`。128/256 从
`31,264` 降到 `25,856`。下降都是 `5,408` bytes。256-token 第一次
decode 的 GPU 字节仍是 `2,018,180`，分配次数仍是 `21`。前四枚是
`[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在 step 70
分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

第一次 decode 仍要创建 scratch。不声称整条 decode 零分配。profiler
采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits
没有升成 Float32。
