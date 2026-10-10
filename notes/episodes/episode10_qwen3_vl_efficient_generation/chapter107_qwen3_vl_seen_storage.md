# Chapter 107 — Qwen3-VL decode 复用 cache 重叠检查

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

每一步 decode 都会检查各层 K/V 是否重叠。检查本身要建一份
56 个 storage 的数组，开头和结尾各建一次。

这份数组现在留在 cache 上。下一次检查先清空再填入同样的 storage。
重叠时仍然失败。28 层函数改为直接读取 parameters 和 cache，
层内计算没有变。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_seen_storage.json`，
SHA256 为
`0e9e3ec5332667677e006ddda68de743ff39071effd1bdcdae4624ad7eae3273`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `13,360`、
`13,360` 和 `13,344`。256-token 稳态一步从 Chapter 106 的 `16,768`
降到 `13,344`。整步下降是 `3,424` bytes。第一次 decode 的 GPU 分配仍是
`2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based
`21239`，HF `64692`，margin `0.02797699`。

profiler 采样的最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
