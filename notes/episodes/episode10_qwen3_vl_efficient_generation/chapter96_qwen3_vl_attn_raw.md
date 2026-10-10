# Chapter 96 — Qwen3-VL decode attention 准备和写回的直接启动

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 95 之后，attention 里还剩准备 kernel 和 context 写回。两次启动
合计 `352` bytes。

参数留在具体盒子里，直接调用 `cuLaunchKernel`。每层的 query 和 K/V
parent 会换掉指针。sequence `1 / 2 / 31 / 32 / 33 / 80 / 330 / 331`
的 output、整段 weights、`weight_perm` 和 `maxima` 与原来的路径都是
0 个不一致。另一组 query 和 K/V parent 的 output 与整段 K 也是 0 个
不一致。编译后的 attention 调用是 `0` bytes。28 层少 `352` bytes，
合计 `9,856`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_attn_raw.json`，
SHA256 为
`24cd654c35811cc9375573995687bbfc47fb207237a28d2b9b1d7d8f0b616896`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `250,288`。
256-token 稳态一步从 Chapter 95 的 `260,144` 降到 `250,288`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

这次 profiled 最后一步里，最大阶段仍是 mlp_gate_up。Episode 10 保持 Open。
最终 logits 没有升成 Float32。
