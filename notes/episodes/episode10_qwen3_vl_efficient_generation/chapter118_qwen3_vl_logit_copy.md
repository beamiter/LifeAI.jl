# Chapter 118 — Qwen3-VL 主机 logits 复用一块缓冲

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

`Array` 每次选择都新分配一条 vocab 向量。BF16 logits 是连续的
`(vocab, 1, 1)`，所以留一块主机缓冲，用同步的 `cuMemcpyDtoH` 抄进去。
异步拷贝在真实 logits 上会让驱动崩掉，所以没有用。数值和 `Array` 一致。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_logit_copy.json`，
SHA256 为
`1a5ee12eadee76b467864460e202d2a371532eb5637337e1c06056abc882722c`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 仍是 `0`。
warmup 的第一次主机选择是 `303,944` bytes，把缓冲建出来。之后的
选择是 `0`。256-token 样本的主机选择从 Chapter 117 的 `308,232`
降到 `0`。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

缓冲只建一次，不声称整条 decode 零分配。profiler 采样的最大阶段
仍是 mlp_gate_up。Episode 10 保持 Open。最终 logits 没有升成 Float32。
