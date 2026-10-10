# Chapter 82 — Qwen3-VL decode K/V 的同一次拷贝

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 81 之后，attention 把 K 和 V 从 cache 布局拷进 scratch 是两次
kernel，单 token 写回 cache 也是两次。下标赋值相同，各是 `288` bytes。

两次都收成一次启动。attention 那次是 `176` bytes，写回也是 `176` bytes。
整段 K 和 V，包括没写到的位置，与原来的两次 kernel 以及 `copyto!` 都是
0 个不一致。CPU 仍是两次 `permutedims!` 和两次 `copyto!`。

一层少两次 `112` bytes。28 层是 `112 × 56 = 6,272`。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_kv_pair.json`，
SHA256 为
`adbfdda29a4aa59a751cd48f5547822b8d21fd947225efdd5b28225a68e9da6c`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `388,496`。
256-token 稳态一步从 Chapter 81 的 `394,768` 降到 `388,496`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
