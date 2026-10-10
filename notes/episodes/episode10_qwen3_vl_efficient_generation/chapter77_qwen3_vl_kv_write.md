# Chapter 77 — Qwen3-VL decode 单 token K/V 写入

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 76 之后，K/V 写入紧挨着 attention。单 token 的一次 `copyto!` 是
`1,568` bytes。decode 每次写 K 和 V 各一块，形状是
`(head_dim, kv_heads, 1, 1)` 进 capacity 上的一个位置。

GPU 的单 token 写入改成两次拷贝 kernel：

`dest[lane, head, index, 1] = src[lane, head, 1, 1]`

多 token 的 prefill 仍走 `copyto!`。CPU 也仍走 `copyto!`。

RTX 4090 D 上，index `1 / 80 / 331` 的整段 K 和 V，包括没写到的位置，与
`copyto!` 都是 0 个不一致。两次 kernel 一起是 `288` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_kv_write.json`，
SHA256 为
`20917f6fd9f56e3f2cfa8752c14f857d7d72ad99268c35ac6dc16a17347fc792`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 是 `441,024`、`441,072`、
`441,024`。256-token 稳态一步从 Chapter 76 的 `513,152` 降到 `441,024`。
`kv_write` 从 `110,976` 降到 `34,080`。第一次 decode 的 GPU 分配仍是
`2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。32-token
没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，HF `64692`，
margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
