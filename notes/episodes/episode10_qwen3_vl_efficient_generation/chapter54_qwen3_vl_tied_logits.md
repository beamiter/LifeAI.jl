# Chapter 54 — Qwen3-VL decode tied logits 常量分配

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-08
>
> 关闭：2026-10-08

## 结果

`vocab_logits` 是一次 tied GEMM：`transpose(embedding) * hidden`，输出
`(151936, 1, 1)` BF16，`303,872` bytes。加上 `mul!` 的 `8` bytes，就是剖面里的
`303,880` bytes、`3` 次分配。

BF16 单 token decode 现在把这块输出留在 cache 上。返回的 logits 与这块 buffer
是同一个数组，下一次 decode 会覆盖它。调用方在下一次 decode 之前完成读取。
GPU `mul!` 与原来的投影逐位一致。CPU 把原来的投影抄进同一块 buffer。

驻留 scratch 是 `303,872` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_tied_logits.json`，SHA256 为
`5123640b5cbd751eb5c7c794566b5c40027f9f7e0ae21b7654775568e7a0e429`。
position `107 / 203 / 331` 的最后一步都是 `243,393` bytes、`1,492` 次分配。
`vocab_logits` 降到 `8` bytes、`2` 次分配。相对 Chapter 53 的 `547,265 /
1,493`，正好少了 `303,872` bytes 和 `1` 次分配。前四枚是
`[1987, 2169, 375, 265]`。128/256 仍在 step 70 分叉，margin `0.02797699`。

下一问是 `qkv_projection`，`230,048` bytes、`252` 次分配、`28` 次调用。
Episode 10 保持 Open。最终 logits 没有升成 Float32。
