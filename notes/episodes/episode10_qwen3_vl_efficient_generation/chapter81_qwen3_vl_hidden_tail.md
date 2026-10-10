# Chapter 81 — Qwen3-VL decode hidden abs2 的尾部归约

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 80 之后，hidden RMSNorm 的 `sum!(abs2)` 是 `656` bytes。长度 `2048`
先分成 4 块，尾部 `sum!(identity)` 把 `(1, 1, 1, 4)` 收成 `(1, 1, 1)`，这一步
是 `512` bytes。

尾部在单 warp 内能覆盖时改走和 `sum!` 相同的 warp 树。`n = 1:32` 的结果逐位
一致。更长的尾部仍走 `sum!`。整段 hidden abs2 仍和 `sum!(abs2)` 逐位一致。
Q/K 的单 block 归约没有改。

这次尾部是 `144` bytes。hidden abs2 从 `656` 降到 `288`，一次 hidden
RMSNorm 从 `1,072` 降到 `704`。每步有 28 次 pre、28 次 post 和 1 次 final，
共少 `368 × 57 = 20,976` bytes。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_hidden_tail.json`，
SHA256 为
`73e26bcfc57fea24cd2ba006b1aa51fe2ca5b245bb73f00c32766f6151dd9721`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 都是 `394,768`。
256-token 稳态一步从 Chapter 80 的 `415,744` 降到 `394,768`。第一次 decode
的 GPU 分配仍是 `2,018,180` bytes、`56` 次。前四枚是 `[1987, 2169, 375, 265]`。
32-token 没有分叉。128/256 仍在 step 70 分叉：LifeAI zero-based `21239`，
HF `64692`，margin `0.02797699`。

profiled 最后一步里，最大阶段仍是 attention。Episode 10 保持 Open。最终
logits 没有升成 Float32。
