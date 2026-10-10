# Chapter 105 — Qwen3-VL decode 单 token 层走热路径

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 104 之后，一层里剩下的分配来自每段 stage 的 `if/else`。
返回类型合在一起，结果被装箱。

没有 profiler、scratch 都在、而且这一步只有一个 BF16 token 时，
decoder 层改走一条只有热路径的函数。顺序仍是 hidden norm、QKV、
QK norm、RoPE、K/V 写入、attention、attention 输出 residual、post norm、
gate/up、SiLU、down residual。这些调用和改前热路径是同一批函数。
attention 输出到 `(hidden, 1, 1)` 的 reshape 按 scratch 复用。
有 profiler 时仍走原来的 stage 函数。

专用一层的 hidden、K、V 与原来的 stage 函数逐位一致。这一层从
`2,528` bytes 降到 `0`。整步下降是 `107,520` bytes，不按
`2,528 × 28` 记。

一次 repeat 的关闭采样在
`.../chapter47-long-oracle/qwen3_vl_allocation_decode_block.json`，
SHA256 为
`414590cfcf94ced3dd4d15f83cdd5a392e8bd584d07e2d4f52d6dc516c894135`。
`closed=false`，`acceptance_fixture=false`。position `107 / 203 / 331`
的最后一步 GPU 仍是 `0` bytes、`0` 次分配。CPU 分别是 `22,096`、
`22,096` 和 `22,080`。256-token 稳态一步从 Chapter 104 的 `129,600`
降到 `22,080`。第一次 decode 的 GPU 分配仍是 `2,018,180` bytes、`56` 次。
前四枚是 `[1987, 2169, 375, 265]`。32-token 没有分叉。128/256 仍在
step 70 分叉：LifeAI zero-based `21239`，HF `64692`，margin `0.02797699`。

profiler 采样仍走原来的 stage 函数，最大阶段仍是 mlp_gate_up。
Episode 10 保持 Open。最终 logits 没有升成 Float32。
