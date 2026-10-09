# Chapter 68 — Qwen3-VL decode attention softmax 合成 kernel

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-09
>
> 关闭：2026-10-09

## 结果

Chapter 67 之后，attention 里 scale、exp 和 weight 仍是几条广播。试过把这三步收成
kernel，`maximum!` 和 `sum!` 的归约留在原处。sequence `1`、`80`、`330`、`331`
上，输出和 CPU 广播路径相比，`2048` 个元素全部不同。GPU 字节是 `0`，但数值
不是同一条 BF16 路径。

因此代码仍使用原来的广播。没有接进整模型 profile，也没有改 step 70 的分叉。

Episode 10 保持 Open。最终 logits 没有升成 Float32。
