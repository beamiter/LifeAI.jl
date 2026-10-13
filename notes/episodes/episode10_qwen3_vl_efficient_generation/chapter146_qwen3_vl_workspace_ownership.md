# Chapter 146 — Qwen3-VL CUDA 工作区的请求所有权

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-13
>
> 关闭：2026-10-13

## 问题

Chapter 145 修复了参数绑定，但相同参数的两个请求仍可能共享全局 CUDA
shells 和 Ready。最小 GPU 复现只分配工作区：先创建 capacity=8 的 cache，
再创建 capacity=16 的 cache。第二次创建后，两者的 Ready 是同一对象；
第一个 cache 的 attention capacity 变成 16，K scratch 从 `(8, 8, 1)`
变成 `(8, 16, 1)`，指针发生变化。第二个请求写入 embedding scratch，也会
覆盖第一个请求的内容。六项独立性断言全部失败。

这不是并发专属问题。顺序交替使用两个仍然存活的 cache，也会破坏请求隔离。

## 修改

CUDA shells 用弱引用记录所属 cache。三个复用入口只有在 owner 就是当前
cache、参数绑定也一致时才允许复用。全局 Ready 还必须等于当前 cache
已经安装的 Ready。

owner 被 GC 回收也不授权其他请求复用这组可变数组。调用者可能仍保留返回的
logits，而 logits 本身就是 scratch；仅检查 cache 是否存活不能证明数组
已无人使用。同一 cache 的 decode/reset 继续复用原来的分配，新 cache
则取得独立工作区。

长生成报告同时补记 `ext/LifeAICUDAExt.jl` 和
`src/models/qwen3_vl_attention_scratch.jl` 的 SHA256，避免关键 CUDA 实现
遗漏在源码证据之外。

## 验证

RTX 4090 D 上，新增工作区回归 `26/26` 通过，覆盖两个 live cache 的所有
scratch 数组、指针、shape、capacity、内容独立性，同 cache reset，以及
cache 回收后仍保留的输出。测试纳入 `LIFEAI_TEST_CUDA=true` 的扩展集合。

修复后参数绑定 CUDA 回归仍为 `456/456`。真实 2B 的 32/128/256-token
BF16 独立进程复测，各一次 repeat，所有生成 token 和逐步 logits SHA256
与 Chapter 145 版本一致。32-token 与 HF oracle exact，128/256 保持原来的
step 70 分叉。两次报告都明确 `acceptance_fixture=false`。

| 生成长度 | 首次 decode CPU bytes | 首次 decode GPU bytes | GPU 分配次数 | 后续每步 CPU/GPU bytes |
| --- | ---: | ---: | ---: | --- |
| 32 | 10,528 | 1,043,332 | 2 | 0 / 0 |
| 128 | 10,528 | 1,461,124 | 2 | 0 / 0 |
| 256 | 10,528 | 2,018,180 | 2 | 0 / 0 |

这两次 GPU 分配是当前请求独立的 BF16/Float32 父缓冲。不同 cache 不能再
借用上一请求仍然可见的数组，因此不再沿用此前跨请求首次 decode 零分配的
结论。同一 cache 的 reset 复用保留。

修复后原始报告 SHA256 为
`0ff9ed9b4d4714c13cd45775a478ac214eb15b48415a5537b459176f2ba559ef`。
[比较摘要与源码哈希](../../../benchmark_results/qwen3_vl_workspace_ownership/summary.json)
保留了两次运行的分配、token/logits 哈希和冻结 oracle 关联；原始报告保存在
同目录，未作为验收 fixture 提交。Episode 10 保持 Open。
