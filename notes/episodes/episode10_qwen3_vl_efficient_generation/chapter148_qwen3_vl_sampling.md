# Chapter 148 — Qwen3-VL 可重放采样生成

> 所属 Episode：Episode 10 — Qwen3-VL 高效生成
>
> 状态：Closed
>
> 开启：2026-10-15
>
> 关闭：2026-10-15

## 接口与行为

Chapter 147 已完成高层生成接口的显式 static cache 复用。本章为
`generate_hf_qwen3_vl_tokens` 与 raw 图文入口 `generate_hf_qwen3_vl` 加入
`strategy=:sample`。默认仍为 `:greedy`，既有 token、logits 和 cache 行为保持。

采样顺序是 temperature 缩放、稳定 top-k、top-p nucleus，再按词表顺序累积
概率采样。完全相同的 logit 以原 token index 排序。未指定采样参数时使用
`temperature=1.0f0`、`top_k=vocab_size`、`top_p=1.0f0`。

```julia
using Random: MersenneTwister

state = init_qwen3_vl_static_kv_cache(parameters; capacity=512)
result = generate_hf_qwen3_vl_tokens(
    parameters, input_ids, rope_layout;
    vision_features,
    cache=state,
    strategy=:sample,
    temperature=0.7,
    top_k=20,
    top_p=0.8,
    rng=MersenneTwister(148),
    max_new_tokens=32,
    capture_distribution=true,
)

# 用记录下来的 uniform 在同一 cache 上重放同一请求。
replay = generate_hf_qwen3_vl_tokens(
    parameters, input_ids, rope_layout;
    vision_features,
    cache=state,
    strategy=:sample,
    temperature=0.7,
    top_k=20,
    top_p=0.8,
    sample_uniforms=[step.sample_uniform for step in result.trace],
    max_new_tokens=length(result.generated_ids),
    capture_distribution=true,
)
```

raw 图文入口使用相同关键字，例如：

```julia
result = generate_hf_qwen3_vl(
    vision_parameters, text_parameters, tokenizer, messages;
    cache=state,
    strategy=:sample,
    temperature=0.7,
    top_k=20,
    top_p=0.8,
    rng=MersenneTwister(148),
    max_new_tokens=32,
    capture_distribution=true,
)
```

`sample_uniforms` 必须至少覆盖 `max_new_tokens`，所有元素（包括未消费的尾部）
都必须为非 Bool、有限且位于 `[0, 1)` 的实数。显式 uniform 保留 Float64
精度，且不消耗 RNG。未提供固定序列时从 RNG 取样；零 token 请求不取样。
trace 记录 `sample_uniform`、`sampled_probability`、`candidate_count` 与实际
采样参数；`capture_distribution=true` 另保存候选 token、过滤后 logits 和概率。
这可能显著增加长生成报告大小，默认关闭。greedy 拒绝采样参数、uniform 与
分布捕获，避免把配置静默忽略。

## 失败边界

配置和全部 uniform 在已有 cache reset、raw 图像处理与 vision 前向之前验证。
temperature 与 top-p 转为 Float32 后必须仍为有限正数；下溢到零也拒绝。
Chapter 147 的完整输入准备与 cache 所有权规则继续适用：失败保留旧请求，
同一个可变 cache 由调用方顺序使用。

EOS 与最大输出长度沿用既有语义：第一枚 token 来自 prefill，后续只把已选
前缀送入 decode；停止 token 会出现在返回的生成序列中。生成最后一枚 token
不会再执行多余 decode，因此 cache position 只包含实际处理过的前缀。

## 已完成验证

CPU 专项 `1345/1345` 通过，测试执行 `39.55` 秒（含编译）：

- 独立离线 oracle 自行实现过滤、softmax 与词表顺序 CDF，不用被测 helper
  生成预期结果；覆盖 Float32/BFloat16、dynamic/static 与显式 cache reuse。
- 默认 greedy 对照、固定 RNG 与 uniform 重放、显式序列不消耗 RNG。
- 相同 logits 的 top-k/top-p tie、`prevfloat(0.5)` 的 Float64 CDF 边界。
- 新增 24 项 raw 图文入口成功路径回归。
- EOS、零 token、cache position，以及配置错误时 cache 内容、工作区与
  bookkeeping 不变；raw 接口的 vision sentinel 不被触发。

## 真实 GPU 验证

RTX PRO 5000 72GB Blackwell 上，真实 Qwen3-VL-2B BF16 使用固定 256 个
Float64 uniform、`temperature=0.8f0`、`top_k=20`、`top_p=0.9f0` 与 capacity
`332`。32/128/256-token 请求分别到达 position `107/203/331`，`rope_delta=-56`。
每档 fresh 与 reused 的 token、逐步 logits、采样分布和有效 K/V 哈希全部一致；
同一 cache/storage 的身份检查通过。独立采样 oracle 全部通过，`top_k=1` 与
greedy 结果相同。测试显式禁用 EOS 停止以覆盖完整长度；长序列中的特殊 token
是诊断轨迹，不作为回答质量证据。

[真实 GPU 报告](../../../benchmark_results/qwen3_vl_sampling/summary.json) SHA256：
`e8243043be75ccea7bd4ce351b98d19790b7a7ff44d54bbd0fe77b64b99bbaca`。
报告冻结模型及 reference 校验和、源码哈希、uniform 序列与各档输出哈希；不声称
跨 HF dtype exact 或性能改善。

## 数值边界与修复

验证中修复了 `top_p=1` 的尾部候选丢失：Float32 累积和可能在尚有很小正概率
候选时提前舍入为 1。现在 `top_p=1` 不运行 nucleus 截断，保留全部 top-k
候选；`top_p<1` 继续沿用稳定降序和最小累积 cutoff。

独立 oracle 用 Float64 计算，生产采样使用 Float32 分布；两者在 cutoff 附近
不能无条件声称逐位相同。因此 verifier 显式计算 top-p 与 categorical CDF 的
边界距离，并要求距离超过 `8 * eps(Float32) * 候选数` 的诊断 guard。
三档轨迹的最小 top-p 距离为 `0.0008668232`，最大 guard 为 `0.00001907349`；
最小 categorical 距离为 `0.0042179220`，最大 guard 为 `0.000008583069`。
这些冻结样本通过了裕量检查；不能外推为所有边界输入都跨精度 exact。

## 集成验收与关闭

2026-10-15 DSW 默认 suite 退出状态为 0，`16,825` 项 assertion 通过，另有 2 项
opt-in 跳过；Chapter 148 的 `1345` 项（包括新增 raw 成功路径 `24` 项）已纳入
该次完整运行。默认 suite 的 opt-in 跳过不替代本章单独完成的真实 GPU 验证。
DSW 是浅克隆，36 项历史 Git 对象比对因对象不可用未执行；注册 snapshot
比对通过。因此这份计数对应当前 DSW 完整默认运行，不能与旧本地 `16,816`
计数混用，也不能解释为历史对象全部完成验证。

CPU 专项、真实 GPU 和默认集成门禁均通过，本章于 2026-10-15 Closed。
Episode 10 保持 Open，本章不以性能改进作为关闭条件。

现有 BF16 自由生成在 step 70 与 HF Float32 分叉是既有跨 dtype 边界；本章
不以跨 dtype token exact 作为采样关闭条件，也不声称改善模型质量或消除该
差异。multi-image、video、padding、长上下文和完整执行图编译仍是后续工作。
