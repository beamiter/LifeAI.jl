# Chapter 23 — Qwen3 XLA 设备端采样

> 状态：Closed（2026-08-01）

## Open：核心问题

Chapter 16 已经把 greedy 的 argmax 放进 executable，宿主每 token 只取回一个
整数；但 `:sample` 至今仍然每步把整条 151,936 维 logits 传回宿主再做
temperature / top-k / top-p。官方 Qwen3 默认就是采样
（`do_sample=true`、`temperature=0.6`、`top_k=20`、`top_p=0.95`），
所以日常 profile 目前只能用 greedy。

能否把整个采样策略 lower 进同一个编译好的 decode executable，使采样路径
与 greedy 一样每 token 只跨越一个整数，并且在给定同一串 uniform 时与已被
HF 验证过的宿主策略 **逐 token 相同**？

## 预期结果

本阶段 Close 时，应当可以展示或验证：

1. 一个设备可 trace 的采样策略：temperature / top-k / top-p / inverse-CDF
   全部在 executable 内完成，宿主每步只传入一个 uniform、取回一个 token。
2. 该策略在普通 Array 上与 Chapter 09 已验证的宿主策略
   （`_sampling_distribution` + `_sample_categorical`）逐 token 一致，
   差异边界（浮点、并列）被显式刻画而不是掩盖。
3. 真实 Qwen3-0.6B 在 CUDA XLA 上：同一 uniform 序列下
   `:sample` 与 `:device_sample` 生成完全相同的 token 序列，且后者吞吐
   显著高于前者。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 设备可 trace 采样策略 | 模型 / 工程 | `_device_sample_next_index` 与 fail-closed 校验 | 离线随机 logits 对拍宿主策略 | 完成 |
| XLA prefill/decode 采样 kernel | 工程 | packed sample step，uniform 作为设备输入 | Reactant CPU 编译 + 数值一致 | 完成 |
| session `:device_sample` 策略 | 工程 | 静态 top-k、traced temperature/top-p、uniform replay | 单元测试 + 真实权重脚本 | 完成 |
| 真实硬件验收 | 学习 / 工程 | 0.6B CUDA XLA token 一致 + 吞吐对比 JSON | 冻结原始指标 | 完成 |

## Close 条件

- 设备策略与宿主策略在固定 uniform 下逐 token 对拍通过；不一致的情形
  （例如第 k 名精确并列）被显式测试钉死并写清语义差异。
- temperature ≤ 0、top_k ≤ 0、top_p ∉ (0, 1]、uniform ∉ [0, 1)、
  top_k 与 session 编译常量不符等全部 fail closed。
- Reactant 上编译出的采样 executable 与同输入的宿主参考实现给出同一 token。
- 真实 0.6B 在 CUDA XLA 上完成同 uniform 序列的 `:sample` /
  `:device_sample` token 一致性与吞吐对比，原始 JSON 进入仓库记录。
- Chapter 23 专项与默认完整测试通过。

## 学习重点

- 要理解的概念：为什么宿主往返而不是算力决定了采样解码的吞吐；
  inverse-CDF 采样为什么必须固定遍历顺序才能复现同一 token。
- 要亲手实现的关键组件：无排序的 top-k 提取（k 次 masked findmax）、
  nucleus 保留规则的等价改写（保留 t ⟺ 严格排在 t 之前的质量 < top_p）、
  设备端 inverse-CDF 的无分支选择。
- 要验证的假设：采样策略可以在不引入设备端 RNG 的前提下完全设备化——
  宿主只提供一个 uniform，随机性语义、replay 能力和既有测试全部不变。

## 风险与取舍

- 本 Chapter 不做设备端 RNG（counter-based）、不做 batch > 1 采样、
  不改 HTTP 服务的默认策略。
- 设备策略以 **恰好 k 个候选、并列取最小 index** 为约定；宿主 / HF 的
  `scores .< threshold` 会保留与第 k 名并列的全部 token。BF16 logits 上
  并列并非不可能，必须实测这一差异的实际影响而不是假设它不存在。
- 本机是 RTX 5080（16 GiB），不是 notes 中 4090 D 的沙箱环境；真实验收
  以 0.6B 为准。

## 实验与过程记录

- 2026-08-01：Chapter 23 Open。基线默认测试 `5,582 / 5,582` 通过。
- 实现选择：**不排序**。整词表排序（151,936）每 token 一次代价不可忽略，
  改成 `top_k` 次「max + 取并列中最大 index + mask」提取，只用 reduction
  和 elementwise select。nucleus 改写为「严格排在候选之前的质量 < top_p」，
  与宿主的「升序 cumulative ≤ 1 − top_p 前缀删除」等价；最终 inverse-CDF
  只在 ≤ top_k 个候选上按 **词表 index 顺序** 走，因为其余 token 概率为 0，
  不影响宿主实现里的顺序遍历结果。
- **并列方向是真 bug，不是理论问题**：第一版用 `findmax`（并列取最小
  index），3,000 组随机 logits 对拍宿主策略有 86 组不一致。HuggingFace 与
  宿主用稳定升序排序再反读，并列时 **较大 index 排在前面**，改成
  `maximum(ifelse.(work .== value, index_column, 0))` 后降到 36 组，
  其中 35 组是「第 k 名分数存在精确并列」这一已知契约差异，
  另 1 组是把 logits 量化到 0.25 网格后制造出的大量并列所致。
  在没有精确并列的 1,000+ 组随机 logits 上 **零不一致**（已写成测试）。
- 契约差异（显式保留、已测试钉死）：宿主 / HF 用
  `scores .< threshold` 保留与第 k 名并列的全部 token，设备策略保留恰好
  `top_k` 个。测试 `tie_at_k = [3, 3, 3, -20]`、`top_k = 2` 下宿主可达
  `{1, 2, 3}`，设备只能达 `{2, 3}`，且设备结果集始终是宿主结果集的子集。
- 随机性仍然留在宿主：每步宿主只送 4 字节 uniform、取回一个整数，
  RNG 语义、seed 与 replay 能力与既有 `:sample` 完全一致，因此可以用
  同一串 uniform 对拍两条路径。设备端 RNG 不在本 Chapter 范围。
- `top_k` 是编译期常量（决定提取轮数），temperature 与 top_p 是运行期
  设备标量，改采样温度或 nucleus 不需要重新编译；`top_k` 与编译常量不符
  时 fail closed。
- 真实权重验收（本机 RTX 5080 16 GiB、driver `595.71.05`、
  CUDA runtime `13.1.0`、Reactant CUDA backend，Qwen3-0.6B revision
  `c1899de289a04d12100db370d81485cdf75e47ca`）：29-token chat prompt、
  官方 `temperature=0.6 / top_k=20 / top_p=0.95`、同一串 48 个 uniform
  replay，两条路径都在第 38 个 token 命中 EOS，
  **38 / 38 token 完全一致，无分歧步**。
  decode 吞吐 `237.23` vs `23.66` tok/s，**10.03×**；
  prefill `0.0061` vs `0.0444` s。设备采样已经贴近 Chapter 16 冻结的
  greedy 246 tok/s，即采样不再比 greedy 慢一个量级。
  编译代价上升：prefill `74.4` s、decode `21.9` s
  （宿主采样路径为 `15.3` / `14.9` s）。
  两条路径各自 warm/steady 两次 replay 结果逐 token 相同。
- 结论修正了一个直觉：`:sample` 慢十倍不是采样算法贵，而是每 token 传回
  151,936 维 logits 贵。策略本身（20 轮 reduction + O(k²) 标量）在设备上
  几乎免费。
- Chapter 23 离线专项 `81 / 81`，加 Reactant CPU 编译对拍 `91 / 91`；
  默认完整测试 `5,663 / 5,663`（Chapter 22 基线为 `5,582`）。

复现验收：

```bash
julia --project=. --startup-file=no \
  scripts/verify_qwen3_xla_device_sampling.jl \
  /home/yj/models/huggingface/Qwen/Qwen3-0.6B/c1899de289a04d12100db370d81485cdf75e47ca \
  benchmark_results/week23/qwen3_0_6b_xla_device_sampling.json

# 可选覆盖：LIFEAI_WEEK23_BACKEND / LIFEAI_WEEK23_CONTEXT / LIFEAI_WEEK23_TOKENS
```

冻结证据：

```text
benchmark_results/week23/qwen3_0_6b_xla_device_sampling.json
SHA256 f05a9d62d6eacc8a9b59746ccd2e1dd068aaa60040d80c2ebf54ef0dfd83c1f6
```

## Close 回顾

- **完成了什么**：新增设备可 trace 的 temperature / top-k / top-p /
  inverse-CDF 采样策略（`src/generation/xla_sampling.jl`）、XLA
  prefill/decode 采样 kernel、session 的 `:device_sample` 策略与
  `sample_uniforms` replay，以及真实权重验收脚本。采样解码的宿主流量从
  每 token 151,936 个 BF16 降到 4 字节进 / 8 字节出。
- **验证证据**：无精确并列的 1,000+ 组随机 logits 上与宿主策略零不一致；
  4,000 次固定分布抽样的经验分布与解析概率 max-abs < 0.02；Reactant 编译
  产物与宿主参考逐 uniform 相同；真实 0.6B CUDA XLA 同 uniform replay
  38/38 token 一致、10.03× 吞吐。
- **没有完成及原因**：设备端 RNG、batch > 1 采样、把 `:device_sample`
  设为 HTTP 服务默认策略、8B 上的同类验收都不在本 Chapter 范围（8B BF16
  常驻超出本机 5080 的 16 GiB）。
- **最重要的认知变化**：并列（tie）的方向是可执行的语义，不是理论细节。
  `findmax` 取最小 index 看起来无害，实测却让 3,000 组随机 logits 中
  86 组选出不同 token；HuggingFace 的稳定升序排序意味着并列时较大 index
  排前。另一个认知是「把策略搬上设备」不需要把随机数也搬上设备——宿主
  每步送一个 uniform，既保留 RNG 语义与 replay，又拿到全部吞吐收益。
- **是否满足 Close 条件**：是。验收报告顶层 `closed=true`。
- **带到下一 Chapter 的问题**：`:device_sample` 是否直接成为 XLA 常驻服务的
  默认策略（需要 8B 上的同类验收与 EOS/停止语义复核）；还是先做设备端
  counter-based RNG，把最后 4 字节的宿主输入也去掉？
