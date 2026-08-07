# Chapter 18 — Qwen3 Activation-Aware INT4 校准

> 状态：Closed
>
> 开启记录：2026-07-29
>
> 关闭记录：2026-07-29
>
> 依赖基线：[`Chapter 17 — Qwen3 Reconstruction-Calibrated INT4 与预算化混合精度`](../../episodes/episode04_efficient_inference_and_quantization/chapter17_qwen3_calibrated_int4.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Chapter 17 证明 weight-reconstruction MSE 即使降低同布局的
> full-logits 全局误差，也可能把 14B greedy 从 RTN 的 16/16 降到
> 4/16。本周不继续调纯权重代理，而引入与评测 prompt 分离的校准 token，
> 用真实线性层输入激活的二阶矩加权 INT4 clipping error。

## 核心问题

> 在不改变 Chapter 17 权重格式、混合精度计划和 BF16 计算契约的前提下，
> 能否用校准输入的 per-channel activation second moment 构造可复现、
> fail-closed 的 diagonal activation-aware INT4 scale search，并在冻结
> Qwen3-14B / RTX 4090 D 上判断它能否保留 mixed RTN 的 16/16 greedy？

## 设计决策

- **只声称 diagonal activation-aware，不冒充 AWQ/GPTQ**：目标函数为
  `Σ E[x_j²] · (w_j - q_j·s)²`，只使用输入二阶矩，不使用完整 Hessian、
  block reconstruction、权重重排或端到端 loss。
- **校准集与评测 prompt 分离**：冻结独立的多语种/代码/数学 token
  matrix、tokenizer revision 与 checksum；Chapter 17 的 8-token parity
  prompt 不进入校准序列，避免把 prompt-specific tuning 写成泛化。
- **量化格式与预算不变**：仍输出 Chapter 16/17 的 packed groupwise INT4，
  activation stats 只影响 scale candidate 选择，不增加部署参数 bytes。
- **计划仍是唯一策略源**：`:activation_mse` 作为 INT4 calibration
  mode；streamed loader 与 BF16 参数树量化接收同一
  `ActivationCalibration`，缺失 target、维度错配、NaN/负权重全部
  fail closed。
- **真实行为决定结论**：以 Chapter 17 同布局 mixed RTN 16/16、mixed
  weight-MSE 4/16 为固定对照；即使 activation-weighted reconstruction
  局部改善，只要 greedy 没改善，就按负结果关闭。

## 实现范围

- `ActivationCalibration` 与 `activation_second_moment` 公共契约。
- `calibrate_hf_qwen3_activations`：以 native BF16 语义逐层流式读取
  Qwen3 权重，采集 Q/K/V/O、gate/up/down 与 untied LM head 的输入
  second moments，不常驻完整 BF16 参数树。
- `LinearQuantizationSpec(...; calibration=:activation_mse)`：在冻结
  clipping candidates 内最小化 activation-weighted reconstruction error。
- `load_hf_qwen3_quantized` / `quantize_bf16_parameters` 共用校准统计；
  max-abs 与 Chapter 17 weight-MSE 默认行为零回归。
- 独立校准 token fixture、计划 JSON、真实 GPU 验证脚本输出与硬件
  asset contract。

## Close 条件

- synthetic fixture 上每个 row/group 的 activation-weighted error 不高于
  `1.0` max-abs candidate，并至少有一个 activation-skewed case 严格改善。
- 校准统计的 target、layer、input dimension、finite/non-negative 与
  token count 全部严格校验；缺失 activation stats 不允许静默回退 RTN。
- BF16 参数树量化与 safetensors streamed loader 对同一 activation plan
  产生逐 tensor 相同结果；Chapter 16/17 legacy/maxabs/MSE 行为不回归。
- 校准 token 与 Chapter 17 evaluation prompt 的 provenance/checksum
  冻结；默认离线套件全部通过。
- Qwen3-14B 在 RTX 4090 D 上完成同 mixed layout 的 activation-aware
  实测，记录 calibration/load、tensor/VRAM、logits/decode/greedy；
  结果无论正负都与 Chapter 17 RTN/MSE 同口径比较。

## 非目标

- 不实现或宣称完整 AWQ、GPTQ、SmoothQuant 或 Hessian/blockwise PTQ。
- 不实现 fused/dequant-on-the-fly INT4 GEMM；吞吐优化继续独立排期。
- 不把评测 prompt 用作校准集，不以 full-logits error 代替 greedy。
- 不提交模型权重、reference 或大体积 activation stats。

## 过程记录

### 2026-07-29：Open

- Chapter 17 保持 Closed；mixed RTN 16/16 是本周必须守住的行为 baseline，
  weight-MSE 4/16 是必须解释而不能覆盖的负证据。
- 冻结实现顺序：校准统计契约 → BF16 流式采集 → activation-weighted
  scale search → 两 loader 一致性 → 独立校准 token → 14B GPU 对照。

### 2026-07-29：实现与离线验证

- 新增 `ActivationCalibration`、`activation_second_moment` 与
  `calibrate_hf_qwen3_activations`；每层 Q/K/V 共享 attention norm
  输入统计，O、gate/up、down 分别采集真实线性输入，untied LM head
  采集 final norm 后输入。
- 校准器仍逐层读取 checkpoint，不常驻完整 BF16 权重树；CPU 路径保持
  Chapter 14 位精确语义，加速路径通过 `to_device` / `to_host` callback
  使用 Chapter 15 device-generic BF16 算子，因此核心包不依赖 CUDA。
- `:activation_mse` 只改变每个 INT4 row/group 的 clipping candidate
  选择；packed bytes、scale shape、反量化与 BF16 compute 全部不变。
- 冻结 8×32 多语种/代码/数学 token fixture：
  SHA256 `0c61c44e8dcebf0eaea75ff320ebdf9f9ab377f20276f5430abf54c654cbb6f2`；
  tokenizer 为 Transformers 4.51.0 `Qwen2TokenizerFast`，模型 revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18`，明确验证 Chapter 17 evaluation
  token sequence 不在选中语料中。
- synthetic case 中 activation-weighted error 严格低于 max-abs 与
  weight-MSE；所有 row/group 因 candidate 含 `1.0` 而局部不劣于 RTN。
  缺失 target、错层、错维度、NaN、负值、零质量 group 和错误 plan
  usage 全部 fail closed。
- 合成 Qwen3 tied/untied fixture 上 CPU loop/CPU accelerated 统计逐值一致，
  BF16 参数树量化与 safetensors streamed loader 逐 tensor 相同；
  Chapter 14—18 回归全部通过。

### 2026-07-29：Qwen3-14B / RTX 4090 D

冻结环境与 Chapter 17 相同：RTX 4090 D、driver 570.153.02、CUDA.jl 6.2.1；
本地 ModelScope 权重逐文件匹配 HuggingFace revision，复用冻结的
Transformers 4.51.0 / Torch 2.7.1+cpu BF16 reference。

| 指标 | mixed RTN（Chapter 17） | mixed weight-MSE（Chapter 17） | mixed activation-MSE |
| --- | ---: | ---: | ---: |
| tensor tree | 12.093 GiB | 12.093 GiB | 12.093 GiB |
| calibration | — | weight-only | 256 tokens / 248.32 s |
| load | 259.60 s | 273.59 s | 489.23 s |
| VRAM used | 22.597 GiB | 22.521 GiB | 21.474 GiB |
| logits max abs | 4.3203 | 3.6289 | **3.4238** |
| logits mean abs | 0.42933 | **0.36034** | 0.42865 |
| decode max abs | 1.125 | 1.0156 | 1.125 |
| greedy agreement | **16/16** | 4/16 | 4/16 |
| first divergence | — | 5 | 5 |

VRAM 是 allocator 状态相关的单进程观测，不把小幅差异解释为参数格式变化；
三组精确 tensor bytes 相同。activation-MSE 的 max error 最低，但 mean
几乎回到 RTN，generation 与 weight-MSE 一样从第 5 token 分歧。这个结果
否定了“只需把 weight MSE 换成 diagonal activation-weighted MSE 就能恢复
greedy”的本周假设。

## Close 回顾

- **完成了什么**：独立校准语料、严格统计契约、CPU/GPU 逐层采集、
  activation-weighted INT4 scale search、两种 loader 一致性和真实 14B
  三方对照全部完成。
- **验证证据**：默认套件 `5086 / 5086`、Chapter 18 专项 `133 / 133`；
  校准 token、plan、generator、模型 revision、BF16 reference 与硬件指标
  均冻结 checksum。4090 D 真实结果为 logits max/mean
  `3.4238 / 0.42865`、greedy 4/16。
- **没有完成及原因**：没有实现完整 AWQ/GPTQ/Hessian/block
  reconstruction，也没有 fused INT4 GEMM；它们从 Open 起就是非目标，
  不能借 diagonal 统计的实现宣称已完成。
- **最重要的认知变化**：activation 信息本身不够，目标函数、跨层误差
  传播和离散 argmax margin 才是生成保真的关键；降低某个重建代理甚至
  full-logits max error，仍可能完全不改变错误的自回归分支。
- **是否满足 Close 条件**：满足。实现与 fail-closed 条件完成，legacy
  行为无回归，真实 GPU 对照完成；质量假设被证伪但 Close 条件明确允许
  正负结果。
- **带到下一 Chapter 的问题**：若继续量化，优先选择“直接约束层输出/生成
  margin 的 blockwise 方法”或 fused quantized GEMM，而不是继续微调
  diagonal clipping ratios；是否投入完整 AWQ/GPTQ 应与吞吐目标一起决策。
