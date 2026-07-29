# Week 18 — Qwen3 Activation-Aware INT4 校准

> 状态：Open
>
> 开启记录：2026-07-29
>
> 依赖基线：[`Week 17 — Qwen3 Reconstruction-Calibrated INT4 与预算化混合精度`](week17_qwen3_calibrated_int4.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Week 17 证明 weight-reconstruction MSE 即使降低同布局的
> full-logits 全局误差，也可能把 14B greedy 从 RTN 的 16/16 降到
> 4/16。本周不继续调纯权重代理，而引入与评测 prompt 分离的校准 token，
> 用真实线性层输入激活的二阶矩加权 INT4 clipping error。

## 核心问题

> 在不改变 Week 17 权重格式、混合精度计划和 BF16 计算契约的前提下，
> 能否用校准输入的 per-channel activation second moment 构造可复现、
> fail-closed 的 diagonal activation-aware INT4 scale search，并在冻结
> Qwen3-14B / RTX 4090 D 上判断它能否保留 mixed RTN 的 16/16 greedy？

## 设计决策

- **只声称 diagonal activation-aware，不冒充 AWQ/GPTQ**：目标函数为
  `Σ E[x_j²] · (w_j - q_j·s)²`，只使用输入二阶矩，不使用完整 Hessian、
  block reconstruction、权重重排或端到端 loss。
- **校准集与评测 prompt 分离**：冻结独立的多语种/代码/数学 token
  matrix、tokenizer revision 与 checksum；Week 17 的 8-token parity
  prompt 不进入校准序列，避免把 prompt-specific tuning 写成泛化。
- **量化格式与预算不变**：仍输出 Week 16/17 的 packed groupwise INT4，
  activation stats 只影响 scale candidate 选择，不增加部署参数 bytes。
- **计划仍是唯一策略源**：`:activation_mse` 作为 INT4 calibration
  mode；streamed loader 与 BF16 参数树量化接收同一
  `ActivationCalibration`，缺失 target、维度错配、NaN/负权重全部
  fail closed。
- **真实行为决定结论**：以 Week 17 同布局 mixed RTN 16/16、mixed
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
  max-abs 与 Week 17 weight-MSE 默认行为零回归。
- 独立校准 token fixture、计划 JSON、真实 GPU 验证脚本输出与硬件
  asset contract。

## Close 条件

- synthetic fixture 上每个 row/group 的 activation-weighted error 不高于
  `1.0` max-abs candidate，并至少有一个 activation-skewed case 严格改善。
- 校准统计的 target、layer、input dimension、finite/non-negative 与
  token count 全部严格校验；缺失 activation stats 不允许静默回退 RTN。
- BF16 参数树量化与 safetensors streamed loader 对同一 activation plan
  产生逐 tensor 相同结果；Week 16/17 legacy/maxabs/MSE 行为不回归。
- 校准 token 与 Week 17 evaluation prompt 的 provenance/checksum
  冻结；默认离线套件全部通过。
- Qwen3-14B 在 RTX 4090 D 上完成同 mixed layout 的 activation-aware
  实测，记录 calibration/load、tensor/VRAM、logits/decode/greedy；
  结果无论正负都与 Week 17 RTN/MSE 同口径比较。

## 非目标

- 不实现或宣称完整 AWQ、GPTQ、SmoothQuant 或 Hessian/blockwise PTQ。
- 不实现 fused/dequant-on-the-fly INT4 GEMM；吞吐优化继续独立排期。
- 不把评测 prompt 用作校准集，不以 full-logits error 代替 greedy。
- 不提交模型权重、reference 或大体积 activation stats。

## 过程记录

### 2026-07-29：Open

- Week 17 保持 Closed；mixed RTN 16/16 是本周必须守住的行为 baseline，
  weight-MSE 4/16 是必须解释而不能覆盖的负证据。
- 冻结实现顺序：校准统计契约 → BF16 流式采集 → activation-weighted
  scale search → 两 loader 一致性 → 独立校准 token → 14B GPU 对照。

## Close 回顾

- **完成了什么**：
- **验证证据**：
- **没有完成及原因**：
- **最重要的认知变化**：
- **是否满足 Close 条件**：
- **带到下一 Week 的问题**：
