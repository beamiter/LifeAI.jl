# Week 17 — Qwen3 Reconstruction-Calibrated INT4 与预算化混合精度

> 状态：Open
>
> 开启记录：2026-07-29
>
> 依赖基线：[`Week 16 — Qwen3 XLA BF16 Compiled Decode 与 INT8/INT4 量化`](week16_qwen3_xla_decode_quant.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Week 16 已解决 8B INT8 / 14B INT4 在 16.3 GiB GPU 上的
> 驻留问题，但 plain RTN INT4 的 14B greedy 仅 4/16，且现有
> `int8_projections` 只能按投影名统一覆盖全部层，不能表达 LM head 或
> 层级敏感度。本周先把量化策略从单一全局开关升级为可校准、可组合、
> 可预算的公共能力。

## 核心问题

> 在不改变 BF16 计算契约的前提下，能否用 reconstruction-calibrated
> INT4 scale 与按层/投影的 INT4、INT8、BF16 计划，系统降低量化误差，
> 并在没有大模型资产或 GPU 的环境中仍以确定性离线证据验证算法、
> loader 一致性和存储预算？

## 设计决策

- **先校准权重重建，不伪装成 activation-aware**：本周实现按
  output-channel / input-group 独立选择 clipping ratio 的 MSE scale
  search；没有校准语料和激活统计时，不使用 AWQ/GPTQ 名称。
- **量化计划是唯一策略源**：默认规则、按 projection override、按
  `(layer, projection)` override 采用固定优先级；LM head 作为显式
  target，避免 Week 16 `int8_projections` 无法覆盖它。
- **保持兼容**：既有 `scheme` / `group` / `int8_projections` 调用继续
  可用，并在内部转换为量化计划；流式 loader 与 BF16 树量化必须解析
  同一计划。
- **预算是参数 tensor bytes，不混入 allocator**：提供真实量化树的
  tensor-byte 统计与 Qwen3 dense topology 的静态估算。GPU allocator、
  cache 和反量化临时量另行记录，不能偷换成“模型显存”。
- **硬件证据边界显式化**：当前执行环境没有 `/home/yj/models` 与可用
  NVIDIA driver；本周 Close 只接受真实离线结果，不复制或推测新的
  14B GPU fidelity 数字。真实模型验证脚本保持可复现，留到资产机器。

## 实现范围

- `LinearQuantizationSpec` / `QuantizationPlan`：支持 `:int4`、`:int8`、
  `:bf16`，支持默认、投影级与层级 override，输入严格校验。
- INT4 `:mse` reconstruction calibration：在冻结的 clipping candidates
  中为每个 row/group 选择误差最小 scale；RTN `:maxabs` 保持位兼容。
- 流式 `load_hf_qwen3_quantized` 与 `quantize_bf16_parameters` 共用计划，
  LM head 与 tied embedding 行为有专项测试。
- `quantized_parameter_bytes` 与 `estimate_qwen3_quantized_bytes`：
  分别统计已构造参数树和不加载权重时的 dense topology 预算。
- Week 17 离线 fixture：outlier 权重、层级/投影 override、streamed vs
  in-memory、一致性、预算估算和错误输入 fail-closed。
- 真实模型验证脚本扩展：输出量化计划、calibration、tensor bytes、
  logits/decode/greedy 指标，供资产机器直接复核。

## Close 条件

- `:maxabs` INT4 保持 Week 16 的 packing/dequant 语义；`:mse` 对每个
  group 的重建平方误差不高于冻结 candidate 中的 max-abs baseline，
  且 outlier fixture 获得严格改善。
- 层级 override > projection override > default 的优先级被测试钉死；
  `lm_head` 可独立选择 INT8/BF16；非法 layer、projection、scheme、
  group 与 calibration 全部 fail closed。
- BF16 参数树量化与 safetensors 流式加载对同一计划产生逐 tensor
  相同的量化结果；既有 Week 16 API 与默认数值行为不回归。
- 实际树 tensor-byte 统计与 topology 静态估算在合成 Qwen3 fixture
  上精确一致；冻结 Qwen3-14B 的纯 INT4 与混合精度理论预算。
- 默认离线测试全部通过；真实 14B GPU 指标只在资产与驱动可用时更新，
  本环境缺失不得被写成通过。

## 非目标

- 不实现 GPTQ、AWQ、激活/KV cache 量化或量化训练。
- 不实现 fused INT4 GEMM；每 token 全量反量化吞吐仍是后续独立问题。
- 不宣称仅凭 weight reconstruction MSE 可以保证 greedy token 一致。
- 不下载或提交大模型权重，不伪造 14B GPU residency/fidelity 结果。

## 过程记录

### 2026-07-29：Open

- Week 16 保持 Closed；Week 17 承接其 plain RTN INT4 fidelity 边界。
- 当前工作区确认无持久模型资产、无可用 NVIDIA driver；资源边界在
  Open 时写入，避免 Close 时用旧机器结果冒充本周实验。
- 实现顺序冻结为：量化计划 → MSE scale search → 两条 loader 共用 →
  tensor-byte 预算 → 离线回归 → 可复现真实模型脚本。

## Close 回顾

- **完成了什么**：
- **验证证据**：
- **没有完成及原因**：
- **最重要的认知变化**：
- **是否满足 Close 条件**：
- **带到下一 Week 的问题**：
