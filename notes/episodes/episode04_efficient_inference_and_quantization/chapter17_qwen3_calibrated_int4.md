# Chapter 17 — Qwen3 Reconstruction-Calibrated INT4 与预算化混合精度

> 状态：Closed
>
> 开启记录：2026-07-29
>
> 关闭记录：2026-07-29
>
> 依赖基线：[`Chapter 16 — Qwen3 XLA BF16 Compiled Decode 与 INT8/INT4 量化`](../../episodes/episode04_efficient_inference_and_quantization/chapter16_qwen3_xla_decode_quant.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Chapter 16 已解决 8B INT8 / 14B INT4 在 16.3 GiB GPU 上的
> 驻留问题，但 plain RTN INT4 的 14B greedy 仅 4/16，且现有
> `int8_projections` 只能按投影名统一覆盖全部层，不能表达 LM head 或
> 层级敏感度。本周先把量化策略从单一全局开关升级为可校准、可组合、
> 可预算的公共能力。

## 核心问题

> 在不改变 BF16 计算契约的前提下，能否用 reconstruction-calibrated
> INT4 scale 与按层/投影的 INT4、INT8、BF16 计划，系统降低量化误差，
> 并用确定性离线证据验证算法、loader 一致性和存储预算，再在可用
> NVIDIA GPU 上以冻结的 Qwen3-14B 权重验证驻留与生成行为？

## 设计决策

- **先校准权重重建，不伪装成 activation-aware**：本周实现按
  output-channel / input-group 独立选择 clipping ratio 的 MSE scale
  search；没有校准语料和激活统计时，不使用 AWQ/GPTQ 名称。
- **量化计划是唯一策略源**：默认规则、按 projection override、按
  `(layer, projection)` override 采用固定优先级；LM head 作为显式
  target，避免 Chapter 16 `int8_projections` 无法覆盖它。
- **保持兼容**：既有 `scheme` / `group` / `int8_projections` 调用继续
  可用，并在内部转换为量化计划；流式 loader 与 BF16 树量化必须解析
  同一计划。
- **预算是参数 tensor bytes，不混入 allocator**：提供真实量化树的
  tensor-byte 统计与 Qwen3 dense topology 的静态估算。GPU allocator、
  cache 和反量化临时量另行记录，不能偷换成“模型显存”。
- **硬件证据边界显式化**：Open 时 sandbox 内的 `nvidia-smi` 失败，
  曾被误判为宿主机没有 NVIDIA driver；用户指出后，经宿主权限复核为
  RTX 4090 D / driver 570.153.02，CUDA.jl 6.2.1 `functional=true`。
  本周因此补做冻结 Qwen3-14B 的真实下载、checksum 与 GPU 实测，不复制
  Chapter 16 的 RTX 5080 数字。

## 实现范围

- `LinearQuantizationSpec` / `QuantizationPlan`：支持 `:int4`、`:int8`、
  `:bf16`，支持默认、投影级与层级 override，输入严格校验。
- INT4 `:mse` reconstruction calibration：在冻结的 clipping candidates
  中为每个 row/group 选择误差最小 scale；RTN `:maxabs` 保持位兼容。
- 流式 `load_hf_qwen3_quantized` 与 `quantize_bf16_parameters` 共用计划，
  LM head 与 tied embedding 行为有专项测试。
- `quantized_parameter_bytes` 与 `estimate_qwen3_quantized_bytes`：
  分别统计已构造参数树和不加载权重时的 dense topology 预算。
- Chapter 17 离线 fixture：outlier 权重、层级/投影 override、streamed vs
  in-memory、一致性、预算估算和错误输入 fail-closed。
- 真实模型验证脚本扩展：输出量化计划、calibration、tensor bytes、
  logits/decode/greedy 指标，供资产机器直接复核。

## Close 条件

- `:maxabs` INT4 保持 Chapter 16 的 packing/dequant 语义；`:mse` 对每个
  group 的重建平方误差不高于冻结 candidate 中的 max-abs baseline，
  且 outlier fixture 获得严格改善。
- 层级 override > projection override > default 的优先级被测试钉死；
  `lm_head` 可独立选择 INT8/BF16；非法 layer、projection、scheme、
  group 与 calibration 全部 fail closed。
- BF16 参数树量化与 safetensors 流式加载对同一计划产生逐 tensor
  相同的量化结果；既有 Chapter 16 API 与默认数值行为不回归。
- 实际树 tensor-byte 统计与 topology 静态估算在合成 Qwen3 fixture
  上精确一致；冻结 Qwen3-14B 的纯 INT4 与混合精度理论预算。
- 默认离线测试全部通过；冻结 Qwen3-14B 的模型文件必须逐文件 checksum
  一致，且真实 GPU 结果同时记录 tensor bytes、allocator/运行时显存、
  logits/decode argmax 与 16-step greedy，不能只写“可驻留”。

## 非目标

- 不实现 GPTQ、AWQ、激活/KV cache 量化或量化训练。
- 不实现 fused INT4 GEMM；每 token 全量反量化吞吐仍是后续独立问题。
- 不宣称仅凭 weight reconstruction MSE 可以保证 greedy token 一致。
- 大模型权重和 reference 只下载到仓库外的持久目录，不提交进 Git；
  不伪造 14B GPU residency/fidelity 结果。

## 过程记录

### 2026-07-29：Open

- Chapter 16 保持 Closed；Chapter 17 承接其 plain RTN INT4 fidelity 边界。
- Open 时工作区确实没有已下载模型资产；sandbox 内 `nvidia-smi`
  返回驱动不可用，但这不代表宿主机没有 GPU。
- 实现顺序冻结为：量化计划 → MSE scale search → 两条 loader 共用 →
  tensor-byte 预算 → 离线回归 → 可复现真实模型脚本。

### 2026-07-29：硬件范围更正

- 用户指出宿主机驱动可用并授权联网下载权重；宿主复核确认
  `NVIDIA GeForce RTX 4090 D`、driver `570.153.02`、总显存
  `25,238,568,960` bytes，CUDA.jl 6.2.1 可正常执行。
- Open commit `2b89a73` 保留了当时的错误判断；Close 不改写历史 commit，
  而在本记录中明确说明 sandbox 视图与宿主硬件事实的差异。
- 权重改从 ModelScope 的官方 `Qwen/Qwen3-14B` 仓库下载，并逐文件与
  HuggingFace 冻结 revision
  `40c069824f4251a91eefaf281ebe4c544efd3e18` 的 SHA256 对照；只有字节
  一致后才允许生成 reference 和运行 GPU parity。

### 2026-07-29：实现与离线验证

- 新增公共 `LinearQuantizationSpec` / `QuantizationPlan`：支持 INT4、
  INT8、BF16，解析优先级固定为 layer override > projection override >
  default；LM head 可独立配置，tied embedding 不重复构造 head。
- INT4 新增确定性的 per-row/group reconstruction-MSE clipping search；
  candidate `1.0` 保证每组不会劣于 max-abs baseline，默认 max-abs RTN
  packing 与数值行为保持不变。
- 流式 safetensors loader 与 BF16 参数树量化共用同一计划；新增真实树
  tensor-byte 统计和不加载权重的 Qwen3 topology 估算，合成 tied/untied
  fixture 上逐 tensor、逐 byte 相同。
- 默认套件 `4953 / 4953` 通过，Chapter 16 `78 / 78`、Chapter 17 最终专项
  `83 / 83`；小型 MSE-INT4/INT8/BF16 混合树另在 RTX 4090 D 完成 CUDA
  forward/greedy smoke。

### 2026-07-29：Qwen3-14B 真实 GPU 对照

- ModelScope 下载的 5 个配置/tokenizer/index 文件与 8 个权重分片
  （`29,536,665,640` bytes）全部匹配冻结 HuggingFace revision 的
  SHA256。BF16 reference 使用 Transformers 4.51.0 /
  Torch 2.7.1+cpu，reference checksum 已冻结进 Chapter 17 fixture。
- 实测设备为 RTX 4090 D，driver 570.153.02，CUDA.jl 6.2.1，总显存
  `25,238,568,960` bytes。下表的 tree 是 tensor payload，VRAM 包含
  allocator、cache 与反量化临时量，两者不能互换。

| 计划 | host tree | 实测 VRAM | full logits max / mean abs | decode max abs | argmax（full/decode） | 16-step greedy | warm tok/s |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 全 INT8 | 14.487 GiB | 23.453 GiB | 1.4375 / 0.13747 | 0.40625 | 对 / 对 | **16/16** | 0.816 |
| mixed MSE-INT4 | 12.093 GiB | 22.521 GiB | 3.62891 / 0.36034 | 1.01563 | 对 / 对 | **4/16**（第 5 token） | 0.377 |
| 同布局 mixed RTN-INT4 | 12.093 GiB | 22.597 GiB | 4.32031 / 0.42933 | 1.125 | 对 / 对 | **16/16** | 0.377 |

- 三组的静态估算与真实 host tensor bytes 逐 byte 相同。全 INT8 距设备
  总显存只剩约 54 MiB，属于本机可复现但没有部署安全余量的上界；mixed
  计划保留约 1 GiB 级运行余量。
- MSE 对照得到明确的负结果：它降低了相同 mixed layout 的 full-logits
  max/mean error，却把 greedy 从 RTN 的 16/16 降到 4/16。重建误差或
  单次 full-logit 全局误差都不能代替自回归生成验证；本周不把 MSE
  calibration 宣称为质量提升。

## Close 回顾

- **完成了什么**：完成校准式 INT4 scale search、统一的层/投影/LM-head
  混合精度计划、两条 loader 共用策略、参数预算 API，以及 14B 的三组
  4090 D 对照。
- **验证证据**：默认 `4953 / 4953`；Chapter 17 `83 / 83`；模型/reference
  checksum；全 INT8 与 mixed RTN 均 16/16 greedy，mixed MSE 的负结果
  同样冻结进 fixture。
- **没有完成及原因**：没有得到“reconstruction MSE 提升生成 fidelity”
  的正结果；真实对照反而否定了它。量化 GEMM、activation-aware
  calibration、GPTQ/AWQ、KV cache/激活量化仍不在本周范围。
- **最重要的认知变化**：计划和可验证预算比单一 calibration heuristic
  更可靠；低重建误差、低全局 logit error 与序列级 argmax 稳定性不是
  单调关系，必须保留端到端 greedy 对照。
- **是否满足 Close 条件**：满足。算法不劣于候选 baseline 的局部契约、
  legacy 兼容、两 loader 一致性、预算精确性、离线回归和真实硬件边界
  均已有证据；质量结论按实测写成负结果。
- **带到下一 Chapter 的问题**：若继续量化，优先做 activation-aware
  sensitivity / AWQ 类校准与量化 GEMM；以 mixed RTN 16/16 作为质量
  baseline，以全 INT8 的 54 MiB 余量作为不可直接部署的显存上界。
