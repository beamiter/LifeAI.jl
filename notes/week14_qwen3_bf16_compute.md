# Week 14 — Qwen3 Native BF16 Mixed-Precision Compute

> 状态：Open
>
> 开启记录：2026-07-25
>
> 依赖基线：[`Week 13 — Qwen3 Streamed Loading and 8B/14B/32B Real-Weight Parity`](week13_qwen3_streamed_large_weights.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Week 07—13 的所有真实验证都以 Float32 计算，BF16 只是权重
> 存储格式。本周把 BF16 升级为真正的推理计算路径——这是 HuggingFace
> 官方 Qwen3 推理的默认精度，也是参数内存减半、让 8B 在本机全量驻留
> 并实际生成的前提。

## 核心问题

> LifeAI 能否以与 HuggingFace Transformers BF16 推理**相同的混合精度
> 语义**执行 Qwen3 dense 前向与增量解码，使逐层 hidden、logits 在
> BF16 量级容差内对齐，并产生相同的 greedy token 序列？

"native BF16 compute" 不是把所有算子无脑改成 BF16。Transformers 4.51.0
的 Qwen3 BF16 推理是一套明确的混合精度约定（本周实现必须逐条镜像）：

1. **RMSNorm / QK-Norm**：输入升到 Float32 归一化，再转回 BF16 后与
   BF16 scale 相乘（`modeling_qwen3.py` L72—76）。
2. **attention softmax**：分数用 BF16 存储，softmax 在 Float32 里做，
   结果转回 BF16（L162）。
3. **RoPE**：cos/sin 表以 Float32 生成（显式关闭 autocast），转成
   BF16 后在 BF16 里应用（L340—346）。
4. **线性层**：BF16 权重与激活，硬件/内核内部以 Float32 累加，输出
   舍入回 BF16。LifeAI 以 `BF16(F32(W) × F32(x))` 仿真——输入本就是
   BF16，Float32 GEMM 累加 + 输出舍入与 oneDNN BF16 GEMM 语义一致，
   差异只剩求和顺序（Week 13 已建立对应容差方法论）。

## 设计决策

- **独立 BF16 推理路径**，不改动既有 Float32 前向、KV cache 与
  Reactant/XLA 路径：新增显式混合精度算子与 `bf16_*` 前向/decode/
  greedy 函数，复用 `GPTModel` 结构与 loader 的 shape/名称校验。
  Week 06—13 的全部历史结论零风险。
- loader 增加 `weight_dtype=BFloat16`：BF16 safetensors 位保真读入
  （不经过 Float32 往返），参数树常驻内存减半。
- 逐位一致不再是跨框架目标（BF16 GEMM 求和顺序不可控），改为：
  LifeAI BF16 路径自身确定性 + 与 HF BF16 reference 的 BF16 量级
  容差 + greedy token 序列完全一致。

## 本周资源边界（先冻结再执行）

| variant | BF16 参数内存 | 30 GiB RAM 全量驻留 | 本周计划 |
| --- | ---: | --- | --- |
| 0.6B / 1.7B / 4B | 1.2 / 3.2 / 7.5 GiB | 是 | 逐层 parity + greedy |
| 8B | 15.3 GiB | 是（首次可全量驻留） | 逐层 parity + greedy |
| 14B / 32B | 27.5 / 61.0 GiB | 否（14B 无转换余量） | 不做，保持边界 |

Python reference 侧 `torch_dtype=bfloat16` 全部四个尺寸均可直载
（8B BF16 ≈ 15.3 GiB），无需 disk offload。

## 实现范围

- `Project.toml` 引入 `BFloat16s`；safetensors 解码支持
  `target_dtype=BFloat16`（BF16 位保真、F32 舍入），
  `load_hf_qwen3_model(...; weight_dtype=BFloat16)` 产出 BF16 参数树。
- 新增 BF16 混合精度推理模块：显式 `_bf16_linear` / `_bf16_rmsnorm` /
  QK-Norm / RoPE / attention（F32 softmax）/ SwiGLU 算子，
  `bf16_forward_trace`、BF16 dynamic KV cache 的 prefill/decode 与
  greedy 生成入口；语义逐条对照本页"核心问题"清单。
- `export_qwen3_reference.py` 增加 `--compute-dtype bfloat16` 与
  greedy 步数选项：逐层 hidden/logits/decode 以 BF16 原位保存，另录
  N 步 greedy token ids。
- 为 0.6B / 1.7B / 4B / 8B 生成 BF16 reference 并逐层对齐；量测后
  冻结 BF16 量级容差（block 沿用 Week 13 的尺度感知判据）。
- 记录 BF16 与 F32 路径的 logits 偏差量级（同模型自对照），以及
  0.6B/8B BF16 常驻内存实测。
- `test/test_week14.jl`：离线合成模型的 BF16 算子/路径测试 +
  `LIFEAI_QWEN3_*_MODEL_DIR` opt-in BF16 integration。

## 验证分层

| 证据层 | 目标状态 |
| --- | --- |
| BF16 解码位保真与 dtype 传播 | 默认离线覆盖 |
| BF16 混合精度算子语义（norm/softmax/RoPE/linear） | 默认离线覆盖 |
| BF16 路径确定性与 vs F32 偏差有界 | 默认离线覆盖 |
| 0.6B / 1.7B / 4B / 8B 逐层 + decode parity | BF16 量级容差内全部对齐 |
| greedy token 序列 vs HF BF16 | 四尺寸 N 步完全一致 |
| 8B BF16 全量驻留生成 | 实测内存并实际产出文本 |
| 既有 F32 / 流式 / XLA 路径 | 零改动，无回归 |

## Close 条件

- BF16 权重加载位保真；BF16 参数树常驻内存相对 F32 减半（实测记录）。
- 四个尺寸（0.6B/1.7B/4B/8B）逐层 hidden、final hidden、logits、
  dynamic decode 在显式冻结的 BF16 量级容差内对齐，block 采用尺度
  感知判据；argmax 与 N 步 greedy token 序列与 HF BF16 完全一致。
- 8B 以 BF16 全量驻留完成一次真实 greedy 生成（本机首个 >4B 全量
  驻留生成），内存实测记录在案。
- BF16 路径两次运行结果逐位一致（自身确定性）；与 F32 路径的偏差
  量级有记录。
- Week 14 测试进入默认套件；默认全套与既有 opt-in（分进程协议）
  无回归。
- 文档明确：14B/32B BF16 全量驻留仍超内存；BF16 训练、CUDA/XLA
  BF16、更低精度量化均未实现。

## 非目标

- 不做 BF16 训练/反向、CUDA 或 Reactant/XLA 的 BF16 路径。
- 不做 14B/32B 的 BF16 全量驻留或 BF16 流式路径。
- 不做 FP8/INT8/INT4 量化、GGUF、AWQ/GPTQ。
- 不做 BF16 的 sampling replay、长文本吞吐 benchmark 或 text 端到端
  chat 闭环；greedy token 一致性是本周的生成证据边界。
- 不改动既有 Float32 与流式路径的任何数值行为。

## 过程记录

### 2026-07-25：Open

- Week 13 保持 Closed；Week 14 承接 native BF16 compute。
- 已核对 Transformers 4.51.0 `modeling_qwen3.py` 的混合精度语义
  （RMSNorm F32 归一化、softmax F32、RoPE 表 F32、线性 BF16），
  作为实现契约逐条冻结进本页。
- 资源边界冻结：BF16 全量驻留上限 8B（15.3 GiB）；14B/32B 出界。
