# Chapter 14 — Qwen3 Native BF16 Mixed-Precision Compute

> 状态：Closed
>
> 开启记录：2026-07-25
>
> 关闭记录：2026-07-26
>
> 依赖基线：[`Chapter 13 — Qwen3 Streamed Loading and 8B/14B/32B Real-Weight Parity`](../../episodes/episode03_model_family_and_large_weights/chapter13_qwen3_streamed_large_weights.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Chapter 07—13 的所有真实验证都以 Float32 计算，BF16 只是权重
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
   差异只剩求和顺序（Chapter 13 已建立对应容差方法论）。

## 设计决策

- **独立 BF16 推理路径**，不改动既有 Float32 前向、KV cache 与
  Reactant/XLA 路径：新增显式混合精度算子与 `bf16_*` 前向/decode/
  greedy 函数，复用 `GPTModel` 结构与 loader 的 shape/名称校验。
  Chapter 06—13 的全部历史结论零风险。
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
  冻结 BF16 量级容差（block 沿用 Chapter 13 的尺度感知判据）。
- 记录 BF16 与 F32 路径的 logits 偏差量级（同模型自对照），以及
  0.6B/8B BF16 常驻内存实测。
- `test/episodes/episode04_efficient_inference_and_quantization/chapter14_qwen3_bf16_compute/test_qwen3_bf16_compute.jl`：离线合成模型的 BF16 算子/路径测试 +
  `LIFEAI_QWEN3_*_MODEL_DIR` opt-in BF16 integration。

## 验证分层

| 证据层 | 最终状态 |
| --- | --- |
| BF16 解码位保真与 dtype 传播 | 默认离线覆盖（8 项） |
| BF16 混合精度算子语义（norm/softmax/RoPE/linear） | 默认离线覆盖（5 项） |
| BF16 路径确定性、cached decode 逐位等价、vs F32 偏差有界 | 默认离线覆盖（11 项） |
| 0.6B / 1.7B / 4B / 8B 逐层 + decode parity | 全部在冻结容差内，argmax 零失配 |
| greedy token 序列 vs HF BF16 | 四尺寸 16 步全部完全一致 |
| 8B BF16 全量驻留生成 | 完成；参数树 15.26 GiB，峰值 RSS 19.0 GiB |
| 既有 F32 / 流式 / XLA 路径 | 前向语义零改动，默认全套无回归 |

## Close 条件

- BF16 权重加载位保真；BF16 参数树常驻内存相对 F32 减半（实测记录）。
- 四个尺寸（0.6B/1.7B/4B/8B）逐层 hidden、final hidden、logits、
  dynamic decode 在显式冻结的 BF16 量级容差内对齐，block 采用尺度
  感知判据；argmax 与 N 步 greedy token 序列与 HF BF16 完全一致。
- 8B 以 BF16 全量驻留完成一次真实 greedy 生成（本机首个 >4B 全量
  驻留生成），内存实测记录在案。
- BF16 路径两次运行结果逐位一致（自身确定性）；与 F32 路径的偏差
  量级有记录。
- Chapter 14 测试进入默认套件；默认全套与既有 opt-in（分进程协议）
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

- Chapter 13 保持 Closed；Chapter 14 承接 native BF16 compute。
- 已核对 Transformers 4.51.0 `modeling_qwen3.py` 的混合精度语义
  （RMSNorm F32 归一化、softmax F32、RoPE 表 F32、线性 BF16），
  作为实现契约逐条冻结进本页。
- 资源边界冻结：BF16 全量驻留上限 8B（15.3 GiB）；14B/32B 出界。

### 2026-07-25：实现与首批验证

- 新增 `src/models/bf16_inference.jl`（独立 BF16 路径）与 loader
  `weight_dtype=BFloat16`（BF16 位保真直读）；`BFloat16s` 进入依赖。
  逐算子镜像 HF 混合精度契约，每次 elementwise 运算显式舍入。
- 合成模型离线测试 24 项通过，含一个强不变式：**BF16 cached decode 与
  全量 BF16 forward 的对应位置 logits 逐位相等**（相同 BF16 运算按相同
  顺序作用于相同值）；BF16 与 F32 路径偏差非零且有界；F32 树误入 BF16
  入口 fail closed。
- `export_qwen3_reference.py` 新增 `--compute-dtype bfloat16` 与
  `--greedy-steps`（greedy 独立 prefill，避免 decode fixture 污染
  cache）；四个尺寸的 HF BF16 reference 已导出。
- **0.6B / 1.7B / 4B BF16 parity 通过**：embedding 位精确；16 步
  greedy token 序列与 HF BF16 **全部完全一致**；logits max-abs
  0.66 / 0.72 / 0.41（mean 0.07 / 0.05 / 0.03），argmax 全一致——
  BF16 量级的跨框架偏差，语义（token 选择）零漂移。BF16 参数树
  1.11 / 3.20 / 7.49 GiB，为 F32 的一半。
- **8B 两次 OOM 教训与修复**：15.3 GiB 参数树常驻时，(1) 每层线性把
  整块权重转成 Float32 的临时数组叠加 GC 滞后使 RSS 越界；(2) kernel
  日志显示本机还有其他进程占用内存，实际可用预算 ≈ 18 GiB 而非
  30 GiB，`total-vm 25 GB / anon-rss 19.3 GB` 时即被 KILL。修复：
  `_bf16_linear` 改为按输出行**分块**升精度（Float32 临时恒为几十
  MiB）+ `_bf16_forward_pass` 逐层 `GC.gc(false)`。分块后 0.6B—4B
  的全部 parity 数值与整块实现**逐位相同**（分块只切输出行，不改变
  任何累加维度），8B 峰值 RSS 压到 19.0 GiB 内完成。
- **四尺寸 BF16 parity 全部通过**（Transformers 4.51.0 BF16 reference，
  argmax 全一致，16 步 greedy token 序列全部完全一致）：

  | variant | logits max/mean | decode max | blocks scaled | BF16 树 | 峰值 RSS | forward |
  | --- | --- | --- | --- | ---: | ---: | ---: |
  | 0.6B | 0.656 / 0.073 | 0.516 | 1.49e-2 | 1.11 GiB | 3.8 GiB | 34.7 s |
  | 1.7B | 0.719 / 0.052 | 0.447 | 9.87e-3 | 3.21 GiB | 6.9 GiB | 78.5 s |
  | 4B | 0.414 / 0.027 | 0.219 | 7.91e-3 | 7.49 GiB | 11.9 GiB | 182.6 s |
  | 8B | 0.422 / 0.040 | 0.203 | 3.53e-3 | 15.26 GiB | 19.0 GiB | 286.6 s |

  8B 以 BF16 全量驻留完成 16 步 greedy 生成——本机首个 >4B 全量驻留
  生成。embedding 四尺寸均位精确（0.0）。
- 容差冻结：embedding `1e-3`、blocks scaled `5e-2`、final hidden
  `4.0`、logits / decode `2.0`（BF16 量级绝对容差 + Chapter 13 尺度感知
  block 判据）；连同 greedy 序列、内存实测冻结进离线 fixture。
- BF16 opt-in 进程协议：0.6B/1.7B/4B 可同进程（累计树 ≈ 11.8 GiB，
  `Pkg.test` 实测通过）；8B 在"完整套件 + integration"同进程下仍被
  OOM KILL（套件自身堆 + 19 GiB 峰值 + 本机其他进程），最终协议为
  **独立测试文件进程**（只跑 `test_qwen3_bf16_compute.jl`），命令记入
  `../../local_model_assets.md`。

### 2026-07-26：验证与 Close

- 默认离线全套 `4729 / 4729` 通过（Chapter 14 离线专项 `77 / 77`），
  Chapter 05—13 计数与各自 Close 时一致，无回归。
- BF16 小尺寸 opt-in（0.6B/1.7B/4B 同进程 `Pkg.test`）：Chapter 14 专项
  `205 / 205` 通过。
- 8B BF16 opt-in 独立进程 + `--heap-size-hint=2G`：`125 / 125` 通过，
  峰值 RSS 实测 **18.53 GiB**——heap hint 强制激进 GC 是在共享内存
  机器上装下 15.26 GiB 参数树的必要条件，已写入复现命令。

## Close 回顾

- **完成了什么**：BF16 从存储格式升级为真正的推理路径——逐算子镜像
  Transformers 4.51.0 的混合精度契约；0.6B/1.7B/4B/8B 四尺寸与 HF
  BF16 逐层对齐、argmax 零失配、16 步 greedy token 序列完全一致；
  参数常驻内存减半，8B（15.26 GiB 树）完成本机首个 >4B 全量驻留
  greedy 生成。
- **验证证据**：四尺寸 logits max-abs ≤ 0.72（mean ≤ 0.073）、blocks
  scaled ≤ 1.49e-2、embedding 位精确；BF16 cached decode 与全量前向
  逐位等价的离线不变式；默认全套 `4729 / 4729`、小尺寸 opt-in
  `205 / 205`、8B 独立进程 `125 / 125`。
- **没有完成及原因**：14B/32B BF16 全量驻留（27.5 / 61.0 GiB）仍超
  内存；BF16 训练、CUDA/XLA BF16、量化、sampling replay 与吞吐
  benchmark 按计划未做——本周聚焦数值语义与验证，不承诺生产吞吐。
- **最重要的认知变化**：其一，"native BF16"的本质是一套混合精度契约
  而非单一 dtype 开关——哪里升 F32、哪里舍回 BF16 决定了能否与官方
  推理逐 token 一致；其二，跨框架 BF16 的正确性标尺是 token 级行为
  （greedy 序列、argmax）加 BF16 量级容差，而不是逐位数值；其三，
  在参数树占满大半 RAM 的场景，临时数组分块 + 显式 GC + heap hint
  三者共同决定可行性，缺一即 OOM。
- **是否满足 Close 条件**：是。位保真加载、四尺寸 parity 与 greedy
  一致、8B 全量驻留生成、确定性与 F32 偏差记录、默认套件接入与
  无回归、文档边界均已落实。
