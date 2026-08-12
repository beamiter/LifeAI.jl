# Chapter 16 — Qwen3 XLA BF16 Compiled Decode 与 INT8/INT4 量化

> 状态：Closed
>
> 开启记录：2026-07-26
>
> 关闭记录：2026-07-26
>
> 依赖基线：[`Chapter 15 — Qwen3 BF16 CUDA / XLA Accelerated Inference`](../../episodes/episode04_efficient_inference_and_quantization/chapter15_qwen3_bf16_accel.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Chapter 15 的 CUDA eager BF16 生成吞吐 8—15 tok/s，瓶颈在
> 逐步 kernel launch 与 dynamic cache 的 `cat` 重分配；同时 8B BF16
> （15.26 GiB）仍进不了 16.3 GiB VRAM。本周两条线：XLA 编译
> static-cache decode 把吞吐拉一个量级；INT8/INT4 权重量化打开 8B+
> 的 GPU 驻留。

## 核心问题

> 1. BF16 混合精度契约能否在固定形状 static KV cache 下编译为单个
>    XLA decode 可执行文件，逐 token 行为与 eager 路径一致，且 steady
>    吞吐达到 eager 的 10 倍量级？
> 2. RTN 权重量化（INT8 per-channel / INT4 group-128）能否把 8B 乃至
>    14B 装进 16.3 GiB VRAM，并把量化引入的 token 级行为漂移量化清楚
>    而不是掩盖？

## 设计决策

- **XLA decode 镜像既有 F32 static 模式**：traced position、
  `cos_cache[:, position]` 动态取位、`cache[..., write_position, :] =`
  动态写入、`key_position .<= valid_length` 有效前缀掩码——Chapter 02—09
  已验证的编译形态，换成 Chapter 14/15 的 BF16 算子语义。
- **量化是权重-only 的 RTN**：INT8 对称 per-output-channel；INT4 对称
  group-128、两值一字节打包。embedding 保持 BF16（gather 位精确），
  全部线性层（含 lm_head）量化。计算路径：GPU 上按层解包/反量化为
  BF16 后走既有 `_bf16a_linear`——量化只改变权重驻留格式，不改变
  计算契约。
- **验证基准分层**：量化模型先与自家 BF16 路径对照（隔离量化误差），
  再与 HF BF16 reference 对照 token 级行为；INT4 若出现 greedy 漂移，
  记录逐 token 一致率与首个分歧位置，不降低标准掩盖。
- 14B 缺 BF16 reference：用 Chapter 13 的 accelerate disk offload 以
  `--compute-dtype bfloat16` 导出（14B BF16 29.5 GiB 超 RAM，offload
  可行已验证）。

## 本周资源边界（先冻结再执行）

| 配置 | 权重驻留 | 16.3 GiB VRAM | 计划 |
| --- | ---: | --- | --- |
| 0.6B BF16 XLA static decode | 1.1 GiB + XLA 缓冲 | 可行 | 必做 |
| 4B BF16 XLA static decode | 7.5 GiB + XLA 缓冲 | 边缘 | stretch |
| 8B INT8 | ≈ 8.2 GiB + BF16 embedding | 可行 | 必做 |
| 14B INT4 g128 | ≈ 7.4 GiB + scales + BF16 embedding | 可行 | 必做（行为漂移如实记录） |
| 32B INT4 | ≈ 16.4 GiB | 出界 | 不做 |

## 实现范围

- BF16 static KV cache 与 traced decode step（单 token）：prefill 编译
  填充 cache，decode 编译复用同一 executable；0.6B 与 eager CUDA 的
  greedy 序列逐 token 对照，warm/steady 吞吐与编译成本记录。
- `quantize` API：BF16 树 → INT8/INT4 量化树（打包 + scales）；
  量化树可整体 `cu()` 上载；`_bf16a_linear` 对量化权重的 dequant
  分派；量化/反量化 round-trip 与打包正确性离线测试。
- 8B INT8 GPU 驻留：VRAM 实测；vs 自家 CPU BF16 logits 与 vs HF BF16
  reference（week14-bf16-parity）的 parity/greedy；吞吐记录。
- 14B BF16 offload reference 导出（层输出 + 16 步 greedy）；14B INT4
  GPU 驻留：logits 偏差、argmax/greedy 一致率、吞吐、VRAM。
- `test/episodes/episode04_efficient_inference_and_quantization/chapter16_qwen3_xla_decode_quant/test_qwen3_xla_decode_quantization.jl`：量化数值语义与 round-trip 离线测试 + 合成
  模型量化推理对照；GPU/XLA 真实验证 opt-in。

## 验证分层

| 证据层 | 最终状态 |
| --- | --- |
| XLA BF16 decode vs HF greedy 序列 | 0.6B 两条编译路径 16 步全对 |
| XLA decode steady 吞吐 | **246 tok/s = eager 的 16.1×**（目标 ≥10× 达成） |
| 量化 round-trip / 打包语义 | 默认离线覆盖（61 项） |
| 8B INT8 GPU：VRAM / parity / greedy | 8.22 GiB 驻留；argmax 全对，greedy 14/16（近平局） |
| 14B INT4 GPU：VRAM / parity / greedy 一致率 | 8.38 GiB 驻留；prefill argmax 对，greedy 4/16 如实冻结 |
| 既有全部路径 | 零改动，默认全套无回归 |

## Close 条件

- 0.6B XLA BF16 compiled decode 与 eager CUDA greedy 逐 token 一致，
  steady 吞吐实测并对照 eager；编译成本记录。
- INT8/INT4 量化树的构造、打包、反量化有离线数值测试；量化只作用于
  线性层权重且计算契约不变有明确陈述。
- 8B INT8 在 GPU 驻留并完成 16 步 greedy：VRAM、吞吐、与 HF BF16
  reference 的 logits 偏差和 greedy 一致性全部冻结。
- 14B INT4 在 GPU 驻留并完成 16 步 greedy：同上；若 greedy 与 BF16
  reference 不完全一致，逐 token 一致率与首个分歧位置记录在案。
- 默认全套与既有 opt-in 无回归；文档明确 32B、量化训练、GPTQ/AWQ
  等边界。

## 非目标

- 不做 GPTQ/AWQ/校准式量化、FP8、KV cache 量化或激活量化。
- 不做 32B GPU 驻留、多卡或权重切分。
- 不做 XLA 端到端 chat/sampling 闭环；compiled greedy decode 是本周
  XLA 证据边界。
- 不改动既有 CPU BF16 / F32 / 流式路径的数值行为。

## 过程记录

### 2026-07-26：Open

- Chapter 15 保持 Closed；Chapter 16 承接 XLA decode 编译与量化两条线。
- F32 XLA static 模式（traced position/动态写/前缀掩码）确认可镜像；
  量化方案冻结为 RTN INT8 per-channel 与 INT4 group-128。
- 资源边界冻结如上表；32B 出界。

### 2026-07-26：XLA compiled decode

- 新增 `src/models/bf16_xla.jl`：BF16 static cache 的 traced prefill 与
  单 token decode（traced position、动态 RoPE 取位、动态 cache 写、
  有效前缀掩码），批 1。三个编译产物：prefill、返回 logits 的
  decode、**设备端 greedy step**（argmax 在 executable 内，token 与
  position 以同型 1 元素数组回馈，宿主每 token 只取回一个整数）。
- 关键性能教训：**宿主往返比整个前向还贵**——纯编译调用 3.4 ms，但
  `copyto!` + 全量 logits 取回把每步拖到 115 ms。设备端 greedy 闭环
  后：**steady 4.06 ms/token ≈ 246 tok/s**，为 CUDA eager
  （15.3 tok/s）的 **16.1 倍**，量级目标达成。logits 路径（每步取回
  151936 维向量做宿主 argmax）为 25.8 tok/s，用于 parity 复核。
- 正确性：两条路径的 16 步 greedy 均与 HF BF16 完全一致；编译成本
  prefill 44.4 s / decode 16.5 s / greedy 14.2 s。
- 排坑记录：0 维 traced 数组缺 `one`/`Array` 方法、1 元素数组的
  `[1]` 触发 scalar-indexing 禁令（用 `sum` 提取 traced 标量）、
  greedy 回馈的 position 必须与输入同为 1 元素数组否则第二次调用
  类型不匹配。

### 2026-07-26：量化实现与 8B INT8

- 新增 `src/models/quantized.jl`：RTN INT8 per-channel 与 INT4
  group-wise（相邻两列一字节打包，+8 偏移）；`load_hf_qwen3_quantized`
  流式逐层量化加载（8B 全程无需 BF16 全树驻留，宿主峰值 15.4 GiB）；
  量化权重经 `_bf16a_linear` 分派反量化为 BF16 后走既有计算契约。
  离线 round-trip / 打包 / 合成模型对照 61 项通过。
- **8B INT8 GPU 驻留达成**：量化树 8.22 GiB，VRAM 总用量 15.4 GiB；
  prefill/decode argmax 与 HF BF16 全对，logits max-abs 1.91
  （mean 0.12）。
- 大权重反量化的 OOM 教训：14B 的 lm_head 单块 BF16 反量化（1.49 GiB）
  在内存池 99% 时直接 OOM——量化线性层改为**按输出行 8192 分块**
  反量化 + 分块 gemm，且 INT4 解包融合进单个广播消除全宽 F32 中间体。
- 分块的副作用（如实记录）：CUBLAS 按矩阵形状选 kernel，split-K 归约
  顺序变化使 BF16 logits 漂移 ±ulp——8B INT8 greedy 从单块实现的
  16/16 变为分块实现的 **14/16**（第 15 个 token 近平局翻转；prefill
  与 decode argmax 仍全对）。量化后 logit 边距收窄，这类翻转是量化
  相似度指标的固有噪声，按测量值冻结。

### 2026-07-26：14B INT4 与量化质量边界

- **14B INT4 g128 GPU 驻留达成**：量化树 8.38 GiB、VRAM 总用量
  15.2 GiB——原本 55 GiB（F32）/ 27.5 GiB（BF16）的模型跑进了
  16.3 GiB 显卡。prefill logits argmax 与 HF BF16 一致。
- **质量如实记录**：16 步 greedy 一致率 **4/16**（第 5 个 token 首次
  分歧），decode argmax 不一致，logits mean-abs 0.89（8B INT8 为
  0.12）。plain RTN INT4 的噪声已达 14B logit 边距量级，greedy 轨迹
  在几步后混沌发散——驻留问题解决了，无校准 INT4 的生成保真是明确
  的下一个边界。
- 两个否定结果也记录在案：g64（更小组、理论误差更小）一致率反而
  0/16——量化噪声与边距同量级时 agreement 是近平局抽签，组大小的
  微小变化即可翻转轨迹；混合精度（down/o_proj INT8）三次尝试均因
  宿主内存竞争 OOM KILL，未获得数据（逐投影低峰值加载器已就位，
  留待空闲内存窗口）。
- 反量化吞吐（0.11—0.61 tok/s）远低于 BF16 eager——每 token 重复
  反量化全部权重是带宽瓶颈，本周非目标（驻留验证），量化 gemm /
  缓存化留待后续。

### 2026-07-26：验证与 Close

- 默认离线全套 `4870 / 4870` 通过（Chapter 16 离线专项 `78 / 78`：
  round-trip 48 + 量化前向 13 + 资产 contract 17），Chapter 05—15 计数
  与各自 Close 时一致，无回归。
- GPU/XLA 实测由三个验证脚本冻结进 fixture；量化 GPU 验证为独占
  任务并有宿主内存前置检查，protocol 记入 `../../local_model_assets.md`。

## Close 回顾

- **完成了什么**：两条线均达标。(1) XLA BF16 static-cache decode 编译
  完成，设备端 greedy 闭环 steady **246 tok/s**（eager 的 16.1 倍），
  16 步 greedy 与 HF BF16 完全一致；(2) RTN 量化让 **8B（INT8，
  8.22 GiB）与 14B（INT4，8.38 GiB）首次驻留 16.3 GiB GPU**，8B token
  级行为近乎无损（argmax 全对、greedy 14/16 仅近平局），14B INT4 的
  质量损失被精确量化而非掩盖。
- **验证证据**：XLA 三个编译产物与吞吐冻结进 fixture；两个量化配置的
  VRAM/parity/greedy 一致率/分歧位置冻结；离线 61 项量化语义测试 +
  默认全套回归。
- **没有完成及原因**：14B 混合精度实验因宿主内存竞争三次 OOM 未获
  数据；量化吞吐优化（量化 gemm、反量化缓存）、无损化 INT4（需校准
  类方法，本周非目标）、32B 驻留均留待后续。
- **最重要的认知变化**：其一，编译推理的瓶颈从来不在 kernel 而在
  宿主往返——argmax 进 executable、token/position 设备闭环一步把
  53 ms 压到 4 ms；其二，量化的"驻留问题"和"保真问题"是两个独立
  问题：INT8 几乎白拿显存减半，而无校准 INT4 的噪声一旦达到 logit
  边距量级，greedy 一致率就退化为近平局抽签（g64 比 g128 更差即为
  明证），此时诚实的指标是一致率+分歧位置而不是一个丢失信息的
  "通过/失败"；其三，GPU 上任何形状变化（分块）都可能换 kernel 改
  归约顺序，BF16 下的"确定性"只对固定形状成立。
- **是否满足 Close 条件**：是。XLA 吞吐 ≥10× 达成且 greedy 全对；
  量化离线测试、8B/14B 驻留与行为记录、默认回归、边界文档均落实。
