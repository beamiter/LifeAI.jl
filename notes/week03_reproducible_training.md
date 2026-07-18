# Week 03 — Reproducible Training and Evaluation

> 状态：Closed
>
> 关闭记录：2026-07-18

## Open：核心问题

> 能否让同一个小 GPT 实验可保存、可恢复、可评估、可比较，并为后续模型结构和真实语料实验建立可信基线？

Week 02 已经跑通训练、生成和 KV Cache。Week 03 不继续堆叠模型结构，而是先补齐实验基础设施：收束增量推理、保存完整训练状态、建立无泄漏的 validation 与 perplexity，并增加梯度裁剪。

## 对初步路线的判断

整体方向合理，但不建议在同一个 Week 同时修改训练系统、Transformer 结构、Tokenizer 和训练语料。这样会一次改变过多变量，出现质量或性能变化时难以定位原因。

更合理的依赖顺序是：

```text
Week 03：可恢复、可评估、可比较
    ↓
Week 04：RMSNorm、SwiGLU、embedding/lm_head 权重共享
    ↓
Week 05：byte-level baseline、BPE 与中文数据管线
    ↓
Week 06：GQA、较真实的小型中文语料训练与综合 benchmark
```

其中 KV Cache 与增量推理已经实现。本阶段的目标是补齐 correctness matrix、公共接口和性能基线，而不是再写一套 cache。

## 预期结果

本阶段 Close 时，应当可以展示或验证：

1. 一个训练任务可以保存 checkpoint，重新加载后得到相同 logits，并从原 step 继续训练。
2. 数据按原始 token stream 划分 train / validation，不因滑动窗口重叠发生数据泄漏。
3. 可以计算 token-weighted validation loss 和 perplexity，并在训练过程中记录它们。
4. 可以启用 global gradient norm clipping，并观察裁剪前后的梯度范数。
5. full forward、动态 KV Cache 和静态 KV Cache 有一致性测试与最小性能报告。

## 计划

| 工作项 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- |
| KV Cache 与增量推理收口 | correctness / benchmark 脚本与结果记录 | eager、dynamic、static logits 对齐；区分编译时间、prefill 和 decode 指标 | 已验证 |
| Checkpoint 保存与加载 | 版本化 checkpoint payload 和 save/load API | round-trip 后 config、tokenizer、parameters、states、optimizer state、step 一致 | 已验证 |
| 断点续训 | resume API 与确定性测试 | 连续训练与中断恢复训练在受控条件下结果一致或在明确容差内一致 | 已验证 |
| Train / validation 划分 | 基于原始 token stream 的 split 工具 | 两侧窗口不跨越 split boundary，validation 不参与参数更新 | 已验证 |
| Evaluation / perplexity | 无梯度 evaluation loop | 按 token 数加权的 NLL 正确，`perplexity = exp(mean_nll)` | 已验证 |
| 梯度裁剪 | global norm clipping 配置、指标和测试 | 超阈值梯度被裁剪，未超阈值梯度保持不变 | 已验证 |
| 示例与文档 | 可中断再恢复的最小训练示例 | 从保存到恢复、评估和生成可以完整运行 | 已完成 |

## 四后端性能对比

新增 `scripts/benchmark_week03.sh`，以独立进程对比以下四条路径：

| 名称 | 训练后端 | 设备 | 推理路径 |
| --- | --- | --- | --- |
| `cpu` | Zygote | CPU | eager 固定形状 KV Cache |
| `gpu` | Zygote | NVIDIA CUDA GPU | eager 固定形状 KV Cache |
| `xla_cpu` | Reactant + Enzyme | XLA CPU | 编译后的固定形状 KV Cache |
| `xla_gpu` | Reactant + Enzyme | XLA GPU | 主表使用固定形状 KV Cache；附加 no-cache / dynamic / static 对比 |

直接运行完整对比：

```bash
./scripts/benchmark_week03.sh
```

结果写入 `benchmark_results/week03-<timestamp>/`：

- `summary.md`：四后端横向表格。
- `cpu.tsv` 等：可继续用脚本或表格软件分析的长格式原始指标。
- `cpu.log` 等：每个独立进程的完整日志。
- `status.tsv`：成功、失败或缺少硬件后端的状态。

默认指标包括：

- 训练首步延迟（包含该后端首次编译/执行）、post-compile GC、warm-up 原始耗时、稳态 p50/p90/min/max、tokens/s。
- KV Cache prefill 首次与稳态延迟。
- KV Cache 单 token decode 首次与稳态 p50/p90、tokens/s。
- XLA+GPU no-cache、dynamic cache、static cache 的 cold decode 总时间、稳态延迟/吞吐和 executable 数。
- 与 host full-forward reference 的 prefill/decode 最大绝对误差。
- 参数量、理论参数/KV Cache 字节数、进程峰值 RSS。
- `nvidia-smi` 可用时的轮询峰值 GPU 显存。
- 训练、prefill、decode 逐样本原始耗时；可用这些数组复核分位数和定位离群 iteration。

每个后端必须使用相同模型、输入 shape、随机种子和样本数。脚本默认把后端放进独立 Julia 进程，避免 CUDA、Reactant/XLA 的编译缓存和内存池互相污染。首步与稳态应分别解读，不能把 XLA 编译时间混入 steady-state 吞吐。

训练 cold step 后会先执行一次显式 GC，再运行 `LIFEAI_BENCH_WARMUP_STEPS` 个不计入稳态统计的 step（默认 3），随后再显式 GC 并开始正式采样。两次 GC 和每个 warm-up step 的耗时都会写入 TSV。这样可隔离 XLA GPU 在 cold compile/first run 后稳定出现的一次性 runtime settling；该事件属于 warm-up，不应进入 steady p90。正式采样默认 30 次，避免 10 个样本下单个离群点通过线性插值显著扭曲 p90。

跨设备 correctness 默认使用 `isapprox(atol=5e-3, rtol=5e-3)`；这是为了容纳 GPU/XLA 不同归约与融合顺序带来的浮点误差。原始最大绝对误差仍写入 TSV。需要严格复现实验时可设置 `LIFEAI_BENCH_ATOL` 和 `LIFEAI_BENCH_RTOL`。

### XLA+GPU 的三种 Cache 模式

`xla_gpu` 结果会额外生成一张三模式表：

| 模式 | 计算方式 | Shape / executable 特征 |
| --- | --- | --- |
| No cache | 每个 token 都对完整增长上下文重新 full forward | 每种 context length 一个 executable |
| Dynamic cache | 保存 K/V，但通过 `cat` 随 token 增长物理数组 | 每种 cache length 一个 decode executable |
| Static cache | 预分配 `max_seq_len` K/V，通过逻辑 position 控制有效前缀 | 一个 prefill executable + 一个可复用 decode executable |

由于 no-cache 和 dynamic cache 的 shape 每个 token 都会改变，cold pass 必须为每个位置分别编译。为防止默认 `decode_tokens=64` 意外触发 128 个以上额外编译，三模式对比默认只使用前 4 个 decode token；三种模式内部使用的 token 完全相同。正式测试可以显式放大：

```bash
LIFEAI_BENCH_BACKENDS=xla_gpu \
LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS=16 \
LIFEAI_BENCH_WARMUP_STEPS=3 \
LIFEAI_BENCH_SAMPLES=30 \
./scripts/benchmark_week03.sh
```

也可以只运行 KV Cache 示例，不包含训练 benchmark：

```bash
LIFEAI_BENCH_XLA=true \
LIFEAI_XLA_BACKEND=gpu \
LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS=8 \
julia --project=. examples/benchmark_kv_cache.jl
```

为避免第一个模式独自承担 Reactant/Lux 初始化成本，三模式开始前会用不同 batch shape 做一次公共 runtime warmup。该时间单独报告，warmup executable 不会被三种目标 workload 复用。

三模式的 cold decode 总时间包含该 decode 序列需要的全部目标 shape 编译和首次执行；steady-state 则复用 cold pass 已生成的所有 executable。每一步都会把 logits materialize 到 host 作为同步边界，以模拟逐 token 生成必须拿到本步结果后才能继续的依赖。

可以用环境变量调整实验规模。例如：

```bash
LIFEAI_BENCH_SAMPLES=30 \
LIFEAI_BENCH_WARMUP_STEPS=3 \
LIFEAI_BENCH_EMBED_DIM=256 \
LIFEAI_BENCH_NUM_HEADS=8 \
LIFEAI_BENCH_NUM_LAYERS=6 \
LIFEAI_BENCH_SEQ_LEN=256 \
LIFEAI_BENCH_BATCH_SIZE=16 \
LIFEAI_BENCH_PROMPT_TOKENS=256 \
LIFEAI_BENCH_DECODE_TOKENS=128 \
./scripts/benchmark_week03.sh
```

只跑部分后端：

```bash
LIFEAI_BENCH_BACKENDS="cpu xla_cpu" ./scripts/benchmark_week03.sh
```

建议正式采样前关闭其他高负载任务，固定 `JULIA_NUM_THREADS`、CPU governor 和 GPU power state，并至少使用 30 个稳态样本。GPU 显存是 `nvidia-smi` 的离散轮询峰值，短暂峰值可能漏采；进程 RSS 也不等价于设备显存。

## 关键设计约束

### 1. Checkpoint 保存的是实验状态，不只是模型权重

建议 checkpoint 至少包含：

- format version。
- 可重建 `GPTModel` 的模型配置。
- parameters 和 Lux states。
- optimizer rule / optimizer state 与 global step。
- Tokenizer 类型、词表和特殊 token 配置。
- 训练配置与已经完成的 epoch / batch position。
- 为确定性恢复所需的 RNG 状态或等价 seed / data-order 信息。
- 可选的最近 train / validation metrics。

设备数组应转换为 host representation 后再保存，加载时再由 `TrainerGPT.device` 放回目标设备，避免 checkpoint 与某个 XLA / GPU 设备绑定。

### 2. Validation 必须先切分原始序列

当前 `DatasetLoader` 使用滑动窗口。若先生成全部重叠窗口再随机划分，相邻样本会共享大量 token，造成 validation leakage。

正确顺序应是：

```text
raw text / token stream
    ↓ split boundary
train token stream      validation token stream
    ↓ separate windowing
train loader            validation loader
```

Tokenizer 应只从 train split 拟合；validation 中的未知内容通过明确的 unknown / byte fallback 策略处理。

### 3. Perplexity 必须按 token 聚合

不能简单平均不同 batch 的 mean loss，尤其最后一个 batch 大小可能不同。evaluation 应累加 token-level negative log-likelihood 和有效 token 数，再计算：

```text
mean_nll = total_nll / total_tokens
perplexity = exp(mean_nll)
```

### 4. KV Cache benchmark 分离编译与运行

至少分别记录：

- logits / greedy generation 与 full forward 的一致性。
- prefill latency。
- decode latency 或 tokens/s。
- cache shape / 理论存储量。
- XLA 首次编译成本与 executable 复用后的 steady-state 成本。

测试模型较小时，计时噪声可能大于真实收益，因此结果应保留输入长度、生成长度、batch size、模型配置、后端和 warmup 条件。

## Close 条件

只有以下条件全部满足后才关闭 Week 03：

- Checkpoint round-trip 后，同一输入的 logits 在明确容差内一致。
- 受控实验中，连续训练与 checkpoint resume 的 step、optimizer state 和最终 loss 一致或在明确容差内一致。
- Train / validation 在 token stream 层先切分，并有防止窗口跨界的测试。
- Validation loss 与 perplexity 有手算小样本测试。
- Global gradient norm clipping 有“发生裁剪”和“不需裁剪”两类测试。
- KV Cache correctness matrix 通过，并产出区分 warmup / compile / steady-state 的最小 benchmark。
- 默认测试套件全部通过；XLA 专项结果与未覆盖后端被单独说明。
- 最小示例可以完成 train → validate → save → load → resume → generate。

## 后续阶段建议

### Week 04：模型基本组件现代化

按可独立开关和独立测试的方式加入：

1. RMSNorm。
2. SwiGLU。
3. embedding / lm_head 权重共享。

每个组件都应与 Week 03 建立的固定 baseline 做参数量、loss、速度和兼容性比较。不要把三个变化只作为一个不可拆分的新模型版本。

### Week 05：Tokenizer 与数据管线

- Byte-level tokenizer 适合作为零 OOV、实现简单的 correctness baseline。
- 中文 UTF-8 通常一个汉字需要多个 byte token，序列会明显变长，因此它不应直接被视为最终训练方案。
- BPE 更适合较真实的中文小模型训练，但需要定义 normalization、special tokens、训练语料边界、词表持久化和确定性规则。
- 建议先用 byte-level 建立 round-trip 和 OOV 基线，再实现 BPE，并比较 tokens/character、词表大小、压缩率和 validation perplexity。

### Week 06：GQA 与较真实的小型中文训练

- GQA 会改变 K/V head 数和 cache 布局，应复用 Week 03 的 full / dynamic / static correctness 与 benchmark 框架。
- 先验证 `num_query_heads % num_kv_heads == 0`、head broadcasting、RoPE 与 cache shape，再训练。
- 中文语料实验应记录来源、清洗规则、去重、train/validation 边界、Tokenizer 版本、模型配置和 checkpoint。
- 这一阶段再比较 MHA 与 GQA 的参数量、KV Cache 大小、decode latency 和 validation perplexity，结论会更可信。

## 学习重点

- 要理解的概念：完整训练状态、确定性恢复、数据泄漏、token-weighted metrics、global gradient norm、steady-state benchmark。
- 要亲手实现的关键组件：checkpoint schema、resume path、evaluation loop、gradient clipping、KV benchmark harness。
- 要验证的假设：先建立稳定测量与恢复能力，会显著降低后续模型结构和 Tokenizer 实验的比较成本。

## 风险与取舍

- XLA device state 不一定适合直接序列化，checkpoint 必须明确 host/device 边界。
- bitwise resume 可能受设备 kernel 和并行归约影响；测试应先在确定性 CPU 路径建立强保证，再为 XLA 定义合理容差。
- 若同时加入数据 shuffle，必须把 sampler 顺序与进度纳入恢复语义；否则先保持确定性顺序，避免扩大本阶段范围。
- Benchmark 先追求可重复和可解释，不追求一次得到最终性能结论。

## 实验与过程记录

### 自动化验证

2026-07-18 运行默认测试套件：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

结果为 654 / 654 通过，其中 Week 03 新增的可复现实验测试为 57 / 57。覆盖：

- token stream 级 train / validation 划分与边界检查；
- token-weighted NLL、validation loss 与 perplexity；
- global gradient norm clipping 的裁剪与不裁剪分支；
- checkpoint round-trip、logits 一致性和确定性 resume；
- full forward、dynamic cache、static cache correctness matrix；
- benchmark 返回结构、warm-up 与 steady-state 指标。

默认命令不包含由 `LIFEAI_TEST_XLA=true` 控制的专项测试，不能仅凭 654 项默认测试声称所有 XLA 路径都已被该命令覆盖。

### 四后端基线

最新一次本机 benchmark 使用相同模型、输入 shape、随机种子，3 个 warm-up step 和 30 个正式样本；原始产物保存在被 `.gitignore` 排除的 `benchmark_results/` 中，关键结果在此固化：

| 后端 | 状态 | correctness | 训练稳态 p50 / p90 ms | 训练 tokens/s | static cache decode 稳态 ms/token |
| --- | --- | --- | ---: | ---: | ---: |
| CPU | ok | true | 376.46 / 448.06 | 2720.0 | 0.54 |
| CUDA GPU | ok | true | 9.88 / 11.20 | 103613.6 | 1.33 |
| XLA CPU | ok | true | 41.96 / 43.06 | 24405.2 | 0.22 |
| XLA GPU | ok | true | 2.76 / 3.13 | 370711.9 | 0.53 |

测试配置为 vocab 512、`d_model=128`、4 heads、4 layers、训练序列 128、batch 8、prompt 128、decode 64。该结果是当前机器上的工程基线，不外推为其他模型规模或硬件上的普遍结论。

XLA GPU 的相同 4-token decode 对比中，no-cache、dynamic cache、static cache 均与 reference 对齐；三者分别产生 5、5、2 个目标 executable，steady decode p50 分别为 1.12、1.30、0.86 ms/token。结果支持“固定 shape 能减少重编译并改善当前 workload 的 decode 稳态延迟”，同时也显示编译冷启动成本必须与稳态指标分开解释。

## Close 回顾

- **完成了什么**：建立版本化、设备无关的 checkpoint 与恢复路径；补齐无泄漏数据划分、token-weighted evaluation / perplexity、global gradient norm clipping；形成 full / dynamic / static KV Cache correctness 与四后端 benchmark；提供 train → validate → save → load → resume → generate 示例。
- **验证证据**：默认测试 654 / 654 通过，Week 03 专项 57 / 57；最新四后端 benchmark 均为 `ok` 且 correctness 为 `true`；checkpoint 测试验证 logits、参数、optimizer state、step 和恢复后 loss 的一致性。
- **没有完成及原因**：没有加入 RMSNorm、SwiGLU、权重共享、Tokenizer 或真实语料；这些是为保持变量可归因而主动留给后续 Week 的范围，不是 Week 03 缺项。默认测试命令未包含显式启用的 XLA 专项套件，其覆盖边界已单独记录。
- **最重要的认知变化**：性能结论必须同时保留 correctness reference、cold compile、warm-up、steady-state 和 shape / executable 数；“可运行”只有在训练状态、数据边界和指标口径都能复现时才成为可信实验基线。
- **是否满足 Close 条件**：是，Week 03 于 2026-07-18 Closed。
- **带到下一 Week 的问题**：如何在不破坏 legacy baseline、checkpoint 和 KV Cache / XLA 路径的前提下，独立引入并比较 RMSNorm、SwiGLU 与 embedding / LM head 权重共享。
