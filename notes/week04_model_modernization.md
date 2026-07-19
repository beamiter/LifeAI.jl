# Week 04 — Modern GPT Building Blocks

> 状态：Closed
>
> 关闭记录：2026-07-19
>
> 依赖基线：[`Week 03 — Reproducible Training and Evaluation`](week03_reproducible_training.md) 已 Closed。

## 已确认的执行边界

2026-07-18 审阅通过，按以下边界执行：

1. **范围**：本 Week 只做 RMSNorm、SwiGLU、embedding / LM head 权重共享及其兼容性与对照实验；不提前加入 BPE、GQA、真实语料或 agent 能力。
2. **兼容策略**：默认构造仍是 LayerNorm + GELU + untied head；Week 03 的模型配置和 checkpoint 必须可加载。
3. **实验矩阵**：所有单变量组合做 correctness 与 CPU 对照，四后端只比较 legacy baseline 与三项全开的 modern 组合，避免产生 20 组高成本编译实验。
4. **SwiGLU 宽度**：显式 `mlp_hidden_dim` 始终优先；未显式指定时，GELU 保持 `4d`，SwiGLU 建议使用接近参数量匹配的 `round(Int, 8d / 3)`，并在结果中记录实际参数量。

## Open：核心问题

> 能否把 RMSNorm、SwiGLU 和 embedding / LM head 权重共享作为三个可独立切换、可恢复、可比较的模型能力加入现有 GPT，同时保持 legacy baseline 以及 full forward、KV Cache 和 XLA 路径一致？

Week 03 已经解决实验“能否复现和比较”的问题。Week 04 使用该基线做受控结构实验：每次只改变一个组件，先证明公式、梯度和接口正确，再观察参数量、训练行为和推理性能变化，不因 tiny corpus 上的一次 loss 波动宣称模型质量提升。

## 预期结果

本阶段 Close 时，应当可以展示或验证：

1. `GPTModel` 通过独立配置选择 LayerNorm / RMSNorm、GELU MLP / SwiGLU MLP、untied / tied output projection；旧调用的默认行为不变。
2. 三个新能力分别有 reference formula、shape、参数树、有限梯度和至少一个训练 step 的测试。
3. baseline、三个单变量变体和 modern 组合均通过 full forward / dynamic cache / static cache logits 对齐。
4. modern 组合可以完成 train → validate → save → load → resume → generate，并能迁移加载 Week 03 checkpoint。
5. 形成固定配置下的参数量、validation loss / perplexity、训练吞吐、prefill / decode 延迟与 checkpoint 大小对照表。

## 建议配置接口

```julia
GPTModel(
    vocab_size,
    d_model,
    num_heads,
    num_layers;
    norm_type=:layernorm,    # :layernorm | :rmsnorm
    mlp_type=:gelu,          # :gelu | :swiglu
    tie_embeddings=false,
    mlp_hidden_dim=nothing,
    norm_epsilon=1.0f-5,
    ...
)
```

设计约束：

- 三个新开关的默认值必须重建 Week 03 架构，避免“升级依赖即改变 baseline”。
- `norm_type` 同时控制每个 block 的两个 pre-norm 和 final norm，`norm_epsilon` 对两类 norm 保持相同语义。
- SwiGLU 使用 `down(silu(gate(x)) .* up(x))`；`mlp_hidden_dim` 表示 gate / up 的共同中间宽度。
- tied 模式只有一份 embedding / output kernel；若 `use_bias=true`，只保留独立 output bias，不复制 output kernel。
- full forward、dynamic cache 和 static / XLA cache 复用同一个 output projection helper，避免三条路径分别实现 tying 语义。
- checkpoint schema 升级时提供显式 v1 → v2 迁移；旧配置缺少新字段时按 legacy 默认值解释，不静默猜测其他值。

## 对照矩阵

| 名称 | Norm | MLP | Output | 作用 |
| --- | --- | --- | --- | --- |
| `baseline` | LayerNorm | GELU | untied | Week 03 固定参考 |
| `rmsnorm_only` | RMSNorm | GELU | untied | 隔离 normalization 变化 |
| `swiglu_only` | LayerNorm | SwiGLU | untied | 隔离 gated MLP 变化 |
| `tied_only` | LayerNorm | GELU | tied | 隔离参数共享变化 |
| `modern` | RMSNorm | SwiGLU | tied | 验证组合兼容性 |

比较分两层：

- **完整单变量矩阵**：五组配置都运行公式 / 梯度 / train-step / cache correctness，并在 CPU Zygote 上做固定步数、固定数据划分的训练与 validation 对照。
- **后端兼容矩阵**：只比较 `baseline` 与 `modern`，覆盖 CPU、CUDA GPU、XLA CPU、XLA GPU；分别记录 cold compile、warm-up 和 steady-state，不把编译时间混入吞吐。

训练结果至少使用 3 个固定 seed，报告均值与范围；性能结果使用同一 seed、输入 shape、warm-up 数和正式样本数。所有表格同时记录实际参数量和 MLP hidden width，避免把参数规模差异误当成组件收益。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 固化 legacy baseline | 工程 | 配置、参数量与输出结构 fixture | 旧构造默认配置不变，legacy 测试与新增 fixture 全部通过 | 已验证 |
| RMSNorm | 模型 / 学习 | 可复用 Lux layer、block / final norm 接入和原理记录 | 与手算公式对齐；shape / dtype 正确；梯度有限；epsilon 生效 | 已验证 |
| SwiGLU | 模型 / 学习 | gated MLP layer、类型化 MLP 构造与宽度策略 | 与手算公式对齐；三投影参数树正确；显式/默认 hidden width 可复现 | 已验证 |
| embedding / LM head 权重共享 | 模型 | 单一共享 kernel、可选 output bias、统一 projection helper | 无重复 kernel；参数量按预期减少；输入与输出两条梯度汇入共享参数 | 已验证 |
| 配置与 checkpoint 迁移 | 工程 | `gpt_config` 新字段、checkpoint v2、v1 迁移测试 | legacy checkpoint 恢复 logits；modern checkpoint 可 round-trip / resume | 已验证 |
| 推理路径兼容 | 模型 / 工程 | full、dynamic、static / XLA 共用的 norm / MLP / projection 语义 | 五组配置的 eager cache 对齐；modern XLA 训练与解码编译对齐 | 已验证 |
| 可控训练与性能实验 | 工程 / 学习 | 五组 CPU A/B 与 baseline-vs-modern 四后端报告 | 固定数据、seed、step、shape；记录参数量、loss、ppl、吞吐、延迟和冷启动 | 已完成 |
| 示例与文档 | 学习 | modern 配置最小示例、设计取舍与结果记录 | train → validate → checkpoint → resume → cached generate 完整运行 | 已验证 |

## 推进顺序

```text
冻结 legacy fixture 与参数统计
    ↓
RMSNorm 独立实现与测试
    ↓
SwiGLU 独立实现与测试
    ↓
权重共享 + 输出投影路径收口
    ↓
checkpoint v1/v2 与 cache / XLA 集成
    ↓
单变量实验 + baseline / modern 四后端对照
```

每个组件在独立测试通过后再进入组合配置；如果组合失败，应先回到最小单变量配置定位，不同时调整三个实现。

## Close 条件

只有以下条件全部满足后才能关闭 Week 04：

- 默认 `GPTModel(...)` 仍构建 LayerNorm + GELU + untied head，Week 03 的现有测试与 legacy fixture 全部通过。
- RMSNorm 与 SwiGLU 分别通过手算 reference、shape、异常输入、有限梯度和 train-step 测试。
- tied 模式参数树中只有一份共享 kernel，logits 与手算投影一致，参数量相对 untied 精确减少 `vocab_size * d_model`。
- 五组配置的 full forward、dynamic cache 和 static cache logits 在明确容差内一致。
- modern 组合完成 train → validate → save → load → resume → generate；Week 03 checkpoint 通过 v1 → v2 迁移后 logits 一致。
- baseline 与 modern 至少在 Zygote CPU 和一个 XLA backend 上完成训练 / 推理 smoke；当前机器可用的其他后端结果或未覆盖原因被单独记录。
- 对照报告固定数据划分、seed、训练 step、输入 shape、warm-up 和样本数，并同时报告参数量、validation loss / perplexity、训练吞吐、prefill / decode 延迟和 checkpoint 大小。
- 默认测试全部通过；XLA 专项测试与硬件覆盖边界单独说明。
- 文档记录结论的适用范围，不把 tiny model / tiny corpus 结果外推为模型质量定论。

## 学习重点

- **要理解的概念**：LayerNorm 与 RMSNorm 的统计差异；GLU / SwiGLU 的门控机制与参数预算；输入输出 embedding tying 的优化与表示含义。
- **要亲手实现的关键组件**：RMSNorm、SwiGLU MLP、共享 output projection、checkpoint schema migration。
- **要验证的假设**：现代组件可以在保持可复现基线和推理正确性的前提下独立组合；参数量、速度与 validation 指标的变化能够被现有实验框架清楚归因。

## 非目标

- 不实现 byte-level tokenizer、BPE、GQA 或新语料管线。
- 不调整 optimizer、学习率策略、数据顺序或训练规模来“配合”某个变体。
- 不重写 Attention / KV Cache，不引入量化、分布式训练或服务框架。
- 不进入 memory、planning、tools、multimodal 或 embodied agent loop。

## 风险与取舍

- Lux 参数树天然按子层组织，权重共享若通过复制数组实现会产生两个 optimizer slot；必须验证只有一个可训练 kernel。
- cache 路径目前分别执行 final norm 与 LM head；若只修改 full forward，会出现生成路径语义漂移，因此先收口公共 projection helper。
- SwiGLU 在相同 hidden width 下比 GELU 多一组投影；实验必须使用明确的宽度策略并报告实际参数量。
- RMSNorm 的平方均值和低精度累加可能影响数值稳定性；至少在 Float32 建立强 reference，并记录低精度 / XLA 容差。
- tiny corpus 的 loss 方差可能大于组件差异；使用多个固定 seed，并把实验定位为工程基线而非质量排名。
- checkpoint schema 变化不能依赖 Julia 结构“碰巧兼容”；需要显式版本迁移和旧 fixture。

## 实验与过程记录

### 实现与兼容性

- 新增 `RMSNormLayer`：沿 `dims=1` 计算 root mean square，只有可训练 scale、没有 bias；block 内两个 pre-norm 与 final norm 使用同一 `norm_type`。
- 新增 `SwiGLU`：独立 gate / up / down 三个投影，使用 `swish(gate) .* up`；未显式指定 hidden width 时采用 `round(Int, 8d / 3)`。
- 新增 `TiedOutputProjection`：输出投影直接复用 `token_embedding.weight`，参数树中不再存在第二份 kernel；可选 output bias 保留在 `lm_head`。
- full forward、dynamic KV Cache 和 static / XLA KV Cache 统一调用 `_project_logits`，避免 tied 语义在三条路径分叉。
- `gpt_config` 新增 `norm_type`、`mlp_type`、`tie_embeddings`。默认值仍为 LayerNorm + GELU + untied，固定 seed 下 legacy 参数树和 logits 与显式 legacy 构造完全一致。
- checkpoint format 升级为 v2；loader 显式接受 v1，将缺失的新配置迁移为 legacy 默认值。测试验证迁移前后 logits、参数、optimizer state 和 step 一致。

### 自动化验证

2026-07-19 运行：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
LIFEAI_TEST_XLA=true julia --project=. -e 'using Pkg; Pkg.test()'
```

- 默认测试 765 / 765 通过，其中 Week 04 专项 111 / 111。
- 显式 Reactant/XLA 专项 30 / 30 通过；加上默认套件共 795 项。
- Week 04 专项覆盖 RMSNorm / SwiGLU reference formula、异常输入、有限梯度、共享梯度合流、精确参数量差、legacy fixture、五配置 train-step/cache matrix、modern checkpoint round-trip / resume / generation 和 checkpoint v1 → v2 迁移。
- XLA 专项覆盖 modern 组合的 XLA CPU 训练、固定 shape prefill / decode 和 full-forward logits 对齐。
- `examples/modern_gpt.jl` 实跑完成 train → validate → save → load → resume → cached generate。

### 五配置 CPU 受控实验

运行 `scripts/benchmark_week04.sh`。单变量层使用 `d_model=32`、4 heads、2 layers、序列 16、batch 4、3 个固定 seed 和 20 个训练 step：

| 配置 | MLP width | 参数量 | Checkpoint KiB | 最终 validation loss | 最终 PPL mean [range] | 训练 tokens/s |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| baseline | 128 | 27520 | 328.0 | 0.5310 | 1.701 [1.666, 1.727] | 36110.1 |
| rmsnorm_only | 128 | 27360 | 325.7 | 1.1866 | 3.316 [2.681, 3.932] | 40465.2 |
| swiglu_only | 85 | 27456 | 327.5 | 0.4412 | 1.556 [1.485, 1.621] | 34647.5 |
| tied_only | 128 | 26208 | 312.5 | 0.2400 | 1.271 [1.244, 1.296] | 39194.2 |
| modern | 85 | 25984 | 309.7 | 0.9212 | 2.516 [2.412, 2.716] | 38560.0 |

五组配置、全部 seed 的 full / dynamic / static cache correctness 均通过。该实验没有为任一变体调整 optimizer、学习率、训练步数或数据顺序。

结果没有支持“把三个现代组件全部打开就必然改善 validation”的简单结论：短训练中 `tied_only` 最好，`modern` 差于 baseline，`rmsnorm_only` 最差。tied 配置还因复用当前 embedding 初始化而具有更高的初始 PPL。它们是后续初始化与训练策略实验的线索，不是修改本 Week 固定基线来追求更好数字的理由。

### Baseline vs Modern 四后端

性能层使用 vocab 512、`d_model=64`、4 heads、2 layers、训练序列 64、batch 4、prompt 64、decode 16、3 个 warm-up 和 30 个正式样本：

| 配置 | 后端 | correctness | 参数量 | 训练 cold ms | 训练 p50 / p90 ms | 训练 tokens/s | Prefill p50 ms | Decode p50 ms/token |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | CPU | true | 164480 | 26212.47 | 9.01 / 13.10 | 28401.8 | 0.51 | 0.07 |
| baseline | CUDA GPU | true | 164480 | 50055.63 | 5.04 / 5.54 | 50843.0 | 0.89 | 0.71 |
| baseline | XLA CPU | true | 164480 | 131721.54 | 4.25 / 5.78 | 60269.4 | 0.65 | 0.04 |
| baseline | XLA GPU | true | 164480 | 137435.22 | 2.06 / 2.39 | 123988.1 | 1.27 | 0.40 |
| modern | CPU | true | 131520 | 26436.31 | 9.46 / 13.55 | 27052.5 | 0.54 | 0.08 |
| modern | CUDA GPU | true | 131520 | 50125.74 | 5.27 / 6.09 | 48539.3 | 0.99 | 0.74 |
| modern | XLA CPU | true | 131520 | 122636.78 | 4.10 / 6.33 | 62386.3 | 0.46 | 0.04 |
| modern | XLA GPU | true | 131520 | 126284.07 | 1.18 / 1.53 | 216736.5 | 0.85 | 0.30 |

modern 在该配置下减少 32960 个参数，约 20.0%。CPU 与 eager CUDA 的稳态训练吞吐略降，XLA CPU 略升，XLA GPU 明显上升；这些差异同时包含 RMSNorm、SwiGLU、权重共享和较小参数树的共同影响。它们只适用于当前 tiny workload 与当前机器，不能外推到真实模型规模。

原始 TSV、逐样本时延和日志保存在被 `.gitignore` 排除的 `benchmark_results/week04-20260718-final/`；关键配置与结果已在本文固化。

## Close 回顾

- **完成了什么**：以独立配置加入 RMSNorm、SwiGLU 和 embedding / LM head 权重共享；保留 legacy 默认；完成统一输出投影、checkpoint v2 与 v1 迁移、五配置 cache / train matrix、modern XLA 路径、端到端示例和两层 benchmark。
- **验证证据**：默认测试 765 / 765、Week 04 专项 111 / 111、显式 XLA 专项 30 / 30；五配置全部 cache-correct；baseline/modern 的 CPU、CUDA GPU、XLA CPU、XLA GPU 八组结果全部 `ok` 且 correctness 为 `true`。
- **没有完成及原因**：没有加入 BPE、GQA、真实语料、低精度专项或重新调参；前三项属于已确认非目标，低精度和调参需要独立变量与更真实训练规模，不应为改善本 Week tiny benchmark 数字而临时扩展范围。
- **最重要的认知变化**：现代组件的工程价值、参数效率、后端性能和短程 validation 表现是四个不同问题。结构能正确组合且在 XLA GPU 上更快，不代表默认超参数下的 tiny validation 一定更好；初始化也会与权重共享产生直接耦合。
- **是否满足 Close 条件**：是，Week 04 于 2026-07-19 Closed。
- **带到下一 Week 的问题**：进入 byte-level / BPE 与中文数据管线前，是否应同时冻结一套适合 tied embedding 的零中心初始化基线；Tokenizer 的 normalization、special token、版本持久化和无泄漏数据边界如何定义。
