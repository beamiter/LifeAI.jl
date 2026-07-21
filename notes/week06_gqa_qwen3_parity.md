# Week 06 — GQA, QK-Norm and Qwen3 Structural Parity

> 状态：Open
>
> 开启记录：2026-07-21
>
> 依赖基线：[`Week 05 — Versioned Tokenizers and Chinese Data Pipeline`](week05_tokenizer_data_pipeline.md) 已 Closed。
>
> 长期目标：本 Week 是「复现 Qwen3 系列并从 HuggingFace 加载权重验证」三阶段计划的第一步。Week 06 完成结构 parity，Week 07 计划完成 safetensors 权重加载与 logits 对齐，Week 08 计划完成 HF tokenizer 导入与 text→text 端到端验证。

## 已确认的执行边界

1. **范围**：本 Week 聚焦 Grouped-Query Attention、QK-Norm 和 Qwen3 dense 结构所需的配置能力；不加载真实 HuggingFace 权重，不导入 HF tokenizer，不加入 MoE、量化、FlashAttention 或分布式训练。
2. **兼容策略**：`n_kv_heads` 缺省等于 `n_heads`，`use_qk_norm` 缺省为 `false`；现有全部调用、Week 03 / 04 / 05 checkpoint 和默认训练示例在缺省配置下行为与 logits 完全不变。
3. **正确性策略**：GQA 先以显式 repeat-KV 的 reference 实现建立语义，再验证高效实现与 reference 逐元素一致；`n_kv_heads == n_heads` 必须精确退化为现有 MHA。
4. **KV Cache 策略**：动态与静态 cache 只存储 `n_kv_heads` 份 K/V；full forward、动态 cache、静态 cache 三路 correctness matrix 必须扩展覆盖 GQA 配置。
5. **对照口径**：GQA 的收益以 KV cache 内存占用和 decode 吞吐衡量，quality 对照沿用 Week 05 的 bits-per-byte 与固定 seed 口径；不为 GQA 调整其他超参数。
6. **参考架构**：以 Qwen3-0.6B 的结构字段为 parity 目标（GQA 16Q/8KV、head_dim=128 独立于 d_model、QK-Norm、RMSNorm、SwiGLU、no bias、rope_theta=1e6、tied embedding）；本 Week 用缩小的同构配置验证，不追求真实规模。

## Open：核心问题

> 能否在不改变现有 MHA 默认行为和历史 checkpoint 的前提下，加入 GQA 与 QK-Norm，使模型结构与 Qwen3 dense 完全同构，并让三路 KV Cache correctness matrix、checkpoint round-trip、XLA 路径和 benchmark 体系全部覆盖新配置？

Week 04 已经证明结构开关可以独立配置和对照，Week 05 冻结了文本入口。Week 06 把结构对齐到 Qwen3：这是权重加载（Week 07）的前置条件——只有每个模块的形状、归一化位置和计算顺序与 Qwen3 一致，safetensors 权重才有落点。因此本 Week 的验收核心是结构正确性与兼容性，而不是训练质量。

## Qwen3 dense 结构对照

| 结构要素 | Qwen3 dense | LifeAI.jl 现状 | 本 Week 动作 |
| --- | --- | --- | --- |
| RMSNorm (pre-norm) | ✓ eps=1e-6 | Week 04 已有 | 校验 eps 可配 |
| SwiGLU MLP | ✓ | Week 04 已有 | 无 |
| RoPE theta=1e6 | ✓ | `rope_theta` 可配 | 无 |
| head_dim 独立于 d_model | ✓ (128) | `head_dim` 可配 | 无 |
| attention 无 bias | ✓ | `use_bias=false` | 无 |
| tied embedding | ✓ (0.6B/1.7B/4B) | Week 04 已有 | 无 |
| **GQA** | ✓ 16Q/8KV | 无 | **实现** |
| **QK-Norm** | ✓ per-head RMSNorm | 无 | **实现** |

## 建议接口

```julia
MultiHeadAttention(d_model, n_heads;
    n_kv_heads=n_heads,      # n_heads % n_kv_heads == 0
    head_dim=nothing,
    use_qk_norm=false,       # per-head RMSNorm on Q/K before RoPE
    qk_norm_eps=1f-6,
    use_bias=false, ...)

gpt_config(...; n_kv_heads=n_heads, use_qk_norm=false, ...)
```

兼容约束：

- Q 投影输出 `n_heads * head_dim`，K/V 投影输出 `n_kv_heads * head_dim`；`n_heads` 必须能被 `n_kv_heads` 整除，违反立即报错。
- QK-Norm 在 head reshape 之后、RoPE 之前施加，对每个 head 的 `head_dim` 维做 RMSNorm；Q 和 K 使用独立可学习 scale（对应 Qwen3 的 `q_norm` / `k_norm`）。
- KV cache（动态与静态）按 `n_kv_heads` 分配；attention 计算时按 group 广播或 repeat，两种路径结果必须一致。
- checkpoint config 新增 `n_kv_heads`、`use_qk_norm`、`qk_norm_eps` 字段；旧 checkpoint 缺省迁移为 `n_kv_heads=n_heads`、`use_qk_norm=false`，恢复后 logits 不变。
- 为 Week 07 预留：`gpt_config` 字段与 HF `config.json` 字段（`num_key_value_heads`、`head_dim`、`rope_theta`、`rms_norm_eps`、`tie_word_embeddings` 等）的对应关系写入文档，作为权重加载的契约草案。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 冻结 MHA 回归 fixture | 工程 | 现有配置下 attention 输出与 checkpoint logits fixture | 固定 seed 下新代码对旧配置输出逐元素不变 | 计划中 |
| GQA reference 实现 | 模型 / 学习 | 显式 repeat-KV 的参考实现与语义测试 | `n_kv_heads==n_heads` 精确等于 MHA；MQA（`n_kv_heads=1`）边界正确 | 计划中 |
| GQA 高效实现 | 模型 | 分组广播实现、K/V 投影缩减 | 与 reference 逐元素一致；参数量按预期减少 | 计划中 |
| QK-Norm | 模型 / 学习 | per-head RMSNorm、独立 q/k scale、位置在 RoPE 前 | 与手写 reference 一致；关闭时输出不变 | 计划中 |
| KV Cache 适配 | 模型 / 工程 | 动态 / 静态 cache 按 `n_kv_heads` 存储 | full / dynamic / static 三路 correctness matrix 覆盖 GQA + QK-Norm 配置 | 计划中 |
| checkpoint 与迁移 | 工程 | 新 config 字段、旧 checkpoint 缺省迁移 | 旧 checkpoint 恢复 logits 不变；新配置 round-trip / resume / generate | 计划中 |
| XLA 路径 | 工程 | GQA + QK-Norm 的 Reactant 训练与静态 cache 生成 | XLA smoke：shape 稳定、与 CPU 结果一致 | 计划中 |
| qwen3_shape 端到端 | 学习 / 工程 | 缩小的 Qwen3 同构配置（GQA + QK-Norm + 全部 modern 开关）示例 | train → validate → save → load → resume → cached generate 全通 | 计划中 |
| GQA benchmark | 工程 | cache 内存与 decode 吞吐对照脚本 | 固定配置下报告 KV cache 大小减半与 decode 收益；correctness 为 true | 计划中 |
| HF config 契约草案 | 工程 / 数据 | `gpt_config` ↔ HF `config.json` 字段映射文档 | 字段覆盖 Qwen3-0.6B config；歧义项（layout、dtype）显式标注 | 计划中 |

## 推进顺序

```text
冻结 MHA / checkpoint 回归 fixture
    ↓
GQA reference（显式 repeat-KV）
    ↓
GQA 高效实现 + 等价性验证
    ↓
QK-Norm（reference → 集成）
    ↓
KV Cache（动态 / 静态）按 n_kv_heads 适配
    ↓
checkpoint 字段 + 迁移
    ↓
XLA smoke + qwen3_shape 端到端
    ↓
GQA benchmark + HF config 契约草案
```

## Close 条件

只有以下条件全部满足后才能关闭本阶段：

- 缺省配置（`n_kv_heads=n_heads`、`use_qk_norm=false`）下，全部现有测试通过且旧 fixture logits 逐元素不变。
- GQA 高效实现与 repeat-KV reference 在多组 `(n_heads, n_kv_heads)` 组合（含 MQA 边界）下逐元素一致；非法整除组合立即报错。
- QK-Norm 与手写 reference 一致，位置固定在 head reshape 之后、RoPE 之前，q/k scale 独立可学习。
- full forward、动态 KV Cache、静态 KV Cache 三路 correctness matrix 覆盖至少一个 GQA 配置和一个 GQA + QK-Norm 配置，全部通过。
- 动态与静态 cache 的内存布局只包含 `n_kv_heads` 份 K/V，并有测试断言其形状。
- Week 03 / 04 / 05 checkpoint 可恢复且 logits 不变；含 GQA + QK-Norm 的新 checkpoint 完成 round-trip / resume / cached generation。
- GQA + QK-Norm 组合在至少一个 XLA backend 上完成训练与生成 smoke，且与 CPU 结果一致。
- 缩小的 qwen3_shape 配置完成端到端示例；benchmark 报告 KV cache 内存与 decode 吞吐对照。
- `gpt_config` ↔ HF `config.json` 字段映射文档完成，覆盖 Qwen3-0.6B 全部结构字段。
- 默认测试全部通过；长时 benchmark 保持显式 opt-in。

## 学习重点

- **要理解的概念**：GQA / MQA / MHA 的谱系与 KV cache 内存权衡；QK-Norm 对训练稳定性的作用及其与 QKV bias 的替代关系；head_dim 与 d_model 解耦的意义；HF 权重布局（row-major `(out, in)`）与 Julia 列主序的对应。
- **要亲手实现的关键组件**：repeat-KV reference、分组广播 attention、per-head QK-Norm、GQA KV cache 布局、checkpoint 迁移。
- **要验证的假设**：GQA 在 `n_kv_heads==n_heads` 时可精确退化；cache 内存按 `n_kv_heads / n_heads` 缩减且 decode 吞吐可测量提升；结构 parity 完成后 Week 07 的权重加载不再需要改动模块计算顺序。

## 非目标

- 不下载或加载真实 HuggingFace 权重（Week 07）。
- 不实现 safetensors / bfloat16 读取或 HF 参数名映射（Week 07）。
- 不导入 HF tokenizer.json，不做 GPT-2 byte↔unicode 映射或 regex pre-tokenization（Week 08）。
- 不加入 MoE、量化、FlashAttention、sliding window、分布式训练或新 optimizer。
- 不为 GQA 配置调整训练超参数以追求更好的 tiny corpus 数字。

## 风险与取舍

- GQA 的 repeat / broadcast 两种实现路径在自动微分和 XLA 编译下行为可能不同；以 reference 等价性测试而不是抽查数值锁定语义。
- QK-Norm 的位置（RoPE 前 vs 后）和作用维度（per-head vs 全宽）在不同模型谱系中不一致；必须按 Qwen3 语义固定并写入测试，否则 Week 07 权重对不上。
- 静态 cache 形状依赖 `n_kv_heads`，会改变 XLA executable shape；benchmark 必须区分 cold compile 与 steady-state，避免把编译差异当成 GQA 收益。
- 旧 checkpoint 迁移若默认值处理不当，会静默改变历史模型行为；迁移必须有 logits 逐元素回归测试。
- tiny 配置下 GQA 的 quality 影响不可外推；本 Week 只对 cache 内存和吞吐下结论，quality 差异记录但不排名。

## 实验与过程记录

按推进顺序记录 reference 公式、等价性测试结果、cache 形状断言、checkpoint 迁移证据、XLA smoke 输出和 benchmark 数据。所有对照先写清比较单位与固定变量。

## Close 回顾

- **完成了什么**：
- **验证证据**：
- **没有完成及原因**：
- **最重要的认知变化**：
- **是否满足 Close 条件**：
- **带到下一 Week 的问题**：
