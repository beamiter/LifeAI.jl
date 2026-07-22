# Week 07 — HuggingFace Weight Loading and Qwen3 Logits Parity

> 状态：Open
>
> 开启记录：2026-07-22
>
> 依赖基线：[`Week 06 — GQA, QK-Norm and Qwen3 Structural Parity`](week06_gqa_qwen3_parity.md) 已 Closed。
>
> 长期目标：本 Week 是「复现 Qwen3 → HF 权重加载 → 推理验证」三阶段计划的第二步。Week 06 已完成结构 parity；Week 07 接入真实权重并完成 token-id→logits 数值对齐；Week 08 再导入 HF tokenizer，完成 text→text 端到端验证。

## 已确认的执行边界

1. **参照模型**：以 HuggingFace `Qwen/Qwen3-0.6B` dense 模型为真实权重验收对象；结构字段与参数名以 [`qwen3_hf_config_mapping.md`](qwen3_hf_config_mapping.md) 为输入契约。
2. **计算精度**：读取官方 BF16 权重后转换为 Float32；数值对齐以相同 BF16 权重、Float32 计算的 HuggingFace reference 为基准，不把 BF16 与 Float32 计算差异混入架构判断。
3. **Tokenizer 边界**：本 Week 只接受固定 token-id fixture。HF 0-based token id 在进入 LifeAI embedding 前显式转换为 1-based；`tokenizer.json`、regex pre-tokenization、byte↔unicode 与 special-token 文本语义属于 Week 08。
4. **下载边界**：默认测试不联网、不下载真实模型。真实 Qwen3-0.6B 对齐通过显式环境变量与本地模型目录 opt-in；小型合成 safetensors 与 reference fixture 纳入默认测试。
5. **兼容策略**：现有 RoPE interleaved 配对保持默认；Qwen3 使用新增的 `rope_style=:rotate_half`。旧 constructor、checkpoint、full forward、dynamic/static KV Cache 与 XLA 路径不得静默改变。
6. **格式范围**：支持 safetensors 中本阶段需要的 BF16/F32 张量、严格 header/offset/shape 校验、单文件模型；若目录提供 `model.safetensors.index.json`，加载器应支持按索引读取分片。拒绝 pickle 权重。
7. **模型范围**：只承诺 Qwen3 dense CausalLM；不实现 MoE、量化、sliding-window attention、FlashAttention、分布式加载或训练。

## Open：核心问题

> 能否在保持 LifeAI.jl 历史模型与推理路径兼容的前提下，安全读取 HuggingFace Qwen3-0.6B 的 config 与 BF16 safetensors，把每个参数无歧义地映射到 Lux 参数树，并让固定 token IDs 的逐层 hidden states、full-forward logits 和 KV-cache decode logits 与 HuggingFace Float32 reference 在明确容差内一致？

Week 06 已证明结构 shape、GQA head 路由和 QK-Norm 位置一致，但“结构同名”不等于“数值同构”。本 Week 必须把 RoPE 配对、tensor 存储顺序、embedding 索引基准、tied lm head 与 cache 绝对位置逐项钉死；最终验收依据是逐层数值证据，而不是成功读到权重或只对上 shape。

## 预期接口

```julia
model = load_hf_qwen3_model(
    model_dir;
    max_seq_len=256,
    weight_dtype=Float32,
)

# 返回可直接用于 eager / dynamic cache / static cache 的 Lux 模型数据。
model.model
model.parameters
model.states
model.config

# HF token ids 是 0-based；显式转换后才进入 LifeAI。
tokens = hf_token_ids([1, 9707, 13])
logits, states = model.model(tokens, model.parameters, model.states)
```

底层能力保持可独立测试：

```julia
config = load_hf_qwen3_config("config.json"; max_seq_len=256)
tensors = load_safetensors("model.safetensors"; target_dtype=Float32)
parameters = load_hf_qwen3_parameters(model, tensors)
```

约束：

- `rope_style` 仅接受 `:interleaved` / `:rotate_half`，并进入 `gpt_config` 与 checkpoint round-trip；旧 config 缺省为 `:interleaved`。
- `load_hf_qwen3_config` 必须验证 `model_type=qwen3`、dense/full-attention/no-dropout 等当前能力边界，并允许调用方将 `max_seq_len` 限制为不大于 `max_position_embeddings` 的正整数。
- 参数加载必须报告 missing / unexpected / duplicate / shape mismatch / unsupported dtype，不允许用初始化随机值悄悄补齐。
- tied 模型复用 `embed_tokens`；untied 模型必须存在并加载 `lm_head.weight`。
- safetensors 返回 Julia 语义 shape 的数组；row-major 文件字节到 column-major Julia 数组的转换由格式层完成，模型映射层只处理 HF/Lux 语义轴差异。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| RoPE rotate_half | 模型 / 工程 | 可配置 `rope_style`，覆盖 eager 与两类 cache/XLA 调用 | 与手写/HF reference 一致；interleaved legacy fixture 逐元素不变 | 计划中 |
| HF config 解析 | 模型 / 工程 | `config.json` → `GPTModel` / `gpt_config`，含严格能力校验 | Qwen3 缩小 config 映射正确；不支持字段立即报错 | 计划中 |
| safetensors 基础读取 | 工程 / 学习 | header、offset、shape、BF16/F32、单文件与 index 分片读取 | 合成文件逐元素还原；损坏/越界/重叠/错误 dtype 被拒绝 | 计划中 |
| HF 参数映射 | 模型 / 工程 | embedding、28 层 attention/MLP/norm、final norm、tied/untied head → Lux 参数树 | 每类 tensor 使用非对称数值 fixture 验证 shape 与前向语义；missing/unexpected 为错误 | 计划中 |
| token-id 边界 | 模型 / 工程 | 0-based HF ids → 1-based LifeAI ids 的显式 helper | 边界 0 / vocab-1 正确，负数/越界拒绝；logits vocabulary 轴对应关系写入测试 | 计划中 |
| HF reference fixture | 学习 / 工程 | 固定 token ids、逐层 hidden states、final norm、logits 与 cache decode 参考数据/导出说明 | fixture 可离线复现，记录 transformers/model revision 与 dtype 口径 | 计划中 |
| Qwen3-0.6B 对齐 | 模型 / 工程 | 真实模型 opt-in integration test 与示例 | 逐层 hidden states、full logits 在记录的 atol/rtol 内一致 | 计划中 |
| KV Cache 对齐 | 模型 / 工程 | HF 与 LifeAI prefill/decode reference | full / dynamic / static 的逐位置 logits 一致，下一 token argmax 一致 | 计划中 |
| checkpoint 与回归 | 工程 | `rope_style` config 迁移、加载后 checkpoint round-trip | Week 01—06 默认测试通过；旧 checkpoint logits 不变 | 计划中 |
| 内存与运行记录 | 工程 | BF16→F32 加载峰值、模型参数量、首次/稳态推理记录 | 不保留无必要的完整权重副本；结果写入本 Week 实施记录 | 计划中 |

## 推进顺序

```text
冻结 legacy RoPE / checkpoint fixture
    ↓
rotate_half reference → eager → dynamic/static/XLA cache
    ↓
HF config 严格解析 + 0-based token-id 边界
    ↓
safetensors header / dtype / 布局读取
    ↓
逐类参数映射数值 fixture
    ↓
逐层 HF reference（embedding → blocks → final norm → logits）
    ↓
Qwen3-0.6B full forward 对齐
    ↓
KV-cache decode 对齐 + checkpoint round-trip
    ↓
默认回归、opt-in integration、内存记录与 Close 回顾
```

## Close 条件

只有以下条件全部满足后才能关闭本阶段：

- `rope_style=:rotate_half` 与独立 reference 数值一致，并覆盖 full forward、dynamic KV Cache、static KV Cache；默认 `:interleaved` 的现有 fixture 与旧 checkpoint logits 不变。
- Qwen3 config 的全部结构字段正确转换；不支持的 model type、sliding window、attention dropout、非法 `max_seq_len` 等输入有明确错误。
- safetensors BF16/F32 单文件读取与 index 分片路径有离线测试；header 长度、JSON、offset、shape、文件边界、dtype 错误均被拒绝。
- embedding、RMSNorm、Q/K/V/O、QK-Norm、gate/up/down、final norm、tied/untied lm head 的参数映射全部有非对称数值测试，不只检查 shape。
- loader 对 missing、unexpected、duplicate 和 shape mismatch 参数 fail closed；加载完成后没有任何随机初始化参数残留。
- 固定 token IDs 下，LifeAI 与 HuggingFace Qwen3-0.6B reference 的逐层 hidden states、final hidden state 和 logits 在文档记录的 Float32 `atol`/`rtol` 内一致。
- 同一 fixture 的 full forward、dynamic cache、static cache 与 HF cache decode 逐位置 logits 在容差内一致，下一 token argmax 一致。
- 真实模型测试显式 opt-in、默认离线套件不下载权重；真实权重路径缺失时给出可操作提示而不是让默认测试失败。
- 加载后的 Qwen3 参数能够保存为现有 checkpoint、恢复后 logits 一致；旧 checkpoint 缺失 `rope_style` 时迁移为 `:interleaved`。
- 默认测试全部通过；显式 XLA smoke 至少覆盖 `rotate_half` 的编译 full/decode 路径；记录运行命令、通过数与真实模型 revision。

## 学习重点

- **要理解的概念**：safetensors 的 8-byte header length、JSON metadata 与相对 data offsets；BF16 位级转换；row-major 文件存储与 Julia column-major 数组的语义重建；HF `rotate_half` RoPE；PyTorch/Lux parameter tree 与 tied weights。
- **要亲手实现的关键组件**：严格 safetensors reader、Qwen3 config validator、HF→Lux 参数树映射、逐层 reference 捕获与差异定位、token-id 索引边界。
- **要验证的假设**：Week 06 结构 parity 后无需改动 block 计算顺序；解决 RoPE/布局/索引差异即可使 Qwen3-0.6B logits 对齐；Float32 计算足以把 BF16 权重的跨框架误差控制在稳定容差内。

## 非目标

- 不导入 `tokenizer.json`，不承诺任意文本到 token ids 的 HF 一致性（Week 08）。
- 不实现 Qwen3 MoE、量化权重、GGUF、PyTorch pickle、FlashAttention、sliding window 或分布式加载。
- 不训练或微调 Qwen3-0.6B，不评价语言质量。
- 不要求默认 CI 保存或下载约 0.6B 真实参数；默认测试只使用小型离线 fixture。
- 不因真实模型接入而改变 LifeAI 的 1-based 公共 token API；边界转换必须显式。

## 风险与取舍

- **RoPE shape 对但值全错**：`rotate_half` 与 interleaved 的 cache shape 完全相同，必须用非对称输入和非零位置 fixture 才能发现。
- **“转置”概念混用**：文件字节序转换与模型语义轴转换是两件事；格式层先返回语义 shape，参数层再按 embedding/linear 契约映射，避免重复转置。
- **错误被最终 logits 放大**：只比最终输出定位困难，因此 reference 从 embedding 与每层 residual 输出开始逐级比较，首次分叉即停止归因。
- **内存峰值**：0.6B BF16 权重约 1.2 GB，Float32 参数约 2.4 GB；加载器应避免同时保留随机参数、BF16 全量副本和 Float32 全量副本，真实测试记录峰值。
- **大文件不可纳入仓库**：真实模型与大 logits 不提交；仓库只保存小 fixture、revision/checksum、生成脚本或明确的复现命令。
- **容差漂移**：容差必须由 Float32 reference 实测后冻结，并同时记录 max-abs / mean-abs / argmax；不以放宽容差掩盖结构错误。

## 实验与过程记录

按推进顺序记录 reference 版本、固定 token ids、tensor shape、首次数值分叉层、最大/平均绝对误差、容差、cache 位置与加载内存。每次只改变一个布局或语义假设，避免最终 logits 对不上时同时猜测多个原因。

## Close 回顾

- **完成了什么**：待 Close。
- **验证证据**：待 Close。
- **没有完成及原因**：待 Close。
- **最重要的认知变化**：待 Close。
- **是否满足 Close 条件**：否，当前为 Open。
- **带到下一 Week 的问题**：HF tokenizer 导入与 text→text 端到端一致性（Week 08）。
