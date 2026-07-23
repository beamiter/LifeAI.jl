# Week 10 — GPT-2 Architecture, HuggingFace Weights and Text Parity

> 状态：Open
>
> 开启记录：2026-07-23
>
> 依赖基线：[`Week 09 — Qwen3 Sampling Fidelity and Real Inference Performance`](week09_qwen3_sampling_performance.md) 已 Closed。
>
> 近期主线：持续复现经典/SOTA 模型与论文，用独立 reference、真实权重和性能原始记录检验 LifeAI.jl 的多架构模型组装能力。

## Open：核心问题

> 在 Qwen3 dense 的结构→权重→tokenizer→生成→性能闭环已经完成后，LifeAI.jl 能否不复制一套孤立模型代码，而是扩展共享组件，严格复现经典 GPT-2 124M 的 learned absolute position、GPT-2 MLP/attention 语义、HuggingFace Conv1D 权重布局和 byte-level BPE，并得到逐层与 text-to-text parity？

GPT-2 与 Qwen3 同为 decoder-only Transformer，但关键约定并不相同：GPT-2 使用 learned absolute position embedding、LayerNorm、带 bias 的 fused QKV Conv1D、GELU-New、标准 MHA 和 GPT-2 byte-level BPE；Qwen3 使用 RoPE、RMSNorm、无 bias GQA/QK-Norm、SwiGLU 和另一套 tokenizer/chat 约定。本 Week 的价值不是再证明一次“Transformer 能运行”，而是验证 LifeAI 的共享抽象能否承载第二种真实架构，并把差异留在显式配置与严格 HF adapter 中。

## 已确认的执行边界

1. **目标模型**：以 HuggingFace `openai-community/gpt2`（124M）为目标；第一项工作先冻结完整 revision、必要文件、Transformers 版本与 checksum。未冻结 revision 前不得生成正式 reference。
2. **复现含义**：本 Week 复现论文架构和官方 checkpoint 的推理数值/文本行为；不下载 WebText、不从零预训练，也不声称复现论文中的 zero-shot quality 数字。
3. **共享组件优先**：learned absolute position、GELU-New、bias 和 LayerNorm epsilon 作为显式通用配置加入现有模型组件；GPT-2 缺省不能改变 Qwen3、legacy GPT 或历史 checkpoint。
4. **adapter 严格隔离**：新增独立 `load_hf_gpt2_bundle` / GPT-2 config-tokenizer adapter；不在 Qwen3 loader 中堆叠模型名分支。通用 safetensors、id 边界和 generation/cache 能力应复用。
5. **权重布局显式转换**：HF GPT-2 `Conv1D.weight` 采用 `(in, out)` 语义，fused `c_attn` 同时包含 Q/K/V；导入时显式 split + transpose 到 LifeAI 投影布局，并用小矩阵 fixture 钉死方向。
6. **位置边界 fail closed**：position ids 从 0 开始，learned position table 长度固定为 `n_positions`；full forward、prefill 和 decode 使用一致的绝对位置，超过上限立即报错。
7. **tokenizer 不借用 Qwen 语义**：严格实现 GPT-2 regex + byte-to-unicode + BPE merge + ByteLevel decode 与 `<|endoftext|>`；不加入 chat template，不把 GPT-2 包装成 instruction/chat 模型。
8. **reference 分层**：先用离线小 fixture 验证组件和 layout，再用冻结 checkpoint 比较 tokenizer、embedding、逐 block hidden、final norm、logits、KV cache 和 greedy text。
9. **真实资产持久化**：模型与 reference 放在 `/home/yj/models/huggingface/openai-community/gpt2/<revision>/`，不放 `/tmp`，不提交权重进仓库；仓库只保留脚本、checksum、小 fixture 和必要 benchmark。
10. **性能不混淆**：GPT-2 124M 与 Qwen3-0.6B 参数量、词表和结构不同；可记录 CPU/CUDA/XLA 数据检验框架路径，但不把跨模型 tok/s 直接解释为架构优劣。

## 目标结构契约

| 要素 | GPT-2 124M 目标 | LifeAI 当前状态 | Week 10 动作 |
| --- | --- | --- | --- |
| vocabulary / context | 50,257 / 1,024 | vocab、max_seq_len 可配 | 严格 config 校验 |
| hidden / layers / heads | 768 / 12 / 12 | 已可配 | shape contract |
| position | learned absolute | RoPE 或无位置编码 | 新增 learned absolute |
| normalization | pre-LayerNorm，eps 1e-5 | LayerNorm 已有 | 冻结 epsilon/bias 语义 |
| attention | MHA，fused QKV，bias | 运行时为独立 Q/K/V | loader split + transpose |
| MLP | 4× hidden，GELU-New，bias | GELU / SwiGLU | 新增精确 GELU-New |
| embeddings / LM head | tied | tied 已有 | 导入与 identity 检查 |
| tokenizer | GPT-2 byte-level BPE | Byte-BPE 与 Qwen tokenizer 已有 | 严格 GPT-2 adapter |
| generation | causal LM，EOS 50256 | full/dynamic/static/XLA | text/cache parity |

## 预期接口

```julia
bundle = load_hf_gpt2_bundle(
    model_dir;
    revision,
    max_seq_len=1024,
)

ids = encode(bundle.tokenizer, "Hello, 世界"; add_special_tokens=false)
text = decode(bundle.tokenizer, ids)

result = generate_hf_text(
    bundle,
    "The meaning of life is";
    strategy=:greedy,
    cache=:dynamic,
    max_new_tokens=16,
)
```

公共 token id 继续使用 LifeAI 1-based；HF reference 和文件边界显式转换为 0-based。GPT-2 adapter 必须返回 tokenizer/config/权重 checksum 与完整 revision，未知或不匹配的目标 fail closed。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 冻结 checkpoint 与论文契约 | 模型 / 学习 | model id、revision、文件清单/checksum、架构差异说明 | 与 HF config、Transformers 实现和 GPT-2 论文逐项核对 | 计划中 |
| learned absolute position | 模型 / 工程 | token + position embedding、full/cache 绝对位置 | 手算 fixture；full/prefill/decode position 完全一致；越界失败 | 计划中 |
| GPT-2 激活与 bias 契约 | 模型 / 工程 | GELU-New、LayerNorm eps、attention/MLP bias 配置 | 与 PyTorch 小张量 reference 对齐；legacy/Qwen 回归不变 | 计划中 |
| GPT-2 config 与权重导入 | 模型 / 工程 | strict config parser、Conv1D split/transpose、tied LM head | missing/unexpected/shape/dtype/layout 错误 fail closed | 计划中 |
| GPT-2 tokenizer | 数据 / 工程 | regex、byte-to-unicode、BPE、special token、decode | ASCII/Unicode/空白/控制字符 corpus 的 ids/spans/bytes 与 HF 一致 | 计划中 |
| 逐层 HF reference | 模型 / 学习 | Python exporter + Julia verifier | embedding、12 blocks、final hidden、logits 在冻结容差内 | 计划中 |
| KV cache 与 greedy parity | 模型 / 工程 | full/dynamic/static，按条件覆盖 XLA | 多 prompt 的逐步 logits/argmax/token/stop/text 一致 | 计划中 |
| 真实性能基线 | 框架 / 工程 | 124M load/prefill/decode/cache 原始记录 | 固定 prompt、decode、同步边界和 samples；correctness 必须先通过 | 计划中 |
| 文档与复盘 | 学习 | 架构映射、实验失败、结论与下一模型选择标准 | 所有结论可追溯到代码、reference、测试或原始数据 | 计划中 |

## 推进顺序

```text
冻结 revision / config / tokenizer / 论文契约
    ↓
learned absolute position + GELU-New + bias 小 fixture
    ↓
strict config + Conv1D/fused-QKV 权重映射
    ↓
token ids → embedding → 逐 block → final logits parity
    ↓
GPT-2 tokenizer parity
    ↓
full / dynamic / static greedy text parity
    ↓
CPU/CUDA/XLA 据实验证 + Close 复盘
```

## Close 条件

只有以下条件全部满足后才能关闭 Week 10：

- `openai-community/gpt2` 的完整 revision、必要文件、Transformers 版本和 SHA256 已冻结；模型/reference 均位于 `/home/yj/models/` 持久目录。
- learned absolute position、GELU-New、LayerNorm epsilon 和 bias 语义分别有独立 PyTorch/手写 fixture；旧默认、Qwen3 logits 和历史 checkpoint 无回归。
- strict GPT-2 config 与权重 loader 覆盖全部参数；HF Conv1D `(in, out)`、fused QKV split 和 tied LM head 有方向/identity 测试，异常输入 fail closed。
- tokenizer 在至少 8 组覆盖 ASCII、Unicode、连续空白、换行、控制字符、byte fallback 和 special token 的 corpus 上，与冻结 HF reference 的 strings/spans/ids/bytes 完全一致。
- 真实 GPT-2 124M 至少一个 prompt 的 embedding、12 个 block、final hidden 和 full logits 与 HF Float32 reference 在冻结容差内；记录每层误差而不是只报最终 argmax。
- full/dynamic/static 至少 8 个真实 greedy generation step 的 logits、token ids、停止位置和文本一致；learned position 在 cache decode 中使用正确的绝对位置。
- 默认测试全部通过；Reactant-XLA 至少完成缩小 GPT-2 同构模型 smoke。真实 124M XLA 若受编译/显存限制，必须记录可复现边界，不伪造通过。
- 至少完成 CPU 的 3 组 prompt length benchmark；CUDA 可用时完成对应 correctness/steady-state。所有计时区分 load、cold、warm-up、steady-state 和同步边界。
- 文档明确区分“官方 checkpoint 推理/架构复现”与“从零训练/论文质量复现”，不把未执行的 WebText 训练或 zero-shot evaluation 写成已验证。

## 学习重点

- **要理解的概念**：GPT-2 pre-LN 架构；learned absolute position 与 RoPE 的 cache 差异；Conv1D 历史命名及 `(in, out)` 权重布局；GELU-New；byte-level BPE 的 space marker 与 reversible byte mapping。
- **要亲手实现的关键组件**：learned position embedding、精确 GELU-New、GPT-2 config/weight adapter、fused-QKV layout converter、GPT-2 tokenizer/reference exporter。
- **要验证的假设**：现有 `GPTModel` 可以通过少量显式配置承载 GPT-2，而无需复制整套模型；Qwen3 建立的逐层/reference/cache 验证方法可以复用到第二种架构；真正的通用性问题主要出现在位置编码、权重布局和 tokenizer，而不是 attention 公式本身。

## 非目标

- 不从零训练 GPT-2，不下载或重建 WebText，不复现论文 zero-shot benchmark 数字。
- 不覆盖 GPT-2 Medium/Large/XL，不做多尺寸 scaling-law 结论。
- 不实现 cross-attention、sequence classification 等非 causal-LM head。
- 不加入 fine-tuning、LoRA、RLHF、量化、FlashAttention、分布式训练或 speculative decoding。
- 不为 GPT-2 添加 chat template、system/user roles、tools 或 agent loop。
- 不把本 Week 扩展为统一支持任意 HuggingFace 架构；先严格完成第二个真实模型，再抽取经验证的公共接口。

## 风险与取舍

- GPT-2 HF `Conv1D` 的名称像卷积，但实际是矩阵乘；最危险的错误是 shape 能对上而 transpose/split 方向错误，因此必须先做可读的小矩阵 fixture。
- learned position embedding 会进入 checkpoint、full forward 和两类 KV cache；若只修 full forward，decode 在第一个增量 token 后才会分叉，逐 step reference 必不可少。
- GPT-2 tokenizer 与 Qwen byte-BPE 共享字节映射概念，但 regex、special token 和 prefix-space 行为不同；过早“泛化”可能产生静默语义漂移，先以严格 adapter 完成 parity。
- Transformers 的 attention backend 可能改变浮点归约顺序；reference 必须冻结实现/版本、Float32 计算和容差，并优先比较逐层误差与 token 决策。
- 124M 很适合快速实验，但它不能代表更大 GPT-2 的质量或显存曲线；性能数据只用于框架路径和 cache 行为，不做 scaling 外推。

## 实验与过程记录

### 2026-07-23：Open

- Week 09 Close commit `7a440f2` 已推送到 `origin/main`；工作区仅有用户未跟踪的 `artifacts/`，不纳入 Week 10。
- 选择 GPT-2 124M 而不是立即扩大 Qwen3 尺寸：它能以较低下载/运行成本引入 learned absolute position、GELU-New、带 bias fused QKV Conv1D 和不同 byte-level BPE，能更直接检验多架构组装能力。
- 本阶段先 Open 计划，尚未下载 GPT-2、冻结 revision、修改模型代码或生成 reference；计划项均保持“计划中”。

## Close 回顾

- **完成了什么**：待 Close 时填写。
- **验证证据**：待 Close 时填写。
- **没有完成及原因**：待 Close 时填写。
- **最重要的认知变化**：待 Close 时填写。
- **是否满足 Close 条件**：否，Week 10 刚刚 Open。
- **带到下一 Week 的问题**：待 Close 时填写。
