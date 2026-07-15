# Week 02 — Minimal GPT, XLA Training, and KV-Cached Decoding

> 周期：2026-07-08 — 2026-07-14

## 本周目标

Week 01 已经完成 Attention、RoPE 和 TransformerBlock，本周的核心问题是：

> 能否把已有组件组合成一个真正可训练、可生成，并能高效增量推理的最小 GPT？

本周工作不仅打通了从文本到生成结果的完整链路，还继续推进到 Reactant/XLA 训练、动态 KV Cache 和适合编译复用的固定形状 KV Cache。

## 本周结果

### 1. 完成最小 decoder-only GPT

- 实现 `GPTModel`，串联 token embedding、多层 `TransformerBlock`、final LayerNorm 和 LM head。
- 延续统一的张量约定：

```text
tokens: (seq_len, batch)
hidden: (d_model, seq_len, batch)
logits: (vocab_size, seq_len, batch)
```

- 在启用 RoPE 时不再添加独立 learned positional embedding，位置信息由各层 Attention 注入。
- 增加 token id、序列长度、模型配置和输出 shape 等验证与测试。

### 2. 补齐数据闭环

- 实现字符级 `Tokenizer`，支持词表拟合、encode、decode 和 unknown token。
- 实现滑动窗口 `DatasetLoader`，生成 next-token prediction 所需的输入与目标。
- 支持 `seq_len`、`batch_size`、`stride`、shuffle 和 `drop_last`。
- 保持 Julia/Lux 使用的 1-based token id，并在进入设备或编译路径前完成合法性检查。

数据流由此形成：

```text
raw text
   ↓ fit / encode
token ids
   ↓ sliding windows + batching
(input tokens, target tokens)
   ↓ GPTModel + next-token loss
loss / gradients
```

### 3. 打通训练和文本生成

- 实现稀疏 next-token cross entropy，通过索引选择目标 log probability，避免构造 dense one-hot target。
- 实现 `TrainerGPT`、train state 初始化、单步训练和多 epoch 训练循环。
- 默认常规训练路径使用 Zygote。
- 测试证明 tiny GPT 能在固定 batch 上得到有限梯度并显著降低 loss。
- 实现 greedy、temperature 和 top-k sampling，以及普通自回归生成。
- 更新 `examples/minigpt.jl`，形成中文字符级 GPT 的训练与生成示例。

到这里，Week 01 提出的“最小完整 GPT”目标已经形成端到端闭环；不过当前验证重点是闭环正确性与 loss 下降，还不是模型质量或规模能力。

### 4. 增加 Reactant/XLA 训练后端

- 为 `TrainerGPT` 增加 `backend=:xla` 路径，使用 Reactant device 和 Enzyme 自动微分。
- 将 forward、loss、gradient 和 optimizer update 放入可编译训练步骤。
- 增加 CPU / GPU / TPU 后端选择。
- 对 XLA 固定输入 shape 做显式检查，提示使用 `drop_last=true` 保持 batch shape 稳定。
- 将依赖输入数据的 token 合法性检查放在 host 侧，避免 tracing 中出现数据相关的 Julia 控制流。
- 示例默认使用 XLA GPU，也允许通过 `LIFEAI_XLA_BACKEND=cpu` 切换后端。

这里的关键认识是：XLA 优化的不只是单个算子，而是尽量稳定的完整计算图；shape 改变会直接影响编译缓存能否复用。

### 5. 实现动态 KV Cache

- 新增每层 `LayerKVCache` 和请求级 `GPTKVCache`，将请求状态与 Lux 模型状态分离。
- 实现 prompt `prefill`、单 token `decode_step` 和 `generate_cached`。
- 在增量解码中只计算新 token 的 Q/K/V，并复用历史 K/V。
- RoPE 使用 cache position 作为 `start_pos`，保证新增 token 使用正确的绝对位置。
- 支持单 batch 与 batched cache。
- 用 full forward 对齐测试验证 prefill logits、逐 token logits 和 greedy generation 结果。

动态 cache 会沿序列维度增长，接口直观，适合普通 eager CPU/GPU 执行，也是验证增量推理语义的基准实现。

### 6. 实现面向 XLA 的固定形状 KV Cache

- 新增 `StaticLayerKVCache` 和 `StaticGPTKVCache`，一次性预分配到 `max_seq_len`。
- decode 时原位写入下一个 cache slot，物理 tensor shape 始终不变。
- 使用 valid-prefix mask 隐藏尚未写入的 cache 区域。
- 使用 tracked position 表示当前逻辑长度，使位置变化不导致解码图 shape 变化。
- 新增 `XLAKVDecoder`，分别处理 prompt prefill 编译和固定形状的单 token decode 编译。
- 实现 `generate_xla_cached!`，使后续 token 能复用同一个 decode executable。

本周由此形成两条互相校验的 cache 路径：

| 路径 | 存储方式 | 主要价值 |
| --- | --- | --- |
| Dynamic KV Cache | 随 token 数增长 | 语义直接，便于开发和 correctness reference |
| Static KV Cache | 预分配 `max_seq_len` | shape 稳定，适合 Reactant/XLA 编译复用 |

### 7. 沉淀可视化学习笔记

新增或完善 Pluto notebook，覆盖：

- KV Cache 为什么能减少重复计算。
- prefill 与 decode 的职责和计算差异。
- RoPE 在增量解码中为什么必须使用绝对位置。
- decode 阶段 causal mask 的常见误区。
- 动态 cache 与静态 cache 的 shape 差异。
- static KV Cache 如何适配 Reactant/XLA。

同时修复 notebook 的项目环境激活方式，并为图表补充中文字体设置。

## 时间线

| 日期 | 主要进展 |
| --- | --- |
| 07-10 | 完成 GPTModel、Tokenizer、DatasetLoader 及对应测试 |
| 07-12 | 完成 loss、训练循环、采样生成、MiniGPT 示例和 Reactant/XLA 训练路径 |
| 07-13 | 完成动态 KV Cache、固定形状 KV Cache 与 XLA 增量解码 |
| 07-14 | 集中补充 prefill/decode、RoPE、KV Cache 与 XLA 的 Pluto 学习笔记 |

## 验证状态

本周结束后的 2026-07-15 复核中，默认测试套件共有 597 项通过：

| 测试集 | 通过数 |
| --- | ---: |
| Attention | 278 |
| RoPE | 127 |
| TransformerBlock | 44 |
| GPT | 33 |
| Tokenizer | 19 |
| DatasetLoader | 50 |
| GPT training and generation | 10 |
| GPT KV cache | 36 |

Reactant/XLA KV Cache 测试由 `LIFEAI_TEST_XLA=true` 显式启用，不属于上述默认 597 项，因此不能据此声称所有 XLA 硬件后端都已完成实机验证。

## 本周关键学习

1. **模型闭环比孤立组件更能暴露问题。** Embedding、张量布局、loss 索引、梯度、采样任何一处不一致，都会在端到端训练中显现。
2. **prefill 和 decode 是两种不同的计算形态。** prefill 处理整个 prompt，需要 causal relation；decode 只处理一个新 token，可以直接看到 cache 中全部有效历史。
3. **KV Cache 不只是保存数组。** 它还必须携带序列位置、batch size 和每层一致性，并正确处理 RoPE 的绝对位置。
4. **编译后端要求从动态思维转向静态 shape。** eager 路径中自然增长的 cache，会导致 XLA 重编译；预分配和 valid-prefix mask 是复用编译图的关键。
5. **性能实现必须有 correctness reference。** 动态 cache、静态 cache 和 full forward 的 logits 对齐测试，比单纯观察生成文本更可靠。
6. **可视化是学习过程的一部分。** 将 cache 增长、mask 和位置编码画出来后，更容易发现实现中的边界条件。

## 尚未解决

- 尚未建立 checkpoint 保存/加载和可重复的实验配置。
- 尚未形成 perplexity、生成质量、吞吐量、首 token 延迟和单 token 延迟等系统 benchmark。
- XLA 专项测试依赖实际后端环境，默认测试没有覆盖全部 XLA 路径。
- 当前模型仍是用于验证闭环的字符级 tiny GPT，不代表可用于真实任务的模型能力。
- 动态与静态 KV Cache 已实现正确性对齐，但仍需要进一步收束公共接口与性能测量。
- 尚未进入 memory、planning、tools、multimodal 和 embodied agent 层。

## 下一里程碑

下一阶段应先把“能运行”提升为“可复现、可比较”：

> 建立 checkpoint、evaluation 和 benchmark，使同一个小模型能够完成训练、保存、加载、生成，并比较 full forward、动态 KV Cache 与静态 XLA KV Cache 的正确性和性能。

在模型基础闭环稳定后，再开始定义与具体设备无关的 observation、action、memory 和 agent loop，为后续桌面机器人、机器狗和具身机器人实验建立共同接口。
