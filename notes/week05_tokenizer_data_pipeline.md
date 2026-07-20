# Week 05 — Versioned Tokenizers and Chinese Data Pipeline

> 状态：Open
>
> 开启记录：2026-07-20
>
> 依赖基线：[`Week 04 — Modern GPT Building Blocks`](week04_model_modernization.md) 已 Closed。

## 已确认的执行边界

1. **范围**：本 Week 聚焦 byte-level baseline、deterministic byte-BPE、Tokenizer artifact 版本化和中文数据管线；不加入 GQA、MoE、量化、分布式训练或 agent 能力。
2. **兼容策略**：现有字符级 `Tokenizer`、Week 03 / Week 04 checkpoint 和默认训练示例必须继续可加载；新接口不能把历史 checkpoint 变成隐式不可恢复状态。
3. **文本策略**：默认不静默修改 Unicode；normalization 必须作为显式配置写入 artifact 和 dataset manifest。训练、验证与生成使用同一份可追踪配置。
4. **数据边界**：先按 document 划分 train / validation，再只用 train split 拟合 Tokenizer；不得先拼接、训练 BPE 后再切分，也不得让同一文档跨 split。
5. **比较口径**：不同 Tokenizer 的 per-token loss / perplexity 不直接横向排名；跨 Tokenizer 主要使用 negative log-likelihood per byte、bits per byte、原始字符/字节吞吐和上下文覆盖率。
6. **语料策略**：仓库只提交许可清晰的小型 fixture、manifest 和构建脚本；大文件、受限语料及下载缓存不进入 Git 历史。

## Open：核心问题

> 能否在不破坏现有字符级模型与 checkpoint 的前提下，建立可逆、确定、可版本化且无 OOV 的 byte-level / byte-BPE Tokenizer，以及来源、变换、切分和指纹均可追踪的中文数据管线，并用跨 Tokenizer 可比较的指标完成端到端训练验证？

Week 04 已经证明模型结构变化可以独立配置、恢复和比较。Week 05 将同样的工程纪律扩展到文本入口：先冻结 Tokenizer 与数据契约，再实现 byte baseline 和 BPE，最后接入 checkpoint、训练、validation 与 generation。目标不是在一个 tiny corpus 上追求最低 perplexity，而是建立后续中文训练可以长期复用的输入与评估基础。

## 预期结果

本阶段 Close 时，应当可以展示或验证：

1. 统一 Tokenizer 接口同时支持 legacy character、byte-level 和 byte-BPE；token id 保持 Julia / Lux 友好的 1-based 语义。
2. Byte tokenizer 对任意有效 UTF-8 文本无 OOV、可精确 round-trip；byte-BPE 在固定语料和配置下产生确定的 vocabulary、merge ranks 和 fingerprint。
3. Tokenizer artifact 具有显式 schema version、算法类型、normalization、special tokens、vocabulary / merges、训练配置和内容指纹，可独立保存、加载与校验。
4. 中文数据管线以 document 为基本单位记录来源、许可、原始 checksum、变换配置、split 规则和输出统计；Tokenizer 只在 train split 上拟合。
5. checkpoint 能保存并恢复任意受支持 Tokenizer；旧 v1 / v2 character checkpoint 迁移后 logits、训练进度和生成行为保持兼容。
6. character / byte / byte-BPE 在固定语料和模型配置下完成 tokenizer-only 与端到端对照，报告压缩率、吞吐、上下文覆盖、参数量、checkpoint 大小和 bits per byte。
7. 至少一个中文最小示例完成 manifest → build dataset → train → validate → save → load → resume → generate。

## 建议接口与 artifact 契约

```julia
abstract type AbstractTokenizer end

encode(tokenizer, text; add_special_tokens=false)
decode_bytes(tokenizer, ids; skip_special_tokens=false)
decode(tokenizer, ids; errors=:strict, skip_special_tokens=false)
vocab_size(tokenizer)
special_token_id(tokenizer, name)
tokenizer_config(tokenizer)
tokenizer_fingerprint(tokenizer)
save_tokenizer(path, tokenizer)
load_tokenizer(path)
```

兼容约束：

- 现有 `Tokenizer` 保留为 character tokenizer，不要求用户立刻改名或迁移调用。
- 新增 `ByteTokenizer` 与 `ByteBPETokenizer`，公共训练、生成和 checkpoint API 接受 `AbstractTokenizer`。
- byte vocabulary 以 256 个原始 byte 为基础；special token 使用独立、显式、稳定的 id，不占用 byte 的可逆映射。
- byte-BPE token 存储 byte sequence，而不是假设每个 token 都是合法 UTF-8 字符串。
- `decode_bytes` 始终可逆；`decode` 对不完整或非法 UTF-8 序列提供明确的 `:strict` / `:replace` 策略，不静默吞掉字节。
- artifact 中记录 `id_base=1`、schema version、normalization、special-token map、vocabulary、merge ranks、trainer config 和 corpus fingerprint。
- vocabulary 中的 byte sequence 使用稳定的二进制或 hex 表示，避免依赖 Julia 类型名和默认序列化布局。

## 数据管线契约

建议每个数据版本至少固化：

```text
dataset name / version
source id / source location / license
raw file SHA-256
text encoding
normalization and filtering config
document identity and boundary policy
split method / seed / split fingerprint
tokenizer fingerprint
raw bytes / Unicode scalar count / token count
encoded artifact checksum
```

处理顺序固定为：

```text
source documents
    ↓ verify checksum / encoding / license metadata
explicit normalization and filtering
    ↓
deterministic document-level train / validation split
    ↓
fit tokenizer on train documents only
    ↓
encode each split with explicit EOS / boundary semantics
    ↓
write versioned manifest and encoded artifacts
    ↓
construct independent DatasetLoader instances
```

## 对照矩阵

| 名称 | Tokenizer | 作用 |
| --- | --- | --- |
| `character_legacy` | 当前 character tokenizer | 兼容性与历史基线 |
| `byte_baseline` | 256-byte alphabet + special tokens | 无 OOV、完全可逆参考 |
| `byte_bpe` | train-only deterministic byte-BPE | 评估压缩率与有效上下文提升 |

比较分两层：

- **Tokenizer-only**：round-trip、determinism、vocab / artifact size、tokens per byte、tokens per Unicode scalar、encode / decode throughput、未见文本覆盖。
- **End-to-end**：固定数据版本、模型配置、optimizer、seed 和训练步数，报告实际参数量、token throughput、raw bytes / characters throughput、context coverage、validation NLL per byte / bits per byte、checkpoint 大小和 generation 可解码性。

per-token perplexity 只在同一 Tokenizer 内用于训练过程跟踪；跨 Tokenizer 结论以 byte-normalized 指标为主。所有端到端结果至少使用 3 个固定 seed，并同时记录固定 token context 覆盖了多少原始 bytes / characters。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 固化 legacy character fixture | 工程 | 当前 vocabulary、encode/decode、checkpoint payload 与训练输出 fixture | 固定 seed 下旧调用、旧 checkpoint 和 logits 不变 | 计划中 |
| 抽象 Tokenizer 公共接口 | 模型 / 工程 | `AbstractTokenizer`、公共 encode/decode/vocab/special-token API | character、byte、BPE 共用训练和生成入口，无调用分叉 | 计划中 |
| 定义 special token 与 normalization 语义 | 模型 / 数据 | 显式配置、稳定 id、错误策略和文档 | 配置可 round-trip；未知配置或冲突立即报错 | 计划中 |
| 实现 byte-level baseline | 模型 / 学习 | `ByteTokenizer`、byte encode/decode、UTF-8 策略 | 中英文、emoji、组合字符和随机有效 UTF-8 round-trip；无 OOV | 计划中 |
| 实现 deterministic byte-BPE | 模型 / 学习 | trainer、merge ranks、encoder/decoder、tie-break 规则 | 同语料与配置重复训练产生相同 vocab、merges 和 fingerprint | 计划中 |
| Tokenizer artifact v1 | 工程 | 独立 save/load、schema、fingerprint、兼容校验 | 三类 Tokenizer 保存加载后配置与编码完全一致；篡改 checksum 被拒绝 | 计划中 |
| 建立中文 corpus manifest | 数据 / 工程 | 来源、许可、checksum、变换和统计 schema；tiny fixture | 同一输入与配置产生相同 dataset fingerprint；缺失许可或 checksum 明确失败 | 计划中 |
| 文档级无泄漏 split 与编码 | 数据 | deterministic split、EOS / boundary 语义、encoded artifacts | train/validation document id 不相交；BPE 不读取 validation；窗口不跨越未标记边界 | 计划中 |
| DatasetLoader 集成 | 数据 / 训练 | 从 versioned encoded split 构建 loader 的公共入口 | character / byte / BPE 均完成 train-step 和 validation；shape 与 token 范围正确 | 计划中 |
| checkpoint 与迁移 | 工程 | 支持多 Tokenizer 的新 checkpoint schema、v1/v2 migration | legacy checkpoint 恢复一致；新 checkpoint round-trip / resume / generate | 计划中 |
| 跨 Tokenizer evaluation | 工程 / 学习 | bits-per-byte、raw throughput、context coverage 和对照脚本 | 固定数据、seed、模型与步骤；不直接横比 token PPL | 计划中 |
| 中文端到端示例与文档 | 学习 | manifest → dataset → train → resume → generate 示例 | 默认测试通过；CPU 与至少一个 XLA backend smoke 通过 | 计划中 |

## 推进顺序

```text
冻结 legacy tokenizer / checkpoint fixture
    ↓
公共接口 + special token / normalization 契约
    ↓
ByteTokenizer reference implementation
    ↓
Deterministic byte-BPE trainer and encoder
    ↓
Tokenizer artifact + fingerprint
    ↓
Document-level Chinese data manifest and split
    ↓
DatasetLoader / checkpoint / generation integration
    ↓
Tokenizer-only + end-to-end controlled comparison
```

Byte baseline 必须先于 BPE 完成；BPE 的每个 token 都应能展开为 byte sequence，并能用 byte tokenizer 作为 reference 验证。数据 pipeline 必须先完成 document split，再调用 BPE trainer，测试中应加入故意只出现在 validation 的 pair，证明它不会进入 merge table。

## Close 条件

只有以下条件全部满足后才能关闭本阶段：

- 默认字符级 API 与现有训练示例保持兼容，Week 03 / Week 04 checkpoint fixture 可显式迁移并恢复一致结果。
- ByteTokenizer 对测试集合和随机有效 UTF-8 文本实现 `decode(encode(text)) == text`；内容 byte 不会映射到 unknown token。
- Byte-BPE 具有确定 tie-break；相同 train corpus、normalization、special tokens、vocab target 和 seed 产生完全相同 artifact 与 fingerprint。
- Tokenizer artifact 不依赖 Julia 结构的偶然布局，保存加载后 token ids、special ids、vocabulary、merges 和配置完全一致。
- normalization 的默认值和可选值明确；任何会改变文本的 normalization 都进入 manifest、fingerprint 和实验报告。
- train / validation 在 document id 上无交集；Tokenizer 只在 train split 上拟合；validation-only byte pair 不进入 BPE merge ranks。
- 数据 manifest 包含来源、许可、raw checksum、transform config、split fingerprint、tokenizer fingerprint 和输出 checksum；构建过程可重复。
- encoded dataset 的文档边界具有显式 EOS 或等价标记，DatasetLoader 不会静默创建跨未标记文档的样本。
- character、byte、byte-BPE 均通过 reference、异常输入、artifact round-trip、train-step、validation、checkpoint resume 和 cached generation 测试。
- 对照报告同时记录 vocab size、实际模型参数量、tokens / bytes / characters、encode/decode throughput、训练 tokens/s、raw bytes/s、上下文覆盖、NLL per byte / bits per byte 和 checkpoint 大小。
- 至少 3 个固定 seed 的 CPU 对照完成；byte-BPE 组合在至少一个 XLA backend 上完成训练与生成 smoke。
- 默认测试全部通过；新增的外部语料下载或长时 benchmark 必须是显式 opt-in，不进入默认测试。
- 文档明确实验适用范围，不把 tiny corpus 结果外推为中文模型质量结论。

## 学习重点

- **要理解的概念**：Unicode scalar、UTF-8 byte 与 grapheme 的区别；byte fallback；BPE merge learning / ranking；special token；normalization；document-level leakage；tokenizer-independent language-model evaluation。
- **要亲手实现的关键组件**：ByteTokenizer、deterministic byte-BPE trainer / encoder、Tokenizer artifact、dataset manifest、document split 和 bits-per-byte evaluation。
- **要验证的假设**：byte baseline 能提供无 OOV、可逆参考；BPE 能在不泄漏 validation 的前提下降低 token density、扩大固定 token context 的原始文本覆盖；版本化 artifact 能让训练、checkpoint 和数据结果长期可复现。

## 非目标

- 不实现 SentencePiece unigram、WordPiece、morphological segmentation 或多语言大词表。
- 不加入 GQA、MoE、量化、FlashAttention、分布式训练或新的 optimizer / scheduler。
- 不为某个 Tokenizer 临时调整模型深度、宽度或初始化以追求更好 validation 数字。
- 不建立大规模爬虫，不把来源不明、许可不清或包含敏感数据的语料提交到仓库。
- 不进入 memory、planning、tools、multimodal 或 embodied agent loop。

## 风险与取舍

- 中文字符在 UTF-8 中通常占多个 byte；byte baseline 会显著增加 token 数，但它是无 OOV、可逆和验证 BPE 的必要参考，而不是最终效率目标。
- Unicode normalization 可能合并或改变字符序列；默认不应静默 normalize，任何规范化都必须进入版本和 fingerprint。
- Byte-level generation 可能在中间 step 形成不完整 UTF-8；内部应以 bytes 保真，展示层再应用明确的 strict / replacement 策略。
- BPE pair frequency 相同的 tie-break 若依赖 Dict 遍历顺序，会破坏确定性；排序规则必须进入测试与 artifact schema。
- 不同 vocabulary 会改变 embedding / LM head 参数量和 XLA executable shape；实验必须记录实际参数量与 cold compilation，不能把规模变化误当成 tokenizer 收益。
- 不同 Tokenizer 的 token perplexity 数值不在同一尺度；若报告 token PPL，必须与 bits per byte 并列且禁止直接横向排名。
- 文档拼接会产生伪造跨文档上下文；必须用 EOS、边界掩码或独立 packing 规则显式处理。
- `Serialization` 适合 checkpoint 内部状态，但不应成为独立 Tokenizer artifact 的唯一长期格式；artifact 需要显式 schema 和稳定 byte 表示。
- 数据许可和删除请求属于工程边界；manifest 记录来源与许可，仓库默认只保存可公开再分发的 tiny fixture。

## 实验与过程记录

按推进顺序记录 reference formula / algorithm、输入语料指纹、Tokenizer 配置、数据 manifest、测试结果、异常、性能数据和下一步。所有实验先写清比较单位是 token、byte、Unicode scalar 还是 document，避免指标含义漂移。

## Close 回顾

- **完成了什么**：
- **验证证据**：
- **没有完成及原因**：
- **最重要的认知变化**：
- **是否满足 Close 条件**：
- **带到下一 Week 的问题**：
