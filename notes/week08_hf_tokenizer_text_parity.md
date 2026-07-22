# Week 08 — HuggingFace Qwen3 Tokenizer and Text-to-Text Parity

> 状态：Closed
>
> 开启记录：2026-07-22
>
> 关闭记录：2026-07-22
>
> 依赖基线：[`Week 07 — HuggingFace Weight Loading and Qwen3 Logits Parity`](week07_hf_weight_loading.md) 已 Closed。
>
> 长期目标：本 Week 是「复现 Qwen3 → HF 权重加载 → 推理验证」三阶段计划的第三步。Week 06 已完成结构 parity，Week 07 已完成真实权重与 token-id→logits parity；Week 08 导入 HF tokenizer，并把验证边界推进到 text→text。

## 已确认的执行边界

1. **参照对象**：继续使用 Week 07 验证过的 `Qwen/Qwen3-0.6B`，模型 revision、tokenizer 文件 revision 与 reference 版本必须一起记录，避免“权重来自一个 revision、tokenizer 来自另一个 revision”。
2. **公共索引语义**：LifeAI 的 tokenizer 和模型公共 API 继续使用 1-based token id；HF fixture 保留 0-based ids，边界处显式转换，不允许两套索引在 BPE、special token 或生成循环内部混用。
3. **Tokenizer 范围**：只实现 Qwen3 当前文件实际使用的 HF Tokenizers JSON 子集，包括 BPE vocabulary / merges、byte↔unicode 映射、目标 regex pre-tokenization、ByteLevel 行为、added/special tokens 与 decoder。遇到未实现的 normalizer、pre-tokenizer、post-processor、decoder 或模型类型必须 fail closed。
4. **Chat 范围**：支持 Qwen3 常规 system/user/assistant 消息、`add_generation_prompt` 与 thinking 开关所需的确定性渲染；不在本 Week 实现通用 Jinja 引擎，也不承诺任意第三方 chat template。工具调用和 tool-role 消息留给后续 agent/tool 阶段。
5. **生成口径**：端到端验收使用 greedy decoding、固定最大新 token 数和明确 stop-token 集合；不要求 PyTorch/Julia 随机数生成器在 temperature/top-k sampling 下产生相同序列。
6. **下载边界**：默认测试完全离线，使用小型合成 tokenizer JSON 与手写 fixture；真实 Qwen3 tokenizer、权重和 Transformers reference 通过本地目录与环境变量显式 opt-in，不在测试中联网下载。
7. **兼容策略**：现有 `Tokenizer`、`ByteTokenizer`、`ByteBPETokenizer`、artifact v1、checkpoint v1/v2 和生成入口不得静默改变。HF tokenizer 作为新的明确类型接入统一接口。

## Open：核心问题

> 能否在不破坏 LifeAI 现有 1-based Tokenizer 契约的前提下，严格复现 HuggingFace Qwen3 的文本切分、byte-level BPE、added/special token 和基础 chat template 语义，使同一文本得到完全相同的 token IDs，并让 LifeAI 加载 Qwen3-0.6B 后以 full、dynamic cache 和 static cache 生成与 Transformers 完全相同的 greedy token 序列及最终文本？

Week 07 已证明：只要输入 token ids 相同，模型各层 hidden states、logits 和单步 cache decode 都能在明确容差内对齐。Week 08 因此不再猜测模型结构，而是把输入输出边界逐层拆开：Unicode 文本 → regex 分段 → UTF-8 bytes → byte-unicode symbols → BPE merges → added/special tokens → 1-based ids，以及生成 ids → bytes → 文本。首次分叉必须能定位到具体阶段。

## 预期接口

```julia
tokenizer = load_hf_qwen3_tokenizer(model_dir)

# LifeAI 公共 API 返回 1-based ids。
ids = encode(tokenizer, "你好，LifeAI!"; add_special_tokens=false)
text = decode(tokenizer, ids; skip_special_tokens=false)

messages = [
    (role="system", content="You are a helpful assistant."),
    (role="user", content="用一句话介绍 Julia。"),
]
prompt = apply_qwen3_chat_template(
    tokenizer,
    messages;
    add_generation_prompt=true,
    enable_thinking=false,
)

bundle = load_hf_qwen3_bundle(model_dir; max_seq_len=256)
result = generate_hf_text(
    bundle,
    prompt;
    max_new_tokens=32,
    strategy=:greedy,
    cache=:dynamic,
)
result.token_ids
result.text
result.stop_reason
```

底层能力保持可独立验证：

```julia
tokenizer = load_hf_qwen3_tokenizer(model_dir; revision=revision)
config = tokenizer_config(tokenizer)
pretokenized = hf_qwen3_pretokenize(tokenizer, text)

# reference 中保留 HF 0-based ids；测试边界显式比较。
@assert encode(tokenizer, text) .- 1 == reference_ids_0_based
```

接口约束：

- `encode` / `decode` 继续实现 `AbstractTokenizer` 的统一协议；默认行为不因调用 HF tokenizer 而反向修改其他 tokenizer。
- imported vocabulary 的第一个 HF id `0` 在 LifeAI 中为 `1`，最后一个 HF id `vocab_size-1` 为 `vocab_size`；added token 使用同一全局映射。
- `decode_bytes` 必须先提供 byte-exact 结果；`decode` 再以显式 `:strict` / `:replace` UTF-8 策略构造 Julia `String`，避免用替换字符掩盖 token 边界问题。
- 加载器记录 tokenizer 文件 checksum、模型/revision provenance 和行为开关；文件缺失、字段冲突、重复 id/token、越界 merge 或未知 pipeline component 均为错误。
- chat template API 只承诺文档列出的 Qwen3 常规消息子集；超出范围的 role、tool payload 或模板 revision 应给出可操作错误，而不是近似渲染。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 冻结 tokenizer 契约 | 模型 / 学习 | 记录目标 revision 的 `tokenizer.json`、`tokenizer_config.json`、special/generation config 结构与 checksum | 与 Transformers 实际 pipeline、special ids、stop ids 逐字段核对 | 已完成 |
| 严格配置解析 | 工程 | HF tokenizer 文件解析、支持子集校验与 provenance | 缺失/重复/冲突字段及未知组件 fail closed；路径测试不联网 | 已完成 |
| byte↔unicode 映射 | 模型 / 学习 | GPT-2/Qwen 使用的 256-byte 可逆 unicode alphabet | 0x00–0xff 全覆盖、双向一一映射、任意 bytes round-trip | 已完成 |
| regex pre-tokenization | 模型 / 学习 | 与目标 tokenizer 相同的 Unicode/数字/空白/换行分段 | 中英混合、组合字符、emoji、代码、连续空白和 CRLF span 与 HF 一致 | 已完成 |
| imported BPE | 模型 / 工程 | 按外部 vocab / merge rank 编码，不重新训练或重排 id | 非对称小 fixture 逐 merge 验证；真实文本 ids 与 AutoTokenizer 完全一致 | 已完成 |
| added/special tokens | 模型 / 工程 | added token 匹配、BOS/EOS/PAD 与 skip-special decode 语义 | 重叠 token、文本中 literal special token、边界空白及非法 id 覆盖 | 已完成 |
| decoder 与 UTF-8 | 模型 / 工程 | token → byte symbols → bytes → String 的严格逆变换 | byte-exact fixture；中文/emoji/组合字符 round-trip；非法 UTF-8 策略明确 | 已完成 |
| Tokenizer artifact/checkpoint | 工程 | HF tokenizer 类型的版本化、自包含保存恢复与 fingerprint | round-trip 后 ids/bytes/fingerprint 不变；旧 artifact/checkpoint fixtures 不变 | 已完成 |
| 基础 chat template | 模型 / 工程 | Qwen3 常规 messages renderer 与 thinking/generation-prompt 开关 | 固定 system/user/assistant cases 的渲染文本和 ids 与 HF 完全一致 | 已完成 |
| tokenizer reference | 学习 / 工程 | Python 导出脚本与离线 reference schema，记录版本、文本、spans、ids、decode | reference 可重复生成；默认仓库只保存小 fixture/生成说明 | 已完成 |
| text→ids 真实对齐 | 模型 / 工程 | Qwen3 tokenizer opt-in integration test | 多语种/代码/Unicode corpus 的 ids 逐项一致，decode bytes/text 一致 | 已完成 |
| greedy 逐步生成 | 模型 / 工程 | Transformers 与 LifeAI 的每步 token/logits/stop trace | raw prompt 与基础 chat prompt 的每个 generated id、stop step、最终文本一致 | 已完成 |
| 三路推理对齐 | 模型 / 工程 | no-cache/full、dynamic cache、static cache 的多 token text generation | 三路 token 序列彼此一致且与 HF greedy reference 一致 | 已完成 |
| XLA 与全量回归 | 工程 | host tokenizer + XLA static decode smoke、默认/显式测试记录 | Week 01—07 默认测试通过；显式 XLA 套件通过；legacy tokenizer 无回归 | 已完成 |

## 推进顺序

```text
冻结目标 tokenizer revision / checksum / pipeline
    ↓
byte↔unicode 256 项可逆 fixture
    ↓
regex 分段 span parity
    ↓
外部 vocabulary + merge ranks 的 BPE parity
    ↓
added/special token + decoder parity
    ↓
真实文本 encode/decode reference
    ↓
基础 Qwen3 chat template parity
    ↓
文本 → ids → Qwen3-0.6B → greedy ids → 文本
    ↓
full / dynamic / static / XLA 多步生成验证
    ↓
artifact/checkpoint、默认回归、opt-in integration 与 Close 回顾
```

## Close 条件

只有以下条件全部满足后才能关闭本阶段：

- 目标 Qwen3 tokenizer 的 revision、输入文件、checksum、Transformers/tokenizers 版本与 pipeline 结构已经记录；真实测试不会隐式使用缓存中的其他 revision。
- byte↔unicode 映射对全部 256 个 byte 一一可逆；regex pre-tokenizer 对中英混合、代码、emoji、组合字符、空格、tab、LF/CRLF 和输入首尾边界与 HF reference 一致。
- imported BPE 严格遵循外部 vocabulary 和 merge rank；不训练、不补词、不重排 id；unknown token、重复 token/id、非法 merge 和不支持组件有明确错误。
- HF normal/added/special token 的匹配优先级、literal special-token 输入、BOS/EOS/PAD、`add_special_tokens` 与 `skip_special_tokens` 行为均被测试钉死。
- `decode_bytes` 与 HF byte-level decoder 一致；`decode` 的 UTF-8 策略明确。固定 corpus 的 encode ids、decode bytes 和合法文本 round-trip 与 Transformers 完全一致。
- HF 0-based fixture 与 LifeAI 1-based API 只在边界显式转换；tokenizer 输出可直接进入 Week 07 模型，vocabulary 最小/最大 id 均有测试。
- Qwen3 常规 system/user/assistant chat prompt 在文档承诺的 `add_generation_prompt` / thinking 组合下，其渲染文本和 token ids 与 HF `apply_chat_template` 完全一致；未支持的 tools/roles fail closed。
- 至少一组 raw prompt 和一组基础 chat prompt 在真实 Qwen3-0.6B 上逐步 greedy 生成：LifeAI 与 Transformers 每一步 token id、停止位置和最终文本完全一致，并记录首步/全程 logits 误差与 argmax。
- LifeAI no-cache/full、dynamic KV Cache、static KV Cache 产生相同的多 token 序列；显式 XLA smoke 覆盖 host tokenizer → static prefill/decode → decode text 路径。
- 新 HF tokenizer 能以版本化 artifact/checkpoint 保存恢复且 fingerprint 稳定；现有 character/byte/byte-BPE artifacts 和 checkpoint v1/v2 行为、ids 与生成 fixture 无回归。
- 默认测试不联网、不依赖真实权重或外部 Python；真实 tokenizer/model integration 显式 opt-in，并在缺少本地文件时给出可操作提示。
- 默认测试、显式 XLA 测试和真实 Qwen3 integration 全部通过，测试数、命令、版本、fixture 文本和结果写入本 Week 实施记录。

## 学习重点

- **要理解的概念**：HF Tokenizers pipeline 的 normalizer → pre-tokenizer → model → post-processor → decoder 分层；GPT-2 byte-unicode alphabet；BPE rank 与稳定 tie-break；added token 在 pre-tokenization 前后的语义；Unicode code point、grapheme、UTF-8 byte offset 与 Julia string index 的区别。
- **要亲手实现的关键组件**：严格 tokenizer JSON 子集解析、byte↔unicode 映射、Qwen3 regex 分段、外部 BPE encode、added/special token 匹配、byte-exact decoder、基础 chat renderer、逐步生成 trace。
- **要验证的假设**：Week 05 的 byte/BPE 抽象可以复用核心 merge 思路，但 HF imported tokenizer 需要独立的 vocabulary、added-token 与 pre-tokenization 契约；Week 07 logits parity 足以让固定 prompt 的 greedy 多步 token 序列持续一致。

## 非目标

- 不实现通用 HuggingFace Tokenizers 框架；不支持 WordPiece、Unigram、SentencePiece 或任意第三方 tokenizer JSON pipeline。
- 不实现完整 Jinja 解释器、Qwen3 tools/tool-role 模板、JSON schema 工具注入或 agent tool loop；这些能力进入后续智能体阶段。
- 不要求 temperature、top-k 或其他随机采样在 Transformers 与 Julia 之间逐 token 一致；只验证各自采样逻辑的既有 correctness。
- 不实现训练或修改 Qwen3 官方 tokenizer，不向 vocabulary 添加项目自定义 token。
- 不以几条成功生成文本评价模型知识、对话质量、推理能力或安全性。
- 不把真实 Qwen3 权重、完整 tokenizer/reference 缓存或大规模生成结果提交到仓库。

## 风险与取舍

- **看似同为 byte-BPE，边界仍可完全不同**：Week 05 的 trainable byte-BPE 与 HF imported BPE 在 regex 分段、byte symbols、added tokens 和 id 排列上都有独立语义，不能只替换 vocabulary 就宣称兼容。
- **Unicode 索引错位**：Rust/Python reference、Julia `String` 和 regex match 的 offset 单位可能不同。内部应明确使用 UTF-8 byte span，并对组合字符、emoji 与 CRLF 保存原始切片证据。
- **特殊 token 吞噬普通文本**：added token 的 longest-match、左右空白和 normalized/special 标志会绕过普通 BPE；必须用重叠与 literal token fixture 验证优先级。
- **单 token 不一定是合法 UTF-8**：逐 token decode 可能产生不完整字节序列；诊断和流式路径应累计 bytes，不得在每步提前插入 replacement character。
- **greedy 分叉会累积**：即使单步 logits 误差很小，接近并列的 argmax 也可能使后续上下文完全不同。生成 trace 应在每一步记录 top candidates、margin 与首次分叉，而不只比较最终字符串。
- **chat template 范围膨胀**：官方 template 同时包含 thinking 与 tools 分支。本 Week 冻结无 tools 的常规消息子集，对超出范围的输入明确拒绝，避免悄悄近似官方行为。
- **外部文件漂移**：同一模型名可随仓库更新而改变 tokenizer/config；checksum 与 revision 是 fixture 的一部分，不能只记录 repo 名称。

## 实验与过程记录

### 实现结果

- 新增 `HFQwen3Tokenizer <: AbstractTokenizer`，严格读取本地 `tokenizer.json`、`tokenizer_config.json` 与 `generation_config.json`，不联网、不回退到近似 pipeline。
- 实现完整 256-byte GPT-2/Qwen byte↔unicode alphabet、NFC、Qwen3 固定 regex、ByteLevel pre-tokenization、imported BPE rank、26 个 added token、special-token skip 与 byte-exact decoder；`hf_qwen3_pretokenize` 同时暴露 HF 0-based character offsets 和 UTF-8 byte offsets。
- tokenizer JSON 中有 151,643 个 BPE token、151,387 个 merges 和 26 个 added token，因此 tokenizer 长度为 151,669；Qwen3-0.6B embedding vocabulary 为 151,936，额外 267 个模型保留槽位不可由 tokenizer 产生。artifact/checkpoint 兼容性据此允许模型词表大于 imported tokenizer，但拒绝反向越界。
- imported tokenizer 可通过 tokenizer artifact 和 checkpoint 自包含保存恢复；fingerprint 同时绑定三份源文件 checksum 与 revision。legacy character/byte/byte-BPE artifact/checkpoint 行为保持不变。
- 实现 Qwen3 无 tools 的 system/user/assistant chat 分支、`add_generation_prompt` 与 `enable_thinking`；chat template 以已验证的 revision hash fail closed。完整 Jinja、tools/tool-role 明确保留为非目标。
- 新增 `load_hf_qwen3_bundle` 与 `generate_hf_text`，统一模型、tokenizer、EOS 集合和 greedy trace；支持 full/no-cache、dynamic KV Cache 与 static KV Cache，返回 prompt/new/all ids、completion、stop reason、top-two margin 及可选完整 logits。

### 真实 tokenizer 契约

- 模型/tokenizer：`Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`。
- `tokenizer.json` SHA-256：`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`。
- `tokenizer_config.json` SHA-256：`d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101`。
- `generation_config.json` SHA-256：`2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2`。
- reference 环境：PyTorch 2.7.1+cpu、Transformers 4.51.0、tokenizers 0.21.4、safetensors 0.5.3；BF16 权重转 Float32 计算。
- 真实 corpus 覆盖英文缩写/数字、中文、NFD→NFC、emoji/ZWJ/肤色/国旗、tab/空格/CRLF、代码和 literal special tokens。6 组 normalized text、ByteLevel symbols、character spans、ids 与 decode 全部和 HF 完全一致。
- 4 组基础 chat case 覆盖 thinking on/off、system+user、多轮 history 和不添加 generation prompt；渲染字符串与 token ids 均逐项完全一致。

### text→text 与逐步 logits 结果

- raw prompt `LifeAI is` 的 HF/LifeAI 0-based greedy ids 均为 `[264, 5339, 429, 5707]`，completion 均为 `" a platform that provides"`，按长度停止。
- no-thinking chat prompt 的 greedy ids 均为 `[14990, 151645]`，completion 均为 `"hello"`，第二步命中 EOS；LifeAI full、dynamic、static 三路 token 序列与文本完全一致。
- 共 6 个生成 step 的下一 token argmax 全部一致。raw 四步 max-abs 为 `1.67e-5`—`1.76e-5`；chat 两步为 `5.054474e-5` / `4.887581e-5`。全局 max-abs 为 `5.054474e-5`，各 step mean-abs 的平均为 `4.507615e-6`，低于复用 Week 07 的 logits 容差 `atol=5e-3, rtol=5e-4`。
- Julia bundle 加载实测 9.59 s、累计分配约 8.73 GB；完整验证进程峰值 RSS 5,441,156 KiB（约 5.19 GiB）。Python reference 导出峰值 RSS 4,802,692 KiB（约 4.58 GiB）。

### 复现与回归

真实 tokenizer、权重与 reference 缓存不进入仓库。准备好同 revision 的本地模型目录后：

```bash
python scripts/export_qwen3_text_reference.py \
  --model-dir /path/to/Qwen3-0.6B \
  --output-dir /tmp/lifeai_qwen3-text-reference \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --max-new-tokens 4

julia --project=. scripts/verify_qwen3_text_parity.jl \
  /path/to/Qwen3-0.6B \
  /tmp/lifeai_qwen3-text-reference
```

2026-07-22 验证记录：

- 默认 `Pkg.test()`：4086 / 4086 通过，其中 Week 08 离线专项 61 / 61。
- `LIFEAI_TEST_XLA=true Pkg.test()`：默认套件通过，Reactant/XLA 52 / 52；Week 08 新增 host tokenizer → XLA static generation 3 / 3。
- 设置 `LIFEAI_QWEN3_MODEL_DIR` 与 `LIFEAI_QWEN3_TEXT_REFERENCE_DIR` 后，真实 tokenizer/text generation integration 70 / 70。
- 默认测试保持离线；未知 normalizer/regex/model/ByteLevel/chat revision、重复或越界 vocab/merge/added token、special/generation config 冲突均 fail closed。

## Close 回顾

- **完成了什么**：完成 Qwen3 HF tokenizer 的严格本地导入、可逆 byte pipeline、imported BPE、added/special tokens、基础 chat、artifact/checkpoint、三路 greedy text generation 与 XLA 接入；Qwen3 三阶段计划从结构、权重/logits 到 tokenizer/text 已闭环。
- **验证证据**：默认 4086 / 4086、XLA 52 / 52、真实 integration 70 / 70；6 组真实 tokenizer、4 组 chat、两组生成共 6 step 的 ids/文本/argmax 全部与 Transformers 一致，全局 logits max-abs `5.054474e-5`。
- **没有完成及原因**：未实现通用 Jinja、tools/tool-role、随机采样跨框架复现和 agent loop；这些内容在 Open 时即明确属于非目标，避免把 tokenizer parity 扩张成完整智能体系统。
- **最重要的认知变化**：模型 vocabulary size 不等于 tokenizer length；Qwen3-0.6B 有 267 个不可生成的保留 embedding 槽位。另一个关键点是 added tokens 在普通 NFC/regex/BPE 之前保持原文语义，必须单独切分，否则 literal chat tokens 会被错误拆解。
- **是否满足 Close 条件**：是。约定范围内全部 Close 条件已完成，并于 2026-07-22 关闭 Week 08。
- **带到下一 Week 的问题**：Qwen3 结构→权重→文本复现已完成；下一阶段可转向最小 agent loop、对话状态、工具接口与 replay，但当前不自动 Open 新 Week。
