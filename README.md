# LifeAI.jl

> 构建有生命感的 AGI，并让它走进真实世界。

LifeAI.jl 是一个长期演进的 Julia/Lux 研究与工程项目。项目不止于实现一个语言模型，而是希望逐步构建能够持续感知、记忆、学习、决策和行动的智能系统，并将这些能力用于机器狗、桌面机器人、具身机器人以及其他可能的智能载体。

这里也是一份公开的学习与构建记录：从关键模型组件开始，理解原理、亲手实现、测试验证、分析性能，再逐步组合成更完整的智能体。

## 项目宗旨

LifeAI.jl 沿四条相互连接的主线持续积累：

1. **模型基础组件**：持续学习和实现 Attention、位置编码、Transformer、训练、推理加速等可复用能力。
2. **智能体核心**：逐步加入记忆、规划、工具使用、反思、多模态理解和持续学习能力。
3. **具身闭环**：让模型通过统一接口连接感知与行动，先在可验证的环境中运行，再走向机器狗、桌面机器人和其他实体设备。
4. **学习过程记录**：用 Chapter 推动小步交付，用 Episode 串联完整能力主线，沉淀结果、实验、失败和认知变化。

这里所说的“有生命感”，不是只让模型表现得像某种人格，而是让系统在长期互动中体现出连续性、状态、记忆、主动性、适应性以及与环境真实连接的行动能力。

## 当前状态

**阶段判断：Qwen3 dense family 真实权重 parity 与 BF16 GPU 推理全闭环，Qwen3-Embedding-0.6B 与最小 dense exact semantic memory 也已完成真实 BF16 parity；RTX 4090 D 上的 dense 容量上限仍是已实证生成的 14B mixed RTN，日常部署选择 Qwen3-8B BF16。原始 Qwen3-30B-A3B 现已具备 40K-capacity BF16 GPU resident/offload session、global/layer-balanced device expert cache、scalar 与 grouped WMMA 直接消费分散 BF16 cache matrices 的 generation-safe CUDA dispatch，以及经过 storage I/O 验证的当前层 bounded parallel miss reads。grouped-scattered 对相同 grouped 数值契约逐位一致，2/32-token request 相对 active-3D materialization 加速 `10.745× / 6.103×`。Qwen3-VL-2B-Instruct 已完成双 registry 资产契约、raw image fast processor、content-list chat、vision tower、main/DeepStack visual injection、三轴 mRoPE 与完整 decoder prefill 的真实 Float32 GPU strict parity；当前仍不包含 multimodal KV cache、增量 decode 或 image-to-text generation。智能体侧已具备工具、跨请求持久记忆、由环境终态评分并可联合 replay 的 observation/action 闭环，以及 clean successful 环境事件的显式写回、fresh-load exact-spec 检索和无反馈后续执行。**

Chapter 01—44 均已 Closed，Episode 06/07 已 Closed、Episode 08/09 已 Open。[`Chapter 44 — Qwen3-VL processor/chat 与 mRoPE decoder prefill`](notes/episodes/episode09_qwen3_vl_multimodal_perception/chapter44_qwen3_vl_multimodal_prefill.md) 在 Chapter 43 的 625-tensor vision 契约上补齐 raw image decode/UInt8 bicubic、content-list chat、64-token placeholder、T/H/W mRoPE、main visual replacement、三层 DeepStack injection 和 28-layer decoder prefill。ModelScope revision `ae9985b…9c53` / Hugging Face revision `78448d…becd` 与完整资产 hash 保持不变；Float32/BF16 reference SHA256 为 `d7d3b58c…b60f5` / `711749d9…cb5ae`。RTX 4090 D 两次 80-stage gate 全部通过：sequence `76`、image tokens `64`、raw max-abs `0`，Float32 final/logits max-abs `0.00094986 / 0.00083363`、warm `0.703 s`；BF16 为 `6.9375 / 7.4375`、warm `0.751 s`，只记作跨后端边界。Chapter 45 将继续 dynamic KV、单 token decode 与真实 greedy image-to-text generation。

[`Chapter 42 — 显式环境事件记忆写回与因果验收`](notes/episodes/episode08_environment_action_loop/chapter42_environment_event_memory.md) 在 ModelScope Qwen3-8B / RTX 4090 D 上让 8 个 full-feedback writer 以 BFS 最短路 `8/8` clean success，显式写出 8 条事件；fresh-load 后 CPU Qwen3-Embedding raw top-1 / exact-spec 都是 `8/8`。三个无中间反馈 reader 臂为 no-memory `0/8`、retrieved `7/8`、等 token 且机械 non-solving 的 mirrored distractor `0/8`，相关记忆相对两组对照的双侧精确 McNemar p 均为 `0.015625`。`grid/08` 是正确检索后的执行失败，不是 retrieval miss；32 条轨迹已在不加载两套模型且不追加 journal 的第二进程完整 replay。环境事件记忆条目已关闭；Episode 08 仍保留跨 adapter timeout / execution failure / e-stop / idempotent action safety 缺口。

[`Chapter 41 — Qwen3 MoE grouped scattered expert-cache dispatch`](notes/episodes/episode06_qwen3_moe_and_model_expansion/chapter41_qwen3_moe_grouped_scattered.md) 在官方 30B 同进程 2-token 与 32-token 三路对照中，让 grouped WMMA 直接读取 generation-safe expert pointer tables；两条 grouped 路径的 prefill/decode logits、逐层 routes 与每次 I/O/cache traffic 全部 exact，并消除 `10.560 / 16.930 GB` logical active concat。它选择的下一架构 Qwen3-VL 已由 Chapter 43 启动并关闭 vision-only 第一阶段。

[`Chapter 40 — 确定性 GridWorld observation/action 与真实反馈闭环`](notes/episodes/episode08_environment_action_loop/chapter40_deterministic_gridworld.md) 首次用环境终态而非模型自述判定任务成功：冻结 8 题的 BFS oracle 与 Qwen3-4B full-feedback 都是 `8/8`、44 个动作、0 个非法动作，模型逐题走出最短路；首轮 prompt 完全相同的 feedback-withheld 臂为 `0/8`、67 个非法/失败动作，配对 McNemar p=`0.0078125`。第二进程重算 148 / 148 prompts、140 / 140 tool outcomes、132 / 132 transitions 与 16 / 16 final states。

[`Chapter 39 — 跨请求、可回放的语义记忆闭环`](notes/episodes/episode07_agent_closed_loop/chapter39_persistent_semantic_memory.md) 已把版本化 append-only journal、fresh-load 严格恢复、exact semantic index 和 retrieval context 接入真实 Qwen3：冻结 12 个 private-fact tasks 上 retrieval top-1 / recall@1 均为 `12/12`，no-memory / retrieved / 等 token 干扰三臂为 `0/12 / 12/12 / 0/12`，两组精确 McNemar p 均为 `0.00048828125`；第二进程 36 / 36 条完整 prompt/context 轨迹 replay 一致。默认专项为 `100 / 100`，加真实 Qwen3-4B tokenizer 为 `112 / 112`。

[`Chapter 38 — 工具到底帮不帮得上忙`](notes/episodes/episode07_agent_closed_loop/chapter38_qwen3_tool_task_success.md) 在同一批 150 道 GSM8K 上做了一次配对对照：无工具基线 `141/150 = .940`，声明计算器后 `137/150 = .913`，再加一句要求使用工具的 system message 仍是 `137/150`。**掉的题几乎全部没有调用工具**（`tool-declared` 掉 6 题、调用过工具的 0 题），所以测到的是「在 prompt 里声明工具」这件事本身对 greedy 解码的扰动，而不是工具的效用；两次下降的精确 McNemar p 为 `0.29` / `0.39`，150 题分辨不了 `2.67` 个点。

[`Chapter 37 — Qwen3 dense 任务质量基线`](notes/episodes/episode07_agent_closed_loop/chapter37_qwen3_task_quality.md) 第一次回答「复现出来的模型答对率是多少」：冻结 MMLU 200 题 + GSM8K 150 题，0.6B/1.7B/4B 的 MMLU loglikelihood 为 `.365 / .395 / .555`，4B 的 GSM8K 为 `141/150 = .940`。同时给出一条本章最重要的负结果——**MMLU loglikelihood 的逐题决策在 BF16 下不可复现**，既不跨 dtype，也不跨设备，甚至不跨同一实现里两条数学等价的写法：协议对齐后与 HuggingFace fp32 参照的一致率为 `193/200 = 96.5%`；同一份权重同一协议，CUDA 给 `73`、CPU 给 `74`、fp32 给 `70`；我们自己 fast 与 general 两条路径之间也有 `6/200` 翻转。accuracy 的极差 2 个点仍远小于 Wilson 区间宽度，标题数字可用，但逐题一致的主张不成立。

[`Chapter 36 — Qwen3 tools chat template HF parity 与 4B 工具调用闭环`](notes/episodes/episode07_agent_closed_loop/chapter36_qwen3_tools_chat_template.md) 用官方 Jinja 模板经 CPython 渲染的外部 reference，把 chat template 的全部分支做到逐字节 parity（30 / 30 case，真实 4B tokenizer 下 token id 亦逐位相同），并修复了 Chapter 08 遗留的两条真实分歧：历史 assistant 的 `<think>` 块未剥离、以及无 user 消息时 `last_query_index` 方向相反。开启 `--thinking` 的多轮聊天从第 2 轮起会命中前者。

[`Chapter 35 — Qwen3 MoE bounded host projection-buffer reuse`](notes/episodes/episode06_qwen3_moe_and_model_expansion/chapter35_qwen3_moe_host_buffer_reuse.md) 在明确 CUDA pageable、pinned 与 CPU identity 的不同 ownership 后，以 72 MiB final-matrix pool 配合 Chapter 34 的 24 MiB raw pool。30B-A3B English32 同进程 A/B 的 cold/revisit Julia allocation 从约 `28.967 GB` 降到 `0.288 GB`（`99.01%`），时延分别加速 `1.315× / 1.296×`，所有输出 exact；CPU 与 pinned async 自动保持无 final pool。

目前已经具备：

- 手写与批量 scaled dot-product attention、因果遮罩和 Multi-Head Attention。
- 支持 legacy interleaved 与 HF rotate_half 配对的 RoPE、learned absolute position、pre-norm TransformerBlock 和 decoder-only GPTModel。
- 可独立切换的 LayerNorm / RMSNorm、GELU / GELU-New / SwiGLU、projection bias / LM-head bias、untied / tied embedding-output projection；legacy 默认保持不变。
- 字符级 Tokenizer、DatasetLoader、next-token loss 和训练循环。
- 无泄漏 train / validation 划分、token-weighted evaluation / perplexity 和 global gradient norm clipping。
- checkpoint v2、设备无关保存/加载、确定性断点续训和 v1 legacy checkpoint 迁移。
- 基于 Zygote 的常规训练，以及 Reactant/Enzyme 驱动的 XLA 训练路径。
- greedy、temperature、top-k、top-p 文本生成；Qwen3 可严格读取官方 generation config，并支持固定 uniform 流的可重放采样。
- 动态 KV Cache 的 prefill / decode，以及面向 XLA 的固定形状 KV Cache 和编译后增量解码。
- Qwen3 BF16 常驻 session：固定容量 KV、bounded chunked prefill、
  last-token-only vocabulary projection、EOS/采样、chat history token
  预算与最老 turn-pair 裁剪；4K/4090 D profile 已完成真实权重验收。
- Qwen3-8B BF16 XLA single-residency session：safetensors 逐层直接组装
  packed compact tree，只做一次递归 device transfer；固定形状 64-token
  prefill 与单 token decode 共用 4K 静态 KV。4090 D 日常 CLI 当前明确为
  batch-1 greedy，sampling 仍走 CUDA eager 路径。
- Qwen3-8B XLA loopback resident service：严格的 `/healthz` 与
  Ollama-compatible `/api/generate` 子集支持 buffered/NDJSON、UTF-8
  安全 streaming、context/options fail-closed 和 single-flight；轻量
  client 只读取 tokenizer，权重与 compiled executable 常驻 server。
- full / dynamic / static KV Cache correctness matrix，以及 CPU、CUDA GPU、XLA CPU、XLA GPU 四后端 benchmark。
- 原始 Qwen3 MoE 的 Float32 top-k router、逐 expert SwiGLU、full/dynamic/static cache，以及 Float32/native BF16 的路由后 active-expert 流式读取。官方 30B-A3B 的 48 层 prompt/cache-decode parity 已冻结：Float32 top-8 集合 `1,152 / 1,152` 一致、BF16 路由槽位重合 `95.92%`，两种模式最终 argmax 均一致。`HFQwen3MoEOffloadSession` 让 attention/router/norm/LM head 与 40K KV 常驻 GPU，expert 在路由后按层读取并用局部 id 交给 scalar 或 grouped WMMA；最坏工作集硬下限 `7.166 GiB`。device cache 可选 global/layer-balanced LRU，按实际 expert-layer bytes 预算，支持跨请求保留、显式清空和运行时重配；scalar 与 grouped CUDA 都可用 scattered pointer-table dispatch 消除 active tensor 拼接，有界复用 pointer/workspace state，并配置 forced-GC cadence。cache miss 可显式选择 bounded parallel host reads 与 CUDA pinned upload；32-token trace 的 reader 默认上限为 8。严格相邻 batch read 已实现，但真实 workload 证明 coalescing 更慢，默认仍逐 tensor 读取；同 dtype多维 decode、raw payload 与 pageable final host-matrix allocation 已依次消除。
- Qwen3-VL-2B-Instruct 的双 registry immutable revision、13-file checksum、
  strict text/vision config 与完整 625-tensor BF16 shape/payload contract；raw
  PNG/JPEG/RGB decode、官方 CPU UInt8 bicubic、normalize、grid/token count 与
  merge-order patchify；content-list chat、VL tokenizer profile、placeholder
  expansion 和 T/H/W mRoPE；3D patch embedding、position interpolation、vision
  RoPE、24-layer per-temporal-frame packed attention、三个 DeepStack mergers、
  main merger，以及 main visual replacement/DeepStack injection 后的 28-layer
  decoder prefill。Float32 RTX 4090 D 已完成 Transformers 4.57.0 的统一
  80-stage strict gate；BF16 单独记录跨后端边界。
- 六个官方 Qwen3 dense 规格的 immutable revision/config checksum、自动识别、显式 variant 校验、精确参数量；严格的 BF16/F32 safetensors 单文件/分片读取、HF 参数映射与显式 0-based token-id 边界转换。
- Qwen3-Embedding-0.6B 的独立 immutable revision/8-asset checksum、
  151,669-vocabulary/32K contract、embedding tokenizer 尾 token 与
  base-model safetensors namespace；BF16 hidden-state 前向不分配
  vocabulary logits 或保留 KV cache。
- instruction-aware query、变长批 attention mask、last-token pooling、
  L2 normalization 与 1024/512/256/128/64 维 MRL；内存内 dense exact
  cosine semantic memory 可关联 metadata，并有真实 notes 检索示例。
- 版本化 append-only agent memory journal：保存 source text 与 string metadata，
  fresh load 严格校验 schema/sequence/ID/checksum，启动时重建 exact embedding
  index；retrieval query、命中 ID/score、注入内容与 prompt 均带 sha256，可离线 replay。
  冻结私有事实三臂实测为 `0/12 / 12/12 / 0/12`，相关记忆同时胜过无记忆与等 token 干扰；
  clean successful GridWorld episode 还可经显式 policy 写为 canonical event，同 run ID 幂等，
  authoritative spec/source binding 和生产 exact-spec context 均 fail closed。
- Qwen3 weight-only INT8 per-channel / packed groupwise INT4，以及统一的
  `QuantizationPlan`：可按 one-based layer、projection 与独立 LM head
  选择 INT4/INT8/BF16，streamed/in-memory 路径共用策略；真实树统计与
  topology 估算逐 byte 对齐。`ActivationCalibration` 可用独立 token
  逐层采集每个线性输入的 second moment，并以 `:activation_mse` 做
  diagonal activation-aware clipping；这不是完整 AWQ/GPTQ。
- Qwen3-0.6B 逐层 hidden states、full logits、dynamic/static cache decode 的真实 HF reference parity；真实权重测试显式 opt-in，默认测试保持离线。
- Qwen3-0.6B 的 16-step 官方 sampled reference parity、position 40,959 独立 Transformers RoPE fixture，以及 CPU/CUDA/Reactant-XLA GPU 真实推理 benchmark。
- 严格的 Qwen3 HF tokenizer：NFC、目标 regex、ByteLevel、imported BPE、added/special tokens、artifact/checkpoint 与 provenance fingerprint。
- 官方 Qwen3 chat template 的全部分支：`# Tools` 头、assistant `tool_calls`、`tool`
  角色的 `<tool_response>` 合并、think 块拆分与 `last_query_index` 反向扫描，30 / 30
  case 与 CPython + Jinja2 渲染的外部 reference 逐字节相同；`tojson` 按 CPython
  `json.dumps(ensure_ascii=False)` 语义复刻，含 CPython `repr(float)` 与任意精度整数，
  无序 `Dict` 与会窄化数字的 `JSON3.Object` fail closed。tools / tool 角色 /
  `tool_calls` 以官方模板 sha256 为准入条件。
- 不使用 `eval` 的算术求值器与 calculator 工具，以及同批题 A/B 的配对统计（2×2 不一致对 +
  精确 McNemar）；准确率相同的两个臂仍可能有 8/150 题在两个方向上互换，只有配对表能看见。
- 任务质量评测基线：冻结的 MMLU/GSM8K 子集（含许可、上游 row_idx 与 sha256 provenance）、
  loglikelihood 与 generative 两种协议、机械且写死的答案抽取规则，以及强制同时报出
  Wilson 区间、未解析数、截断数与格式合规数的报告口径。
- 最小工具闭环：工具声明与注册、`<tool_call>` 解析与合法性判定、沙箱化内置工具，以及
  「观察 → 决策 → 调用 → 回填 → 再决策」的多 step 循环；每一步记录 prompt sha256 与
  generated ids，可在不加载模型的情况下离线 replay。
- 确定性环境闭环：通用 observation/action/transition 类型、hidden-wall GridWorld、唯一 allowlisted
  `move(direction)` 副作用面、动作预算和环境终态评分；Qwen3-4B full-feedback 8 / 8 最短路完成，
  feedback-withheld 0 / 8，模型 prompt、工具结果和环境状态可联合 replay。Qwen3-8B 的 8 条 clean
  writer 事件在 fresh-load 后 raw/exact-spec `8/8` 命中；无反馈 reader 的 no/retrieved/等 token 干扰
  为 `0/8 / 7/8 / 0/8`，32 条 writer/reader 轨迹可 tokenizer-only replay。
- full/dynamic/static/XLA 的真实 greedy text-generation parity。
- 严格的 GPT-2 config、Float32 safetensors、Conv1D/fused-QKV 映射与 GPT-2 byte-level BPE adapter；冻结 revision/checksum 不匹配时 fail closed。
- GPT-2 124M 的 10 组 tokenizer corpus、embedding、12 层 residual、final hidden、full logits 与 full/dynamic/static 8-step greedy text 均通过 Transformers reference parity。
- GPT-2 124M 的 16/64/256-token CPU/CUDA correctness 与 steady-state benchmark，以及 learned-position/GELU-New XLA 同构 smoke。
- 围绕 Attention、RoPE、prefill/decode 和 KV Cache 的 Pluto 可视化学习笔记。
- 默认测试套件全部通过；Reactant/XLA 专项测试需显式启用。

尚未具备：

- GPT-2 的 WebText 从零训练、论文 zero-shot quality、Medium/Large/XL、cross-attention 与分类 head 未复现；当前完成的是 124M 官方 checkpoint 的 Float32 推理/架构复现。
- Qwen3 MoE 官方 30B-A3B checkpoint 的 40K cache 容量、短/32-token GPU offload、跨请求 device LRU、中英文自然文本 cache-budget sweep、scalar/grouped CUDA scattered cache 与 500-request reuse 生命周期已实跑，但 full-window 40K prefill、长序列生成质量、layer-ahead 异步预取、grouped workspace byte cap、完整 AWQ/GPTQ、
  activation/KV-cache 量化与量化 GEMM；30B 的 48 层 prompt/cache-decode 真实 parity 与 dense 0.6B—32B 真实权重
  parity、native BF16、weight-only INT8/INT4 与 diagonal
  activation-aware clipping 已完成，不能再沿用早期“只验证 0.6B”的边界。
- 通用 Jinja 渲染器：Chapter 36 的实现是按官方模板逐分支手写的 Julia 渲染，并以模板 sha256 作为准入条件，不是通用模板引擎。
- 更完整的质量评估：Chapter 37 只覆盖 0.6B/1.7B/4B 与 8 个 MMLU subject，且全部是 0-shot、
  greedy 单次；8B 及以上、5-shot、多 seed 方差与更大 token 预算下的 generative 上界都未做。
  本章数字是本仓库自己的基线，不能与官方榜单并列。
- ANN/reranker、通用 redaction/摘要/遗忘、并发 writer、规划与反思；Chapter 42 仍是 single-writer
  append-only journal，同 run ID 幂等但不同 run 可重复写同一事实，只接纳 clean successful GridWorld
  episode，不是完整长期记忆系统。
- Qwen3-VL 当前已从 raw PNG/JPEG/RGB image 接到 content-list chat、vision 与
  cache-free decoder prefill，但尚无 multimodal KV cache、增量 decode、greedy/
  sampled image-to-text generation 或 video；真实权重验收只覆盖单图、batch 1、
  sequence 76 和全一 attention mask，padding/multi-image generation、长上下文、
  音频与其他传感器输入也未实现。
- 连续控制、动力学、外部 simulator 与实体机器人 adapter；Chapter 40/42 只有离散确定性 GridWorld，
  生产记忆强制 exact-spec，cross-spec 只用于 diagnostic control。timeout、execution failure、e-stop 与
  idempotent action safety 尚未定义，不等于具备真实设备控制或物理安全。
- 在线或持续学习机制。

更详细的能力盘点、验证范围与建议里程碑见 [`notes/current_status.md`](notes/current_status.md)。

大模型权重与真实 reference 存放在仓库外的持久模型目录，不使用易清理的
`/tmp`，也不提交进 Git；历史资产机器使用 `/home/yj/models/`，当前验证机
使用 `/home/ubuntu/models/modelscope/`。Qwen3 六尺寸（0.6B—32B）、
Qwen3-Embedding-0.6B、Qwen3-VL-2B-Instruct 与 GPT-2 124M 的完整资产和恢复命令见
[`notes/local_model_assets.md`](notes/local_model_assets.md)，六个 Qwen3
dense config 规格见 [`Chapter 11`](notes/episodes/episode03_model_family_and_large_weights/chapter11_qwen3_dense_family.md)，
Qwen3-VL vision-only 边界见 [`Chapter 43`](notes/episodes/episode09_qwen3_vl_multimodal_perception/chapter43_qwen3_vl_vision_architecture.md)，raw processor/chat 与 decoder prefill 边界见 [`Chapter 44`](notes/episodes/episode09_qwen3_vl_multimodal_perception/chapter44_qwen3_vl_multimodal_prefill.md)。

## 演进路线

```text
模型基本组件
    ↓
可训练、可生成、可评估的模型闭环
    ↓
记忆 + 规划 + 工具 + 多模态的智能体核心
    ↓
感知 → 决策 → 行动 → 反馈的具身闭环
    ↓
能长期互动、持续适应的“有生命感”AGI
```

模型组件不会在进入下一阶段后停止建设；它会始终作为底层主线，与智能体和具身实验互相驱动。

## 记录节奏

- **Chapter**：目标驱动的逻辑研发章节，不与自然周绑定。每章先定义核心问题、交付物和 Close 条件；条件满足后立即复盘关闭。
- **Episode**：围绕一条完整能力主线组织多个 Chapter，不与自然月绑定；主题闭环后总结验证证据、关键学习、失败尝试、架构决策和下一卷重点。
- **状态快照**：每次 Chapter Close 后按需更新，只描述仓库此刻真实具备的能力，明确区分“已实现”“已验证”和“尚未开始”。

```text
Open Chapter → 执行与验证 → 满足 Close 条件 → 复盘并更新状态 → Open Next Chapter
```

记录索引与模板见 [`notes/README.md`](notes/README.md)。

## 仓库结构

```text
src/
├── LifeAI.jl      # 按依赖顺序组装实现
├── api.jl         # 按职责分组的公共 API 清单
├── core/          # Attention、RoPE、Transformer、sampling
├── data/          # Tokenizer 与 DatasetLoader
├── io/            # HuggingFace config / safetensors 权重加载
├── models/        # GPT 模型
├── train/         # Zygote / Reactant-XLA 训练
├── generation/    # 共享采样/输入边界、文本生成与 KV Cache
├── agent/         # 工具、持久记忆、环境/action loop
└── eval/          # 任务质量、记忆与环境闭环评测

test/
├── runtests.jl    # 按 Episode / Chapter 分组的统一入口
├── episodes/      # 与 notes/episodes 一一对应的测试与 fixture
└── support/       # 跨 Chapter 复用的测试构造器与资产定位工具
examples/          # 最小训练和生成示例
notebook/          # 可交互的原理与实验记录
notes/
├── README.md      # Episode / Chapter 总目录
├── episodes/      # 按主题卷组织的 Chapter 正文
└── templates/     # Episode / Chapter 模板
```

## 开始使用

安装依赖：

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

运行默认测试：

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

在具备对应 XLA 后端的环境中运行专项测试：

```bash
LIFEAI_TEST_XLA=true julia --project=. -e 'using Pkg; Pkg.test()'
```

运行字符级 GPT 训练与生成示例（默认使用 Reactant/XLA GPU）：

```bash
julia --project=. examples/minigpt.jl
```

运行 RMSNorm + SwiGLU + tied embedding 的可恢复训练示例：

```bash
julia --project=. examples/modern_gpt.jl
```

也可以直接构建可独立开关的 modern 配置：

```julia
model = GPTModel(
    vocab_size,
    d_model,
    num_heads,
    num_layers;
    norm_type=:rmsnorm,
    mlp_type=:swiglu,
    tie_embeddings=true,
)
```

RTX 4090 D 上启动 Chapter 20 的 Qwen3-8B BF16 XLA 日常聊天：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_xla_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

默认 profile 是 4,096-token 总 context（最多 3,584-token
prompt/history + 512-token output）、batch 1、non-thinking、greedy；
XLA allocator 默认 fraction 为 `0.87` 且不预分配。首次进程仍需约
1.5 分钟编译 prefill/decode executable；Reactant 0.2.275 当前只持久化
autotune cache，不持久化完整 kernel executable。

需要 temperature/top-k/top-p sampling 时保留 Chapter 19 CUDA eager 入口：

```bash
julia --project=. --startup-file=no \
  scripts/run_qwen3_cuda_chat.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-8B
```

14B mixed RTN 仍是当前已实证的尺寸上限，但 0.377 tok/s 和不足 1 GiB
的短输入余量不适合作为日常默认。

运行 Chapter 36 的工具调用闭环（RTX 5080 16 GiB 上用 Qwen3-4B 验收）：

```bash
julia --project=. --startup-file=no scripts/run_qwen3_tool_loop.jl \
  /path/to/Qwen3-4B/<revision> \
  --variant qwen3_4b --revision <revision> \
  --out benchmark_results/chapter36/tool_loop_trace_run1.jsonl \
  --summary benchmark_results/chapter36/tool_loop_summary_run1.json
```

默认任务集是冻结的 20 题（8 题算术、6 题目录列举、6 题文件读取），文件系统工具被限制在
`--sandbox` 指定的根内。每个 model turn 写一行 JSONL：prompt sha256、prompt token 数、
generated ids、原始 completion、合法性判定、工具调用与返回、prefill/decode 秒数。
`greedy` 下两次独立进程运行的全部 turn 逐位相同。

也可以直接组装工具并渲染官方 tools prompt：

```julia
registry = default_agent_tools(pwd())
prompt = apply_qwen3_chat_template(
    tokenizer,
    [(role="user", content="1+2?")];
    tools=qwen3_tool_specs(registry),
)
calls = parse_qwen3_tool_calls(completion)
result = invoke_agent_tool(registry, first(calls.calls))
```

`tools`、`tool` 角色与 `tool_calls` 要求 tokenizer 携带官方 Qwen3 `chat_template`
（sha256 `a55ee1b1…74d8`，六个 dense 尺寸一致），其他模板一律 fail closed。

查看或严格选择 Qwen3 dense family member：

```julia
for spec in qwen3_dense_specs()
    println(spec.model_id, ": ", qwen3_dense_parameter_count(spec))
end

bundle = load_hf_qwen3_bundle(
    "/path/to/Qwen3-4B";
    variant=:qwen3_4b,
    revision=qwen3_dense_spec(:qwen3_4b).revision,
    max_seq_len=256,
)
```

`variant` 会校验所有架构 shape，但不会下载文件。0.6B—32B 六个 dense
尺寸均已完成真实权重逐层 parity（8B—32B 使用流式验证）；完整 tokenizer
到 text-generation reference 仍以 0.6B 为主，不能把两种验证口径混写。

运行 Qwen3-Embedding-0.6B 的最小 notes 语义检索：

```bash
julia --threads=8 --project=. --startup-file=no \
  examples/qwen3_embedding_memory.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B
```

也可以直接构建和查询内存索引：

```julia
bundle = load_hf_qwen3_embedding_bundle(
    "/path/to/Qwen3-Embedding-0.6B";
    max_seq_len=256,
)
memory = build_qwen3_semantic_memory(
    bundle,
    ["KV cache 避免重复计算历史 token。", "MRL 支持截取 embedding 前若干维。"];
    dimension=512,
)
hits = search_qwen3_semantic_memory(
    bundle,
    memory,
    "如何减少增量解码的重复计算？";
    top_k=1,
)
```

这是 dense exact、进程内 baseline；不包含持久化、ANN、reranker 或 agent
memory policy。Python oracle 只使用仓库 `.venv`，固定依赖见
`requirements/week22-reference.txt`。

加载冻结的 GPT-2 124M 并执行 greedy generation：

```julia
bundle = load_hf_gpt2_bundle(
    "/home/yj/models/huggingface/openai-community/gpt2/607a30d783dfa663caf39e06633721c8d4cfcd7e";
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    max_seq_len=256,
)
result = generate_hf_text(
    bundle,
    "The meaning of life is";
    strategy=:greedy,
    cache=:dynamic,
    max_new_tokens=8,
)
```

没有 NVIDIA GPU 时可以尝试 XLA CPU 后端：

```bash
LIFEAI_XLA_BACKEND=cpu julia --project=. examples/minigpt.jl
```

对比 CPU、CUDA GPU、XLA+CPU 和 XLA+GPU 的训练与 KV Cache 推理性能：

```bash
./scripts/benchmark_week03.sh
```

脚本会分别记录首编译/首次执行、post-compile warm-up 和 steady-state 指标，并保留逐 iteration 原始耗时，生成 TSV 原始数据与 Markdown 汇总。默认使用 3 个 warm-up step 和 30 个正式样本，避免一次性 runtime settling 扭曲 p90。配置项和指标口径见 [`Chapter 03 四后端性能对比`](notes/episodes/episode01_transformer_and_training_foundations/chapter03_reproducible_training.md#四后端性能对比)。

其中 XLA+GPU 会额外对比 no-cache、dynamic KV Cache 和 static KV Cache，并报告不同 shape 导致的 executable 数量与 cold compilation 总成本。

运行 Chapter 04 的五配置 CPU 对照，以及 baseline / modern 四后端 benchmark：

```bash
./scripts/benchmark_week04.sh
```

该脚本使用三个固定 seed 汇总短程 validation 结果，并将 cold compile、warm-up 与 steady-state 性能分开记录。实验用于验证组件可归因性和后端兼容性，不代表真实模型质量排名。
