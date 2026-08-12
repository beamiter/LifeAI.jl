# Chapter 12 — Qwen3 Dense Family Real-Weight Parity

> 状态：Closed
>
> 开启记录：2026-07-24
>
> 关闭记录：2026-07-25
>
> 依赖基线：[`Chapter 11 — Qwen3 Dense Family Completion`](../../episodes/episode03_model_family_and_large_weights/chapter11_qwen3_dense_family.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Chapter 11 把六个官方 dense 尺寸钉死为结构 contract，但真实
> checkpoint 逐层 parity 仍只有 0.6B。本周把真实权重验证扩展到当前机器
> 可实跑的所有尺寸，专门排查"只在其他尺寸暴露"的加载 bug。

## 核心问题

> `load_hf_qwen3_model` 对 0.6B 之外的官方 checkpoint，是否真的能把
> 真实 BF16 权重完整、正确地映射进 Julia 模型，并在逐层 hidden、
> logits 和 KV cache decode 上与 HuggingFace Transformers 对齐？

Chapter 11 的结构 contract（config 识别、topology、参数量）不能替代真实权重
证据。以下加载路径在真实 checkpoint 上仍只被 0.6B 单点覆盖，每一条都可能
藏有只在其他尺寸暴露的 bug：

1. **分片 safetensors index**：0.6B 是单文件 `model.safetensors`；loader 的
   `model.safetensors.index.json` 分片路径此前只有合成 fixture 证据，没有
   真实 checkpoint 实跑过。
2. **`Q width == hidden` 分支**：0.6B/4B/32B 的 Q projection width 大于
   hidden；1.7B 是六个尺寸中唯一 `Q width == hidden`（2048 == 2048）的
   官方 checkpoint，真实权重从未走过这一分支。
3. **更深/更宽的逐层误差累积**：36 层、hidden 2560 的 Float32 误差是否仍
   收敛在 0.6B 建立的容差量级，需要实测而不是外推。

## 本周资源边界（先冻结再执行）

当前机器 30 GiB RAM、RTX 5080 16 GiB。LifeAI 计算 dtype 为 Float32，
完整加载所需参数内存：

| variant | 参数量 | Float32 参数内存 | 30 GiB RAM 下可实跑 |
| --- | ---: | ---: | --- |
| Qwen3-1.7B | 1,720,574,976 | ≈ 6.4 GiB | 是 |
| Qwen3-4B | 4,022,468,096 | ≈ 15.0 GiB | 是（紧张，需单进程串行） |
| Qwen3-8B | 8,190,735,360 | ≈ 30.5 GiB | 否 |
| Qwen3-14B | 14,768,307,200 | ≈ 55.0 GiB | 否 |
| Qwen3-32B | 32,762,123,264 | ≈ 122.1 GiB | 否 |

因此本周真实权重验证范围为 **1.7B 与 4B**：

- 1.7B：tied embedding、`Q width == hidden`、28 层——真实覆盖分支 2。
- 4B：tied embedding、`Q width > hidden`（4096 > 2560）、36 层、分片
  safetensors——真实覆盖分支 1 与更深的误差累积。

完成后六尺寸中 3/6 具备真实逐层 parity；tied embedding 的三个官方尺寸
全部实跑。8B/14B/32B 的真实权重（含 untied LM head 的真实实跑）超出
Float32 全量加载的内存上限，属于后续低精度 / 流式加载工作，本周不以任何
方式伪装成已验证。

## 实现范围

- 按 [`../../local_model_assets.md`](../../local_model_assets.md) 约定下载 Chapter 11 冻结
  revision 的 1.7B（`70d244cc…`）与 4B（`1cfa9a72…`）完整资产（config、
  tokenizer、generation_config、权重与分片 index），记录全部文件 SHA256。
- 复用 `scripts/export_qwen3_reference.py` 为两个尺寸生成 Transformers
  Float32 逐层 reference（同一 token-id fixture，reference 存入各自
  revision 目录的 `lifeai-references/week12-parity/`）。
- 用 `scripts/verify_qwen3_parity.jl` 验证 embedding、每个 block、
  final hidden、full logits 与 dynamic/static cache decode logits 对齐；
  两个尺寸的 variant 必须被识别为对应官方 spec 且参数量精确一致。
- 若任何一层出现超出容差的偏差，定位是 loader 映射、shape 语义还是数值
  累积问题并修复；修复必须回归 0.6B 与 GPT-2 既有 parity。
- 新增 `test/episodes/episode03_model_family_and_large_weights/chapter12_qwen3_dense_real_weights/test_qwen3_dense_real_weights.jl`：离线部分冻结两个尺寸的权重文件 checksum 与
  parity 结论 fixture，默认不联网；真实 integration 经
  `LIFEAI_QWEN3_1_7B_MODEL_DIR` / `LIFEAI_QWEN3_4B_MODEL_DIR` opt-in。

## 验证分层

| 证据层 | 最终状态 |
| --- | --- |
| 1.7B / 4B 资产 revision 与全文件 checksum | 已冻结进 notes 与离线 fixture |
| 真实分片 safetensors index 加载 | 1.7B（2 分片）与 4B（3 分片）均实跑覆盖 |
| `Q width == hidden` 真实权重分支 | 1.7B 实跑覆盖 |
| 逐层 hidden / logits / decode parity | 两尺寸全部在显式容差内，argmax 全一致 |
| 0.6B 与 GPT-2 既有 parity | 默认全套无回归；本周零 src 改动 |
| 8B / 14B / 32B 真实权重 | 明确不可实跑，保持边界陈述 |

## Close 条件

- 1.7B 与 4B 的 immutable revision 完整资产位于 `/home/yj/models/`，全部
  文件 SHA256 记入 `../../local_model_assets.md` 与仓库离线 fixture。
- 两个尺寸经同一 token-id fixture 的逐层 hidden、final hidden、full
  logits、dynamic/static decode logits 全部在显式记录的 Float32 容差内，
  argmax 一致；4B 必须经真实 `model.safetensors.index.json` 分片加载。
- `load_hf_qwen3_model` 对两个尺寸自动识别 variant，参数量与 Chapter 11
  冻结值精确一致；发现的任何尺寸相关 bug 已修复并有针对性测试。
- Chapter 12 opt-in integration 与离线 fixture 测试进入 `test/runtests.jl`，
  默认全套与既有 XLA 专项无回归。
- 文档明确 8B/14B/32B 在 30 GiB RAM 下无法 Float32 实跑，untied LM head
  仍只有缩小模型证据，不写成已验证。

## 非目标

- 不下载 8B / 14B / 32B 巨型权重，不实现流式 / 低精度 / device-offload
  加载来强行覆盖；不伪造任何逐层数据。
- 不做 1.7B / 4B 的 tokenizer text 端到端、sampling replay 或 CPU/GPU
  benchmark；Chapter 08/09 已在 0.6B 钉死这些语义，本周聚焦权重与 logits。
- 不实现 native BF16 compute、量化、MoE、YaRN / RoPE scaling。
- 不改动 Chapter 06—11 已冻结的历史结论与文档。

## 过程记录

### 2026-07-24：Open

- Chapter 11 保持 Closed；Chapter 12 承接真实权重 parity 扩展。
- 冻结资源边界：30 GiB RAM 下 Float32 全量加载上限为 4B；8B+ 明确出界。
- 计划复用 Chapter 07 的 export / verify 脚本与 token-id fixture，不新造
  验证口径。

### 2026-07-25：下载、reference 与逐层 parity

- 按冻结 revision 下载 1.7B（2 分片，约 4.1 GB）与 4B（3 分片，约
  8.0 GB）完整资产；`config.json` SHA256 与 Chapter 11 冻结值逐字节一致，
  tokenizer 三件套与 0.6B 资产字节相同。两个 checkpoint 都是分片
  safetensors——1.7B 也实跑 index 路径，比计划多覆盖一次。
- `export_qwen3_reference.py`（Transformers 4.51.0 / torch 2.7.1+cpu、
  BF16 存储 → Float32 计算）对两个尺寸生成同一 token-id fixture 的
  逐层 reference，存入各 revision 目录 `lifeai-references/week12-parity/`。
- `verify_qwen3_parity.jl` 结果（max-abs，全部 stage argmax 一致）：
  - **1.7B**：variant 识别 `qwen3_1_7b`，参数量 1,720,574,976 精确一致；
    embedding 0.0，28 层 block 最大 `4.88e-3`（mean 均 ≤ `6.01e-5`），
    final hidden `1.72e-4`，full logits `9.35e-5`，dynamic/static decode
    均 `3.05e-5`；加载 19.9 s。
  - **4B**：variant 识别 `qwen3_4b`，参数量 4,022,468,096 精确一致；
    embedding 0.0，36 层 block 最大 `5.13e-3`（末层大激活下的 BF16
    舍入，mean `1.08e-5`），final hidden `1.07e-4`，full logits
    `3.53e-5`，dynamic/static decode 均 `2.00e-5`；真实分片 index 加载
    32.0 s，进程峰值内存在 30 GiB 内。
- **无需修改任何 src 代码**：loader 对两个新尺寸一次通过。Chapter 11 修复
  的 QK-Norm 参数计数与钉死的宽 attention/tied 分支已经覆盖了这两个
  尺寸会触发的差异；本周把该结论从结构证据升级为真实权重证据。
- 冻结容差（对实测值约 2 倍以上余量）进入离线 fixture：blocks `1e-2`、
  final hidden / logits `5e-4`、decode `2e-4`、embedding `1e-4`。
- 新增 `test/episodes/episode03_model_family_and_large_weights/chapter12_qwen3_dense_real_weights/test_qwen3_dense_real_weights.jl`：离线 asset contract（revision/checksum 与
  Chapter 11 specs 交叉核对、分片清单、parity 结果小于容差）默认运行；
  `LIFEAI_QWEN3_1_7B_MODEL_DIR` / `LIFEAI_QWEN3_4B_MODEL_DIR` opt-in
  integration 重算文件尺寸、variant、参数量与全部 parity 断言。

### 2026-07-25：验证与 Close

- 带两个 integration 的完整套件：Chapter 12 专项 `209 / 209` 通过（离线
  85 项 + 1.7B integration 57 项 + 4B integration 67 项），其余各周
  全部通过。
- 默认全套（离线）`4369 / 4369` 通过，其中 Chapter 12 离线专项 `85 / 85`；
  Chapter 05—11 计数与 Close 时完全一致，无回归。
- 本周零 src 代码改动，只新增测试与 fixture；既有 Reactant/XLA 专项
  测的是未变化的 src 路径，不受影响。
- opt-in integration 双尺寸串行运行 1 分 40 秒（含 4B 16 GiB Float32
  加载），单进程峰值内存在 30 GiB 内。

## Close 回顾

- **完成了什么**：把真实 checkpoint 逐层 parity 从 0.6B 扩展到 1.7B 与
  4B，真实覆盖分片 safetensors index（两种分片数）与
  `Q width == hidden` 分支；tied embedding 的三个官方尺寸全部实跑；
  资产、reference、容差与 parity 结果全部冻结进离线 fixture 与 opt-in
  integration。
- **验证证据**：两尺寸逐层 max-abs（1.7B logits `9.35e-5`、4B logits
  `3.53e-5`，decode ≤ `3.05e-5`）、argmax 全一致；默认全套
  `4369 / 4369`，opt-in `209 / 209`。
- **没有完成及原因**：8B/14B/32B 真实权重与 untied LM head 的真实实跑
  仍未执行——Float32 全量加载需要 30.5—122.1 GiB，超出本机 30 GiB
  RAM；这属于后续低精度 / 流式加载工作，不用 shape 证据冒充。
- **最重要的认知变化**：担心的"只在其他尺寸暴露的 loader bug"没有出现
  ——Chapter 11 的结构 contract（QK-Norm 参数计数、宽 attention、tied
  分支钉死）已经提前消化了这两个尺寸的差异点，本周的价值是把这个结论
  从"结构推断"升级为"真实权重证据"，并留下可复跑的验证闭环。
- **是否满足 Close 条件**：是。资产 checksum、双尺寸容差内逐层对齐、
  variant 识别与精确参数量、默认测试接入与无回归、8B+ 边界陈述均已
  落实。
