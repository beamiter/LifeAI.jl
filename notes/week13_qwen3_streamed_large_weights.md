# Week 13 — Qwen3 Streamed Loading and 8B/14B/32B Real-Weight Parity

> 状态：Closed
>
> 开启记录：2026-07-25
>
> 关闭记录：2026-07-25
>
> 依赖基线：[`Week 12 — Qwen3 Dense Family Real-Weight Parity`](week12_qwen3_dense_real_weights.md) 已 Closed，保持历史内容不变。
>
> 近期主线：Week 12 把真实权重 parity 扩展到 30 GiB RAM 内可全量加载的
> 1.7B / 4B 后，8B / 14B / 32B 仍是"结构 contract 有、真实权重无"的
> 最后缺口。本周实现流式 / 逐层加载，让权重远大于 RAM 的 checkpoint
> 也能完成逐层 parity 验证。

## 核心问题

> 不把完整参数树装进内存，能否仍然以与现有 in-memory 路径**完全相同的
> 数值语义**，对 8B / 14B / 32B 官方 checkpoint 完成逐层 hidden、logits
> 与 KV cache decode 的真实权重验证？

三个尺寸全部 untied LM head——这是 Week 11 只有缩小模型证据、真实权重
从未触达的分支。Float32 全量加载需要 30.5 / 55.0 / 122.1 GiB，超出本机
30 GiB RAM，因此必须换加载策略而不是换机器：

1. **Julia 侧**：safetensors 只解析 header 建立 tensor → 分片/偏移 索引，
   逐层按需读取 BF16 → Float32，用现有 `_block_with_kv_cache` 逐层执行
   prefill 与 decode，用完即弃。峰值内存 ≈ 单个最大权重（32B 的
   embedding / lm_head ≈ 3.1 GiB Float32）+ 常数级激活。
2. **Python reference 侧**：Transformers Float32 计算下 32B 同样放不进
   RAM；用 accelerate disk offload（`max_memory` + `offload_folder`）
   生成与 Week 07/12 完全同口径的逐层 Float32 reference。
3. **数值语义不变**：流式路径必须与 in-memory 路径在小模型上**逐位一致**
   （同一算子、同一顺序、同一 Float32 计算），流式只改变权重驻留方式。

## 本周资源边界（先冻结再执行）

- 机器：30 GiB RAM、NVMe 437 GiB 空闲（下载前）。
- 下载：8B（5 分片，≈ 16.4 GB）、14B（8 分片，≈ 29.5 GB）、32B
  （17 分片，≈ 65.1 GB），共 ≈ 111 GB，revision 沿用 Week 11 冻结值。
- Python offload：32B Float32 offload 目录 ≈ 122 GiB，放在
  `/home/yj/models/` 所在盘的临时目录，用完删除；不放 `/tmp`。
- 计算 dtype 仍为 Float32；BF16 只是存储格式。不实现 native BF16 compute。

## 实现范围

- 新增流式 safetensors 读取：header-only 索引（单文件与 index 分片）、
  按名读取单个 tensor、与现有 loader 相同的 fail-closed 校验（missing /
  unexpected / duplicate / shape / dtype / index 一致性）。
- 新增 `stream_hf_qwen3_forward`（命名以实现为准）：逐层流式执行
  embedding → blocks → final norm → logits 的 trace，及基于 dynamic KV
  cache 的单 token decode 第二遍流式执行；复用
  `load_hf_qwen3_config` 的 variant 识别与 `_block_with_kv_cache`。
- 单层参数映射从 `load_hf_qwen3_parameters` 中提炼共用，避免两份映射
  语义漂移。
- `export_qwen3_reference.py` 增加 disk-offload 选项，为 8B / 14B / 32B
  生成逐层 Float32 reference（同一 token-id fixture）。
- 下载三个尺寸冻结 revision 完整资产，记录全部文件 SHA256。
- 流式 vs in-memory 逐位一致性用小 fixture（含人工分片）进入默认离线
  测试；真实三尺寸经 `LIFEAI_QWEN3_8B/14B/32B_MODEL_DIR` opt-in。

## 验证分层

| 证据层 | 最终状态 |
| --- | --- |
| 流式 reader 严格性（missing/unexpected/shape/index） | 默认离线覆盖（44 项） |
| 流式 vs in-memory 逐位一致（trace + decode） | 合成 untied/tied 分片 fixture 默认离线覆盖（23 项）+ 真实 0.6B 逐位相等复核 |
| 8B / 14B / 32B 资产 revision 与全文件 checksum | 已冻结进 notes 与离线 fixture |
| untied LM head 真实权重 | 8B / 14B / 32B 全部实跑通过 |
| 三尺寸逐层 hidden / logits / dynamic decode parity | 全部对齐；block 用尺度感知容差（实测 scaled ≤ 6.9e-7），归一化输出用绝对容差 |
| 流式峰值内存 | 6.94 / 8.84 / 8.87 GiB，对照全量 30.51 / 55.02 / 122.06 GiB |
| 既有 0.6B—4B / GPT-2 parity 与默认测试 | 回归不变（分进程协议下全绿） |

## Close 条件

- 8B / 14B / 32B 冻结 revision 完整资产位于 `/home/yj/models/`，全部文件
  SHA256 记入 `local_model_assets.md` 与离线 fixture；`config.json`
  checksum 与 Week 11 冻结值一致。
- 流式路径与 in-memory 路径在离线 fixture 和至少一个真实小尺寸上
  （trace 全部 stage + dynamic decode）数值完全一致。
- 三个尺寸经同一 token-id fixture 的 embedding、每层 block、final
  hidden、full logits 与 dynamic cache decode logits 全部在显式记录的
  Float32 容差内，argmax 一致；variant 自动识别正确。
- 流式执行峰值内存有实测记录，且显著低于对应 Float32 全量加载需求。
- Week 13 测试进入默认套件且不联网；默认全套与既有专项无回归。
- 文档明确：静态 KV cache 与 XLA 路径仍需全量参数，8B+ 的 static/XLA/
  text/sampling/benchmark 未验证；native BF16 compute、量化仍未实现。

## 非目标

- 不实现 native BF16 / FP16 compute、量化、GGUF、tensor parallel 或
  GPU offload；流式只服务于验证，不承诺生产级吞吐。
- 不做 8B/14B/32B 的静态 cache、XLA、text 端到端、sampling replay 或
  性能 benchmark；dynamic cache decode 是本周的 cache 证据边界。
- 不改动 Week 06—12 已冻结的历史结论。
- 不下载 Qwen3 MoE 或其他架构。

## 过程记录

### 2026-07-25：Open

- Week 12 保持 Closed；Week 13 承接最后三个 dense 尺寸的真实权重验证。
- 三个 repo 冻结 revision 的分片清单已经 HF API 核对：8B 5 分片、
  14B 8 分片、32B 17 分片，共 ≈ 111 GB，已开始后台下载。
- 设计定稿：Julia 侧 header-only 索引 + 逐层读取 + 复用
  `_block_with_kv_cache`；Python 侧 accelerate disk offload 生成同口径
  Float32 reference；流式与 in-memory 的逐位一致性作为第一道闸门。

### 2026-07-25：实现与首批验证

- 新增 `src/io/hf_streaming.jl`：`open_safetensors_reader`（header-only
  索引，沿用单文件/index 的全部 fail-closed 校验）、
  `read_safetensors_tensor`、`stream_hf_qwen3_forward`。embedding 按
  token 行直接从磁盘 gather（行主序 `(vocab, d_model)` 每行连续），
  全程不物化 vocab×d_model 矩阵；单层参数映射提炼为
  `_qwen3_block_parameters` 与 in-memory loader 共用。
- 111 GB 三尺寸资产下载完成；全部文件 SHA256 已计算，三个
  `config.json` 与 Week 11 冻结值一致，tokenizer 三件套与 0.6B 字节
  相同。
- **第一道闸门通过**：真实 0.6B 上流式与 in-memory 的 embedding、28 层
  block、final hidden、logits、dynamic decode **全部逐位相等**；合成
  untied 宽 attention 分片 fixture 与 tied 单文件 fixture 同样逐位
  相等，reader 严格性测试 44 项、逐位一致测试 23 项通过。
- `export_qwen3_reference.py` 新增 `--offload-dir` / `--max-cpu-memory`
  （accelerate disk offload，Float32 计算语义不变）；8B/14B/32B 三份
  逐层 reference 已在 12 GiB CPU 预算下导出。
- **8B 流式 parity 通过**：36 层全部 argmax 一致，final hidden max-abs
  `5.15e-5`、logits `3.05e-5`、dynamic decode `2.86e-5`；untied LM head
  真实权重首次验证。streamed 全程峰值 RSS 6.94 GiB，对照 Float32 全量
  加载需要的 30.51 GiB；耗时 157 s。
- **14B / 32B 流式 parity 通过**：14B logits `2.77e-5`、decode
  `2.05e-5`，峰值 RSS 8.84 GiB（对照 55.02 GiB），283 s；32B logits
  `2.79e-5`、decode `4.01e-5`，final hidden `1.37e-4`，峰值 RSS
  **8.87 GiB（对照 122.06 GiB）**，609 s。全部 stage argmax 一致。
- 容差冻结（约 2 倍以上余量）：blocks `1e-2`、final hidden / logits
  `5e-4`、decode `2e-4`、embedding `1e-4`；连同全部文件 checksum、
  实测 parity 与峰值内存写入离线 fixture。

### 2026-07-25：block 容差量纲修正

- week13 opt-in 在 `Pkg.test` 下复跑时，14B/32B 共 92 项 block 级
  max-abs 断言失败（1.4e-2—2.5e-2 > 1e-2），而同一轮的 final hidden、
  logits、decode、argmax 全部通过；plain `julia` 下同一脚本两次运行
  逐位相同。定位：`Pkg.test` 的 `--check-bounds=yes` 改变 SIMD 归约
  顺序；Qwen3 大模型中层 hidden state 有万级激活 outlier（14B 最大
  12,627、32B 最大 19,620），Float32 在该量级的 ulp 为 1e-3—2e-3，
  数个 ulp 的合法漂移在绝对值上就是 1e-2 量级，相对误差始终 ~1e-6。
- 修正：block 输出（未归一化、量纲随激活幅值）改用尺度感知容差
  `max-abs 差 ≤ 1e-5 × max(1, 该层 reference 最大激活)`，另设 5e-2
  绝对天花板；实测三尺寸 scaled 误差最差 `6.9e-7`，失败轮折算亦仅
  `1.3e-6`。final hidden / logits / decode 经过归一化，量纲稳定，
  维持绝对容差不变。
- 教训：对未归一化的中间激活用绝对容差，等价于在最大 outlier 的 ulp
  上赌运行环境不变；容差必须与被比较量的量纲一致。

### 2026-07-25：OOM 教训与测试进程协议

- 把 Week 12 两个全量加载 integration（4B 峰值 ≈ 16 GiB）与 Week 13
  三个流式 integration 放进**同一个** `Pkg.test` 进程时，进程被内核
  OOM KILL——Julia 堆在多个大 testset 间累积增长，虽然单项峰值都远低
  于 30 GiB，叠加后仍越界。Week 13 单独进程 `545 / 545` 通过。
- 结论与协议：tied 尺寸（`LIFEAI_QWEN3_1_7B/4B_MODEL_DIR`）与 streamed
  尺寸（`LIFEAI_QWEN3_8B/14B/32B_MODEL_DIR`）的 opt-in integration 分
  两个进程运行；默认离线套件不受影响。该边界记入文档，不靠运气。

### 2026-07-25：验证与 Close

- 分进程协议下三轮全绿：week12 opt-in 全套通过（Week 12 专项
  `209 / 209`）；week13 opt-in 全套通过（Week 13 专项 `689 / 689` =
  离线 283 + 三尺寸 integration 406）；默认离线全套 `4652 / 4652`
  （Week 13 离线专项 `283 / 283`）。
- 本周 src 改动限于：新增 `src/io/hf_streaming.jl`、从
  `load_hf_qwen3_parameters` 中提炼 `_qwen3_block_parameters` /
  `_qwen3_validate_semantics` / `_qwen3_validate_tensor_names`（语义
  不变，Week 07—12 离线测试逐项复核通过）；既有 Reactant/XLA 专项
  路径未改动。

## Close 回顾

- **完成了什么**：流式 / 逐层 safetensors 加载让 30 GiB RAM 的机器完成
  了 8B / 14B / 32B（Float32 全量需 30.5—122.1 GiB）的真实权重逐层
  parity；untied LM head 分支从缩小模型证据升级为三个官方 checkpoint
  的真实证据；至此 Qwen3 dense family 六个尺寸全部具备真实权重逐层
  验证。
- **验证证据**：流式 vs in-memory 逐位一致（合成 fixture + 真实
  0.6B）；三尺寸 logits max-abs ≤ `3.05e-5`、decode ≤ `4.01e-5`、
  argmax 全一致；峰值 RSS 最高 8.87 GiB（32B，节省 13.8×）；默认全套
  `4652 / 4652` 与两轮 opt-in 全绿。
- **没有完成及原因**：8B+ 的静态 cache、XLA、text 端到端、sampling
  与 benchmark 未做——流式路径服务于验证而非生产吞吐，静态 cache 与
  XLA 仍需全量参数驻留；native BF16 compute、量化仍为非目标。
- **最重要的认知变化**：其一，"权重大于内存"不是验证的硬边界——
  header-only 索引 + 逐层执行以 ~9 GiB 上限跑通了 122 GiB 的模型，且
  与全量路径逐位一致；其二，绝对容差对未归一化中间激活是错误量纲，
  万级 outlier 上几个 ulp 的合法漂移就能击穿它，容差判据必须跟随被
  比较量的尺度；其三，多个大 testset 同进程的堆累积会 OOM，测试协议
  要显式声明进程边界。
- **是否满足 Close 条件**：是。资产 checksum、逐位一致闸门、三尺寸
  parity、峰值内存实测、默认套件接入与无回归、文档边界均已落实。
