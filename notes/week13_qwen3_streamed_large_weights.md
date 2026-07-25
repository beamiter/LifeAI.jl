# Week 13 — Qwen3 Streamed Loading and 8B/14B/32B Real-Weight Parity

> 状态：Open
>
> 开启记录：2026-07-25
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

| 证据层 | 目标状态 |
| --- | --- |
| 流式 reader 严格性（missing/unexpected/shape/index） | 默认离线覆盖 |
| 流式 vs in-memory 逐位一致（trace + decode） | 小 fixture 默认离线覆盖，0.6B/1.7B/4B 实权重 opt-in 复核 |
| 8B / 14B / 32B 资产 revision 与全文件 checksum | 冻结进 notes 与离线 fixture |
| untied LM head 真实权重 | 8B / 14B / 32B 全部实跑 |
| 三尺寸逐层 hidden / logits / dynamic decode parity | 显式容差内全部对齐 |
| 既有 0.6B—4B / GPT-2 parity 与默认测试 | 回归不变 |

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
