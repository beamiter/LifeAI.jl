# Week 22 — Qwen3-Embedding-0.6B 与最小语义记忆

> 状态：Closed（2026-07-31）

## Open：核心问题

现有 Qwen3 tokenizer、严格 safetensors loader 与 native BF16 推理路径，
能否在不混淆 causal-LM 与 embedding checkpoint 契约的前提下，复现
Qwen3-Embedding-0.6B 的 last-token pooling、归一化与 MRL 语义，并驱动
一个可验证的本地语义检索闭环？

## 预期结果

本阶段 Close 时，应当可以展示或验证：

1. 冻结 Qwen3-Embedding-0.6B revision/config/tokenizer/权重 provenance，
   严格识别其 151,669 vocabulary、32K context 与独立 embedding variant。
2. 原生 BF16 CPU/CUDA 路径支持 instruction formatting、变长批输入、
   last-token pooling、L2 normalization 与可配置 MRL 输出维度。
3. HuggingFace reference 的 token ids、final hidden/normalized embeddings、
   similarity matrix 与 top-k 排名完成 parity，并有最小语义记忆示例。

## 计划

| 工作项 | 所属主线 | 交付物 | 验收方式 | 状态 |
| --- | --- | --- | --- | --- |
| 冻结 embedding checkpoint contract | 模型 | 独立 spec、严格 config/tokenizer/asset 校验 | 离线 fixture + 真实文件 checksum | 完成 |
| 实现 embedding 推理 | 模型 / 工程 | pooling、MRL、批输入、CPU/CUDA API | 单元测试 + HF 数值 parity | 完成 |
| 实现语义记忆 | 智能体 | 内存索引、cosine top-k、notes 检索示例 | 固定 corpus/query 排名 | 完成 |
| 记录性能与边界 | 学习 / 工程 | reference/benchmark JSON、复现脚本 | 冻结原始指标 | 完成 |

## Close 条件

- 官方 0.6B config 与 tokenizer/权重资产有 immutable revision 和 checksum，
  与普通 Qwen3-0.6B 的 vocabulary/context 差异显式 fail closed。
- pooling 同时覆盖无 padding、left padding 与 right padding；零 token、
  非连续 mask、越界 MRL dimension 和零范数输入 fail closed。
- 1024/512/256/128/64 维输出均为先截断后归一化，norm 与 similarity
  在显式容差内对齐 HF reference。
- 至少覆盖中文、英文、代码的固定 query/document corpus；LifeAI 与 HF
  top-k 排名一致，最小语义记忆可返回稳定结果。
- Week 22 专项、默认完整测试以及可用硬件上的真实权重验证全部通过；
  复现命令、性能、未覆盖边界和原始证据进入仓库。

## 学习重点

- 要理解的概念：decoder-only embedding、last-token pooling、instruction
  aware retrieval、Matryoshka Representation Learning 与 cosine retrieval。
- 要亲手实现的关键组件：mask-aware pooling、截断后重新归一化、批处理
  attention mask、top-k semantic memory。
- 要验证的假设：现有 Qwen3 BF16 数值路径可以复用，但 checkpoint contract、
  tokenizer vocabulary 和输出头语义不能直接沿用普通 dense variant。

## 风险与取舍

- 本 Week 不实现 reranker、持久向量数据库、ANN index 或 XLA resident
  embedding service；先冻结 dense exact-search correctness baseline。
- 沙箱内 `nvidia-smi` 可能因设备隔离而失败，不能据此推断宿主机没有
  GPU；必须在获准的宿主环境复核，再决定 CUDA benchmark 是否可执行。
- Python reference 使用仓库内 `.venv`；虚拟环境本体不提交，只提交精确
  requirements 与导出脚本。

## 实验与过程记录

- 2026-07-31：Week 22 Open。官方 revision 冻结为
  `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；模型资产按既有约定存放在
  仓库外持久模型目录。
- 真实 tokenizer 首次加载时严格拒绝了与 causal-LM 不同的
  `Sequence(ByteLevel, TemplateProcessing)` post-processor。最终实现只接受
  官方的 `A + <|endoftext|>` / `A + B + <|endoftext|>` 模板，并把
  `add_special_tokens=true` 的尾 token 行为纳入 token-id parity；没有把
  post-processor 校验泛化为任意 Sequence。
- 真实 safetensors 的 310 个 tensor 使用 base-model namespace
  （`embed_tokens.* / layers.* / norm.*`），而不是 causal-LM 的
  `model.*`。embedding 专用 loader 只接受该 namespace，再映射给共享的
  严格 Qwen3 参数校验；causal-LM loader 未放宽。
- Python reference 使用项目内 `.venv`：Python `3.10.12`、
  PyTorch `2.7.1+cpu`、Transformers `4.51.3`、Tokenizers `0.21.4`。
  8 条文本、35-token padded sequence、280 padded / 181 valid tokens 的
  BF16 load / forward 为 `1.589 / 1.246 s`，峰值 RSS `1238.79 MiB`。
- Intel Core i9-14900K、Julia `1.12.6`、8 Julia/BLAS threads 上，
  8 个资产校验 / native BF16 load / 同批 forward 为
  `2.436 / 8.832 / 7.687 s`，峰值 RSS `3361.09 MiB`。
- token ids 与 attention mask 逐项相同。1024/512/256/128/64 维
  embedding max-abs 分别为
  `0.00360 / 0.00502 / 0.00667 / 0.00826 / 0.00927`；similarity
  max-abs 为
  `0.00652 / 0.00474 / 0.00394 / 0.00830 / 0.00820`，均低于冻结的
  `0.02` 门槛。五档共 15 组完整 top-k 排名全部一致，三个 query 的
  首项分别稳定命中文档 1/2/3，报告顶层 `closed=true`。
- `examples/qwen3_embedding_memory.jl` 实跑仓库四份 notes；默认 query
  的首项是 Week 22（score `0.5927`），随后是 Week 17 / Week 21。
  这是内存内 dense exact cosine baseline，不是持久向量库或 agent 长期记忆。
- 宿主机复核确认 RTX 4090 D、NVIDIA driver `570.153.02` 与 CUDA
  runtime `12.9.0` 可用。相同 8×35 batch 的 CUDA cold / warm forward
  为 `18.959 / 0.0641 s`，参数上传 `1.103 s`；冷/热 embedding max-abs
  为 `0`。GPU token/mask、五档 embedding/similarity、15 组完整 top-k
  与 semantic memory 门禁全部通过，CUDA report 顶层 `closed=true`。
- Week 22 离线专项 `93 / 93`，加真实 CPU 权重专项 `103 / 103`。

复现 reference 与验收：

```bash
/home/ubuntu/.local/bin/uv venv --python /usr/bin/python3 .venv
/home/ubuntu/.local/bin/uv pip install \
  --python .venv/bin/python \
  -r requirements/week22-reference.txt

.venv/bin/python scripts/export_qwen3_embedding_reference.py \
  --model-dir /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B \
  --output test/fixtures/week22_qwen3_embedding/reference.json

julia --threads=8 --project=. --startup-file=no \
  scripts/verify_qwen3_embedding_parity.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B \
  test/fixtures/week22_qwen3_embedding/reference.json \
  benchmark_results/week22/qwen3_embedding_0_6b_cpu.json

LIFEAI_WEEK22_EMBEDDING_DEVICE=cuda \
julia --threads=8 --project=. --startup-file=no \
  scripts/verify_qwen3_embedding_parity.jl \
  /home/ubuntu/models/modelscope/Qwen/Qwen3-Embedding-0.6B \
  test/fixtures/week22_qwen3_embedding/reference.json \
  benchmark_results/week22/qwen3_embedding_0_6b_cuda.json
```

冻结证据：

```text
test/fixtures/week22_qwen3_embedding/reference.json
SHA256 669694c860798fb8496dd1be199fc648a5779fd7860000c81fe8ca3d0153b322

test/fixtures/week22_qwen3_embedding/assets.json
SHA256 f02a10758da8b561a9d111823e26d0f4cca05ad905408d3737842c2342bf7782

benchmark_results/week22/qwen3_embedding_0_6b_cpu.json
SHA256 1e2098c5cfc1ba08e941beb1fdd9c012688ab55a518c528228b2b6b2372fe668

benchmark_results/week22/qwen3_embedding_0_6b_cuda.json
SHA256 5d677d499794bab74c73b22ea0c8342a62fb96f72d942e58eb0655809d839be6
```

## Close 回顾

- **完成了什么**：新增独立 embedding spec/asset/tokenizer contract，
  base-model safetensors 映射、变长批 attention mask、无 logits/无保留 KV
  的 native BF16 hidden-state 前向、mask-aware last-token pooling、五档
  MRL、cosine similarity、dense exact semantic memory，以及可实跑的
  notes 检索示例。
- **验证证据**：8 个官方资产逐 SHA256；PyTorch BF16 oracle 与 Julia
  CPU/CUDA token/mask、五档 embedding/similarity/top-k 全通过；semantic
  memory 完整排名与首项一致；离线专项 93/93、真实 CPU 专项 103/103，
  默认完整测试 5,582/5,582 通过。
- **没有完成及原因**：reranker、ANN、持久化、增量更新、XLA resident
  service 与 agent memory policy 均不在本 Week 范围。
- **最重要的认知变化**：embedding checkpoint 不是“去掉 LM head 的同一
  Qwen3-0.6B”。它还改变 vocabulary/context、tokenizer 尾 token、
  SentenceTransformers namespace 与输出语义；这些差异必须成为独立
  fail-closed contract。与此同时，已有 Qwen3 BF16 block 数值路径确实可
  复用，新增能力集中在输入 mask、最终 hidden 与 pooling 边界。
- **是否满足 Close 条件**：是。真实 CPU 与 RTX 4090 D CUDA 权重验证及
  全部门禁均已通过，两份验收报告顶层均为 `closed=true`。
- **带到下一 Week 的问题**：优先进入 Qwen3-VL 的视觉输入与最小图文
  parity，还是先把 embedding baseline 扩展为持久化/增量 semantic memory？
