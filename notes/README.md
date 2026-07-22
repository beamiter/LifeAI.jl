# LifeAI.jl 研发记录

这里记录 LifeAI.jl 从模型基本组件走向智能体和具身系统的长期演进。

## 索引

- [`current_status.md`](current_status.md)：最新项目状态、能力边界和近期里程碑。
- [`local_model_assets.md`](local_model_assets.md)：本机持久模型目录约定、Qwen3 revision/checksum 与 reference 复现命令。
- [`week01_transformer.md`](week01_transformer.md)：Week 01，项目骨架、Attention、RoPE 与 Transformer 基础。
- [`week02_gpt_xla_kv_cache.md`](week02_gpt_xla_kv_cache.md)：Week 02，最小 GPT、XLA 训练与 KV Cache 增量推理。
- [`week03_reproducible_training.md`](week03_reproducible_training.md)：Week 03（Closed），checkpoint、断点续训、validation / perplexity、梯度裁剪与 KV Cache 基线。
- [`week04_model_modernization.md`](week04_model_modernization.md)：Week 04（Closed），RMSNorm、SwiGLU、embedding / LM head 权重共享及独立对照实验。
- [`week05_tokenizer_data_pipeline.md`](week05_tokenizer_data_pipeline.md)：Week 05（Closed），byte-level / byte-BPE、Tokenizer artifact 版本化与无泄漏中文数据管线。
- [`week06_gqa_qwen3_parity.md`](week06_gqa_qwen3_parity.md)：Week 06（Closed），GQA、QK-Norm 与 Qwen3 dense 结构 parity，Qwen3 复现三阶段计划第一步。
- [`week07_hf_weight_loading.md`](week07_hf_weight_loading.md)：Week 07（Closed），safetensors / BF16 权重加载、HF config 映射与 Qwen3-0.6B logits / KV-cache 对齐。
- [`week08_hf_tokenizer_text_parity.md`](week08_hf_tokenizer_text_parity.md)：Week 08（Closed），HuggingFace Qwen3 tokenizer 导入、基础 chat template 与 text→text greedy generation parity，Qwen3 三阶段复现闭环。
- [`week09_qwen3_sampling_performance.md`](week09_qwen3_sampling_performance.md)：Week 09（Open），Qwen3 官方 temperature/top-k/top-p 采样 fidelity、长位置 correctness 与真实推理性能基线。
- [`qwen3_hf_config_mapping.md`](qwen3_hf_config_mapping.md)：Qwen3 HF `config.json` 与 `gpt_config` 的字段、权重名与布局映射契约（Week 07 已实现并验证）。
- [`weekly/`](weekly/)：Week plan、实验过程和 Close 回顾。
- [`monthly/`](monthly/)：月度总结和跨周能力变化。

## Week 的含义

这里的 Week 是一个**逻辑研发阶段**，不是七天自然周：

1. Open 一个 Week，写清核心问题、预期结果和 Close 条件。
2. 围绕目标实现、学习和验证，不为填满时间而增加任务。
3. Close 条件提前满足时，立即关闭当前 Week 并完成回顾。
4. 未完成项只有在仍然重要时才进入下一个 Week。
5. Close 后更新项目状态，再 Open 下一阶段。

## 记录原则

1. **结果可验证**：尽量附上测试、实验、指标、示例或代码位置。
2. **状态不混写**：明确区分计划中、已实现、已验证和已完成。
3. **记录认知变化**：不仅写做了什么，也写为什么这样做、哪里判断错了。
4. **围绕能力积累**：每项工作说明它增强了模型、智能体、具身闭环或工程基础中的哪一部分。
5. **保留历史语境**：旧计划可以过期，但不重写当时的判断；用新的状态快照说明后续结果。

开启新的 Week 或编写月度总结时，可以分别复制 [`weekly/TEMPLATE.md`](weekly/TEMPLATE.md) 和 [`monthly/TEMPLATE.md`](monthly/TEMPLATE.md)。
