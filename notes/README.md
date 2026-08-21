# LifeAI.jl 研发之书

这里记录 LifeAI.jl 从模型基本组件走向智能体和具身系统的长期演进。研发记录按书本目录组织：一个 **Episode** 汇聚一段完整的能力主线，一个 **Chapter** 负责一个可验证的逻辑研发阶段。

## 目录

- [`current_status.md`](current_status.md)：最新项目状态、能力边界和近期里程碑。
- [`local_model_assets.md`](local_model_assets.md)：本机持久模型目录、revision/checksum 与 reference 复现命令。
- [`qwen3_hf_config_mapping.md`](qwen3_hf_config_mapping.md)：Qwen3 HF 配置、权重名与布局映射契约。
- [`Episode 01 — Transformer 与训练基础`](episodes/episode01_transformer_and_training_foundations/README.md)：Chapter 01–05。
- [`Episode 02 — Qwen3 端到端对齐`](episodes/episode02_qwen3_end_to_end_parity/README.md)：Chapter 06–09。
- [`Episode 03 — 模型家族与大权重验证`](episodes/episode03_model_family_and_large_weights/README.md)：Chapter 10–13。
- [`Episode 04 — 高效推理与量化`](episodes/episode04_efficient_inference_and_quantization/README.md)：Chapter 14–18。
- [`Episode 05 — 部署、记忆与设备采样`](episodes/episode05_deployment_memory_and_sampling/README.md)：Chapter 19–23。
- [`Episode 06 — Qwen3 MoE 与模型架构扩展`](episodes/episode06_qwen3_moe_and_model_expansion/README.md)：Chapter 24–35、41。
- [`Episode 07 — 智能体闭环`](episodes/episode07_agent_closed_loop/README.md)：Chapter 36–39。
- [`Episode 08 — 环境与行动闭环`](episodes/episode08_environment_action_loop/README.md)：Chapter 40、42–。

## Episode 与 Chapter

Episode 是围绕一条能力主线组织的“卷”，不再按自然月归档。它负责给出主题、阅读顺序、阶段结果和下一卷的衔接。

Chapter 是目标驱动的逻辑研发章节，不对应自然周：

1. Open 一个 Chapter，写清核心问题、预期结果和 Close 条件。
2. 围绕目标实现、学习和验证，不为填满时间而增加任务。
3. Close 条件提前满足时，立即关闭当前 Chapter 并完成回顾。
4. 未完成项只有在仍然重要时才进入下一 Chapter。
5. 一个主题形成完整能力闭环后，完成 Episode 回顾并进入下一卷。

## 记录原则

1. **结果可验证**：尽量附上测试、实验、指标、示例或代码位置。
2. **状态不混写**：明确区分计划中、已实现、已验证和已完成。
3. **记录认知变化**：不仅写做了什么，也写为什么这样做、哪里判断错了。
4. **围绕能力积累**：每项工作说明它增强了模型、智能体、具身闭环或工程基础中的哪一部分。
5. **保留历史语境**：旧计划可以过期，但不重写当时的判断；用新的状态快照说明后续结果。

## 新建记录

- 新开研发章节：复制 [`templates/CHAPTER.md`](templates/CHAPTER.md)。
- 新开主题卷：复制 [`templates/EPISODE.md`](templates/EPISODE.md)，并在对应 Episode 目录中维护章节顺序。

历史测试、fixture、benchmark 和外部 reference 仍保留 `weekXX` 技术标识。这些名称是既有可复现资产的稳定路径，不再承担文档组织含义。
