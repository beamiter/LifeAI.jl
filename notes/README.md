# LifeAI.jl 研发记录

这里记录 LifeAI.jl 从模型基本组件走向智能体和具身系统的长期演进。

## 索引

- [`current_status.md`](current_status.md)：最新项目状态、能力边界和近期里程碑。
- [`week01_transformer.md`](week01_transformer.md)：07-01 至 07-07，项目骨架、Attention、RoPE 与 Transformer 基础。
- [`week02_gpt_xla_kv_cache.md`](week02_gpt_xla_kv_cache.md)：07-08 至 07-14，最小 GPT、XLA 训练与 KV Cache 增量推理。
- [`weekly/`](weekly/)：weekly plan、实验过程和周末回顾。
- [`monthly/`](monthly/)：月度总结和跨周能力变化。

## 记录原则

1. **结果可验证**：尽量附上测试、实验、指标、示例或代码位置。
2. **状态不混写**：明确区分计划中、已实现、已验证和已完成。
3. **记录认知变化**：不仅写做了什么，也写为什么这样做、哪里判断错了。
4. **围绕能力积累**：每项工作说明它增强了模型、智能体、具身闭环或工程基础中的哪一部分。
5. **保留历史语境**：旧计划可以过期，但不重写当时的判断；用新的状态快照说明后续结果。

新的一周或月份可以分别复制 [`weekly/TEMPLATE.md`](weekly/TEMPLATE.md) 和 [`monthly/TEMPLATE.md`](monthly/TEMPLATE.md) 开始记录。
