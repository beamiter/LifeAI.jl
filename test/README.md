# LifeAI.jl test suite

测试按能力命名，不按研发时间编号。查看文件名即可判断覆盖范围：

- `test_manual_attention`、`test_mha`、`test_rope`、`test_transformer`：基础模型组件。
- `test_tokenizer_*`：tokenizer、数据管线和 XLA 兼容性。
- `test_gpt2_*`：GPT-2 架构与 HuggingFace parity。
- `test_qwen3_hf_*`：Qwen3 配置、权重、tokenizer 和文本生成。
- `test_qwen3_bf16_*`、`test_qwen3_*quantization*`、`test_qwen3_*int4`：BF16、加速和量化。
- `test_qwen3_*deployment*`、`test_qwen3_resident_http_service`：本地部署与服务。
- `test_qwen3_embedding_memory`、`test_qwen3_device_sampling`：语义记忆和设备端采样。
- `qwen3_moe_router_test`、`qwen3_moe_expert_mixture_test`、`qwen3_moe_weight_loading_test`、`qwen3_moe_cached_decode_test`：MoE 路由、专家混合输出、HF 权重映射与缓存解码。
- `qwen3_moe_transformers_parity_test`：冻结 tiny checkpoint 的 router、逐层 hidden、full logits 与 cached decode 跨框架对齐。

`fixtures/` 同样按被验证的能力命名。冻结 benchmark 和模型 reference 通过文件内容或 metadata 定位，测试不依赖时间编号目录。

默认测试入口：

```bash
julia --project=. test/runtests.jl
```

启用 Reactant/XLA 专项：

```bash
LIFEAI_TEST_XLA=true julia --project=. test/runtests.jl
```
