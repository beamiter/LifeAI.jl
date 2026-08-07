# LifeAI.jl test suite

测试按能力命名，不按研发时间编号。查看文件名即可判断覆盖范围：

- `test_manual_attention`、`test_mha`、`test_rope`、`test_transformer`：基础模型组件。
- `test_tokenizer_*`：tokenizer、数据管线和 XLA 兼容性。
- `test_gpt2_*`：GPT-2 架构与 HuggingFace parity。
- `test_qwen3_hf_*`：Qwen3 配置、权重、tokenizer 和文本生成。
- `test_qwen3_bf16_*`、`test_qwen3_*quantization*`、`test_qwen3_*int4`：BF16、加速和量化。
- `test_qwen3_*deployment*`、`test_qwen3_resident_http_service`：本地部署与服务。
- `test_qwen3_embedding_memory`、`test_qwen3_device_sampling`：语义记忆和设备端采样。
- `qwen3_moe_router_test`、`qwen3_moe_expert_mixture_test`、`qwen3_moe_weight_loading_test`、`qwen3_moe_cached_decode_test`：MoE 路由、专家混合输出、HF 权重映射、双分片流式加载与缓存解码。
- `qwen3_moe_transformers_parity_test`：冻结 tiny checkpoint 的 router、逐层 hidden、full logits、路由后 active-expert streaming 与 cached decode 跨框架对齐。
- `qwen3_moe_real_checkpoint_contract_test`：官方 30B-A3B immutable revision、config/index、16 个权重分片及 opt-in 本地资产完整性。
- `qwen3_moe_sparse_dispatch_test`：只执行被路由 token-expert pair、跳过 inactive expert，并与全 expert oracle 对齐。
- `qwen3_moe_device_sparse_dispatch_test`：紧凑 top-k route 表、route-major batched expert 计算与未选 expert 隔离。
- `qwen3_moe_device_sparse_dispatch_xla_test`：同一紧凑 sparse dispatch kernel 的 Reactant/XLA 编译与数值一致性。
- `qwen3_moe_device_sparse_dispatch_cuda_test`：CUDA indexed kernels 与 dense/route-major 数值一致性、未选 `NaN` expert 隔离及 workspace contract。

`fixtures/` 同样按被验证的能力命名。冻结 benchmark 和模型 reference 通过文件内容或 metadata 定位，测试不依赖时间编号目录。

默认测试入口：

```bash
julia --project=. test/runtests.jl
```

启用 Reactant/XLA 专项：

```bash
LIFEAI_TEST_XLA=true julia --project=. test/runtests.jl
```

启用 CUDA sparse accelerator 专项：

```bash
LIFEAI_TEST_CUDA=true julia --project=. test/runtests.jl
```
