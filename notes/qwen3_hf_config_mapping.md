# Qwen3 HF config.json ↔ LifeAI.jl gpt_config 映射契约

> 状态：Week 06 冻结输入契约；Week 07 已实现并通过真实 Qwen3-0.6B 数值验证；Week 11 已冻结并离线覆盖六个官方 dense 尺寸。
>
> 参照对象：HuggingFace `Qwen/Qwen3-0.6B` 的 `config.json`（`Qwen3ForCausalLM` /
> `model_type: qwen3`）。字段值以 0.6B 为例，规则适用于全部 Qwen3 dense 型号。

## Week 11 官方 dense family 规格

| variant | hidden / Q width | MLP | layers | Q / KV heads | tied | 参数量 |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 0.6B | 1,024 / 2,048 | 3,072 | 28 | 16 / 8 | 是 | 596,049,920 |
| 1.7B | 2,048 / 2,048 | 6,144 | 28 | 16 / 8 | 是 | 1,720,574,976 |
| 4B | 2,560 / 4,096 | 9,728 | 36 | 32 / 8 | 是 | 4,022,468,096 |
| 8B | 4,096 / 4,096 | 12,288 | 36 | 32 / 8 | 否 | 8,190,735,360 |
| 14B | 5,120 / 5,120 | 17,408 | 40 | 40 / 8 | 否 | 14,768,307,200 |
| 32B | 5,120 / 8,192 | 25,600 | 64 | 64 / 8 | 否 | 32,762,123,264 |

`Q width = num_attention_heads * head_dim`。0.6B、4B、32B 证明 attention
内部宽度不能从 residual hidden size 推导；8B+ 则覆盖独立 `lm_head.weight`。
六个冻结 config 的 `head_dim=128`、`num_key_value_heads=8`、
`max_position_embeddings=40960`。公开 API `qwen3_dense_specs()` 保存各模型
immutable revision 与 config SHA256；默认测试使用
`test/fixtures/week11_qwen3_dense_family/specs.json`，不联网。

## 结构字段映射

| HF `config.json` | 0.6B 值 | LifeAI `gpt_config` | 说明 |
| --- | ---: | --- | --- |
| `vocab_size` | 151936 | `vocab_size` | 直接对应 |
| `hidden_size` | 1024 | `d_model` | 直接对应 |
| `num_hidden_layers` | 28 | `num_layers` | 直接对应 |
| `num_attention_heads` | 16 | `num_heads` | 直接对应 |
| `num_key_value_heads` | 8 | `num_kv_heads` | Week 06 新增；GQA 分组 |
| `head_dim` | 128 | `head_dim` | 独立于 `hidden_size ÷ num_heads`（1024/16=64 ≠ 128） |
| `intermediate_size` | 3072 | `mlp_hidden_dim` | SwiGLU hidden 维度 |
| `hidden_act` | `"silu"` | `mlp_type=:swiglu` | HF 的 silu + gate/up/down 即 SwiGLU |
| `rms_norm_eps` | 1e-6 | `norm_epsilon`（配合 `norm_type=:rmsnorm`） | 所有 pre-norm 与 final norm |
| —（qwen3 固定行为） | — | `use_qk_norm=true`、`qk_norm_epsilon` | Qwen3 的 `q_norm`/`k_norm` 使用 `rms_norm_eps`；LifeAI 侧加载时应设 `qk_norm_epsilon = rms_norm_eps` |
| `rope_theta` | 1000000 | `rope_theta` | 直接对应 |
| `max_position_embeddings` | 40960 | `max_seq_len` | 实际加载可取更小值以控制 RoPE cache 与静态 KV cache 内存 |
| `attention_bias` | false | `use_bias=false` | Qwen3 全部投影无 bias |
| `tie_word_embeddings` | true (0.6B/1.7B/4B)、false (8B+) | `tie_embeddings` | 直接对应 |
| `torch_dtype` | `"bfloat16"` | —（加载策略） | Week 07：读 bf16 → 转 Float32 推理；不进入 config |
| `sliding_window` / `use_sliding_window` | null / false | — | Qwen3 dense 不用 SWA；出现 true 应显式报错 |
| `attention_dropout` | 0.0 | — | 推理路径无 dropout；非 0 时应显式报错 |

固定语义（无 HF 字段，属于 Qwen3 架构本身，Week 06 已实现并测试）：

- QK-Norm 位置：head reshape 之后、RoPE 之前；per-head、作用于 `head_dim` 维；
  `q_norm` / `k_norm` 各自有独立可学习 scale（`(head_dim,)`）。
- causal 全注意力（`is_causal=true`）、pre-norm 结构、RoPE 不作用于 V。
- KV 分组语义：query head `h` 使用 KV head `(h-1) ÷ (num_heads ÷ num_kv_heads) + 1`
  （与 HF `repeat_kv` 的连续展开一致；已用 reference 测试钉死）。

## 权重名映射（Week 07 已实现）

| HF 参数名 | LifeAI 参数树路径 | 布局说明 |
| --- | --- | --- |
| `model.embed_tokens.weight` | `ps.token_embedding.weight` | 格式层先还原 HF `(vocab, hidden)` 语义数组；映射层再转为 embedding 所需的 `(hidden, vocab)` |
| `model.layers.{i}.input_layernorm.weight` | `ps.blocks.layer_{i+1}.norm1.scale` | `(hidden,)`；LifeAI 存 `(hidden,1,1)`，需 reshape |
| `model.layers.{i}.self_attn.q_proj.weight` | `ps.blocks.layer_{i+1}.attn.q_proj.weight` | 格式层已从行主序字节重建 `(out,in)` 语义数组；Lux Dense 同为 `(out,in)`，映射层直接使用 |
| `model.layers.{i}.self_attn.k_proj.weight` | `…attn.k_proj.weight` | 同上；out = `num_kv_heads * head_dim` |
| `model.layers.{i}.self_attn.v_proj.weight` | `…attn.v_proj.weight` | 同上 |
| `model.layers.{i}.self_attn.o_proj.weight` | `…attn.o_proj.weight` | 同上；in = `num_heads * head_dim` |
| `model.layers.{i}.self_attn.q_norm.weight` | `…attn.q_norm.scale` | `(head_dim,)` 直接对应 |
| `model.layers.{i}.self_attn.k_norm.weight` | `…attn.k_norm.scale` | `(head_dim,)` 直接对应 |
| `model.layers.{i}.post_attention_layernorm.weight` | `…norm2.scale` | 同 norm1 |
| `model.layers.{i}.mlp.gate_proj.weight` | `…mlp.gate_proj.weight` | SwiGLU gate |
| `model.layers.{i}.mlp.up_proj.weight` | `…mlp.up_proj.weight` | SwiGLU up |
| `model.layers.{i}.mlp.down_proj.weight` | `…mlp.down_proj.weight` | SwiGLU down |
| `model.norm.weight` | `ps.final_norm.scale` | final RMSNorm |
| `lm_head.weight`（untied 型号） | `ps.lm_head.weight` | tied 型号复用 `embed_tokens`；若 safetensors 仍保存重复 `lm_head.weight`，加载器验证其与 embedding 完全相同后丢弃重复副本 |

## 显式标注的歧义与风险项

1. **转置约定**：PyTorch `nn.Linear.weight` 与 Julia/Lux `Dense.weight` 的语义
   shape 都是 `(out, in)`，但文件行主序与 Julia 数组列主序不同。safetensors
   格式层负责从文件字节重建语义数组，参数层不得对 Dense 再次转置；embedding
   因模型接口要求 `(hidden, vocab)` 才在参数层转置。每类权重均已有非对称数值
   fixture，避免只凭 shape 判断。
2. **head 切分顺序**：HF Q/K/V 投影输出按 head-major 排列
   （head h 占据 `(h-1)*head_dim+1 : h*head_dim` 行）；LifeAI 的
   `reshape(·, head_dim, num_heads, …)` 假设相同排列——已与 repeat_kv
   语义一起在 Week 06 测试中钉死，Week 07 直接沿用。
3. **RoPE 配对约定**：HF Qwen3 使用 rotate_half（前半/后半配对），历史
   LifeAI 默认使用相邻偶奇配对（interleaved），两者不等价。Week 07 已选择
   在模型与三类推理路径中增加 `rope_style=:rotate_half`，并用逐层 HF fixture
   验证；legacy config 和 checkpoint 继续默认 `:interleaved`。
4. **dtype**：bf16 权重转 Float32 后与 HF bf16 推理存在固有数值差；对齐
   验证应以 HF float32 参考跑（`torch_dtype=float32`）为基准，并记录容差。
5. **`max_seq_len`**：加载时可小于 `max_position_embeddings`，但必须
   ≥ 验证用序列长度，且写入 checkpoint config。
