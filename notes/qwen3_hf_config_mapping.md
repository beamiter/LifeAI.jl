# Qwen3 HF config.json ↔ LifeAI.jl gpt_config 映射契约（草案）

> 状态：Week 06 交付的契约草案，作为 Week 07 权重加载的输入。
>
> 参照对象：HuggingFace `Qwen/Qwen3-0.6B` 的 `config.json`（`Qwen3ForCausalLM` /
> `model_type: qwen3`）。字段值以 0.6B 为例，规则适用于全部 Qwen3 dense 型号。

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

## 权重名映射（Week 07 待实现）

| HF 参数名 | LifeAI 参数树路径 | 布局说明 |
| --- | --- | --- |
| `model.embed_tokens.weight` | `ps.token_embedding.weight` | HF `(vocab, hidden)` 行主序 ↔ Julia `(hidden, vocab)` 列主序：**数值内存布局一致，需按语义转置核对** |
| `model.layers.{i}.input_layernorm.weight` | `ps.blocks.layer_{i+1}.norm1.scale` | `(hidden,)`；LifeAI 存 `(hidden,1,1)`，需 reshape |
| `model.layers.{i}.self_attn.q_proj.weight` | `ps.blocks.layer_{i+1}.attn.q_proj.weight` | HF `(out,in)` 行主序 → Julia Dense `(out,in)` 列主序：**需转置拷贝**，Week 07 用单元测试逐案钉死 |
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
| `lm_head.weight`（untied 型号） | `ps.lm_head.weight` | tied 型号复用 `embed_tokens` |

## 显式标注的歧义与风险项

1. **转置约定**：PyTorch `nn.Linear.weight` 是 `(out_features, in_features)` 行主序；
   Julia/Lux `Dense.weight` 是 `(out, in)` 列主序。同名 shape 但内存序不同，
   加载时每类权重都必须有数值级 fixture 测试（给定输入 → 双方输出一致），
   不允许只对 shape。
2. **head 切分顺序**：HF Q/K/V 投影输出按 head-major 排列
   （head h 占据 `(h-1)*head_dim+1 : h*head_dim` 行）；LifeAI 的
   `reshape(·, head_dim, num_heads, …)` 假设相同排列——已与 repeat_kv
   语义一起在 Week 06 测试中钉死，Week 07 直接沿用。
3. **RoPE 配对约定**：HF Qwen3 使用 rotate_half（前半/后半配对），LifeAI
   `apply_rope` 使用相邻偶奇配对（interleaved）。**两者不等价**，Week 07
   必须做以下二选一并用逐层 fixture 验证：(a) LifeAI 增加 rotate_half 模式；
   (b) 加载 q/k 权重与 q_norm 时按 permutation 重排 head_dim 维。
   这是最可能导致 logits 对不上的单点。
4. **dtype**：bf16 权重转 Float32 后与 HF bf16 推理存在固有数值差；对齐
   验证应以 HF float32 参考跑（`torch_dtype=float32`）为基准，并记录容差。
5. **`max_seq_len`**：加载时可小于 `max_position_embeddings`，但必须
   ≥ 验证用序列长度，且写入 checkpoint config。
