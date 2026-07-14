### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 20000000-0000-0000-0000-000000000001
begin
    import Pkg

    # LifeAI is the local package in this repository, not a registered package.
    # Explicit activation also tells Pluto not to use its isolated scratch
    # environment for this notebook.
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))

    using LifeAI
    using Lux
    using Random
    using LinearAlgebra
    using PlutoUI
    using Plots

    figure_dir = joinpath(@__DIR__, "assets")
    mkpath(figure_dir)
end

# ╔═╡ 20000000-0000-0000-0000-000000000002
md"""
# 02｜直接运行 LifeAI：prefill 与 decode_step

这份 Notebook 不再使用虚构数组，而是直接运行仓库中的：

- `GPTModel`
- `init_kv_cache`
- `prefill`
- `decode_step`

Notebook 会自动激活仓库根目录的 LifeAI 项目环境。首次运行前如尚未安装依赖，
请在仓库根目录执行 `julia --project=. -e 'using Pkg; Pkg.instantiate()'`。
"""

# ╔═╡ 20000000-0000-0000-0000-000000000003
begin
    rng = Xoshiro(20260714)
    model = GPTModel(
        31,   # vocab_size
        24,   # d_model
        3,    # num_heads
        2;    # num_layers
        max_seq_len=16,
        use_rope=true,
    )
    ps, st = Lux.setup(rng, model)
end

# ╔═╡ 20000000-0000-0000-0000-000000000004
prompt = reshape([2, 5, 7, 11, 13], :, 1)

# ╔═╡ 20000000-0000-0000-0000-000000000005
begin
    full_prompt_logits, _ = model(prompt, ps, st)

    empty_cache = init_kv_cache(model; batch_size=1)
    prefill_logits, prompt_cache, prefill_state = prefill(
        model,
        ps,
        st,
        prompt,
        empty_cache,
    )
end

# ╔═╡ 20000000-0000-0000-0000-000000000006
begin
    prefill_error = abs.(Array(prefill_logits) .- Array(full_prompt_logits))
    maximum(prefill_error)
end

# ╔═╡ 20000000-0000-0000-0000-000000000007
md"""
## 1. prefill 应该与完整 forward 等价

`prefill` 仍然处理整段 prompt，只是额外把每一层的 K/V 保存下来。

因此在相同参数和输入下：

```julia
prefill_logits ≈ model(prompt)
```

当前最大绝对误差为：**$(maximum(prefill_error))**。
"""

# ╔═╡ 20000000-0000-0000-0000-000000000008
begin
    layer_lengths = [length(layer_cache) for layer_cache in prompt_cache.layers]
    p_lengths = bar(
        1:model.num_layers,
        layer_lengths;
        xlabel="Transformer 层",
        ylabel="已缓存 token 数",
        title="prefill 后，每一层都缓存完整 prompt",
        label="cache length",
        xticks=1:model.num_layers,
        ylim=(0, length(prompt) + 1),
        size=(760, 430),
    )
    savefig(p_lengths, joinpath(figure_dir, "04_prefill_layer_lengths.png"))
    p_lengths
end

# ╔═╡ 20000000-0000-0000-0000-000000000009
begin
    first_keys = Array(prompt_cache.layers[1].keys)
    key_norms = [
        norm(@view first_keys[:, head, token, 1])
        for head in 1:size(first_keys, 2), token in 1:size(first_keys, 3)
    ]
    p_keys = heatmap(
        1:size(key_norms, 2),
        1:size(key_norms, 1),
        key_norms;
        xlabel="prompt token 位置",
        ylabel="attention head",
        title="第 1 层缓存中每个 Key 向量的范数",
        colorbar_title="‖K‖",
        size=(760, 430),
    )
    savefig(p_keys, joinpath(figure_dir, "05_layer1_key_norms.png"))
    p_keys
end

# ╔═╡ 20000000-0000-0000-0000-000000000010
next_token = 17

# ╔═╡ 20000000-0000-0000-0000-000000000011
begin
    step_logits, extended_cache, decode_state = decode_step(
        model,
        ps,
        prefill_state,
        next_token,
        prompt_cache,
    )

    extended_prompt = vcat(prompt, reshape([next_token], 1, 1))
    extended_full_logits, _ = model(extended_prompt, ps, st)
    reference_last_logits = extended_full_logits[:, end:end, :]
    decode_error = abs.(Array(step_logits) .- Array(reference_last_logits))
end

# ╔═╡ 20000000-0000-0000-0000-000000000012
md"""
## 2. decode_step 只处理一个新增 token

`decode_step` 的输入形状是 `(1, batch)`，输出 logits 形状是：

```julia
(vocab_size, 1, batch)
```

它的结果应当等价于“把新 token 接在 prompt 后，再完整 forward，取最后一个位置”。

当前最大绝对误差：**$(maximum(decode_error))**。
"""

# ╔═╡ 20000000-0000-0000-0000-000000000013
begin
    p_growth = bar(
        ["prefill 后", "decode_step 后"],
        [length(prompt_cache), length(extended_cache)];
        ylabel="cache.position",
        title="一次 decode_step 只让缓存长度增加 1",
        label="processed tokens",
        ylim=(0, length(extended_cache) + 1),
        size=(650, 430),
    )
    savefig(p_growth, joinpath(figure_dir, "06_decode_cache_growth.png"))
    p_growth
end

# ╔═╡ 20000000-0000-0000-0000-000000000014
begin
    compare_count = min(15, model.vocab_size)
    p_logits = groupedbar_data = nothing
    p_logits = bar(
        1:compare_count,
        Array(reference_last_logits[1:compare_count, 1, 1]);
        label="完整 forward",
        xlabel="token id",
        ylabel="logit",
        title="decode_step 与完整 forward 的最后位置 logits",
        alpha=0.7,
        size=(820, 440),
    )
    scatter!(
        p_logits,
        1:compare_count,
        Array(step_logits[1:compare_count, 1, 1]);
        label="decode_step",
        markersize=5,
    )
    savefig(p_logits, joinpath(figure_dir, "07_decode_logits_equivalence.png"))
    p_logits
end

# ╔═╡ 20000000-0000-0000-0000-000000000015
md"""
## 3. 代码路径翻译成人话

### `prefill`

1. prompt 全部进入模型；
2. 每层计算 prompt 的 Q/K/V；
3. K/V 沿 token 维保存；
4. 返回 prompt 所有位置的 logits。

### `decode_step`

1. 输入只有新 token；
2. 只计算新 token 的 Q/K/V；
3. 新 K/V 追加到每层 cache；
4. 新 Q 与全部历史 K/V 交互；
5. 返回新位置的 logits。
"""

# ╔═╡ 20000000-0000-0000-0000-000000000016
md"""
## 本章输出图片

- `assets/04_prefill_layer_lengths.png`
- `assets/05_layer1_key_norms.png`
- `assets/06_decode_cache_growth.png`
- `assets/07_decode_logits_equivalence.png`
"""

# ╔═╡ Cell order:
# ╠═20000000-0000-0000-0000-000000000001
# ╟─20000000-0000-0000-0000-000000000002
# ╠═20000000-0000-0000-0000-000000000003
# ╠═20000000-0000-0000-0000-000000000004
# ╠═20000000-0000-0000-0000-000000000005
# ╠═20000000-0000-0000-0000-000000000006
# ╟─20000000-0000-0000-0000-000000000007
# ╠═20000000-0000-0000-0000-000000000008
# ╠═20000000-0000-0000-0000-000000000009
# ╠═20000000-0000-0000-0000-000000000010
# ╠═20000000-0000-0000-0000-000000000011
# ╟─20000000-0000-0000-0000-000000000012
# ╠═20000000-0000-0000-0000-000000000013
# ╠═20000000-0000-0000-0000-000000000014
# ╟─20000000-0000-0000-0000-000000000015
# ╟─20000000-0000-0000-0000-000000000016
