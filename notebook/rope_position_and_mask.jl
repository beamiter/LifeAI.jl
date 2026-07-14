### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 30000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))

    using LifeAI
    using LinearAlgebra
    using PlutoUI
    using Plots

    cjk_font = let
        source = readchomp(`fc-match -f '%{file}' "Noto Sans CJK SC"`)
        target = joinpath(tempdir(), "LifeAI-NotoSansCJKSC.ttf")
        ispath(target) || symlink(source, target)
        splitext(target)[1]
    end
    Plots.default(fontfamily=cjk_font)

    figure_dir = joinpath(@__DIR__, "assets")
    mkpath(figure_dir)
end

# ╔═╡ 30000000-0000-0000-0000-000000000002
md"""
# 03｜KV Cache 中最容易出错的地方：RoPE 绝对位置

增量解码时，新 token 的序列长度虽然只有 1，但它的真实位置不是第 1 个位置。

如果 prompt 已经有 5 个 token，那么下一个 token 必须使用第 6 个绝对位置的 RoPE，而不能每次都从位置 1 重新旋转。
"""

# ╔═╡ 30000000-0000-0000-0000-000000000003
begin
    head_dim = 8
    sequence_length = 12
    rope = RoPE(head_dim; max_seq_len=32)

    # 每个位置输入相同向量，方便观察“仅位置变化”带来的旋转。
    repeated = zeros(Float32, head_dim, 1, sequence_length, 1)
    repeated[1, 1, :, 1] .= 1
    repeated[3, 1, :, 1] .= 1
    rotated = apply_rope(repeated, rope; start_pos=1)
end

# ╔═╡ 30000000-0000-0000-0000-000000000004
begin
    p_rope_fast = plot(
        0:(sequence_length - 1),
        vec(rotated[1, 1, :, 1]);
        label="维度 1",
        xlabel="绝对 position",
        ylabel="旋转后的分量",
        title="RoPE：同一个输入向量会随 position 旋转",
        linewidth=3,
        marker=:circle,
        size=(820, 440),
    )
    plot!(
        p_rope_fast,
        0:(sequence_length - 1),
        vec(rotated[2, 1, :, 1]);
        label="维度 2",
        linewidth=3,
        marker=:square,
    )
    savefig(p_rope_fast, joinpath(figure_dir, "08_rope_absolute_position.png"))
    p_rope_fast
end

# ╔═╡ 30000000-0000-0000-0000-000000000005
begin
    p_rope_freq = plot(
        0:(sequence_length - 1),
        vec(rotated[1, 1, :, 1]);
        label="前部维度对",
        xlabel="绝对 position",
        ylabel="旋转后的分量",
        title="不同维度对具有不同旋转频率",
        linewidth=3,
        marker=:circle,
        size=(820, 440),
    )
    plot!(
        p_rope_freq,
        0:(sequence_length - 1),
        vec(rotated[3, 1, :, 1]);
        label="后部维度对",
        linewidth=3,
        marker=:square,
    )
    savefig(p_rope_freq, joinpath(figure_dir, "09_rope_frequency_pairs.png"))
    p_rope_freq
end

# ╔═╡ 30000000-0000-0000-0000-000000000006
@bind cached_tokens Slider(1:15; default=5, show_value=true)

# ╔═╡ 30000000-0000-0000-0000-000000000007
begin
    one_token = repeated[:, :, 1:1, :]
    correct_incremental = apply_rope(one_token, rope; start_pos=cached_tokens + 1)
    wrong_incremental = apply_rope(one_token, rope; start_pos=1)

    full_reference_input = repeat(one_token, 1, 1, cached_tokens + 1, 1)
    full_reference = apply_rope(full_reference_input, rope; start_pos=1)
    correct_reference = full_reference[:, :, end:end, :]

    correct_error = maximum(abs.(correct_incremental .- correct_reference))
    wrong_error = maximum(abs.(wrong_incremental .- correct_reference))
end

# ╔═╡ 30000000-0000-0000-0000-000000000008
md"""
## 正确与错误的增量 RoPE

当前缓存中已经有 **$(cached_tokens)** 个 token，因此新 token 应使用：

```julia
start_pos = cached_tokens + 1
```

结果：

- 使用绝对位置的误差：**$(correct_error)**
- 错误地从位置 1 开始的误差：**$(wrong_error)**

这正是 KV Cache 实现里必须把 `start_pos` 与 `cache.position` 对齐的原因。
"""

# ╔═╡ 30000000-0000-0000-0000-000000000009
begin
    labels = ["正确：start_pos = cache.position + 1", "错误：每步 start_pos = 1"]
    errors = [correct_error, wrong_error]
    p_error = bar(
        labels,
        errors;
        ylabel="与完整序列计算的最大误差",
        title="增量解码必须使用绝对 RoPE 位置",
        label="max abs error",
        xrotation=10,
        size=(850, 460),
    )
    savefig(p_error, joinpath(figure_dir, "10_rope_position_error.png"))
    p_error
end

# ╔═╡ 30000000-0000-0000-0000-000000000010
md"""
## 另一个容易出错的地方：decode 阶段的 causal mask

在完整 forward 中，query 和 key 都有多个位置，需要下三角 causal mask。

在单 token decode 中：

- query 长度是 1；
- cache 中只包含过去和当前位置；
- 因而这个 query 可以直接看见 cache 中所有有效 K/V。

如果还用局部下标 `1:Tq` 与 `1:Tk` 生成 mask，可能错误地只允许它看第一个 key。
"""

# ╔═╡ 30000000-0000-0000-0000-000000000011
begin
    key_count = cached_tokens + 1
    correct_visibility = ones(Float32, 1, key_count)
    wrong_local_mask = Float32[j <= 1 for _ in 1:1, j in 1:key_count]

    p_mask = heatmap(
        vcat(correct_visibility, wrong_local_mask);
        xlabel="cache 中的 key 位置",
        ylabel="方案",
        yticks=([1, 2], ["正确：全部有效前缀", "错误：局部位置比较"]),
        title="单 token decode 的可见范围",
        colorbar=false,
        size=(820, 300),
    )
    savefig(p_mask, joinpath(figure_dir, "11_decode_mask_pitfall.png"))
    p_mask
end

# ╔═╡ 30000000-0000-0000-0000-000000000012
md"""
## 本章输出图片

- `assets/08_rope_absolute_position.png`
- `assets/09_rope_frequency_pairs.png`
- `assets/10_rope_position_error.png`
- `assets/11_decode_mask_pitfall.png`
"""

# ╔═╡ Cell order:
# ╠═30000000-0000-0000-0000-000000000001
# ╟─30000000-0000-0000-0000-000000000002
# ╠═30000000-0000-0000-0000-000000000003
# ╠═30000000-0000-0000-0000-000000000004
# ╠═30000000-0000-0000-0000-000000000005
# ╠═30000000-0000-0000-0000-000000000006
# ╠═30000000-0000-0000-0000-000000000007
# ╟─30000000-0000-0000-0000-000000000008
# ╠═30000000-0000-0000-0000-000000000009
# ╟─30000000-0000-0000-0000-000000000010
# ╠═30000000-0000-0000-0000-000000000011
# ╟─30000000-0000-0000-0000-000000000012
