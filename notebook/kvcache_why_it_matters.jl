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

# ╔═╡ 10000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))

    using PlutoUI
    using Plots
    using Statistics

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

# ╔═╡ 10000000-0000-0000-0000-000000000002
md"""
# 01｜为什么需要 KV Cache？

这一章先不碰复杂实现，只回答一个问题：

> 模型连续生成 token 时，为什么不应该反复重算全部历史？

所有图都会同时显示在 Pluto 中，并保存到当前目录的 `assets/`，可以直接用于知乎文章。
"""

# ╔═╡ 10000000-0000-0000-0000-000000000003
@bind prompt_len Slider(4:2:128; default=32, show_value=true)

# ╔═╡ 10000000-0000-0000-0000-000000000004
@bind new_tokens Slider(1:64; default=24, show_value=true)

# ╔═╡ 10000000-0000-0000-0000-000000000005
begin
    context_lengths = collect(prompt_len:(prompt_len + new_tokens - 1))

    # 这里只画“数量级直觉”，不是具体硬件耗时模型。
    eager_step_cost = context_lengths .^ 2
    cached_step_cost = context_lengths

    eager_total_cost = cumsum(eager_step_cost)
    cached_total_cost = cumsum(cached_step_cost)
end

# ╔═╡ 10000000-0000-0000-0000-000000000006
md"""
## 1. 每生成一个 token，重复计算在哪里？

没有缓存时，第 `L+1` 个 token 到来后，前面 `L` 个 token 会再次经过模型，历史位置的 K/V 也会再次计算。

有缓存时，历史 K/V 已经保存在每层 cache 中，只需要：

1. 为新 token 计算 Q/K/V；
2. 把新 K/V 加入缓存；
3. 让新 Q 与历史 K 做注意力。

因此下面用平方增长和线性增长，展示两种路径的数量级直觉。
"""

# ╔═╡ 10000000-0000-0000-0000-000000000007
begin
    p_step = plot(
        context_lengths,
        eager_step_cost;
        label="整段重算：近似 O(L²)",
        xlabel="当前上下文长度 L",
        ylabel="单步相对计算量",
        title="每生成一个 token 的计算量",
        linewidth=3,
        marker=:circle,
        legend=:topleft,
        size=(820, 460),
    )
    plot!(
        p_step,
        context_lengths,
        cached_step_cost;
        label="KV Cache：近似 O(L)",
        linewidth=3,
        marker=:square,
    )
    savefig(p_step, joinpath(figure_dir, "01_step_cost.png"))
    p_step
end

# ╔═╡ 10000000-0000-0000-0000-000000000008
begin
    p_total = plot(
        1:new_tokens,
        eager_total_cost;
        label="整段重算",
        xlabel="已经生成的新 token 数",
        ylabel="累计相对计算量",
        title="累计计算量会迅速拉开",
        linewidth=3,
        marker=:circle,
        legend=:topleft,
        size=(820, 460),
    )
    plot!(
        p_total,
        1:new_tokens,
        cached_total_cost;
        label="KV Cache",
        linewidth=3,
        marker=:square,
    )
    savefig(p_total, joinpath(figure_dir, "02_total_cost.png"))
    p_total
end

# ╔═╡ 10000000-0000-0000-0000-000000000009
round(eager_total_cost[end] / cached_total_cost[end]; digits=2)

# ╔═╡ 10000000-0000-0000-0000-000000000010
md"""
当前滑块参数下，直觉模型中的累计计算量相差约 **$(round(eager_total_cost[end] / cached_total_cost[end]; digits=2)) 倍**。

上下文越长，重复计算越昂贵，所以 KV Cache 对长文本生成尤其重要。
"""

# ╔═╡ 10000000-0000-0000-0000-000000000011
begin
    visible_len = min(prompt_len, 32)
    causal = Float32[j <= i for i in 1:visible_len, j in 1:visible_len]
    p_causal = heatmap(
        1:visible_len,
        1:visible_len,
        causal;
        xlabel="Key / Value 的历史位置",
        ylabel="Query 位置",
        title="因果注意力：当前位置只能看过去和自己",
        yflip=true,
        colorbar=false,
        size=(650, 560),
    )
    savefig(p_causal, joinpath(figure_dir, "03_causal_attention.png"))
    p_causal
end

# ╔═╡ 10000000-0000-0000-0000-000000000012
md"""
## 2. 一句话理解 KV Cache

> 每一层都保存历史 token 的 Key 和 Value；下一步只为新 token 做增量计算。

它缓存的不是最终文本，也不是 logits，而是每层 attention 已经计算好的历史 K/V。
"""

# ╔═╡ 10000000-0000-0000-0000-000000000013
md"""
## 本章输出图片

运行上面的绘图单元格后会生成：

- `assets/01_step_cost.png`
- `assets/02_total_cost.png`
- `assets/03_causal_attention.png`
"""

# ╔═╡ Cell order:
# ╠═10000000-0000-0000-0000-000000000001
# ╟─10000000-0000-0000-0000-000000000002
# ╠═10000000-0000-0000-0000-000000000003
# ╠═10000000-0000-0000-0000-000000000004
# ╠═10000000-0000-0000-0000-000000000005
# ╟─10000000-0000-0000-0000-000000000006
# ╠═10000000-0000-0000-0000-000000000007
# ╠═10000000-0000-0000-0000-000000000008
# ╠═10000000-0000-0000-0000-000000000009
# ╟─10000000-0000-0000-0000-000000000010
# ╠═10000000-0000-0000-0000-000000000011
# ╟─10000000-0000-0000-0000-000000000012
# ╟─10000000-0000-0000-0000-000000000013
