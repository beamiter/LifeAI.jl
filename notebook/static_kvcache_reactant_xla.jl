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

# ╔═╡ 40000000-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))

    using LifeAI
    using Lux
    using Random
    using PlutoUI
    using Plots

    figure_dir = joinpath(@__DIR__, "assets")
    mkpath(figure_dir)
end

# ╔═╡ 40000000-0000-0000-0000-000000000002
md"""
# 04｜固定形状 KV Cache 与 Reactant/XLA

PR1 的动态缓存通过 `cat` 增长，适合先验证算法正确性。

PR2 进一步引入固定容量缓存：物理 shape 始终是

```julia
(head_dim, num_heads, max_seq_len, batch)
```

变化的只有逻辑位置 `position`。这使单 token decode 更容易复用同一个 XLA executable。
"""

# ╔═╡ 40000000-0000-0000-0000-000000000003
has_pr2 = all(
    name -> isdefined(LifeAI, name),
    (:StaticGPTKVCache, :init_static_kv_cache, :XLAKVDecoder, :xla_prefill!, :xla_decode_step!),
)

# ╔═╡ 40000000-0000-0000-0000-000000000004
md"""
PR2 API 是否已经存在：**$(has_pr2)**

如果这里是 `false`，请先应用 PR2 format-patch；概念图仍然可以正常运行。
"""

# ╔═╡ 40000000-0000-0000-0000-000000000005
@bind capacity Slider(8:8:128; default=32, show_value=true)

# ╔═╡ 40000000-0000-0000-0000-000000000006
@bind logical_position Slider(1:128; default=10, show_value=true)

# ╔═╡ 40000000-0000-0000-0000-000000000007
begin
    used = min(logical_position, capacity)
    occupancy = [i <= used ? 1.0 : 0.15 for i in 1:capacity]
    p_slots = bar(
        1:capacity,
        occupancy;
        xlabel="预分配的 token 槽位",
        ylabel="逻辑状态",
        title="固定容量缓存：shape 不变，只移动 position",
        label="slot",
        ylim=(0, 1.2),
        size=(860, 430),
    )
    vline!(p_slots, [used + 0.5]; label="position = $used", linewidth=3)
    savefig(p_slots, joinpath(figure_dir, "12_static_cache_slots.png"))
    p_slots
end

# ╔═╡ 40000000-0000-0000-0000-000000000008
begin
    steps = 1:16
    initial_length = 8
    dynamic_shapes = initial_length .+ steps
    fixed_shapes = fill(capacity, length(steps))

    p_shapes = plot(
        steps,
        dynamic_shapes;
        xlabel="decode step",
        ylabel="K/V 第三维物理长度",
        title="动态 cat 与固定容量缓存",
        label="动态 cache：每步 shape 改变",
        linewidth=3,
        marker=:circle,
        size=(820, 450),
    )
    plot!(
        p_shapes,
        steps,
        fixed_shapes;
        label="静态 cache：shape 不变",
        linewidth=3,
        marker=:square,
    )
    savefig(p_shapes, joinpath(figure_dir, "13_dynamic_vs_static_shape.png"))
    p_shapes
end

# ╔═╡ 40000000-0000-0000-0000-000000000009
begin
    valid_prefix = Float32[i <= used for i in 1:capacity]
    p_prefix = heatmap(
        reshape(valid_prefix, 1, :);
        xlabel="固定缓存槽位",
        ylabel="",
        yticks=false,
        title="有效前缀 mask：只有 1:position 参与注意力",
        colorbar=false,
        size=(840, 220),
    )
    savefig(p_prefix, joinpath(figure_dir, "14_valid_prefix_mask.png"))
    p_prefix
end

# ╔═╡ 40000000-0000-0000-0000-000000000010
begin
    rng = Xoshiro(20260714)
    demo_model = GPTModel(29, 24, 3, 2; max_seq_len=16, use_rope=true)
    demo_ps, demo_st = Lux.setup(rng, demo_model)
    demo_prompt = reshape([2, 4, 6, 8], :, 1)
end

# ╔═╡ 40000000-0000-0000-0000-000000000011
static_demo = if has_pr2
    cache = LifeAI.init_static_kv_cache(
        demo_model;
        batch_size=1,
        dtype=Float32,
        device=Lux.cpu_device(),
    )
    logits, filled, state = LifeAI.prefill(
        demo_model,
        demo_ps,
        demo_st,
        demo_prompt,
        cache,
    )
    (; logits, filled, state, physical_shape=size(filled.layers[1].keys))
else
    nothing
end

# ╔═╡ 40000000-0000-0000-0000-000000000012
md"""
## 固定形状缓存的关键区别

若 PR2 已应用，上一单元格会真实创建 `StaticGPTKVCache`。

- 逻辑长度：`length(cache)`
- 物理容量：`cache.max_seq_len`
- 单层物理 shape：始终包含完整 `max_seq_len`

prefill 与每次 decode 只会更新内容和 `position`，不会改变 K/V 张量 shape。
"""

# ╔═╡ 40000000-0000-0000-0000-000000000013
if static_demo === nothing
    "等待应用 PR2"
else
    (
        logical_position=length(static_demo.filled),
        capacity=static_demo.filled.max_seq_len,
        physical_shape=static_demo.physical_shape,
    )
end

# ╔═╡ 40000000-0000-0000-0000-000000000014
@bind run_xla_cpu CheckBox(default=false)

# ╔═╡ 40000000-0000-0000-0000-000000000015
md"""
## 可选：真实编译一次 XLA CPU decode

勾选上面的开关后，Notebook 会：

1. 创建 `XLAKVDecoder(...; xla_backend="cpu")`；
2. 编译或复用 prompt-shape 对应的 prefill thunk；
3. 第一次 decode 时编译固定形状 decode thunk；
4. 第二次 decode 复用同一个 thunk。

第一次执行可能较慢，这是编译时间，不是稳定态 token 延迟。
"""

# ╔═╡ 40000000-0000-0000-0000-000000000016
xla_result = if run_xla_cpu && has_pr2
    decoder = LifeAI.XLAKVDecoder(
        demo_model,
        demo_ps,
        demo_st;
        batch_size=1,
        xla_backend="cpu",
    )

    logits0, _, _ = LifeAI.xla_prefill!(decoder, demo_prompt)
    first_thunk_before = decoder.decode_thunk
    logits1, _, _ = LifeAI.xla_decode_step!(decoder, 10)
    first_thunk_after = decoder.decode_thunk
    logits2, _, _ = LifeAI.xla_decode_step!(decoder, 12)

    (
        prefill_thunk_count=length(decoder.prefill_thunks),
        decode_was_uncompiled=first_thunk_before === nothing,
        decode_is_compiled=first_thunk_after !== nothing,
        same_decode_thunk_reused=decoder.decode_thunk === first_thunk_after,
        host_position=decoder.host_position,
        logits_shapes=(size(logits0), size(logits1), size(logits2)),
    )
elseif !has_pr2
    "PR2 API 尚不存在"
else
    "勾选 run_xla_cpu 后执行"
end

# ╔═╡ 40000000-0000-0000-0000-000000000017
md"""
## 工程结论

- 动态 KV Cache：最容易理解和验证，但每步 `cat` 会改变 shape；
- 固定形状 KV Cache：提前分配容量，用 `position` 描述有效前缀；
- XLA decoder：prompt shape 对应 prefill thunk，所有 token 位置复用同一个 decode thunk。

换句话说：

> PR1 解决“不要重复算历史”；PR2 解决“不要因为缓存长度变化而重复编译”。
"""

# ╔═╡ 40000000-0000-0000-0000-000000000018
md"""
## 本章输出图片

- `assets/12_static_cache_slots.png`
- `assets/13_dynamic_vs_static_shape.png`
- `assets/14_valid_prefix_mask.png`
"""

# ╔═╡ Cell order:
# ╠═40000000-0000-0000-0000-000000000001
# ╟─40000000-0000-0000-0000-000000000002
# ╠═40000000-0000-0000-0000-000000000003
# ╟─40000000-0000-0000-0000-000000000004
# ╠═40000000-0000-0000-0000-000000000005
# ╠═40000000-0000-0000-0000-000000000006
# ╠═40000000-0000-0000-0000-000000000007
# ╠═40000000-0000-0000-0000-000000000008
# ╠═40000000-0000-0000-0000-000000000009
# ╠═40000000-0000-0000-0000-000000000010
# ╠═40000000-0000-0000-0000-000000000011
# ╟─40000000-0000-0000-0000-000000000012
# ╠═40000000-0000-0000-0000-000000000013
# ╠═40000000-0000-0000-0000-000000000014
# ╟─40000000-0000-0000-0000-000000000015
# ╠═40000000-0000-0000-0000-000000000016
# ╟─40000000-0000-0000-0000-000000000017
# ╟─40000000-0000-0000-0000-000000000018
