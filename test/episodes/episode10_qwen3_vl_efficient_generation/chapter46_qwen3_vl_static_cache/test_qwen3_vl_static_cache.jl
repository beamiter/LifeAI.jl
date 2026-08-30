using Base64: base64decode
using JSON3
using SHA: sha256
using Test
import MLDataDevices
using LifeAI: Qwen3VLRopeLayout,
    Qwen3VLStaticLayerKVCache,
    Qwen3VLStaticKVCache,
    Qwen3VLTextSpec,
    generate_hf_qwen3_vl_tokens,
    hf_qwen3_vl_text_decode_step,
    hf_qwen3_vl_text_decode_step_static,
    hf_qwen3_vl_text_prefill_cached,
    hf_qwen3_vl_text_prefill_static,
    init_qwen3_vl_kv_cache,
    init_qwen3_vl_static_kv_cache,
    reset_qwen3_vl_static_kv_cache!

const _CH46_TINY_TEXT_SPEC = Qwen3VLTextSpec(
    32,             # vocab_size
    16,             # hidden_size
    32,             # intermediate_size
    4,              # num_hidden_layers
    2,              # num_attention_heads
    1,              # num_key_value_heads
    8,              # head_dim
    1.0e-6,         # rms_norm_eps
    10_000.0,       # rope_theta
    64,             # max_position_embeddings
    true,           # mrope_interleaved
    (2, 1, 1),      # mrope_section
    true,           # tie_word_embeddings
    "silu",
)

struct _Ch46NoAllocate{T} <: AbstractMatrix{T}
    touched::Base.RefValue{Bool}
end

Base.size(::_Ch46NoAllocate) = (1, 1)
Base.getindex(::_Ch46NoAllocate, ::Int, ::Int) = error(
    "Qwen3-VL allocation sentinel must not be read",
)
function Base.similar(
    sentinel::_Ch46NoAllocate,
    ::Type,
    ::Dims,
)
    sentinel.touched[] = true
    error("Qwen3-VL cache allocation ran before arithmetic preflight")
end

struct _Ch46ForeignDevice <: MLDataDevices.AbstractDevice end

struct _Ch46ForeignDeviceArray{T,N,A<:AbstractArray{T,N}} <:
       AbstractArray{T,N}
    data::A
end

Base.size(values::_Ch46ForeignDeviceArray) = size(values.data)
Base.axes(values::_Ch46ForeignDeviceArray) = axes(values.data)
Base.IndexStyle(::Type{<:_Ch46ForeignDeviceArray}) = IndexCartesian()
Base.getindex(values::_Ch46ForeignDeviceArray, indices...) =
    getindex(values.data, indices...)
MLDataDevices.get_device(::_Ch46ForeignDeviceArray) = _Ch46ForeignDevice()

function _ch46_bad_cache_spec(;
    num_hidden_layers=1,
    head_dim=8,
    num_key_value_heads=1,
    max_position_embeddings=1,
)
    return (;
        num_hidden_layers,
        head_dim,
        num_key_value_heads,
        max_position_embeddings,
    )
end

function _ch46_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected the test call to fail")
end

function _ch46_tiny_values(count::Int, offset::Int; scale=0.02f0)
    return Float32[
        scale * sin(0.173f0 * Float32(offset + index))
        for index in 1:count
    ]
end

function _ch46_tiny_hf_matrix(rows::Int, columns::Int, offset::Int; scale=0.02f0)
    values = _ch46_tiny_values(rows * columns, offset; scale)
    return permutedims(reshape(values, columns, rows))
end

function _ch46_tiny_text_parameters()
    spec = _CH46_TINY_TEXT_SPEC
    embedding = permutedims(_ch46_tiny_hf_matrix(32, 16, 10))
    blocks = ntuple(spec.num_hidden_layers) do julia_layer
        offset = 10_000 * julia_layer
        return (;
            norm1=1.0f0 .+ _ch46_tiny_values(16, offset; scale=0.01f0),
            q_weight=_ch46_tiny_hf_matrix(16, 16, offset + 100),
            k_weight=_ch46_tiny_hf_matrix(8, 16, offset + 200),
            v_weight=_ch46_tiny_hf_matrix(8, 16, offset + 300),
            o_weight=_ch46_tiny_hf_matrix(16, 16, offset + 400),
            q_norm=1.0f0 .+ _ch46_tiny_values(8, offset + 500; scale=0.01f0),
            k_norm=1.0f0 .+ _ch46_tiny_values(8, offset + 600; scale=0.01f0),
            norm2=1.0f0 .+ _ch46_tiny_values(16, offset + 700; scale=0.01f0),
            gate_weight=_ch46_tiny_hf_matrix(32, 16, offset + 800),
            up_weight=_ch46_tiny_hf_matrix(32, 16, offset + 900),
            down_weight=_ch46_tiny_hf_matrix(16, 32, offset + 1_000),
        )
    end
    final_norm = 1.0f0 .+
        _ch46_tiny_values(16, 90_000; scale=0.01f0)
    return (; embedding, blocks, final_norm, spec)
end

function _ch46_tiny_prefill_inputs()
    position_ids = reshape(Int[
        0 1 2 2 2 2 4 5
        0 1 2 2 3 3 4 5
        0 1 2 3 2 3 4 5
    ], 3, 8, 1)
    visual_mask = falses(8, 1)
    visual_mask[3:6, 1] .= true
    rope_layout = Qwen3VLRopeLayout(
        position_ids,
        reshape(Int[-2], 1, 1),
        visual_mask,
        trues(8, 1),
    )
    visual_embeddings = permutedims(
        _ch46_tiny_hf_matrix(4, 16, 100_000; scale=0.1f0),
    )
    deepstack = ntuple(3) do index
        permutedims(_ch46_tiny_hf_matrix(
            4,
            16,
            110_000 + 1_000 * (index - 1);
            scale=0.1f0,
        ))
    end
    return (;
        input_ids=collect(1:8),
        rope_layout,
        vision_features=(; visual_embeddings, deepstack),
    )
end

function _ch46_reference()
    path = normpath(joinpath(
        @__DIR__,
        "..",
        "..",
        "episode09_qwen3_vl_multimodal_perception",
        "chapter45_qwen3_vl_dynamic_decode",
        "fixtures",
        "tiny_text_dynamic_decode.json",
    ))
    return JSON3.read(read(path, String))
end

function _ch46_reference_bytes(reference, name::AbstractString)
    entry = reference.tensors[Symbol(name)]
    bytes = base64decode(String(entry.f32_le_base64))
    @test bytes2hex(sha256(bytes)) == String(entry.sha256)
    return bytes, Int.(collect(entry.shape))
end

function _ch46_hf_hidden(reference, name::AbstractString)
    bytes, shape = _ch46_reference_bytes(reference, name)
    batch, tokens, width = shape
    return reshape(collect(reinterpret(Float32, bytes)), width, tokens, batch)
end

function _ch46_hf_cache(reference, phase::AbstractString, layer::Int, kind::String)
    bytes, shape = _ch46_reference_bytes(
        reference,
        "cache.$phase.layer.$layer.$kind",
    )
    batch, kv_heads, tokens, head_dim = shape
    reversed_hf = reshape(
        collect(reinterpret(Float32, bytes)),
        head_dim,
        tokens,
        kv_heads,
        batch,
    )
    return permutedims(reversed_hf, (1, 3, 2, 4))
end

function _ch46_top_two(logits)
    scores = vec(Array(view(logits, :, size(logits, 2), 1)))
    ids = partialsortperm(scores, 1:2; rev=true)
    return (; ids, margin=scores[ids[1]] - scores[ids[2]])
end

function _ch46_storage_bytes(cache)
    bytes = 0
    for layer in cache.layers
        bytes += length(layer.keys) * sizeof(eltype(layer.keys))
        bytes += length(layer.values) * sizeof(eltype(layer.values))
    end
    return bytes
end

function _ch46_storage_refs(cache)
    return (
        keys=map(layer -> layer.keys, cache.layers),
        values=map(layer -> layer.values, cache.layers),
    )
end

function _ch46_assert_storage_identity(cache, refs)
    for layer in eachindex(cache.layers)
        @test cache.layers[layer].keys === refs.keys[layer]
        @test cache.layers[layer].values === refs.values[layer]
    end
end

function _ch46_assert_prefix(reference, cache, phase::String, tokens::Int)
    @test cache.position == tokens
    @test length(cache) == tokens
    for layer in 0:3
        actual = cache.layers[layer + 1]
        expected_key = _ch46_hf_cache(reference, phase, layer, "key")
        expected_value = _ch46_hf_cache(reference, phase, layer, "value")
        @test view(actual.keys, :, :, 1:tokens, :) ≈
            expected_key atol=1.0f-6 rtol=1.0f-6
        @test view(actual.values, :, :, 1:tokens, :) ≈
            expected_value atol=1.0f-6 rtol=1.0f-6
    end
end

@testset "Chapter 46 — static layer storage is strict" begin
    keys = zeros(Float32, 8, 1, 4, 1)
    values = similar(keys)
    fill!(values, 1.0f0)
    layer = Qwen3VLStaticLayerKVCache(keys, values)
    @test layer.keys === keys
    @test layer.values === values
    @test size(layer.keys) == (8, 1, 4, 1)
    @test eltype(layer.values) === Float32

    @test_throws MethodError Qwen3VLStaticLayerKVCache{
        typeof(keys),
        typeof(values),
    }(keys, values)
    @test_throws MethodError Qwen3VLStaticLayerKVCache(1, 2)
    @test_throws DimensionMismatch Qwen3VLStaticLayerKVCache(
        keys,
        zeros(Float32, 8, 1, 4),
    )
    @test_throws DimensionMismatch Qwen3VLStaticLayerKVCache(
        keys,
        zeros(Float32, 8, 1, 3, 1),
    )
    @test_throws ArgumentError Qwen3VLStaticLayerKVCache(
        zeros(Float64, 8, 1, 4, 1),
        zeros(Float64, 8, 1, 4, 1),
    )
    @test_throws ArgumentError Qwen3VLStaticLayerKVCache(
        keys,
        zeros(Float64, 8, 1, 4, 1),
    )
    @test_throws ArgumentError Qwen3VLStaticLayerKVCache(keys, keys)
    @test_throws ArgumentError Qwen3VLStaticLayerKVCache(
        zeros(Float32, 8, 1, 0, 1),
        zeros(Float32, 8, 1, 0, 1),
    )
    bf16_keys = zeros(Core.BFloat16, 8, 1, 4, 1)
    bf16_values = similar(bf16_keys)
    bf16_layer = Qwen3VLStaticLayerKVCache(bf16_keys, bf16_values)
    @test bf16_layer.keys === bf16_keys
    @test bf16_layer.values === bf16_values
end

@testset "Chapter 46 — static cache constructor is strict" begin
    keys = zeros(Float32, 8, 1, 4, 1)
    values = ones(Float32, 8, 1, 4, 1)
    layers = (Qwen3VLStaticLayerKVCache(keys, values),)
    cache = Qwen3VLStaticKVCache(
        layers,
        Int32(2),
        Int16(-1),
        UInt8(1),
        Int128(4),
    )
    @test cache.layers === layers
    @test cache.position === 2
    @test cache.rope_delta === -1
    @test cache.batch_size === 1
    @test cache.capacity === 4
    @test length(cache) == 2
    @test !isempty(cache)

    too_large = big(typemax(Int)) + 1
    for (position, rope_delta, batch_size, capacity, message) in (
        (true, 0, 1, 4, "Qwen3-VL static cache position must be an integer"),
        (0, true, 1, 4, "Qwen3-VL static cache rope_delta must be an integer"),
        (0, 0, true, 4, "Qwen3-VL static cache batch_size must be an integer"),
        (0, 0, 1, true, "Qwen3-VL static cache capacity must be an integer"),
        (
            too_large,
            0,
            1,
            4,
            "Qwen3-VL static cache position is outside the host integer range",
        ),
        (
            0,
            too_large,
            1,
            4,
            "Qwen3-VL static cache rope_delta is outside the host integer range",
        ),
        (
            0,
            0,
            too_large,
            4,
            "Qwen3-VL static cache batch_size is outside the host integer range",
        ),
        (
            0,
            0,
            1,
            too_large,
            "Qwen3-VL static cache capacity is outside the host integer range",
        ),
        (0, 0, 1, 0, "Qwen3-VL static cache capacity must be positive"),
        (0, 0, 1, -1, "Qwen3-VL static cache capacity must be positive"),
        (-1, 0, 1, 4, "Qwen3-VL static cache position must be non-negative"),
        (
            5,
            0,
            1,
            4,
            "Qwen3-VL static cache position must not exceed capacity",
        ),
        (
            0,
            0,
            2,
            4,
            "Qwen3-VL static generation currently supports batch size one",
        ),
        (
            0,
            1,
            1,
            4,
            "an empty Qwen3-VL static cache must have rope_delta == 0",
        ),
    )
        failure = _ch46_captured_error() do
            Qwen3VLStaticKVCache(
                layers,
                position,
                rope_delta,
                batch_size,
                capacity,
            )
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end
    @test_throws MethodError Qwen3VLStaticKVCache{typeof(layers)}(
        layers,
        false,
        true,
        true,
        false,
    )
end

@testset "Chapter 46 — static cache layer collection is strict" begin
    make_layer(shape=(8, 1, 4, 1), dtype=Float32) = begin
        keys = zeros(dtype, shape)
        values = similar(keys)
        fill!(values, one(dtype))
        Qwen3VLStaticLayerKVCache(keys, values)
    end

    first_layer = make_layer()
    second_layer = make_layer()
    layers = (first_layer, second_layer)
    cache = Qwen3VLStaticKVCache(layers, 0, 0, 1, 4)
    @test cache.layers === layers
    @test cache.layers[1] === first_layer
    @test cache.layers[2] === second_layer
    @test cache.layers[1].keys !== cache.layers[2].keys

    @test_throws ArgumentError Qwen3VLStaticKVCache(
        Qwen3VLStaticLayerKVCache[first_layer],
        0,
        0,
        1,
        4,
    )
    @test_throws ArgumentError Qwen3VLStaticKVCache((), 0, 0, 1, 4)
    @test_throws ArgumentError Qwen3VLStaticKVCache((1,), 0, 0, 1, 4)
    @test_throws DimensionMismatch Qwen3VLStaticKVCache(
        (make_layer((8, 1, 3, 1)),),
        0,
        0,
        1,
        4,
    )
    @test_throws DimensionMismatch Qwen3VLStaticKVCache(
        (make_layer((8, 1, 4, 2)),),
        0,
        0,
        1,
        4,
    )
    @test_throws DimensionMismatch Qwen3VLStaticKVCache(
        (first_layer, make_layer((7, 1, 4, 1))),
        0,
        0,
        1,
        4,
    )
    @test_throws ArgumentError Qwen3VLStaticKVCache(
        (first_layer, make_layer((8, 1, 4, 1), Core.BFloat16)),
        0,
        0,
        1,
        4,
    )
    @test_throws ArgumentError Qwen3VLStaticKVCache(
        (first_layer, first_layer),
        0,
        0,
        1,
        4,
    )
    cross_reused_layer = Qwen3VLStaticLayerKVCache(
        first_layer.values,
        first_layer.keys,
    )
    @test_throws ArgumentError Qwen3VLStaticKVCache(
        (first_layer, cross_reused_layer),
        0,
        0,
        1,
        4,
    )
    foreign_layer = Qwen3VLStaticLayerKVCache(
        _Ch46ForeignDeviceArray(zeros(Float32, 8, 1, 4, 1)),
        _Ch46ForeignDeviceArray(ones(Float32, 8, 1, 4, 1)),
    )
    @test_throws ArgumentError Qwen3VLStaticKVCache(
        (first_layer, foreign_layer),
        0,
        0,
        1,
        4,
    )

    cache.layers = (first_layer, first_layer)
    @test_throws ArgumentError reset_qwen3_vl_static_kv_cache!(cache)
    @test_throws ArgumentError reset_qwen3_vl_static_kv_cache!(
        cache;
        clear=true,
    )
end

@testset "Chapter 46 — static cache allocation and error contract" begin
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()
    cache = init_qwen3_vl_static_kv_cache(
        parameters;
        capacity=Int128(10),
        batch_size=big(1),
    )

    @test cache isa Qwen3VLStaticKVCache
    @test isempty(cache)
    @test length(cache) == 0
    @test cache.position == 0
    @test cache.rope_delta == 0
    @test cache.capacity == 10
    @test cache.batch_size == 1
    @test length(cache.layers) == parameters.spec.num_hidden_layers
    @test all(
        layer -> size(layer.keys) == (8, 1, 10, 1),
        cache.layers,
    )
    @test all(
        layer -> size(layer.values) == (8, 1, 10, 1),
        cache.layers,
    )
    @test all(layer -> eltype(layer.keys) === Float32, cache.layers)
    @test all(layer -> eltype(layer.values) === Float32, cache.layers)

    dynamic = init_qwen3_vl_kv_cache(parameters; batch_size=Int32(1))
    @test dynamic.batch_size === 1
    @test length(dynamic.layers) == parameters.spec.num_hidden_layers

    overflow_integer = big(typemax(Int)) + 1
    for invalid_integer in (true, overflow_integer)
        dynamic_error = try
            init_qwen3_vl_kv_cache(42; batch_size=invalid_integer)
            nothing
        catch caught
            caught
        end
        @test dynamic_error isa ArgumentError
        @test occursin("batch_size", sprint(showerror, dynamic_error))

        capacity_error = try
            init_qwen3_vl_static_kv_cache(
                42;
                capacity=invalid_integer,
                batch_size=1,
            )
            nothing
        catch caught
            caught
        end
        @test capacity_error isa ArgumentError
        @test occursin("capacity", sprint(showerror, capacity_error))

        batch_error = try
            init_qwen3_vl_static_kv_cache(
                42;
                capacity=1,
                batch_size=invalid_integer,
            )
            nothing
        catch caught
            caught
        end
        @test batch_error isa ArgumentError
        @test occursin("batch_size", sprint(showerror, batch_error))
    end

    arithmetic_cases = (
        (
            spec=_ch46_bad_cache_spec(num_hidden_layers=true),
            block_count=1,
            needle="num_hidden_layers",
        ),
        (
            spec=_ch46_bad_cache_spec(head_dim=1.0),
            block_count=1,
            needle="head_dim",
        ),
        (
            spec=_ch46_bad_cache_spec(
                num_key_value_heads=overflow_integer,
            ),
            block_count=1,
            needle="num_key_value_heads",
        ),
        (
            spec=_ch46_bad_cache_spec(max_position_embeddings=true),
            block_count=1,
            needle="max_position_embeddings",
        ),
        (
            spec=_ch46_bad_cache_spec(num_hidden_layers=0),
            block_count=0,
            needle="num_hidden_layers must be positive",
        ),
        (
            spec=_ch46_bad_cache_spec(head_dim=0),
            block_count=1,
            needle="head_dim must be positive",
        ),
        (
            spec=_ch46_bad_cache_spec(num_key_value_heads=0),
            block_count=1,
            needle="num_key_value_heads must be positive",
        ),
        (
            spec=_ch46_bad_cache_spec(max_position_embeddings=0),
            block_count=1,
            needle="max_position_embeddings must be positive",
        ),
        (
            spec=_ch46_bad_cache_spec(
                head_dim=typemax(Int),
                num_key_value_heads=2,
            ),
            block_count=1,
            needle="tensor element count",
        ),
        (
            spec=_ch46_bad_cache_spec(
                num_hidden_layers=2,
                head_dim=typemax(Int) ÷ 4 + 1,
            ),
            block_count=2,
            needle="total element count",
        ),
        (
            spec=_ch46_bad_cache_spec(
                head_dim=typemax(Int) ÷ 8 + 1,
            ),
            block_count=1,
            needle="byte count",
        ),
    )
    for case in arithmetic_cases
        touched = Ref(false)
        parameters_with_bad_spec = (;
            embedding=_Ch46NoAllocate{Float32}(touched),
            blocks=ntuple(_ -> nothing, case.block_count),
            final_norm=nothing,
            spec=case.spec,
        )
        arithmetic_error = try
            init_qwen3_vl_static_kv_cache(
                parameters_with_bad_spec;
                capacity=1,
            )
            nothing
        catch caught
            caught
        end
        @test arithmetic_error isa ArgumentError
        @test occursin(case.needle, sprint(showerror, arithmetic_error))
        @test !touched[]
    end

    for invalid_keep in (true, overflow_integer)
        dynamic_guard = init_qwen3_vl_kv_cache(parameters)
        dynamic_error = try
            hf_qwen3_vl_text_prefill_cached(
                parameters,
                42,
                inputs.rope_layout;
                cache=dynamic_guard,
                logits_to_keep=invalid_keep,
            )
            nothing
        catch caught
            caught
        end
        @test dynamic_error isa ArgumentError
        @test occursin("logits_to_keep", sprint(showerror, dynamic_error))
        @test isempty(dynamic_guard)
        @test all(
            layer -> layer.keys === nothing && layer.values === nothing,
            dynamic_guard.layers,
        )

        static_guard = init_qwen3_vl_static_kv_cache(
            parameters;
            capacity=10,
        )
        static_refs = _ch46_storage_refs(static_guard)
        static_error = try
            hf_qwen3_vl_text_prefill_static(
                parameters,
                42,
                inputs.rope_layout;
                cache=static_guard,
                logits_to_keep=invalid_keep,
            )
            nothing
        catch caught
            caught
        end
        @test static_error isa ArgumentError
        @test occursin("logits_to_keep", sprint(showerror, static_error))
        @test isempty(static_guard)
        @test static_guard.rope_delta == 0
        @test all(layer -> all(iszero, layer.keys), static_guard.layers)
        @test all(layer -> all(iszero, layer.values), static_guard.layers)
        @test all(eachindex(static_guard.layers)) do layer
            static_guard.layers[layer].keys === static_refs.keys[layer] &&
                static_guard.layers[layer].values === static_refs.values[layer]
        end
    end

    invalid_integer_layouts = (
        (
            layout=() -> Qwen3VLRopeLayout(
                Bool.(inputs.rope_layout.position_ids .> 0),
                reshape(Int[-6], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL position_ids must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                Float64.(inputs.rope_layout.position_ids),
                inputs.rope_layout.rope_deltas,
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL position_ids must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                inputs.rope_layout.position_ids,
                reshape(Bool[true], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL rope_delta must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                inputs.rope_layout.position_ids,
                reshape(Float64[-2.0], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL rope_delta must be an integer",
        ),
        (
            layout=() -> Qwen3VLRopeLayout(
                inputs.rope_layout.position_ids,
                reshape(BigInt[overflow_integer], 1, 1),
                inputs.rope_layout.visual_mask,
                inputs.rope_layout.attention_mask,
            ),
            message="Qwen3-VL rope_delta is outside the host integer range",
        ),
    )
    for case in invalid_integer_layouts
        static_guard = init_qwen3_vl_static_kv_cache(
            parameters;
            capacity=10,
        )
        static_refs = _ch46_storage_refs(static_guard)
        layout_error = try
            layout = case.layout()
            hf_qwen3_vl_text_prefill_static(
                parameters,
                inputs.input_ids,
                layout;
                vision_features=inputs.vision_features,
                cache=static_guard,
            )
            nothing
        catch caught
            caught
        end
        @test layout_error isa ArgumentError
        @test layout_error isa Exception &&
            occursin(case.message, sprint(showerror, layout_error))
        @test isempty(static_guard)
        @test static_guard.position == 0
        @test static_guard.rope_delta == 0
        @test all(layer -> all(iszero, layer.keys), static_guard.layers)
        @test all(layer -> all(iszero, layer.values), static_guard.layers)
        _ch46_assert_storage_identity(static_guard, static_refs)
    end

    expected_bytes = parameters.spec.num_hidden_layers * 2 *
        parameters.spec.head_dim * parameters.spec.num_key_value_heads *
        cache.capacity * cache.batch_size * sizeof(Float32)
    @test expected_bytes == 2_560
    @test _ch46_storage_bytes(cache) == expected_bytes

    @test_throws ArgumentError init_qwen3_vl_static_kv_cache(
        parameters;
        capacity=0,
    )
    @test_throws UndefKeywordError init_qwen3_vl_static_kv_cache(parameters)
    @test_throws ArgumentError init_qwen3_vl_static_kv_cache(
        parameters;
        capacity=parameters.spec.max_position_embeddings + 1,
    )
    @test_throws ArgumentError init_qwen3_vl_static_kv_cache(
        parameters;
        capacity=10,
        batch_size=2,
    )
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step_static(
        parameters,
        8,
        cache,
    )

    too_small = init_qwen3_vl_static_kv_cache(parameters; capacity=7)
    too_small_refs = _ch46_storage_refs(too_small)
    @test_throws ArgumentError hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=too_small,
    )
    @test isempty(too_small)
    _ch46_assert_storage_identity(too_small, too_small_refs)

    malformed_layout = Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        reshape(Int[-1], 1, 1),
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    )
    malformed = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
    @test_throws ArgumentError hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        malformed_layout;
        vision_features=inputs.vision_features,
        cache=malformed,
    )
    @test isempty(malformed)
end

@testset "Chapter 46 — feature residency fails before static cache writes" begin
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()
    dtype_features = merge(inputs.vision_features, (
        visual_embeddings=Float64.(
            inputs.vision_features.visual_embeddings,
        ),
    ))
    foreign_deepstack = merge(inputs.vision_features, (
        deepstack=Base.setindex(
            inputs.vision_features.deepstack,
            _Ch46ForeignDeviceArray(inputs.vision_features.deepstack[2]),
            2,
        ),
    ))

    for (features, message) in (
        (
            dtype_features,
            "ArgumentError: Qwen3-VL visual_embeddings dtype must match " *
            "text parameter embedding",
        ),
        (
            foreign_deepstack,
            "ArgumentError: Qwen3-VL deepstack[2] device must match " *
            "text parameter embedding",
        ),
    )
        cache = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
        refs = _ch46_storage_refs(cache)
        key_snapshots = map(layer -> copy(layer.keys), cache.layers)
        value_snapshots = map(layer -> copy(layer.values), cache.layers)
        error = _ch46_captured_error(() ->
            hf_qwen3_vl_text_prefill_static(
                parameters,
                inputs.input_ids,
                inputs.rope_layout;
                vision_features=features,
                cache,
            ),
        )
        @test error isa ArgumentError
        @test sprint(showerror, error) == message
        @test isempty(cache)
        @test cache.position == 0
        @test cache.rope_delta == 0
        _ch46_assert_storage_identity(cache, refs)
        for layer in eachindex(cache.layers)
            @test cache.layers[layer].keys == key_snapshots[layer]
            @test cache.layers[layer].values == value_snapshots[layer]
        end
    end
end

@testset "Chapter 46 — reset reuses the allocation" begin
    reference = _ch46_reference()
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
    refs = _ch46_storage_refs(cache)
    fixed_bytes = _ch46_storage_bytes(cache)

    first_prefill, _ = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache,
        logits_to_keep=Int128(0),
    )
    token = _ch46_top_two(first_prefill.logits).ids[1]
    first_decode, _ = hf_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        cache,
    )
    @test cache.position == 9
    @test cache.rope_delta == -2

    returned = reset_qwen3_vl_static_kv_cache!(cache)
    @test returned === cache
    @test isempty(cache)
    @test cache.position == 0
    @test cache.rope_delta == 0
    @test cache.capacity == 10
    _ch46_assert_storage_identity(cache, refs)
    @test _ch46_storage_bytes(cache) == fixed_bytes

    second_prefill, returned = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache,
        logits_to_keep=big(0),
    )
    @test returned === cache
    @test second_prefill.final_hidden ≈
        first_prefill.final_hidden atol=1.0f-6 rtol=1.0f-6
    @test second_prefill.logits ≈
        first_prefill.logits atol=1.0f-6 rtol=1.0f-6
    _ch46_assert_prefix(reference, cache, "prefill", 8)
    second_decode, returned = hf_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        cache,
    )
    @test returned === cache
    @test second_decode ≈ first_decode atol=1.0f-6 rtol=1.0f-6
    _ch46_assert_prefix(reference, cache, "decode.0", 9)
    _ch46_assert_storage_identity(cache, refs)
    @test _ch46_storage_bytes(cache) == fixed_bytes

    # `clear=true` is a deterministic hygiene option for cross-request reuse;
    # it clears storage without replacing it.
    returned = reset_qwen3_vl_static_kv_cache!(cache; clear=true)
    @test returned === cache
    @test isempty(cache)
    @test cache.rope_delta == 0
    @test all(layer -> all(iszero, layer.keys), cache.layers)
    @test all(layer -> all(iszero, layer.values), cache.layers)
    _ch46_assert_storage_identity(cache, refs)
end

@testset "Chapter 46 — fixed storage matches DynamicCache and frozen HF" begin
    reference = _ch46_reference()
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()

    dynamic = init_qwen3_vl_kv_cache(parameters; batch_size=1)
    dynamic_prefill, dynamic = hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=dynamic,
        logits_to_keep=Int128(0),
    )

    wide_layout = Qwen3VLRopeLayout(
        UInt128.(inputs.rope_layout.position_ids),
        Int128.(inputs.rope_layout.rope_deltas),
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    )
    wide_static = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
    wide_prefill, wide_returned = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        wide_layout;
        vision_features=inputs.vision_features,
        cache=wide_static,
        logits_to_keep=0,
    )
    @test wide_returned === wide_static
    @test wide_static.position == 8
    @test wide_static.rope_delta == -2

    static = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
    refs = _ch46_storage_refs(static)
    fixed_bytes = _ch46_storage_bytes(static)
    static_prefill, returned = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=static,
        logits_to_keep=big(0),
    )
    @test returned === static
    @test static.position == 8
    @test static.rope_delta == -2
    @test static_prefill.final_hidden ≈
        _ch46_hf_hidden(reference, "prefill.final_hidden") atol=1.0f-6 rtol=1.0f-6
    @test static_prefill.logits ≈
        _ch46_hf_hidden(reference, "prefill.logits") atol=1.0f-6 rtol=1.0f-6
    @test static_prefill.final_hidden ≈
        dynamic_prefill.final_hidden atol=1.0f-6 rtol=1.0f-6
    @test static_prefill.logits ≈
        dynamic_prefill.logits atol=1.0f-6 rtol=1.0f-6
    @test wide_prefill.final_hidden == static_prefill.final_hidden
    @test wide_prefill.logits == static_prefill.logits
    _ch46_assert_prefix(reference, static, "prefill", 8)
    for layer in eachindex(static.layers)
        @test wide_static.layers[layer].keys == static.layers[layer].keys
        @test wide_static.layers[layer].values == static.layers[layer].values
        @test view(static.layers[layer].keys, :, :, 1:8, :) ≈
            dynamic.layers[layer].keys atol=1.0f-6 rtol=1.0f-6
        @test view(static.layers[layer].values, :, :, 1:8, :) ≈
            dynamic.layers[layer].values atol=1.0f-6 rtol=1.0f-6
    end
    _ch46_assert_storage_identity(static, refs)
    @test _ch46_storage_bytes(static) == fixed_bytes

    first = _ch46_top_two(static_prefill.logits)
    @test first.ids == [8, 24]
    dynamic_logits9, dynamic = hf_qwen3_vl_text_decode_step(
        parameters,
        first.ids[1],
        dynamic,
    )
    static_logits9, returned = hf_qwen3_vl_text_decode_step_static(
        parameters,
        first.ids[1],
        static,
    )
    @test returned === static
    @test static_logits9 ≈
        _ch46_hf_hidden(reference, "decode.0.logits") atol=1.0f-6 rtol=1.0f-6
    @test static_logits9 ≈ dynamic_logits9 atol=1.0f-6 rtol=1.0f-6
    _ch46_assert_prefix(reference, static, "decode.0", 9)
    for layer in eachindex(static.layers)
        @test view(static.layers[layer].keys, :, :, 1:9, :) ≈
            dynamic.layers[layer].keys atol=1.0f-6 rtol=1.0f-6
        @test view(static.layers[layer].values, :, :, 1:9, :) ≈
            dynamic.layers[layer].values atol=1.0f-6 rtol=1.0f-6
    end
    _ch46_assert_storage_identity(static, refs)
    @test _ch46_storage_bytes(static) == fixed_bytes

    second = _ch46_top_two(static_logits9)
    @test second.ids == [8, 24]
    dynamic_logits10, dynamic = hf_qwen3_vl_text_decode_step(
        parameters,
        second.ids[1],
        dynamic,
    )
    static_logits10, returned = hf_qwen3_vl_text_decode_step_static(
        parameters,
        second.ids[1],
        static,
    )
    @test returned === static
    @test static_logits10 ≈
        _ch46_hf_hidden(reference, "decode.1.logits") atol=1.0f-6 rtol=1.0f-6
    @test static_logits10 ≈ dynamic_logits10 atol=1.0f-6 rtol=1.0f-6
    _ch46_assert_prefix(reference, static, "decode.1", 10)
    for layer in eachindex(static.layers)
        @test static.layers[layer].keys ≈
            dynamic.layers[layer].keys atol=1.0f-6 rtol=1.0f-6
        @test static.layers[layer].values ≈
            dynamic.layers[layer].values atol=1.0f-6 rtol=1.0f-6
    end
    _ch46_assert_storage_identity(static, refs)
    @test _ch46_storage_bytes(static) == fixed_bytes

    third = _ch46_top_two(static_logits10)
    @test third.ids == [8, 24]
    @test [first.margin, second.margin, third.margin] ≈ Float32[
        phase.top_two.margin_f32 for phase in reference.metadata.phases
    ] atol=2.0f-7 rtol=2.0f-5

    # Greedy selection consumes prefill logits before the first decode. Three
    # output tokens therefore correspond to two appends and positions 8, 9, 10.
    timeline = [
        (; step=1, token_id=first.ids[1], hf_token_id=first.ids[1] - 1,
            cache_position=8),
        (; step=2, token_id=second.ids[1], hf_token_id=second.ids[1] - 1,
            cache_position=9),
        (; step=3, token_id=third.ids[1], hf_token_id=third.ids[1] - 1,
            cache_position=10),
    ]
    @test [entry.token_id for entry in timeline] == [8, 8, 8]
    @test [entry.hf_token_id for entry in timeline] == [7, 7, 7]
    @test [entry.cache_position for entry in timeline] == [8, 9, 10]

    full_keys = map(layer -> copy(layer.keys), static.layers)
    full_values = map(layer -> copy(layer.values), static.layers)
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step_static(
        parameters,
        third.ids[1],
        static,
    )
    @test static.position == 10
    @test static.rope_delta == -2
    for layer in eachindex(static.layers)
        @test static.layers[layer].keys == full_keys[layer]
        @test static.layers[layer].values == full_values[layer]
    end
    _ch46_assert_storage_identity(static, refs)
    @test_throws ArgumentError hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=static,
    )
end

@testset "Chapter 46 — exact-capacity boundaries are atomic" begin
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()

    prompt_only = init_qwen3_vl_static_kv_cache(parameters; capacity=8)
    prompt_refs = _ch46_storage_refs(prompt_only)
    _, returned = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=prompt_only,
    )
    @test returned === prompt_only
    @test prompt_only.position == 8
    prompt_snapshot = map(layer -> (copy(layer.keys), copy(layer.values)),
        prompt_only.layers)
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step_static(
        parameters,
        8,
        prompt_only,
    )
    @test prompt_only.position == 8
    for layer in eachindex(prompt_only.layers)
        @test prompt_only.layers[layer].keys == prompt_snapshot[layer][1]
        @test prompt_only.layers[layer].values == prompt_snapshot[layer][2]
    end
    _ch46_assert_storage_identity(prompt_only, prompt_refs)

    one_append = init_qwen3_vl_static_kv_cache(parameters; capacity=9)
    prefill, _ = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=one_append,
    )
    token = _ch46_top_two(prefill.logits).ids[1]
    _, returned = hf_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        one_append,
    )
    @test returned === one_append
    @test one_append.position == 9
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        one_append,
    )
    @test one_append.position == 9

    token_validation = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
    _, _ = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=token_validation,
    )
    token_validation_refs = _ch46_storage_refs(token_validation)
    token_validation_snapshot = map(
        layer -> (copy(layer.keys), copy(layer.values)),
        token_validation.layers,
    )
    token_validation.rope_delta = typemax(Int)
    coordinate_error = try
        hf_qwen3_vl_text_decode_step_static(
            parameters,
            8,
            token_validation,
        )
        nothing
    catch caught
        caught
    end
    @test coordinate_error isa ArgumentError
    @test coordinate_error isa Exception && occursin(
        "decode mRoPE coordinate exceeds the host integer range",
        sprint(showerror, coordinate_error),
    )
    @test token_validation.position == 8
    @test token_validation.rope_delta == typemax(Int)
    for layer in eachindex(token_validation.layers)
        @test token_validation.layers[layer].keys ==
            token_validation_snapshot[layer][1]
        @test token_validation.layers[layer].values ==
            token_validation_snapshot[layer][2]
    end
    _ch46_assert_storage_identity(token_validation, token_validation_refs)
    token_validation.rope_delta = -2

    @test_throws ArgumentError hf_qwen3_vl_text_decode_step_static(
        parameters,
        0,
        token_validation,
    )
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step_static(
        parameters,
        parameters.spec.vocab_size + 1,
        token_validation,
    )
    @test_throws DimensionMismatch hf_qwen3_vl_text_decode_step_static(
        parameters,
        [8, 8],
        token_validation,
    )
    @test token_validation.position == 8
end

@testset "Chapter 46 — public greedy generation selects static storage" begin
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()
    static = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        capture_logits=true,
        cache=:static,
        static_capacity=big(10),
    )
    dynamic = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        capture_logits=true,
        cache=:dynamic,
    )

    @test static.cache isa Qwen3VLStaticKVCache
    @test static.cache_mode === :static
    @test static.cache.capacity == 10
    @test static.cache.position == 10
    @test static.cache.rope_delta == -2
    @test static.prefill.input_embeddings === nothing
    @test static.prefill.final_hidden === nothing
    @test dynamic.prefill.input_embeddings === nothing
    @test dynamic.prefill.final_hidden === nothing
    @test static.generated_ids == [8, 8, 8]
    @test static.generated_ids == dynamic.generated_ids
    @test static.token_ids == vcat(collect(1:8), [8, 8, 8])
    @test static.stop_reason === :length
    @test [step.step for step in static.trace] == [1, 2, 3]
    @test [step.token_id for step in static.trace] == [8, 8, 8]
    @test [step.hf_token_id for step in static.trace] == [7, 7, 7]
    @test [step.second_token_id for step in static.trace] == [24, 24, 24]
    for step in eachindex(static.trace)
        @test static.trace[step].logits ≈
            dynamic.trace[step].logits atol=1.0f-6 rtol=1.0f-6
    end

    captured_static = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        capture_logits=true,
        cache=:static,
        static_capacity=10,
        capture_prefill_states=true,
    )
    @test size(captured_static.prefill.input_embeddings) == (16, 8, 1)
    @test size(captured_static.prefill.final_hidden) == (16, 8, 1)
    @test captured_static.prefill.logits == static.prefill.logits
    @test captured_static.generated_ids == static.generated_ids
    @test captured_static.trace == static.trace
    @test captured_static.cache.position == static.cache.position

    inferred = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        cache=:static,
    )
    @test inferred.cache.capacity == 10
    @test inferred.cache.position == 10
    @test inferred.generated_ids == [8, 8, 8]

    zero = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=0,
        cache=:static,
    )
    @test zero.cache_mode === :static
    @test zero.cache.capacity == 8
    @test isempty(zero.cache)

    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        cache=:static,
        static_capacity=9,
    )
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=1,
        cache=:unknown,
    )
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=1,
        cache=:dynamic,
        static_capacity=8,
    )
    for invalid_capacity in (true, big(typemax(Int)) + 1)
        @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
            parameters,
            inputs.input_ids,
            inputs.rope_layout;
            vision_features=inputs.vision_features,
            max_new_tokens=1,
            cache=:static,
            static_capacity=invalid_capacity,
        )
    end
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=typemax(Int),
    )
end
