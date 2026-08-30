using Test
using Random
using Lux
import MLDataDevices
using LifeAI:
    GPTModel,
    GPTKVCache,
    LayerKVCache,
    _append_kv,
    decode_step,
    generate,
    generate_cached,
    init_kv_cache,
    init_static_kv_cache,
    prefill

struct _Ch02OffsetArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    parent::A
    offsets::NTuple{N,Int}
end

Base.size(array::_Ch02OffsetArray) = size(array.parent)
Base.axes(array::_Ch02OffsetArray{T,N}) where {T,N} = ntuple(N) do dimension
    parent_axis = axes(array.parent, dimension)
    offset = array.offsets[dimension]
    return (first(parent_axis) + offset):(last(parent_axis) + offset)
end
Base.IndexStyle(::Type{<:_Ch02OffsetArray}) = IndexCartesian()
function Base.getindex(
    array::_Ch02OffsetArray{T,N},
    indices::Vararg{Int,N},
) where {T,N}
    parent_indices = ntuple(
        dimension -> indices[dimension] - array.offsets[dimension],
        N,
    )
    return getindex(array.parent, parent_indices...)
end

struct _Ch02ForeignDevice <: MLDataDevices.AbstractDevice end

struct _Ch02ForeignDeviceArray{T,N,A<:AbstractArray{T,N}} <:
    AbstractArray{T,N}
    parent::A
end

Base.size(array::_Ch02ForeignDeviceArray) = size(array.parent)
Base.axes(array::_Ch02ForeignDeviceArray) = axes(array.parent)
Base.IndexStyle(::Type{<:_Ch02ForeignDeviceArray}) = IndexCartesian()
Base.getindex(array::_Ch02ForeignDeviceArray, indices...) =
    getindex(array.parent, indices...)
MLDataDevices.get_device(::_Ch02ForeignDeviceArray) = _Ch02ForeignDevice()

function _ch02_captured_error(f)
    try
        f()
    catch error
        return sprint(showerror, error)
    end
    return nothing
end

@testset "LayerKVCache seals dynamic storage invariants" begin
    empty_cache = LayerKVCache()
    @test empty_cache.keys === nothing
    @test empty_cache.values === nothing
    @test isempty(empty_cache)
    @test isempty(LayerKVCache(nothing, nothing))

    keys = reshape(collect(Int32, 1:24), 2, 3, 4, 1)
    values = reshape(collect(Int32, 25:48), 2, 3, 4, 1)
    cache = LayerKVCache(keys, values)
    @test cache.keys === keys
    @test cache.values === values
    @test length(cache) == 4
    @test eltype(cache.keys) == Int32

    @test_throws MethodError LayerKVCache(nothing, values)
    @test_throws MethodError LayerKVCache(keys, nothing)
    @test_throws MethodError LayerKVCache(1, 2)
    @test_throws MethodError LayerKVCache{Nothing,Nothing}(nothing, nothing)
    @test_throws MethodError LayerKVCache{
        typeof(keys),
        typeof(values),
    }(keys, values)
    @test_throws DimensionMismatch LayerKVCache(
        zeros(Float32, 2, 3, 4),
        zeros(Float32, 2, 3, 4),
    )
    @test_throws DimensionMismatch LayerKVCache(
        zeros(Float32, 2, 3, 4, 1),
        zeros(Float32, 2, 3, 5, 1),
    )
    @test_throws ArgumentError LayerKVCache(
        zeros(Float32, 2, 3, 0, 1),
        zeros(Float32, 2, 3, 0, 1),
    )
    @test_throws ArgumentError LayerKVCache(
        _Ch02OffsetArray(zeros(Float32, 2, 3, 4, 1), (1, 0, 0, 0)),
        zeros(Float32, 2, 3, 4, 1),
    )
    @test_throws ArgumentError LayerKVCache(
        zeros(Float32, 2, 3, 4, 1),
        _Ch02OffsetArray(zeros(Float32, 2, 3, 4, 1), (0, 0, 1, 0)),
    )
    @test_throws ArgumentError LayerKVCache(
        zeros(Float32, 2, 3, 4, 1),
        zeros(Float64, 2, 3, 4, 1),
    )
    @test_throws ArgumentError LayerKVCache(
        zeros(Float32, 2, 3, 4, 1),
        _Ch02ForeignDeviceArray(zeros(Float32, 2, 3, 4, 1)),
    )
    @test_throws ArgumentError LayerKVCache(keys, keys)

    first_keys = zeros(Float32, 2, 3, 2, 1)
    first_values = ones(Float32, 2, 3, 2, 1)
    first_cache = _append_kv(LayerKVCache(), first_keys, first_values)
    @test first_cache.keys === first_keys
    @test first_cache.values === first_values
    @test_throws DimensionMismatch _append_kv(
        LayerKVCache(),
        zeros(Float32, 2, 3, 1),
        ones(Float32, 2, 3, 1),
    )
    @test_throws ArgumentError _append_kv(
        LayerKVCache(),
        _Ch02OffsetArray(zeros(Float32, 2, 3, 1, 1), (0, 1, 0, 0)),
        ones(Float32, 2, 3, 1, 1),
    )
    @test_throws DimensionMismatch _append_kv(
        LayerKVCache(),
        zeros(Float32, 2, 3, 1, 1),
        ones(Float32, 2, 4, 1, 1),
    )
    shared = zeros(Float32, 2, 3, 1, 1)
    @test_throws ArgumentError _append_kv(LayerKVCache(), shared, shared)
    @test_throws ArgumentError _append_kv(
        first_cache,
        zeros(Float64, 2, 3, 1, 1),
        ones(Float64, 2, 3, 1, 1),
    )
end

@testset "GPTKVCache seals dynamic container invariants" begin
    empty_layers = (LayerKVCache(), LayerKVCache())
    empty_cache = GPTKVCache(empty_layers, Int32(0), BigInt(2))
    @test empty_cache.layers === empty_layers
    @test empty_cache.position === 0
    @test empty_cache.batch_size === 2
    @test isempty(empty_cache)
    @test length(empty_cache) == 0
    @test_throws MethodError GPTKVCache{typeof(empty_layers)}(
        empty_layers,
        0,
        2,
    )

    shape = (2, 3, 4, 2)
    first_layer = LayerKVCache(
        zeros(Int32, shape),
        ones(Int32, shape),
    )
    second_layer = LayerKVCache(
        fill(Int32(2), shape),
        fill(Int32(3), shape),
    )
    populated_layers = (first_layer, second_layer)
    populated = GPTKVCache(populated_layers, 4, 2)
    @test populated.layers === populated_layers
    @test length(populated) == 4
    @test !isempty(populated)
    @test eltype(populated.layers[1].keys) == Int32

    overflow = big(typemax(Int)) + 1
    invalid_metadata_cases = (
        (
            () -> GPTKVCache(empty_layers, true, 2),
            "ArgumentError: GPT KV cache position must be an integer",
        ),
        (
            () -> GPTKVCache(empty_layers, 1.5, 2),
            "ArgumentError: GPT KV cache position must be an integer",
        ),
        (
            () -> GPTKVCache(empty_layers, overflow, 2),
            "ArgumentError: GPT KV cache position is outside the host integer range",
        ),
        (
            () -> GPTKVCache(empty_layers, -1, 2),
            "ArgumentError: GPT KV cache position must be non-negative",
        ),
        (
            () -> GPTKVCache(empty_layers, 0, false),
            "ArgumentError: GPT KV cache batch_size must be an integer",
        ),
        (
            () -> GPTKVCache(empty_layers, 0, 1.5),
            "ArgumentError: GPT KV cache batch_size must be an integer",
        ),
        (
            () -> GPTKVCache(empty_layers, 0, overflow),
            "ArgumentError: GPT KV cache batch_size is outside the host integer range",
        ),
        (
            () -> GPTKVCache(empty_layers, 0, 0),
            "ArgumentError: GPT KV cache batch_size must be positive",
        ),
    )
    for (build, message) in invalid_metadata_cases
        @test _ch02_captured_error(build) == message
    end

    invalid_layer_cases = (
        (
            () -> GPTKVCache(collect(empty_layers), 0, 2),
            "ArgumentError: GPT KV cache layers must be a tuple",
        ),
        (
            () -> GPTKVCache((), 0, 2),
            "ArgumentError: GPT KV cache layers must not be empty",
        ),
        (
            () -> GPTKVCache((1,), 0, 2),
            "ArgumentError: GPT KV cache layer 1 must be LayerKVCache storage",
        ),
        (
            () -> GPTKVCache((first_layer,), 0, 2),
            "DimensionMismatch: empty GPT KV cache contains layer storage",
        ),
        (
            () -> GPTKVCache((LayerKVCache(),), 1, 2),
            "DimensionMismatch: populated GPT KV cache is missing layer 1 storage",
        ),
        (
            () -> GPTKVCache((first_layer, LayerKVCache()), 4, 2),
            "DimensionMismatch: populated GPT KV cache is missing layer 2 storage",
        ),
        (
            () -> GPTKVCache((first_layer,), 3, 2),
            "DimensionMismatch: GPT KV cache token dimension must match position",
        ),
        (
            () -> GPTKVCache((first_layer,), 4, 1),
            "DimensionMismatch: GPT KV cache batch dimension must match batch_size",
        ),
    )
    for (build, message) in invalid_layer_cases
        @test _ch02_captured_error(build) == message
    end

    wrong_shape_layer = LayerKVCache(
        zeros(Int32, 4, 3, 4, 2),
        ones(Int32, 4, 3, 4, 2),
    )
    @test _ch02_captured_error() do
        GPTKVCache((first_layer, wrong_shape_layer), 4, 2)
    end == "DimensionMismatch: GPT KV cache layer shapes must match"

    wrong_dtype_layer = LayerKVCache(
        zeros(Float32, shape),
        ones(Float32, shape),
    )
    @test _ch02_captured_error() do
        GPTKVCache((first_layer, wrong_dtype_layer), 4, 2)
    end == "ArgumentError: GPT KV cache layer dtypes must match"

    foreign_layer = LayerKVCache(
        _Ch02ForeignDeviceArray(zeros(Int32, shape)),
        _Ch02ForeignDeviceArray(ones(Int32, shape)),
    )
    @test _ch02_captured_error() do
        GPTKVCache((first_layer, foreign_layer), 4, 2)
    end == "ArgumentError: GPT KV cache layer devices must match"

    @test _ch02_captured_error() do
        GPTKVCache((first_layer, first_layer), 4, 2)
    end == "ArgumentError: GPT KV cache layers must use distinct storage"

    cross_reused_layer = LayerKVCache(first_layer.values, first_layer.keys)
    @test _ch02_captured_error() do
        GPTKVCache((first_layer, cross_reused_layer), 4, 2)
    end == "ArgumentError: GPT KV cache layers must use distinct storage"
end

@testset "KV cache prefill and incremental decode" begin
    rng = Xoshiro(20260713)
    model = GPTModel(
        13,
        16,
        2,
        2;
        max_seq_len=8,
        use_rope=true,
    )
    ps, st = Lux.setup(rng, model)

    prompt = reshape([1, 3, 5, 7], 4, 1)
    full_logits, _ = model(prompt, ps, st)

    cache = init_kv_cache(model; batch_size=1)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)

    @test isapprox(cached_logits, full_logits; atol=1.0f-5, rtol=1.0f-4)
    @test length(cache) == size(prompt, 1)
    @test all(layer_cache -> length(layer_cache) == length(cache), cache.layers)

    next_token = 9
    step_logits, cache, cached_state = decode_step(
        model,
        ps,
        cached_state,
        next_token,
        cache,
    )

    extended_prompt = vcat(prompt, reshape([next_token], 1, 1))
    extended_logits, _ = model(extended_prompt, ps, st)

    @test size(step_logits) == (model.vocab_size, 1, 1)
    @test isapprox(
        vec(step_logits[:, 1, 1]),
        vec(extended_logits[:, end, 1]);
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test length(cache) == size(extended_prompt, 1)
end


@testset "Batched KV cache matches full forward" begin
    rng = Xoshiro(23)
    model = GPTModel(19, 16, 2, 2; max_seq_len=6, use_rope=true)
    ps, st = Lux.setup(rng, model)
    prompt = [1 2; 3 4; 5 6]

    full_logits, _ = model(prompt, ps, st)
    cache = init_kv_cache(model; batch_size=2)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)

    @test isapprox(cached_logits, full_logits; atol=1.0f-5, rtol=1.0f-4)

    next_tokens = [7, 8]
    step_logits, cache, _ = decode_step(
        model,
        ps,
        cached_state,
        next_tokens,
        cache,
    )
    extended_prompt = vcat(prompt, reshape(next_tokens, 1, 2))
    extended_logits, _ = model(extended_prompt, ps, st)

    @test size(step_logits) == (model.vocab_size, 1, 2)
    @test isapprox(
        step_logits[:, 1, :],
        extended_logits[:, end, :];
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test length(cache) == 4
end

@testset "KV-cached greedy generation matches eager generation" begin
    rng = Xoshiro(7)
    model = GPTModel(
        17,
        24,
        3,
        2;
        max_seq_len=12,
        use_rope=true,
    )
    ps, st = Lux.setup(rng, model)
    prompt = [2, 4, 6]

    eager_tokens, _ = generate(
        model,
        ps,
        st,
        prompt;
        max_new_tokens=5,
        temperature=0,
    )
    cached_tokens, _ = generate_cached(
        model,
        ps,
        st,
        prompt;
        max_new_tokens=5,
        temperature=0,
    )

    @test cached_tokens == eager_tokens
end

@testset "KV cache validation" begin
    rng = Xoshiro(11)
    model = GPTModel(9, 8, 2, 1; max_seq_len=3, use_rope=true)
    ps, st = Lux.setup(rng, model)
    empty_cache = init_kv_cache(model)
    @test init_kv_cache(model; batch_size=BigInt(2)).batch_size == 2
    @test _ch02_captured_error() do
        init_kv_cache(model; batch_size=true)
    end == "ArgumentError: `batch_size` must be an integer"
    @test _ch02_captured_error() do
        init_kv_cache(model; batch_size=0)
    end == "ArgumentError: `batch_size` must be positive"
    @test _ch02_captured_error() do
        init_kv_cache(model; batch_size=big(typemax(Int)) + 1)
    end == "ArgumentError: `batch_size` is outside the host integer range"

    @test_throws ArgumentError decode_step(model, ps, st, 1, empty_cache)

    wrong_layer_count = GPTKVCache(
        (LayerKVCache(), LayerKVCache()),
        0,
        1,
    )
    @test _ch02_captured_error() do
        prefill(model, ps, st, [1], wrong_layer_count)
    end == "DimensionMismatch: cache layer count does not match model.num_layers"

    wrong_head_shape = (3, model.num_kv_heads, 1, 1)
    wrong_head_cache = GPTKVCache(
        (LayerKVCache(
            zeros(Float32, wrong_head_shape),
            ones(Float32, wrong_head_shape),
        ),),
        1,
        1,
    )
    @test _ch02_captured_error() do
        decode_step(model, ps, st, 1, wrong_head_cache)
    end ==
          "DimensionMismatch: layer 1 cache shape does not match model geometry"

    wrong_head_count_shape = (model.head_dim, 1, 1, 1)
    wrong_head_count_cache = GPTKVCache(
        (LayerKVCache(
            zeros(Float32, wrong_head_count_shape),
            ones(Float32, wrong_head_count_shape),
        ),),
        1,
        1,
    )
    @test _ch02_captured_error() do
        decode_step(model, ps, st, 1, wrong_head_count_cache)
    end ==
          "DimensionMismatch: layer 1 cache shape does not match model geometry"

    overlength_shape = (
        model.head_dim,
        model.num_kv_heads,
        model.max_seq_len + 1,
        1,
    )
    overlength_cache = GPTKVCache(
        (LayerKVCache(
            zeros(Float32, overlength_shape),
            ones(Float32, overlength_shape),
        ),),
        model.max_seq_len + 1,
        1,
    )
    @test _ch02_captured_error() do
        decode_step(model, ps, st, 1, overlength_cache)
    end ==
          "ArgumentError: cache position is outside 0:model.max_seq_len"

    _, full_cache, cached_state = prefill(model, ps, st, [1, 2, 3], empty_cache)
    @test_throws ArgumentError prefill(model, ps, cached_state, [1], full_cache)
    @test_throws ArgumentError decode_step(model, ps, cached_state, 4, full_cache)
    @test_throws ArgumentError generate_cached(
        model,
        ps,
        st,
        [1, 2];
        max_new_tokens=3,
        temperature=0,
    )
end

@testset "KV cache token ids require host-representable integers" begin
    rng = Xoshiro(20260830)
    model = GPTModel(13, 8, 2, 1; max_seq_len=4, use_rope=true)
    ps, st = Lux.setup(rng, model)

    _, int32_cache, _ = prefill(
        model,
        ps,
        st,
        Int32[1, 2],
        init_kv_cache(model),
    )
    @test length(int32_cache) == 2
    _, bigint_cache, bigint_state = prefill(
        model,
        ps,
        st,
        BigInt[1, 2],
        init_kv_cache(model),
    )
    @test length(bigint_cache) == 2
    _, decoded_cache, _ = decode_step(
        model,
        ps,
        bigint_state,
        BigInt(3),
        bigint_cache,
    )
    @test length(decoded_cache) == 3

    overflow = big(typemax(Int)) + 1
    for invalid in (
        Float64[1, 2],
        Bool[true, false],
        Char['\x01', '\x02'],
        BigInt[1, overflow],
    )
        @test_throws ArgumentError prefill(
            model,
            ps,
            st,
            invalid,
            init_kv_cache(model),
        )
    end
    for invalid in (true, Bool[true], Float64[3], Char['\x03'], BigInt[overflow])
        @test_throws ArgumentError decode_step(
            model,
            ps,
            bigint_state,
            invalid,
            bigint_cache,
        )
    end
end


@testset "Static KV cache keeps fixed storage and matches full forward" begin
    rng = Xoshiro(20260714)
    model = GPTModel(23, 24, 3, 2; max_seq_len=10, use_rope=true)
    ps, st = Lux.setup(rng, model)
    prompt = reshape([1, 4, 7, 10], 4, 1)

    cache = init_static_kv_cache(model; batch_size=1)
    key_buffers = map(layer -> layer.keys, cache.layers)
    value_buffers = map(layer -> layer.values, cache.layers)

    full_logits, _ = model(prompt, ps, st)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)

    @test isapprox(cached_logits, full_logits; atol=1.0f-5, rtol=1.0f-4)
    @test length(cache) == size(prompt, 1)
    @test all(
        layer -> size(layer.keys, 3) == model.max_seq_len,
        cache.layers,
    )
    @test all(
        index -> cache.layers[index].keys === key_buffers[index],
        eachindex(cache.layers),
    )
    @test all(
        index -> cache.layers[index].values === value_buffers[index],
        eachindex(cache.layers),
    )

    generated_context = vec(prompt)
    for next_token in (13, 16, 19)
        step_logits, cache, cached_state = decode_step(
            model,
            ps,
            cached_state,
            next_token,
            cache,
        )
        push!(generated_context, next_token)
        reference_logits, _ = model(
            reshape(generated_context, :, 1),
            ps,
            st,
        )

        @test isapprox(
            vec(step_logits[:, 1, 1]),
            vec(reference_logits[:, end, 1]);
            atol=1.0f-5,
            rtol=1.0f-4,
        )
        @test length(cache) == length(generated_context)
        @test all(
            index -> cache.layers[index].keys === key_buffers[index],
            eachindex(cache.layers),
        )
    end
end

@testset "Batched static KV cache matches full forward" begin
    rng = Xoshiro(20260715)
    model = GPTModel(29, 16, 2, 2; max_seq_len=8, use_rope=true)
    ps, st = Lux.setup(rng, model)
    prompt = [1 2; 3 4; 5 6]

    cache = init_static_kv_cache(model; batch_size=2)
    cached_logits, cache, cached_state = prefill(model, ps, st, prompt, cache)
    reference_logits, _ = model(prompt, ps, st)

    @test isapprox(cached_logits, reference_logits; atol=1.0f-5, rtol=1.0f-4)

    next_tokens = [7, 8]
    step_logits, cache, _ = decode_step(
        model,
        ps,
        cached_state,
        next_tokens,
        cache,
    )
    extended_prompt = vcat(prompt, reshape(next_tokens, 1, 2))
    extended_logits, _ = model(extended_prompt, ps, st)

    @test isapprox(
        step_logits[:, 1, :],
        extended_logits[:, end, :];
        atol=1.0f-5,
        rtol=1.0f-4,
    )
    @test length(cache) == 4
    @test all(layer -> size(layer.keys, 3) == 8, cache.layers)
end

@testset "Static KV cache validation" begin
    rng = Xoshiro(20260716)
    model = GPTModel(11, 8, 2, 1; max_seq_len=3, use_rope=true)
    ps, st = Lux.setup(rng, model)
    cache = init_static_kv_cache(model)

    @test_throws ArgumentError decode_step(model, ps, st, 1, cache)
    _, cache, cached_state = prefill(model, ps, st, [1, 2, 3], cache)
    @test_throws ArgumentError prefill(model, ps, cached_state, [1], cache)
    @test_throws ArgumentError decode_step(model, ps, cached_state, 4, cache)
end
