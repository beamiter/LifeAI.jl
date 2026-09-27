using Base64: base64decode
using JSON3
using SHA: sha256
using Test
import LifeAI
import MLDataDevices
using LifeAI: Qwen3VLRopeLayout,
    Qwen3VLTextSpec,
    LayerKVCache,
    generate_hf_qwen3_vl,
    generate_hf_qwen3_vl_tokens,
    hf_qwen3_vl_text_decode_step,
    hf_qwen3_vl_text_prefill_cached,
    init_qwen3_vl_kv_cache,
    load_hf_qwen3_vl_tokenizer,
    qwen3_vl_checkpoint_spec

isdefined(@__MODULE__, :qwen3_tokenizer_fixture_payloads) || include(joinpath(
    @__DIR__,
    "..",
    "..",
    "..",
    "support",
    "qwen3_tokenizer_fixture.jl",
))

const _CH45_TINY_TEXT_SPEC = Qwen3VLTextSpec(
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

function _ch45_tiny_values(count::Int, offset::Int; scale=0.02f0)
    return Float32[
        scale * sin(0.173f0 * Float32(offset + index))
        for index in 1:count
    ]
end

function _ch45_vl_generation_tokenizer()
    payloads = qwen3_tokenizer_fixture_payloads()
    delete!(payloads.tokenizer["model"], "ignore_merges")
    payloads.tokenizer["model"]["merges"] = [
        join(String.(pair), " ") for pair in payloads.tokenizer["model"]["merges"]
    ]
    payloads.generation_config["repetition_penalty"] = 1.0
    return mktempdir() do directory
        write_qwen3_tokenizer_fixture(directory; payloads)
        load_hf_qwen3_vl_tokenizer(
            directory;
            revision="qwen3-vl-preflight-test",
        )
    end
end

function _ch45_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected the test call to fail")
end

struct _CH45VisionComputePoison end

function Base.getproperty(::_CH45VisionComputePoison, ::Symbol)
    error("vision compute was touched")
end

struct _CH45GenerationCachePoison
    spec::Qwen3VLTextSpec
    embedding::Matrix{Float32}
end

Base.propertynames(
    ::_CH45GenerationCachePoison,
    ::Bool=false,
) = (:spec, :blocks, :embedding, :final_norm)

function Base.getproperty(
    parameters::_CH45GenerationCachePoison,
    name::Symbol,
)
    name === :spec && return getfield(parameters, :spec)
    name === :embedding && return getfield(parameters, :embedding)
    error("generation cache allocation was touched")
end

struct _Ch45ForeignDevice <: MLDataDevices.AbstractDevice end

mutable struct _Ch45MutableCheckpoint
    image_token_id::Int
    video_token_id::Int
end

mutable struct _Ch45MutableTextParameters{E,B,N,S}
    embedding::E
    blocks::B
    final_norm::N
    spec::S
end

struct _Ch45ForeignDeviceArray{T,N,A<:AbstractArray{T,N}} <:
       AbstractArray{T,N}
    data::A
end

Base.size(values::_Ch45ForeignDeviceArray) = size(values.data)
Base.axes(values::_Ch45ForeignDeviceArray) = axes(values.data)
Base.IndexStyle(::Type{<:_Ch45ForeignDeviceArray}) = IndexCartesian()
Base.getindex(values::_Ch45ForeignDeviceArray, indices...) =
    getindex(values.data, indices...)
MLDataDevices.get_device(::_Ch45ForeignDeviceArray) = _Ch45ForeignDevice()

struct _Ch45CountedIntArray{N,A<:AbstractArray{Int,N}} <:
       AbstractArray{Int,N}
    data::A
    collections::Base.RefValue{Int}
end

Base.size(values::_Ch45CountedIntArray) = size(values.data)
Base.axes(values::_Ch45CountedIntArray) = axes(values.data)
Base.IndexStyle(::Type{<:_Ch45CountedIntArray}) = IndexCartesian()
Base.getindex(values::_Ch45CountedIntArray, indices...) =
    getindex(values.data, indices...)
Base.copy(values::_Ch45CountedIntArray) =
    _Ch45CountedIntArray(copy(values.data), values.collections)
function Base.collect(values::_Ch45CountedIntArray)
    values.collections[] += 1
    return collect(values.data)
end

# Mathematical matrix emitted by the exporter's row-major
# `tiny_values((rows, columns), offset)` construction.
function _ch45_tiny_hf_matrix(rows::Int, columns::Int, offset::Int; scale=0.02f0)
    values = _ch45_tiny_values(rows * columns, offset; scale)
    return permutedims(reshape(values, columns, rows))
end

function _ch45_tiny_text_parameters()
    spec = _CH45_TINY_TEXT_SPEC
    embedding = permutedims(_ch45_tiny_hf_matrix(32, 16, 10))
    blocks = ntuple(spec.num_hidden_layers) do julia_layer
        offset = 10_000 * julia_layer
        return (;
            norm1=1.0f0 .+ _ch45_tiny_values(16, offset; scale=0.01f0),
            q_weight=_ch45_tiny_hf_matrix(16, 16, offset + 100),
            k_weight=_ch45_tiny_hf_matrix(8, 16, offset + 200),
            v_weight=_ch45_tiny_hf_matrix(8, 16, offset + 300),
            o_weight=_ch45_tiny_hf_matrix(16, 16, offset + 400),
            q_norm=1.0f0 .+ _ch45_tiny_values(8, offset + 500; scale=0.01f0),
            k_norm=1.0f0 .+ _ch45_tiny_values(8, offset + 600; scale=0.01f0),
            norm2=1.0f0 .+ _ch45_tiny_values(16, offset + 700; scale=0.01f0),
            gate_weight=_ch45_tiny_hf_matrix(32, 16, offset + 800),
            up_weight=_ch45_tiny_hf_matrix(32, 16, offset + 900),
            down_weight=_ch45_tiny_hf_matrix(16, 32, offset + 1_000),
        )
    end
    final_norm = 1.0f0 .+
        _ch45_tiny_values(16, 90_000; scale=0.01f0)
    return (; embedding, blocks, final_norm, spec)
end

function _ch45_tiny_prefill_inputs()
    position_ids = reshape(Int[
        0 1 2 2 2 2 4 5
        0 1 2 2 3 3 4 5
        0 1 2 3 2 3 4 5
    ], 3, 8, 1)
    visual_mask = falses(8, 1)
    visual_mask[3:6, 1] .= true
    attention_mask = trues(8, 1)
    rope_layout = Qwen3VLRopeLayout(
        position_ids,
        reshape(Int[-2], 1, 1),
        visual_mask,
        attention_mask,
    )
    visual_embeddings = permutedims(
        _ch45_tiny_hf_matrix(4, 16, 100_000; scale=0.1f0),
    )
    deepstack = ntuple(3) do index
        permutedims(_ch45_tiny_hf_matrix(
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

function _ch45_reference()
    path = joinpath(@__DIR__, "fixtures", "tiny_text_dynamic_decode.json")
    return JSON3.read(read(path, String))
end

function _ch45_reference_bytes(reference, name::AbstractString)
    entry = reference.tensors[Symbol(name)]
    bytes = base64decode(String(entry.f32_le_base64))
    @test bytes2hex(sha256(bytes)) == String(entry.sha256)
    return bytes, Int.(collect(entry.shape))
end

function _ch45_hf_hidden(reference, name::AbstractString)
    bytes, shape = _ch45_reference_bytes(reference, name)
    @test length(shape) == 3
    batch, tokens, width = shape
    values = collect(reinterpret(Float32, bytes))
    return reshape(values, width, tokens, batch)
end

function _ch45_hf_cache(reference, phase::AbstractString, layer::Int, kind::String)
    name = "cache.$phase.layer.$layer.$kind"
    bytes, shape = _ch45_reference_bytes(reference, name)
    @test length(shape) == 4
    batch, kv_heads, tokens, head_dim = shape
    values = collect(reinterpret(Float32, bytes))
    # C-order HF (B, KV, L, D) bytes first reshape to Julia (D, L, KV, B),
    # then place KV before L to obtain LifeAI's (D, KV, L, B) contract.
    reversed_hf = reshape(values, head_dim, tokens, kv_heads, batch)
    return permutedims(reversed_hf, (1, 3, 2, 4))
end

function _ch45_top_two(logits)
    scores = vec(Array(view(logits, :, size(logits, 2), 1)))
    ids = partialsortperm(scores, 1:2; rev=true)
    return (; ids, margin=scores[ids[1]] - scores[ids[2]])
end

function _ch45_assert_cache_matches(reference, cache, phase::String, tokens::Int)
    @test cache.position == tokens
    @test length(cache.layers) == 4
    for layer in 0:3
        actual = cache.layers[layer + 1]
        expected_key = _ch45_hf_cache(reference, phase, layer, "key")
        expected_value = _ch45_hf_cache(reference, phase, layer, "value")
        @test size(actual.keys) == (8, 1, tokens, 1)
        @test size(actual.values) == (8, 1, tokens, 1)
        @test actual.keys ≈ expected_key atol=1.0f-6 rtol=1.0f-6
        @test actual.values ≈ expected_value atol=1.0f-6 rtol=1.0f-6
    end
end

@testset "Chapter 45 — dynamic cache constructor is strict" begin
    layers = ()
    cache = LifeAI.Qwen3VLKVCache(
        layers,
        Int32(2),
        Int16(-1),
        UInt8(1),
    )
    @test cache.layers === layers
    @test cache.position === 2
    @test cache.rope_delta === -1
    @test cache.batch_size === 1
    @test length(cache) == 2
    @test !isempty(cache)

    too_large = big(typemax(Int)) + 1
    for (position, rope_delta, batch_size, message) in (
        (true, 0, 1, "Qwen3-VL KV cache position must be an integer"),
        (0, true, 1, "Qwen3-VL KV cache rope_delta must be an integer"),
        (0, 0, true, "Qwen3-VL KV cache batch_size must be an integer"),
        (
            too_large,
            0,
            1,
            "Qwen3-VL KV cache position is outside the host integer range",
        ),
        (
            0,
            too_large,
            1,
            "Qwen3-VL KV cache rope_delta is outside the host integer range",
        ),
        (
            0,
            0,
            too_large,
            "Qwen3-VL KV cache batch_size is outside the host integer range",
        ),
        (-1, 0, 1, "Qwen3-VL KV cache position must be non-negative"),
        (
            0,
            0,
            2,
            "Qwen3-VL dynamic generation currently supports batch size one",
        ),
        (
            0,
            1,
            1,
            "an empty Qwen3-VL KV cache must have rope_delta == 0",
        ),
    )
        failure = _ch45_captured_error() do
            LifeAI.Qwen3VLKVCache(layers, position, rope_delta, batch_size)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) == "ArgumentError: $message"
    end
    @test_throws MethodError LifeAI.Qwen3VLKVCache{Tuple{}}(
        (),
        false,
        true,
        true,
    )
end

@testset "Chapter 45 — dynamic cache layer collection is strict" begin
    make_layer(shape=(8, 1, 2, 1), dtype=Float32) = begin
        keys = zeros(dtype, shape)
        values = similar(keys)
        fill!(values, one(dtype))
        LayerKVCache(keys, values)
    end

    empty_layers = (LayerKVCache(), LayerKVCache())
    empty_cache = LifeAI.Qwen3VLKVCache(empty_layers, 0, 0, 1)
    @test empty_cache.layers === empty_layers
    @test isempty(empty_cache)

    first_layer = make_layer()
    second_layer = make_layer()
    layers = (first_layer, second_layer)
    cache = LifeAI.Qwen3VLKVCache(layers, 2, -1, 1)
    @test cache.layers === layers
    @test cache.layers[1] === first_layer
    @test cache.layers[2] === second_layer
    @test cache.layers[1].keys !== cache.layers[2].keys

    disjoint_parent = zeros(Float32, 8, 1, 4, 1)
    disjoint_keys = view(disjoint_parent, :, :, 1:2, :)
    disjoint_values = view(disjoint_parent, :, :, 3:4, :)
    @test !Base.mightalias(disjoint_keys, disjoint_values)
    disjoint_layer = LayerKVCache(disjoint_keys, disjoint_values)
    disjoint_layers = (disjoint_layer,)
    disjoint_cache = LifeAI.Qwen3VLKVCache(disjoint_layers, 2, 0, 1)
    @test disjoint_cache.layers === disjoint_layers
    @test disjoint_cache.layers[1].keys === disjoint_keys
    @test disjoint_cache.layers[1].values === disjoint_values

    intra_layer_parent = zeros(Float32, 8, 1, 3, 1)
    intra_layer_keys = view(intra_layer_parent, :, :, 1:2, :)
    intra_layer_values = view(intra_layer_parent, :, :, 2:3, :)
    @test intra_layer_keys !== intra_layer_values
    @test Base.mightalias(intra_layer_keys, intra_layer_values)
    intra_layer = LayerKVCache(intra_layer_keys, intra_layer_values)
    intra_layer_failure = _ch45_captured_error() do
        LifeAI.Qwen3VLKVCache((intra_layer,), 2, 0, 1)
    end
    @test intra_layer_failure isa ArgumentError
    @test sprint(showerror, intra_layer_failure) ==
        "ArgumentError: Qwen3-VL dynamic cache layers must use " *
        "non-overlapping storage"

    cross_layer_parent = zeros(Float32, 8, 1, 8, 1)
    cross_layer_first = LayerKVCache(
        view(cross_layer_parent, :, :, 1:2, :),
        view(cross_layer_parent, :, :, 4:5, :),
    )
    cross_layer_second = LayerKVCache(
        view(cross_layer_parent, :, :, 2:3, :),
        view(cross_layer_parent, :, :, 6:7, :),
    )
    @test Base.mightalias(
        cross_layer_first.keys,
        cross_layer_second.keys,
    )
    cross_layer_failure = _ch45_captured_error() do
        LifeAI.Qwen3VLKVCache(
            (cross_layer_first, cross_layer_second),
            2,
            0,
            1,
        )
    end
    @test cross_layer_failure isa ArgumentError
    @test sprint(showerror, cross_layer_failure) ==
        "ArgumentError: Qwen3-VL dynamic cache layers must use " *
        "non-overlapping storage"

    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        LayerKVCache[LayerKVCache()],
        0,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache((1,), 0, 0, 1)
    @test_throws MethodError LifeAI.Qwen3VLKVCache(
        (LayerKVCache(first_layer.keys, nothing),),
        2,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI.Qwen3VLKVCache(
        (first_layer,),
        0,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI.Qwen3VLKVCache(
        (LayerKVCache(),),
        2,
        0,
        1,
    )
    @test_throws MethodError LifeAI.Qwen3VLKVCache(
        (LayerKVCache(1, 2),),
        2,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI.Qwen3VLKVCache(
        (LayerKVCache(
            zeros(Float32, 8, 1, 2),
            zeros(Float32, 8, 1, 2),
        ),),
        2,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI.Qwen3VLKVCache(
        (LayerKVCache(
            zeros(Float32, 8, 1, 2, 1),
            zeros(Float32, 7, 1, 2, 1),
        ),),
        2,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI.Qwen3VLKVCache(
        (make_layer((8, 1, 3, 1)),),
        2,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI.Qwen3VLKVCache(
        (make_layer((8, 1, 2, 2)),),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (make_layer((0, 1, 2, 1)),),
        2,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI.Qwen3VLKVCache(
        (first_layer, make_layer((7, 1, 2, 1))),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (make_layer((8, 1, 2, 1), Float64),),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (LayerKVCache(
            zeros(Float32, 8, 1, 2, 1),
            zeros(Core.BFloat16, 8, 1, 2, 1),
        ),),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (first_layer, make_layer((8, 1, 2, 1), Core.BFloat16)),
        2,
        0,
        1,
    )

    foreign_layer = LayerKVCache(
        _Ch45ForeignDeviceArray(zeros(Float32, 8, 1, 2, 1)),
        _Ch45ForeignDeviceArray(ones(Float32, 8, 1, 2, 1)),
    )
    foreign_cache = LifeAI.Qwen3VLKVCache((foreign_layer,), 2, 0, 1)
    @test foreign_cache.layers[1] === foreign_layer
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (LayerKVCache(
            first_layer.keys,
            _Ch45ForeignDeviceArray(ones(Float32, 8, 1, 2, 1)),
        ),),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (first_layer, foreign_layer),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (LayerKVCache(first_layer.keys, first_layer.keys),),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (first_layer, first_layer),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI.Qwen3VLKVCache(
        (
            first_layer,
            LayerKVCache(first_layer.values, first_layer.keys),
        ),
        2,
        0,
        1,
    )

    parameters = _ch45_tiny_text_parameters()
    wrong_shape_cache = LifeAI.Qwen3VLKVCache(
        ntuple(_ -> make_layer((7, 1, 2, 1)), 4),
        2,
        0,
        1,
    )
    @test_throws DimensionMismatch LifeAI._validate_qwen3_vl_kv_cache(
        parameters,
        wrong_shape_cache,
    )
    wrong_dtype_cache = LifeAI.Qwen3VLKVCache(
        ntuple(_ -> make_layer((8, 1, 2, 1), Core.BFloat16), 4),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI._validate_qwen3_vl_kv_cache(
        parameters,
        wrong_dtype_cache,
    )
    wrong_device_cache = LifeAI.Qwen3VLKVCache(
        ntuple(_ -> LayerKVCache(
            _Ch45ForeignDeviceArray(zeros(Float32, 8, 1, 2, 1)),
            _Ch45ForeignDeviceArray(ones(Float32, 8, 1, 2, 1)),
        ), 4),
        2,
        0,
        1,
    )
    @test_throws ArgumentError LifeAI._validate_qwen3_vl_kv_cache(
        parameters,
        wrong_device_cache,
    )
end

@testset "Chapter 45 — feature residency fails before dynamic cache compute" begin
    parameters = _ch45_tiny_text_parameters()
    inputs = _ch45_tiny_prefill_inputs()
    dtype_features = merge(inputs.vision_features, (
        visual_embeddings=Float64.(
            inputs.vision_features.visual_embeddings,
        ),
    ))
    foreign_deepstack = merge(inputs.vision_features, (
        deepstack=Base.setindex(
            inputs.vision_features.deepstack,
            _Ch45ForeignDeviceArray(inputs.vision_features.deepstack[1]),
            1,
        ),
    ))
    wrong_visual_shape = merge(inputs.vision_features, (;
        visual_embeddings=inputs.vision_features.visual_embeddings[:, 1:3],
    ))
    nonmatrix_deepstack = merge(inputs.vision_features, (;
        deepstack=Base.setindex(
            inputs.vision_features.deepstack,
            vec(inputs.vision_features.deepstack[1]),
            1,
        ),
    ))

    for (features, error_type, message) in (
        (
            dtype_features,
            ArgumentError,
            "ArgumentError: Qwen3-VL visual_embeddings dtype must match " *
            "text parameter embedding",
        ),
        (
            foreign_deepstack,
            ArgumentError,
            "ArgumentError: Qwen3-VL deepstack[1] device must match " *
            "text parameter embedding",
        ),
        (
            wrong_visual_shape,
            DimensionMismatch,
            "DimensionMismatch: Qwen3-VL main visual feature count does " *
            "not match image placeholders",
        ),
        (
            nonmatrix_deepstack,
            ArgumentError,
            "ArgumentError: Qwen3-VL deepstack[1] must be a matrix",
        ),
    )
        cache = init_qwen3_vl_kv_cache(parameters)
        error = _ch45_captured_error(() ->
            hf_qwen3_vl_text_prefill_cached(
                parameters,
                inputs.input_ids,
                inputs.rope_layout;
                vision_features=features,
                cache,
            ),
        )
        @test isa(error, error_type)
        @test sprint(showerror, error) == message
        @test isempty(cache)
        @test cache.position == 0
        @test cache.rope_delta == 0
        @test all(
            layer -> layer.keys === nothing && layer.values === nothing,
            cache.layers,
        )
    end
end

@testset "Chapter 45 — image-token masks fail before dynamic cache writes" begin
    parameters = merge(
        _ch45_tiny_text_parameters(),
        (; checkpoint=(; image_token_id=2, video_token_id=8)),
    )
    inputs = _ch45_tiny_prefill_inputs()
    input_ids = Int[1, 2, 3, 3, 3, 3, 4, 5]
    missing_image = copy(inputs.rope_layout.visual_mask)
    missing_image[3, 1] = false
    non_image_position = copy(inputs.rope_layout.visual_mask)
    non_image_position[2, 1] = true

    for (visual_mask, message) in (
        (
            missing_image,
            "ArgumentError: Qwen3-VL checkpoint image tokens must be " *
            "marked by visual_mask",
        ),
        (
            non_image_position,
            "ArgumentError: Qwen3-VL visual_mask must only mark attended " *
            "image input tokens",
        ),
    )
        layout = Qwen3VLRopeLayout(
            inputs.rope_layout.position_ids,
            inputs.rope_layout.rope_deltas,
            visual_mask,
            inputs.rope_layout.attention_mask,
        )
        cache = init_qwen3_vl_kv_cache(parameters)
        error = _ch45_captured_error() do
            hf_qwen3_vl_text_prefill_cached(
                parameters,
                input_ids,
                layout;
                vision_features=inputs.vision_features,
                cache,
            )
        end
        @test error isa ArgumentError
        @test sprint(showerror, error) == message
        @test isempty(cache)
        @test cache.position == 0
        @test cache.rope_delta == 0
        @test all(
            layer -> layer.keys === nothing && layer.values === nothing,
            cache.layers,
        )
    end
end

@testset "Chapter 45 — video placeholders fail before dynamic cache writes" begin
    parameters = merge(
        _ch45_tiny_text_parameters(),
        (; checkpoint=(; image_token_id=2, video_token_id=8)),
    )
    inputs = _ch45_tiny_prefill_inputs()
    bound_tokens = Int[1, 2, 3, 3, 3, 3, 4, 5]
    expected = "ArgumentError: Qwen3-VL video placeholders are not " *
        "supported by image-only text prefill"

    for video_position in (2, 3)
        video_tokens = copy(bound_tokens)
        video_tokens[video_position] = 9
        cache = init_qwen3_vl_kv_cache(parameters)
        error = _ch45_captured_error() do
            hf_qwen3_vl_text_prefill_cached(
                parameters,
                video_tokens,
                inputs.rope_layout;
                vision_features=_CH45VisionComputePoison(),
                cache,
            )
        end
        @test error isa ArgumentError
        @test sprint(showerror, error) == expected
        @test isempty(cache)
        @test cache.position == 0
        @test cache.rope_delta == 0
        @test all(
            layer -> layer.keys === nothing && layer.values === nothing,
            cache.layers,
        )
    end
end

@testset "Chapter 45 — frozen HF DynamicCache fixture contract" begin
    reference = _ch45_reference()
    metadata = reference.metadata
    @test String(metadata.oracle) == "qwen3_vl_tiny_dynamic_cache_greedy_decode"
    @test String(metadata.python) == "3.10.12"
    @test String(metadata.numpy) == "1.26.4"
    @test String(metadata.transformers) == "4.57.0"
    @test String(metadata.torch) == "2.7.1+cpu"
    @test String(metadata.attention_implementation) == "eager"
    @test String(metadata.attention_mask_contract) ==
        "explicit_all_ones_every_call"
    @test String(metadata.cache_capture_contract) ==
        "detach_clone_cpu_before_next_dynamic_cache_update"
    @test String(metadata.hf_cache_layout) == "batch,kv_heads,tokens,head_dim"
    @test String(metadata.julia_cache_layout) == "head_dim,kv_heads,tokens,batch"
    @test Int.(collect(metadata.hf_to_julia_permutation_1_based)) == [4, 2, 3, 1]

    @test Int.(collect(metadata.cache_lengths)) == [8, 9, 10]
    @test Int(metadata.prefill_length) == 8
    @test Int(metadata.decode_forward_calls) == 2
    @test Int(metadata.rope_delta) == -2
    @test Int.(collect(metadata.greedy_token_ids_0_based)) == [7, 7, 7]
    @test Int.(collect(metadata.greedy_token_ids_1_based)) == [8, 8, 8]
    @test Int.(collect(metadata.phases[2].mrope_position_ids_thw_0_based)) ==
        [6, 6, 6]
    @test Int.(collect(metadata.phases[3].mrope_position_ids_thw_0_based)) ==
        [7, 7, 7]
    @test Int(metadata.phases[2].physical_cache_position_0_based) == 8
    @test Int(metadata.phases[3].physical_cache_position_0_based) == 9
    @test Int.(collect(metadata.phases[2].attention_mask_shape)) == [1, 9]
    @test Int.(collect(metadata.phases[3].attention_mask_shape)) == [1, 10]

    expected_margins = [
        0.0004043877124786377,
        0.0006773620843887329,
        0.0008579939603805542,
    ]
    @test [Float64(phase.top_two.margin_f32) for phase in metadata.phases] ==
        expected_margins
    @test all(>(0), expected_margins)

    @test size(_ch45_hf_hidden(reference, "prefill.final_hidden")) == (16, 8, 1)
    @test size(_ch45_hf_hidden(reference, "prefill.logits")) == (32, 8, 1)
    for step in 0:1
        @test size(_ch45_hf_hidden(reference, "decode.$step.final_hidden")) ==
            (16, 1, 1)
        @test size(_ch45_hf_hidden(reference, "decode.$step.logits")) ==
            (32, 1, 1)
    end

    # The fixture itself proves snapshot discipline independently of LifeAI:
    # every later HF DynamicCache tensor has a byte-identical old prefix.
    for layer in 0:3, kind in ("key", "value")
        prefill = _ch45_hf_cache(reference, "prefill", layer, kind)
        decode0 = _ch45_hf_cache(reference, "decode.0", layer, kind)
        decode1 = _ch45_hf_cache(reference, "decode.1", layer, kind)
        @test decode0[:, :, 1:8, :] == prefill
        @test decode1[:, :, 1:9, :] == decode0
    end

    # Per-phase hidden tensors are independently tied to the frozen logits.
    parameters = _ch45_tiny_text_parameters()
    for phase in ("prefill", "decode.0", "decode.1")
        hidden = _ch45_hf_hidden(reference, "$phase.final_hidden")
        logits = _ch45_hf_hidden(reference, "$phase.logits")
        projected = reshape(
            transpose(parameters.embedding) * reshape(hidden, 16, :),
            size(logits),
        )
        @test projected ≈ logits atol=2.0f-7 rtol=2.0f-6
    end
end

@testset "Chapter 45 — tiny cached prefill and two decode steps" begin
    reference = _ch45_reference()
    parameters = _ch45_tiny_text_parameters()
    inputs = _ch45_tiny_prefill_inputs()
    cache0 = init_qwen3_vl_kv_cache(parameters; batch_size=1)
    @test isempty(cache0)
    @test cache0.position == 0
    @test cache0.rope_delta == 0
    @test all(layer -> layer.keys === nothing && layer.values === nothing, cache0.layers)
    @test_throws ArgumentError init_qwen3_vl_kv_cache(parameters; batch_size=2)
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step(parameters, 8, cache0)

    wide_layout = Qwen3VLRopeLayout(
        UInt128.(inputs.rope_layout.position_ids),
        Int128.(inputs.rope_layout.rope_deltas),
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    )
    wide_prefill, wide_cache8 = hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        wide_layout;
        vision_features=inputs.vision_features,
        cache=init_qwen3_vl_kv_cache(parameters),
        logits_to_keep=0,
    )

    overflow_integer = big(typemax(Int)) + 1
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
        guard = init_qwen3_vl_kv_cache(parameters)
        layout_error = _ch45_captured_error(() -> begin
            layout = case.layout()
            hf_qwen3_vl_text_prefill_cached(
                parameters,
                inputs.input_ids,
                layout;
                vision_features=inputs.vision_features,
                cache=guard,
            )
        end)
        @test layout_error isa ArgumentError
        @test occursin(case.message, sprint(showerror, layout_error))
        @test isempty(guard)
        @test guard.position == 0
        @test guard.rope_delta == 0
        @test all(
            layer -> layer.keys === nothing && layer.values === nothing,
            guard.layers,
        )
    end

    malformed_layout = Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        reshape(Int[-1], 1, 1),
        inputs.rope_layout.visual_mask,
        inputs.rope_layout.attention_mask,
    )
    @test_throws ArgumentError hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        malformed_layout;
        vision_features=inputs.vision_features,
        cache=cache0,
    )

    prefill, cache8 = hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=cache0,
        logits_to_keep=0,
    )
    @test cache8.position == 8
    @test cache8.rope_delta == -2
    @test wide_cache8.position == cache8.position
    @test wide_cache8.rope_delta == cache8.rope_delta
    @test size(prefill.final_hidden) == (16, 8, 1)
    @test size(prefill.logits) == (32, 8, 1)
    @test wide_prefill.final_hidden == prefill.final_hidden
    @test wide_prefill.logits == prefill.logits
    @test prefill.final_hidden ≈
        _ch45_hf_hidden(reference, "prefill.final_hidden") atol=1.0f-6 rtol=1.0f-6
    @test prefill.logits ≈
        _ch45_hf_hidden(reference, "prefill.logits") atol=1.0f-6 rtol=1.0f-6
    for layer in eachindex(cache8.layers)
        @test wide_cache8.layers[layer].keys == cache8.layers[layer].keys
        @test wide_cache8.layers[layer].values == cache8.layers[layer].values
    end

    full_last_prefill, _ = hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=init_qwen3_vl_kv_cache(parameters; batch_size=1),
        logits_to_keep=1,
    )
    light_prefill, light_cache8 = hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=init_qwen3_vl_kv_cache(parameters; batch_size=1),
        logits_to_keep=1,
        capture_input_embeddings=false,
        capture_final_hidden=false,
    )
    @test light_prefill.input_embeddings === nothing
    @test light_prefill.final_hidden === nothing
    @test light_prefill.logits == full_last_prefill.logits
    @test light_cache8.position == cache8.position
    @test light_cache8.rope_delta == cache8.rope_delta
    for layer in eachindex(light_cache8.layers)
        @test light_cache8.layers[layer].keys == cache8.layers[layer].keys
        @test light_cache8.layers[layer].values == cache8.layers[layer].values
    end

    light_all_logits, _ = hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=init_qwen3_vl_kv_cache(parameters; batch_size=1),
        logits_to_keep=0,
        capture_input_embeddings=false,
        capture_final_hidden=false,
    )
    @test light_all_logits.input_embeddings === nothing
    @test light_all_logits.final_hidden === nothing
    @test light_all_logits.logits == prefill.logits

    overflow_token = big(typemax(Int)) + 1
    for invalid in (
        true,
        Bool[true],
        Float64[8],
        Char['\x08'],
        BigInt[overflow_token],
    )
        @test_throws ArgumentError hf_qwen3_vl_text_decode_step(
            parameters,
            invalid,
            cache8,
        )
    end
    _ch45_assert_cache_matches(reference, cache8, "prefill", 8)
    prefill_keys = map(layer -> copy(layer.keys), cache8.layers)
    prefill_values = map(layer -> copy(layer.values), cache8.layers)

    overflow_cache = LifeAI.Qwen3VLKVCache(
        cache8.layers,
        cache8.position,
        typemax(Int),
        cache8.batch_size,
    )
    coordinate_error = _ch45_captured_error(() ->
        hf_qwen3_vl_text_decode_step(parameters, 8, overflow_cache),
    )
    @test coordinate_error isa ArgumentError
    @test occursin(
        "decode mRoPE coordinate exceeds the host integer range",
        sprint(showerror, coordinate_error),
    )
    @test overflow_cache.position == 8
    @test overflow_cache.rope_delta == typemax(Int)
    for layer in eachindex(overflow_cache.layers)
        @test overflow_cache.layers[layer].keys === cache8.layers[layer].keys
        @test overflow_cache.layers[layer].values === cache8.layers[layer].values
        @test overflow_cache.layers[layer].keys == prefill_keys[layer]
        @test overflow_cache.layers[layer].values == prefill_values[layer]
    end

    choice = _ch45_top_two(prefill.logits)
    @test choice.ids == [8, 24]
    @test choice.margin ≈ 0.0004043877f0 atol=2.0f-7 rtol=2.0f-5

    logits9, cache9 = hf_qwen3_vl_text_decode_step(parameters, choice.ids[1], cache8)
    @test cache9.position == 9
    @test cache9.rope_delta == -2
    @test size(logits9) == (32, 1, 1)
    @test logits9 ≈
        _ch45_hf_hidden(reference, "decode.0.logits") atol=1.0f-6 rtol=1.0f-6
    _ch45_assert_cache_matches(reference, cache9, "decode.0", 9)
    for layer in 1:4
        @test cache9.layers[layer].keys !== cache8.layers[layer].keys
        @test cache9.layers[layer].values !== cache8.layers[layer].values
        @test cache8.layers[layer].keys == prefill_keys[layer]
        @test cache8.layers[layer].values == prefill_values[layer]
        @test cache9.layers[layer].keys[:, :, 1:8, :] == prefill_keys[layer]
        @test cache9.layers[layer].values[:, :, 1:8, :] == prefill_values[layer]
    end
    decode0_keys = map(layer -> copy(layer.keys), cache9.layers)
    decode0_values = map(layer -> copy(layer.values), cache9.layers)

    choice9 = _ch45_top_two(logits9)
    @test choice9.ids == [8, 24]
    @test choice9.margin ≈ 0.0006773621f0 atol=2.0f-7 rtol=2.0f-5
    logits10, cache10 = hf_qwen3_vl_text_decode_step(
        parameters,
        choice9.ids[1],
        cache9,
    )
    @test cache10.position == 10
    @test cache10.rope_delta == -2
    @test size(logits10) == (32, 1, 1)
    @test logits10 ≈
        _ch45_hf_hidden(reference, "decode.1.logits") atol=1.0f-6 rtol=1.0f-6
    _ch45_assert_cache_matches(reference, cache10, "decode.1", 10)
    for layer in 1:4
        @test cache9.layers[layer].keys == decode0_keys[layer]
        @test cache9.layers[layer].values == decode0_values[layer]
        @test cache10.layers[layer].keys[:, :, 1:9, :] == decode0_keys[layer]
        @test cache10.layers[layer].values[:, :, 1:9, :] == decode0_values[layer]
    end
    choice10 = _ch45_top_two(logits10)
    @test choice10.ids == [8, 24]
    @test choice10.margin ≈ 0.0008579940f0 atol=2.0f-7 rtol=2.0f-5

    # A cache is request state, and cached prefill is legal only once.
    @test_throws ArgumentError hf_qwen3_vl_text_prefill_cached(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=cache8,
    )
end

@testset "Chapter 45 — greedy generation cache timeline" begin
    reference = _ch45_reference()
    parameters = _ch45_tiny_text_parameters()
    inputs = _ch45_tiny_prefill_inputs()

    default_checkpoint = (
        text=parameters.spec,
        eos_token_id=7,
        bos_token_id=6,
    )
    default_parameters = merge(parameters, (; checkpoint=default_checkpoint))
    default_stops = generate_hf_qwen3_vl_tokens(
        default_parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=0,
    )
    @test isempty(default_stops.generated_ids)

    for name in (:eos_token_id, :bos_token_id)
        invalid_checkpoint = merge(
            default_checkpoint,
            (; name => parameters.spec.vocab_size),
        )
        failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                merge(parameters, (; checkpoint=invalid_checkpoint)),
                42,
                inputs.rope_layout;
                vision_features=inputs.vision_features,
                max_new_tokens=0,
            )
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL checkpoint $name must be in 0:31"
    end

    generated = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        capture_logits=true,
    )
    @test generated.prompt_ids == collect(1:8)
    @test generated.generated_ids == [8, 8, 8]
    @test generated.token_ids == vcat(collect(1:8), [8, 8, 8])
    @test generated.stop_reason === :length
    @test generated.strategy === :greedy
    @test generated.cache.position == 10
    @test generated.cache.rope_delta == -2
    @test length(generated.trace) == 3
    @test [step.step for step in generated.trace] == [1, 2, 3]
    @test [step.token_id for step in generated.trace] == [8, 8, 8]
    @test [step.hf_token_id for step in generated.trace] == [7, 7, 7]
    @test [step.second_token_id for step in generated.trace] == [24, 24, 24]
    @test all(step -> length(step.logits) == 32, generated.trace)
    @test [Float64(step.margin) for step in generated.trace] ≈ [
        Float64(phase.top_two.margin_f32) for phase in reference.metadata.phases
    ] atol=2.0e-7 rtol=2.0e-5

    # The first token is selected directly from prefill. With three output
    # tokens only two decode calls occur, so the final cache length is 10.
    @test generated.prefill !== nothing
    @test generated.prefill.input_embeddings === nothing
    @test generated.prefill.final_hidden === nothing
    @test size(generated.prefill.logits) == (32, 1, 1)
    @test generated.trace[1].logits ≈
        vec(_ch45_hf_hidden(reference, "prefill.logits")[:, end, 1])
    @test generated.trace[2].logits ≈
        vec(_ch45_hf_hidden(reference, "decode.0.logits"))
    @test generated.trace[3].logits ≈
        vec(_ch45_hf_hidden(reference, "decode.1.logits"))

    captured = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        capture_logits=true,
        capture_prefill_states=true,
    )
    @test size(captured.prefill.input_embeddings) == (16, 8, 1)
    @test captured.prefill.final_hidden ≈
        _ch45_hf_hidden(reference, "prefill.final_hidden") atol=1.0f-6 rtol=1.0f-6
    @test captured.prefill.logits == generated.prefill.logits
    @test captured.generated_ids == generated.generated_ids
    @test captured.trace == generated.trace
    @test captured.cache.position == generated.cache.position
    for layer in eachindex(captured.cache.layers)
        @test captured.cache.layers[layer].keys == generated.cache.layers[layer].keys
        @test captured.cache.layers[layer].values == generated.cache.layers[layer].values
    end

    stopped = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=3,
        stop_token_ids=[8],
    )
    @test stopped.generated_ids == [8]
    @test stopped.stop_reason === :eos
    @test stopped.cache.position == 8
    @test length(stopped.trace) == 1
    @test stopped.prefill.input_embeddings === nothing
    @test stopped.prefill.final_hidden === nothing

    zero = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=Int128(0),
    )
    @test isempty(zero.generated_ids)
    @test zero.token_ids == collect(1:8)
    @test zero.prefill === nothing
    @test isempty(zero.cache)
    overflow_length = big(typemax(Int)) + 1
    for invalid_length in (true, overflow_length)
        failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                parameters,
                42,
                inputs.rope_layout;
                vision_features=inputs.vision_features,
                max_new_tokens=invalid_length,
            )
        end
        @test failure isa ArgumentError
        @test occursin("max_new_tokens", sprint(showerror, failure))
    end
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=-1,
    )
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=58,
    )
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        max_new_tokens=1,
        stop_token_ids=[33],
    )
    overflow_stop = big(typemax(Int)) + 1
    for invalid_stops in (
        Float64[8],
        Bool[true],
        Char['\x08'],
        BigInt[overflow_stop],
    )
        @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
            parameters,
            inputs.input_ids,
            inputs.rope_layout;
            vision_features=inputs.vision_features,
            max_new_tokens=0,
            stop_token_ids=invalid_stops,
        )
    end

    # A prompt may occupy the complete physical context when only the first
    # token from prefill logits is requested. The selected token is returned
    # without being appended; an actual decode attempt remains illegal.
    full_context_ids = fill(1, parameters.spec.max_position_embeddings)
    full_positions = repeat(
        reshape(collect(0:(length(full_context_ids) - 1)), 1, :, 1),
        3,
        1,
        1,
    )
    full_layout = Qwen3VLRopeLayout(
        full_positions,
        reshape(Int[0], 1, 1),
        falses(length(full_context_ids), 1),
        trues(length(full_context_ids), 1),
    )
    boundary = generate_hf_qwen3_vl_tokens(
        parameters,
        full_context_ids,
        full_layout;
        max_new_tokens=1,
        stop_token_ids=Int[],
    )
    @test length(boundary.generated_ids) == 1
    @test boundary.cache.position == parameters.spec.max_position_embeddings
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step(
        parameters,
        only(boundary.generated_ids),
        boundary.cache,
    )
end

@testset "Chapter 45 — generation prompt fails before cache allocation" begin
    inputs = _ch45_tiny_prefill_inputs()
    parameters = _ch45_tiny_text_parameters()
    full_context_ids = fill(1, parameters.spec.max_position_embeddings)
    full_positions = repeat(
        reshape(collect(0:(length(full_context_ids) - 1)), 1, :, 1),
        3,
        1,
        1,
    )
    overfull_layout = Qwen3VLRopeLayout(
        full_positions,
        reshape(Int[1], 1, 1),
        falses(length(full_context_ids), 1),
        trues(length(full_context_ids), 1),
    )
    @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
        parameters,
        full_context_ids,
        overfull_layout;
        max_new_tokens=1,
        stop_token_ids=Int[],
    )
    poison = _CH45GenerationCachePoison(
        _CH45_TINY_TEXT_SPEC,
        parameters.embedding,
    )
    cache_options = (
        (; cache=:dynamic),
        (; cache=:static, static_capacity=7),
    )

    # Prompt/layout mismatch is a public input error even when no prefill is
    # requested. It must not reach either dynamic or static cache creation.
    for options in cache_options, requested in (0, 1)
        failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                poison,
                inputs.input_ids[1:7],
                inputs.rope_layout;
                vision_features=inputs.vision_features,
                max_new_tokens=requested,
                stop_token_ids=Int[],
                options...,
            )
        end
        @test failure isa DimensionMismatch
        @test sprint(showerror, failure) ==
            "DimensionMismatch: Qwen3-VL visual mask does not match input_ids"
    end

    invalid_features = merge(
        inputs.vision_features,
        (; visual_embeddings=inputs.vision_features.visual_embeddings[:, 1:3]),
    )
    dtype_features = merge(
        inputs.vision_features,
        (; visual_embeddings=Float64.(
            inputs.vision_features.visual_embeddings,
        )),
    )
    device_features = merge(
        inputs.vision_features,
        (; visual_embeddings=_Ch45ForeignDeviceArray(
            inputs.vision_features.visual_embeddings,
        )),
    )
    for options in (
        (; cache=:dynamic),
        (; cache=:static, static_capacity=8),
    )
        missing_failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                poison,
                inputs.input_ids,
                inputs.rope_layout;
                max_new_tokens=1,
                stop_token_ids=Int[],
                options...,
            )
        end
        @test missing_failure isa ArgumentError
        @test sprint(showerror, missing_failure) ==
            "ArgumentError: Qwen3-VL visual placeholders require vision features"

        shape_failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                poison,
                inputs.input_ids,
                inputs.rope_layout;
                vision_features=invalid_features,
                max_new_tokens=1,
                stop_token_ids=Int[],
                options...,
            )
        end
        @test shape_failure isa DimensionMismatch
        @test sprint(showerror, shape_failure) ==
            "DimensionMismatch: Qwen3-VL main visual feature count does not " *
            "match image placeholders"

        for (features, message) in (
            (
                dtype_features,
                "ArgumentError: Qwen3-VL visual_embeddings dtype must match " *
                "text parameter embedding",
            ),
            (
                device_features,
                "ArgumentError: Qwen3-VL visual_embeddings device must match " *
                "text parameter embedding",
            ),
        )
            residency_failure = _ch45_captured_error() do
                generate_hf_qwen3_vl_tokens(
                    poison,
                    inputs.input_ids,
                    inputs.rope_layout;
                    vision_features=features,
                    max_new_tokens=1,
                    stop_token_ids=Int[],
                    options...,
                )
            end
            @test residency_failure isa ArgumentError
            @test sprint(showerror, residency_failure) == message
        end
    end

    # Zero-token generation retains its intentional no-vision-compute path.
    zero = generate_hf_qwen3_vl_tokens(
        _ch45_tiny_text_parameters(),
        inputs.input_ids,
        inputs.rope_layout;
        max_new_tokens=0,
        stop_token_ids=Int[],
    )
    @test isempty(zero.generated_ids)
    @test isempty(zero.cache)

    padded_attention = copy(inputs.rope_layout.attention_mask)
    padded_attention[end, 1] = false
    padded_layout = Qwen3VLRopeLayout(
        inputs.rope_layout.position_ids,
        reshape(Int[-3], 1, 1),
        inputs.rope_layout.visual_mask,
        padded_attention,
    )
    padded_zero = generate_hf_qwen3_vl_tokens(
        parameters,
        inputs.input_ids,
        padded_layout;
        max_new_tokens=0,
        stop_token_ids=Int[],
    )
    @test isempty(padded_zero.generated_ids)
    @test padded_zero.prefill === nothing
    @test isempty(padded_zero.cache)
end

@testset "Chapter 45 — generation preflights the mRoPE decode horizon" begin
    parameters = _ch45_tiny_text_parameters()
    limit = parameters.spec.max_position_embeddings
    prompt = [1, 2]
    function high_layout(maximum_coordinate)
        positions = repeat(
            reshape(Int[maximum_coordinate - 1, maximum_coordinate], 1, 2, 1),
            3,
            1,
            1,
        )
        delta = maximum_coordinate + 1 - length(prompt)
        return Qwen3VLRopeLayout(
            positions,
            reshape(Int[delta], 1, 1),
            falses(2, 1),
            trues(2, 1),
        )
    end

    error_message = "ArgumentError: Qwen3-VL generated mRoPE coordinates " *
        "exceed max_position_embeddings"
    cache_options = (
        (; cache=:dynamic),
        (; cache=:static, static_capacity=4),
    )

    # The final legal prompt coordinate leaves room for the token selected
    # from prefill logits, but no room to append that token through decode.
    exhausted = high_layout(limit - 1)
    cache_poison = _CH45GenerationCachePoison(
        parameters.spec,
        parameters.embedding,
    )
    for options in cache_options
        vision_failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                parameters,
                prompt,
                exhausted;
                vision_features=_CH45VisionComputePoison(),
                max_new_tokens=2,
                stop_token_ids=Int[],
                options...,
            )
        end
        @test sprint(showerror, vision_failure) == error_message

        cache_failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                cache_poison,
                prompt,
                exhausted;
                max_new_tokens=2,
                stop_token_ids=Int[],
                options...,
            )
        end
        @test sprint(showerror, cache_failure) == error_message
    end

    # max_new_tokens=0 skips prefill and max_new_tokens=1 consumes only its
    # logits, so neither request needs a decode coordinate beyond the prompt.
    for requested in (0, 1), options in cache_options
        result = generate_hf_qwen3_vl_tokens(
            parameters,
            prompt,
            exhausted;
            max_new_tokens=requested,
            stop_token_ids=Int[],
            options...,
        )
        @test length(result.generated_ids) == requested
        @test result.cache.position == (requested == 0 ? 0 : length(prompt))
    end

    # One coordinate below the limit admits exactly one decode append.
    one_decode = high_layout(limit - 2)
    for options in cache_options
        legal = generate_hf_qwen3_vl_tokens(
            parameters,
            prompt,
            one_decode;
            max_new_tokens=2,
            stop_token_ids=Int[],
            options...,
        )
        @test length(legal.generated_ids) == 2
        @test legal.cache.position == length(prompt) + 1

        failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                cache_poison,
                prompt,
                one_decode;
                max_new_tokens=3,
                stop_token_ids=Int[],
                options...,
            )
        end
        @test sprint(showerror, failure) == error_message
    end

    # BigInt arithmetic in the guard must reject a horizon whose host-Int
    # additions would overflow, without allocating a correspondingly huge cache.
    names = fieldnames(Qwen3VLTextSpec)
    huge_spec = Qwen3VLTextSpec(ntuple(length(names)) do index
        names[index] === :max_position_embeddings && return typemax(Int)
        return getfield(parameters.spec, names[index])
    end...)
    huge_poison = _CH45GenerationCachePoison(huge_spec, parameters.embedding)
    huge_layout = high_layout(typemax(Int) - 2)
    for options in cache_options
        overflow_failure = _ch45_captured_error() do
            generate_hf_qwen3_vl_tokens(
                huge_poison,
                prompt,
                huge_layout;
                max_new_tokens=3,
                stop_token_ids=Int[],
                options...,
            )
        end
        @test sprint(showerror, overflow_failure) == error_message
    end
end

@testset "Chapter 45 — raw generation preflights mRoPE before vision" begin
    parameters = _ch45_tiny_text_parameters()
    limit = parameters.spec.max_position_embeddings
    prompt = [1, 2]
    tokens = LifeAI._qwen3_vl_token_matrix(prompt)
    function high_layout(maximum_coordinate)
        positions = repeat(
            reshape(Int[maximum_coordinate - 1, maximum_coordinate], 1, 2, 1),
            3,
            1,
            1,
        )
        delta = maximum_coordinate + 1 - length(prompt)
        return Qwen3VLRopeLayout(
            positions,
            reshape(Int[delta], 1, 1),
            falses(2, 1),
            trues(2, 1),
        )
    end

    error_message = "ArgumentError: Qwen3-VL generated mRoPE coordinates " *
        "exceed max_position_embeddings"
    exhausted = high_layout(limit - 1)
    poison = _CH45VisionComputePoison()

    horizon_failure = _ch45_captured_error() do
        LifeAI._qwen3_vl_generation_mrope_horizon_preflight(
            parameters,
            tokens,
            exhausted,
            2,
        )
    end
    @test sprint(showerror, horizon_failure) == error_message

    vision_failure = _ch45_captured_error() do
        LifeAI._qwen3_vl_generation_mrope_horizon_preflight(
            parameters,
            tokens,
            exhausted,
            2,
        )
        LifeAI._qwen3_vl_generation_vision_features(poison, poison, 2)
    end
    @test sprint(showerror, vision_failure) == error_message

    one_decode = high_layout(limit - 2)
    LifeAI._qwen3_vl_generation_mrope_horizon_preflight(
        parameters,
        tokens,
        one_decode,
        2,
    )
end

@testset "Chapter 45 — generation reuses sealed prompt validation" begin
    parameters = _ch45_tiny_text_parameters()
    inputs = _ch45_tiny_prefill_inputs()
    for options in (
        (; cache=:dynamic),
        (; cache=:static, static_capacity=8),
    )
        collections = Ref(0)
        counted_positions = _Ch45CountedIntArray(
            inputs.rope_layout.position_ids,
            collections,
        )
        counted_layout = Qwen3VLRopeLayout(
            counted_positions,
            inputs.rope_layout.rope_deltas,
            inputs.rope_layout.visual_mask,
            inputs.rope_layout.attention_mask,
        )
        generated = generate_hf_qwen3_vl_tokens(
            parameters,
            inputs.input_ids,
            counted_layout;
            vision_features=inputs.vision_features,
            max_new_tokens=1,
            stop_token_ids=Int[],
            options...,
        )
        @test collections[] == 1
        @test generated.generated_ids == [8]
        @test generated.cache.position == 8
    end

    tokens = LifeAI._qwen3_vl_token_matrix(inputs.input_ids)
    rope_layout = LifeAI._qwen3_vl_generation_rope_snapshot(
        inputs.rope_layout,
    )
    contract = LifeAI._qwen3_vl_generation_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        inputs.vision_features,
        1,
    )
    cache = init_qwen3_vl_kv_cache(parameters)
    changed_parameters = merge(
        parameters,
        (; embedding=copy(parameters.embedding)),
    )
    changed_layout = LifeAI._qwen3_vl_generation_rope_snapshot(rope_layout)
    changed_features = merge(
        inputs.vision_features,
        (; visual_embeddings=copy(
            inputs.vision_features.visual_embeddings,
        )),
    )
    for (source_parameters, source_tokens, source_layout, source_features) in (
        (changed_parameters, tokens, rope_layout, inputs.vision_features),
        (parameters, copy(tokens), rope_layout, inputs.vision_features),
        (parameters, tokens, changed_layout, inputs.vision_features),
        (parameters, tokens, rope_layout, changed_features),
    )
        @test_throws ArgumentError LifeAI._qwen3_vl_text_prefill_cached_prevalidated(
            source_parameters,
            source_tokens,
            source_layout;
            vision_features=source_features,
            cache,
            contract,
        )
        @test isempty(cache)
    end

    mutable_deepstack = Any[inputs.vision_features.deepstack...]
    mutable_features = (;
        visual_embeddings=inputs.vision_features.visual_embeddings,
        deepstack=mutable_deepstack,
    )
    mutable_contract = LifeAI._qwen3_vl_generation_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        mutable_features,
        1,
    )
    mutable_deepstack[1] = zeros(
        Float32,
        parameters.spec.hidden_size,
        size(inputs.vision_features.visual_embeddings, 2) - 1,
    )

    dynamic_guard = init_qwen3_vl_kv_cache(parameters)
    dynamic_failure = _ch45_captured_error() do
        LifeAI._qwen3_vl_text_prefill_cached_prevalidated(
            parameters,
            tokens,
            rope_layout;
            vision_features=mutable_features,
            cache=dynamic_guard,
            contract=mutable_contract,
        )
    end
    @test dynamic_failure isa ArgumentError
    @test sprint(showerror, dynamic_failure) ==
        "ArgumentError: Qwen3-VL prevalidated generation prompt sources changed"
    @test isempty(dynamic_guard)

    static_guard = LifeAI.init_qwen3_vl_static_kv_cache(
        parameters;
        capacity=8,
    )
    static_keys = map(layer -> copy(layer.keys), static_guard.layers)
    static_values = map(layer -> copy(layer.values), static_guard.layers)
    static_failure = _ch45_captured_error() do
        LifeAI._qwen3_vl_text_prefill_static_prevalidated(
            parameters,
            tokens,
            rope_layout;
            vision_features=mutable_features,
            cache=static_guard,
            contract=mutable_contract,
        )
    end
    @test static_failure isa ArgumentError
    @test sprint(showerror, static_failure) ==
        "ArgumentError: Qwen3-VL prevalidated generation prompt sources changed"
    @test isempty(static_guard)
    @test static_guard.position == 0
    @test static_guard.rope_delta == 0
    for layer in eachindex(static_guard.layers)
        @test static_guard.layers[layer].keys == static_keys[layer]
        @test static_guard.layers[layer].values == static_values[layer]
    end

    checkpoint = _Ch45MutableCheckpoint(2, 8)
    checkpoint_parameters = merge(parameters, (; checkpoint))
    checkpoint_tokens = LifeAI._qwen3_vl_token_matrix(
        Int[1, 2, 3, 3, 3, 3, 4, 5],
    )
    checkpoint_contract = LifeAI._qwen3_vl_generation_prompt_contract(
        checkpoint_parameters,
        checkpoint_tokens,
        rope_layout,
        inputs.vision_features,
        1,
    )
    checkpoint.image_token_id = 3

    checkpoint_dynamic_guard = init_qwen3_vl_kv_cache(checkpoint_parameters)
    @test_throws ArgumentError LifeAI._qwen3_vl_text_prefill_cached_prevalidated(
        checkpoint_parameters,
        checkpoint_tokens,
        rope_layout;
        vision_features=inputs.vision_features,
        cache=checkpoint_dynamic_guard,
        contract=checkpoint_contract,
    )
    @test isempty(checkpoint_dynamic_guard)

    checkpoint_static_guard = LifeAI.init_qwen3_vl_static_kv_cache(
        checkpoint_parameters;
        capacity=8,
    )
    checkpoint_static_keys = map(
        layer -> copy(layer.keys),
        checkpoint_static_guard.layers,
    )
    checkpoint_static_values = map(
        layer -> copy(layer.values),
        checkpoint_static_guard.layers,
    )
    @test_throws ArgumentError LifeAI._qwen3_vl_text_prefill_static_prevalidated(
        checkpoint_parameters,
        checkpoint_tokens,
        rope_layout;
        vision_features=inputs.vision_features,
        cache=checkpoint_static_guard,
        contract=checkpoint_contract,
    )
    @test isempty(checkpoint_static_guard)
    for layer in eachindex(checkpoint_static_guard.layers)
        @test checkpoint_static_guard.layers[layer].keys ==
            checkpoint_static_keys[layer]
        @test checkpoint_static_guard.layers[layer].values ==
            checkpoint_static_values[layer]
    end
end

@testset "Chapter 45 — sealed generation parameter replacement is rejected" begin
    base = _ch45_tiny_text_parameters()
    parameters = _Ch45MutableTextParameters(
        base.embedding,
        collect(base.blocks),
        base.final_norm,
        base.spec,
    )
    inputs = _ch45_tiny_prefill_inputs()
    tokens = LifeAI._qwen3_vl_token_matrix(inputs.input_ids)
    rope_layout = LifeAI._qwen3_vl_generation_rope_snapshot(
        inputs.rope_layout,
    )
    contract = LifeAI._qwen3_vl_generation_prompt_contract(
        parameters,
        tokens,
        rope_layout,
        inputs.vision_features,
        1,
    )
    dynamic_guard = init_qwen3_vl_kv_cache(parameters)
    static_guard = LifeAI.init_qwen3_vl_static_kv_cache(
        parameters;
        capacity=8,
    )
    static_key_sources = map(layer -> layer.keys, static_guard.layers)
    static_value_sources = map(layer -> layer.values, static_guard.layers)
    static_keys = map(copy, static_key_sources)
    static_values = map(copy, static_value_sources)

    assert_rejected_before_compute = () -> begin
        dynamic_failure = _ch45_captured_error() do
            LifeAI._qwen3_vl_text_prefill_cached_prevalidated(
                parameters,
                tokens,
                rope_layout;
                vision_features=inputs.vision_features,
                cache=dynamic_guard,
                contract,
            )
        end
        @test dynamic_failure isa ArgumentError
        @test sprint(showerror, dynamic_failure) ==
            "ArgumentError: Qwen3-VL prevalidated generation prompt sources changed"
        @test isempty(dynamic_guard)

        static_failure = _ch45_captured_error() do
            LifeAI._qwen3_vl_text_prefill_static_prevalidated(
                parameters,
                tokens,
                rope_layout;
                vision_features=inputs.vision_features,
                cache=static_guard,
                contract,
            )
        end
        @test static_failure isa ArgumentError
        @test sprint(showerror, static_failure) ==
            "ArgumentError: Qwen3-VL prevalidated generation prompt sources changed"
        @test isempty(static_guard)
        @test static_guard.position == 0
        @test static_guard.rope_delta == 0
        for layer in eachindex(static_guard.layers)
            @test static_guard.layers[layer].keys === static_key_sources[layer]
            @test static_guard.layers[layer].values === static_value_sources[layer]
            @test static_guard.layers[layer].keys == static_keys[layer]
            @test static_guard.layers[layer].values == static_values[layer]
        end
    end

    original_blocks = parameters.blocks
    original_block = original_blocks[3]
    original_final_norm = parameters.final_norm
    parameters.blocks[3] = merge(
        original_block,
        (; q_weight=zeros(Float32, 1, 1)),
    )
    assert_rejected_before_compute()

    parameters.blocks[3] = original_block
    parameters.final_norm = copy(parameters.final_norm)
    assert_rejected_before_compute()

    parameters.final_norm = original_final_norm
    parameters.blocks = copy(original_blocks)
    assert_rejected_before_compute()
end

@testset "Chapter 45 — raw generation image roles are strict" begin
    image = zeros(UInt8, 1, 1, 3)
    valid_messages = [(
        role=SubString("xuser", 2),
        content=Any[(type="image", image=image)],
    )]
    @test LifeAI._qwen3_vl_generation_image(valid_messages) === image

    for invalid_role in (1, true, :user)
        failure = _ch45_captured_error() do
            LifeAI._qwen3_vl_generation_image([(
                role=invalid_role,
                content=Any[(type="image", image=image)],
            )])
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL generation chat role must be a string"
    end
end

@testset "Chapter 45 — raw generation options fail before image compute" begin
    poison = _CH45VisionComputePoison()
    @test LifeAI._qwen3_vl_generation_vision_features(
        poison,
        poison,
        0,
    ) === nothing
    vision_failure = _ch45_captured_error() do
        LifeAI._qwen3_vl_generation_vision_features(
            poison,
            poison,
            1,
        )
    end
    @test vision_failure isa ErrorException
    @test occursin("vision compute was touched", sprint(showerror, vision_failure))

    tokenizer = _ch45_vl_generation_tokenizer()
    text_parameters = (;
        spec=_CH45_TINY_TEXT_SPEC,
        embedding=zeros(Float32, 1, 1),
    )
    vision_parameters = (;
        spec=(; out_hidden_size=_CH45_TINY_TEXT_SPEC.hidden_size),
        patch_weight=zeros(Float32, 1, 1),
    )
    poison_messages = [(
        role="user",
        content=Any[
            (type="image", image=42),
            (type="text", text="Describe."),
        ],
    )]

    normalized = LifeAI._qwen3_vl_generation_preflight(
        text_parameters,
        Int32[8],
        3,
        :static,
        big(10),
        :replace,
    )
    @test normalized.stops == Set([8])
    @test normalized.static_capacity === 10
    @test normalized.max_new_tokens === 3
    @test LifeAI._qwen3_vl_generation_prompt_preflight(
        text_parameters,
        collect(1:8),
        3,
        10,
    ) == 10
    @test_throws ArgumentError LifeAI._qwen3_vl_generation_prompt_preflight(
        text_parameters,
        collect(1:8),
        3,
        9,
    )
    @test_throws ArgumentError LifeAI._qwen3_vl_generation_prompt_preflight(
        text_parameters,
        Int[_CH45_TINY_TEXT_SPEC.vocab_size + 1],
        0,
        nothing,
    )
    @test_throws ArgumentError LifeAI._qwen3_vl_generation_preflight(
        text_parameters,
        Int[],
        0,
        :static,
        true,
        :replace,
    )
    @test_throws ArgumentError LifeAI._qwen3_vl_generation_preflight(
        text_parameters,
        Int[],
        0,
        :static,
        big(typemax(Int)) + 1,
        :replace,
    )

    cases = (
        (
            needle="max_new_tokens",
            options=(; max_new_tokens=true, stop_token_ids=Int[]),
        ),
        (
            needle="max_new_tokens",
            options=(;
                max_new_tokens=big(typemax(Int)) + 1,
                stop_token_ids=Int[],
            ),
        ),
        (
            needle="max_new_tokens",
            options=(; max_new_tokens=-1, stop_token_ids=Int[]),
        ),
        (
            needle="cache",
            options=(; max_new_tokens=0, stop_token_ids=Int[], cache=:bad),
        ),
        (
            needle="stop token",
            options=(;
                max_new_tokens=0,
                stop_token_ids=Int[_CH45_TINY_TEXT_SPEC.vocab_size + 1],
            ),
        ),
        (
            needle="decode_errors",
            options=(;
                max_new_tokens=0,
                stop_token_ids=Int[],
                decode_errors=:bad,
            ),
        ),
        (
            needle="static_capacity",
            options=(;
                max_new_tokens=0,
                stop_token_ids=Int[],
                cache=:dynamic,
                static_capacity=8,
            ),
        ),
    )
    for case in cases
        failure = _ch45_captured_error() do
            generate_hf_qwen3_vl(
                vision_parameters,
                text_parameters,
                tokenizer,
                poison_messages;
                case.options...,
            )
        end
        @test failure isa ArgumentError
        message = sprint(showerror, failure)
        @test occursin(case.needle, message)
        @test !occursin("image payload", message)
    end

    image_failure = _ch45_captured_error() do
        generate_hf_qwen3_vl(
            vision_parameters,
            text_parameters,
            tokenizer,
            poison_messages;
            max_new_tokens=Int128(0),
            stop_token_ids=Int[],
        )
    end
    @test image_failure isa ArgumentError
    @test occursin("image payload", sprint(showerror, image_failure))
end

@testset "Chapter 45 — visual merge token preflight" begin
    parameters = _ch45_tiny_text_parameters()
    spec = qwen3_vl_processor_spec()
    oversized = reshape(
        Int[1, spec.merge_size * 16, spec.merge_size * 16],
        3,
        1,
    )
    @test_throws ArgumentError LifeAI._qwen3_vl_generation_visual_token_preflight(
        parameters,
        oversized,
        spec,
        1,
    )
    @test LifeAI._qwen3_vl_generation_visual_token_preflight(
        parameters,
        oversized,
        spec,
        0,
    ) === nothing
end
