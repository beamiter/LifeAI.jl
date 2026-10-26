using Test
using Random: MersenneTwister

# Keep the real image processor, chat template and decoder. The small tower
# deliberately depends on every image's pixels and preserves image order.
struct _Ch149VisionStub{S,A}
    spec::S
    patch_weight::A
    calls::Base.RefValue{Int}
    features::Base.RefValue{Any}
end

function LifeAI._qwen3_vl_generation_vision_features(
    parameters::_Ch149VisionStub, processed, requested::Int,
)
    requested == 0 && return nothing
    parameters.calls[] += 1
    merge_unit = parameters.spec.spatial_merge_size^2
    feature_count = sum(prod(processed.grid_thw[:, i]) for i in axes(processed.grid_thw, 2)) ÷ merge_unit
    features = zeros(Float32, parameters.spec.out_hidden_size, feature_count)
    patch_start, feature_start = 1, 1
    for image in axes(processed.grid_thw, 2)
        patch_count = prod(processed.grid_thw[:, image])
        image_features = patch_count ÷ merge_unit
        for token in 1:image_features, channel in axes(features, 1)
            first_patch = patch_start + (token - 1) * merge_unit
            pixel = processed.pixel_values[mod1(channel, size(processed.pixel_values, 1)), first_patch]
            features[channel, feature_start + token - 1] =
                0.15f0 * sin(Float32(channel) + pixel) + 0.03f0 * pixel
        end
        patch_start += patch_count
        feature_start += image_features
    end
    dtype = eltype(parameters.patch_weight)
    result = (;
        visual_embeddings=dtype.(features),
        deepstack=ntuple(i -> dtype.(features .* (0.01f0 * i)), 3),
    )
    parameters.features[] = result
    return result
end

function _ch149_fixture(; bf16=false)
    fixture = _ch147_raw_fixture()
    processor = _ch147_rebuild(fixture.processor;
        min_pixels=32 * 32, max_pixels=64 * 64,
    )
    parameters = bf16 ? _ch47_bf16_parameters(fixture.parameters) : fixture.parameters
    patch_weight = bf16 ? BFloat16.(fixture.vision.patch_weight) : fixture.vision.patch_weight
    tower = _Ch149VisionStub(fixture.vision.spec, patch_weight,
        Ref(0), Ref{Any}(nothing))
    first_image = fill(UInt8(17), 32, 32, 3)
    second_image = fill(UInt8(221), 32, 64, 3)
    return merge(fixture, (; parameters, processor, tower, first_image, second_image))
end

function _ch149_messages(first_image, second_image; separate=false)
    if separate
        return Any[
            (role="user", content=Any[(type="image", image=first_image),
                (type="text", text="First image.")]),
            (role="assistant", content="I see it."),
            (role="user", content=Any[(type="text", text="Compare."),
                (type="image", image=second_image)]),
        ]
    end
    return Any[(role="user", content=Any[
        (type="image", image=first_image), (type="text", text="Compare."),
        (type="image", image=second_image),
    ])]
end

function _ch149_generate(fixture, messages; kwargs...)
    options = merge((;
        processor_spec=fixture.processor, max_new_tokens=3,
        stop_token_ids=Int[], capture_logits=true,
    ), (; kwargs...))
    return generate_hf_qwen3_vl(fixture.tower, fixture.parameters,
        fixture.tokenizer, messages; options...)
end

@testset "Chapter 149 — ordered image processing and metadata" begin
    fixture = _ch149_fixture()
    (; first_image, second_image, processor) = fixture
    for separate in (false, true)
        messages = _ch149_messages(first_image, second_image; separate)
        images = LifeAI._qwen3_vl_generation_images(messages)
        @test length(images) == 2
        @test images[1] === first_image
        @test images[2] === second_image
        combined = LifeAI._qwen3_vl_generation_process_images(images, processor)
        first = LifeAI._qwen3_vl_generation_process_image(first_image, processor)
        second = LifeAI._qwen3_vl_generation_process_image(second_image, processor)
        @test combined.pixel_values == hcat(first.pixel_values, second.pixel_values)
        @test combined.grid_thw == hcat(first.grid_thw, second.grid_thw)
        @test combined.original_sizes == [(32, 32), (32, 64)]
        @test combined.resized_sizes == [first.resized_size, second.resized_size]
        result = _ch149_generate(fixture, messages)
        @test result.image_grid_thw == combined.grid_thw
        @test result.original_image_sizes == combined.original_sizes
        @test result.resized_image_sizes == combined.resized_sizes
        @test result.original_image_size === nothing
        @test result.resized_image_size === nothing
        @test count(result.rope_layout.visual_mask) == 3
        @test length(result.generated_ids) == 3
        @test result.cache.position == length(result.prompt_ids) + 2
    end
    single = Any[(role="user", content=Any[(type="image", image=first_image)])]
    result = _ch149_generate(fixture, single)
    @test result.original_image_size == (32, 32)
    @test result.resized_image_size == (32, 32)
    @test result.original_image_sizes == [(32, 32)]
    @test result.resized_image_sizes == [(32, 32)]
end

@testset "Chapter 149 — multi-image cached decoding and replay" begin
    for bf16 in (false, true), separate in (false, true), sampled in (false, true)
        fixture = _ch149_fixture(; bf16)
        messages = _ch149_messages(fixture.first_image, fixture.second_image; separate)
        sampling = sampled ? (;
            strategy=:sample, temperature=0.7, top_k=7, top_p=0.83,
            sample_uniforms=[0.173, 0.637, 0.913], capture_distribution=true,
        ) : (;)
        dynamic = _ch149_generate(fixture, messages; cache=:dynamic, sampling...)
        static = _ch149_generate(fixture, messages;
            cache=:static, static_capacity=1024, sampling...)
        @test static.generated_ids == dynamic.generated_ids
        @test static.trace == dynamic.trace
        @test static.prefill.logits == dynamic.prefill.logits
        _ch145_check_prefix(static.cache, dynamic.cache)
        reused = _ch149_generate(fixture, messages; cache=static.cache, sampling...)
        @test reused.cache === static.cache
        @test reused.generated_ids == dynamic.generated_ids
        @test reused.trace == dynamic.trace
        _ch145_check_prefix(reused.cache, dynamic.cache)
        direct = generate_hf_qwen3_vl_tokens(fixture.parameters,
            dynamic.prompt_ids, dynamic.rope_layout;
            vision_features=fixture.tower.features[], max_new_tokens=3,
            stop_token_ids=Int[], capture_logits=true, sampling...)
        @test direct.trace == dynamic.trace
        if sampled
            for step in dynamic.trace
                oracle = _ch148_oracle(step.logits, step.sample_uniform;
                    temperature=0.7, top_k=7, top_p=0.83)
                @test step.token_id == oracle.token
            end
        end
    end
    # Equal image geometries isolate pixel order from token/layout geometry.
    fixture = _ch149_fixture()
    dark = fixture.first_image
    bright = fill(UInt8(221), size(dark))
    forward = _ch149_generate(fixture, _ch149_messages(dark, bright))
    forward_features = copy(fixture.tower.features[].visual_embeddings)
    backward = _ch149_generate(fixture, _ch149_messages(bright, dark))
    @test forward.prompt_ids == backward.prompt_ids
    @test forward_features == reverse(fixture.tower.features[].visual_embeddings; dims=2)
    @test forward.prefill.logits != backward.prefill.logits
end

@testset "Chapter 149 — multi-image preflight preserves an existing request" begin
    fixture = _ch149_fixture()
    cache = init_qwen3_vl_static_kv_cache(fixture.parameters; capacity=1024)
    valid = _ch149_messages(fixture.first_image, fixture.second_image)
    _ch149_generate(fixture, valid; cache)
    before = _ch147_snapshot(cache)
    calls = fixture.tower.calls[]
    invalid_requests = [
        (ArgumentError, Any[(role="user", content="No image.")]),
        (ArgumentError, _ch149_messages(fixture.first_image, 42)),
        (ArgumentError, _ch149_messages(fixture.first_image,
            joinpath(mktempdir(), "missing-image.png"))),
        (DimensionMismatch, _ch149_messages(fixture.first_image, zeros(UInt8, 8, 8, 2))),
        (ArgumentError, Any[(role="user", content=Any[(type="image", image=fixture.first_image),
            (type="image_url", image_url="https://invalid.example/image.png")])]),
        (ArgumentError, Any[(role="user", content=Any[(type="image", image=fixture.first_image),
            (type="video", video="not-a-video")])]),
        (ArgumentError, Any[(role="user", content=Any[(type="image", image=fixture.first_image)]),
            (role="assistant", content=Any[(type="image", image=fixture.second_image)])]),
    ]
    for (error_type, messages) in invalid_requests
        @test_throws error_type _ch149_generate(fixture, messages; cache)
        @test fixture.tower.calls[] == calls
        _ch147_unchanged(cache, before)
    end
    @test_throws ArgumentError _ch149_generate(fixture, valid; cache,
        strategy=:sample, sample_uniforms=[0.1, NaN, 0.3])
    @test fixture.tower.calls[] == calls
    _ch147_unchanged(cache, before)
    small = init_qwen3_vl_static_kv_cache(fixture.parameters; capacity=8)
    _ch147_seed_text(fixture.parameters, small)
    small_before = _ch147_snapshot(small)
    @test_throws ArgumentError _ch149_generate(fixture, valid; cache=small)
    @test fixture.tower.calls[] == calls
    _ch147_unchanged(small, small_before)

    failure = _Ch147VisionFailure(fixture.vision.spec,
        fixture.vision.patch_weight, Ref(false))
    @test_throws ErrorException generate_hf_qwen3_vl(failure,
        fixture.parameters, fixture.tokenizer, valid;
        processor_spec=fixture.processor, cache, max_new_tokens=3,
        stop_token_ids=Int[])
    @test failure.touched[]
    _ch147_unchanged(cache, before)

    zero = _ch149_generate(fixture, valid; cache, max_new_tokens=0)
    @test fixture.tower.calls[] == calls
    @test isempty(zero.generated_ids)
    @test zero.prefill === nothing
    @test zero.cache.position == 0
    @test zero.original_image_sizes == [(32, 32), (32, 64)]
end

@testset "Chapter 149 — single/multiple/single cache storage reuse" begin
    for bf16 in (false, true)
        fixture = _ch149_fixture(; bf16)
        cache = init_qwen3_vl_static_kv_cache(fixture.parameters; capacity=1024)
        storage = map(layer -> (layer.keys, layer.values), cache.layers)
        single = Any[(role="user", content=Any[
            (type="image", image=fixture.first_image), (type="text", text="Describe.")])]
        multiple = _ch149_messages(fixture.first_image, fixture.second_image)
        for messages in (single, multiple, single)
            actual = _ch149_generate(fixture, messages; cache)
            expected = _ch149_generate(fixture, messages; cache=:dynamic)
            @test actual.cache === cache
            @test actual.generated_ids == expected.generated_ids
            @test actual.trace == expected.trace
            _ch145_check_prefix(cache, expected.cache)
            for (layer, (keys, values)) in zip(cache.layers, storage)
                @test layer.keys === keys
                @test layer.values === values
            end
        end
    end
end

@testset "Chapter 149 — exact expanded context and total visual budget" begin
    fixture = _ch149_fixture()
    messages = _ch149_messages(fixture.first_image, fixture.second_image)
    reference = _ch149_generate(fixture, messages)
    limit = length(reference.prompt_ids) + 2 # three outputs process two appends
    spec = _ch147_rebuild(fixture.parameters.spec; max_position_embeddings=limit)
    checkpoint = _ch147_rebuild(fixture.parameters.checkpoint; text=spec)
    parameters = merge(fixture.parameters, (; spec, checkpoint))
    bounded = merge(fixture, (; parameters))
    exact = _ch149_generate(bounded, messages; cache=:static, static_capacity=limit)
    @test exact.cache.position == limit
    @test exact.generated_ids == reference.generated_ids
    @test exact.trace == reference.trace
    before = _ch147_snapshot(exact.cache)
    calls = fixture.tower.calls[]
    @test_throws ArgumentError _ch149_generate(bounded, messages;
        cache=exact.cache, max_new_tokens=4)
    @test fixture.tower.calls[] == calls
    _ch147_unchanged(exact.cache, before)

    # Each grid individually fits, while their combined merged tokens do not.
    grid = Int[1, 64, 64]
    grids = repeat(reshape(grid, 3, 1), 1, 3)
    @test qwen3_vl_image_token_count(Tuple(grid); spec=fixture.processor) <=
        fixture.parameters.spec.max_position_embeddings
    @test_throws ArgumentError LifeAI._qwen3_vl_generation_visual_token_preflight(
        fixture.parameters, grids, fixture.processor, 1,
        ones(Int, fixture.parameters.spec.max_position_embeddings))
end

@testset "Chapter 149 — real tiny vision isolates image attention" begin
    for dtype in (Float32, BFloat16)
        parameters = _ch43_tiny_parameters(dtype)
        first_pixels = _ch43_values(dtype, (24, 4), 50; scale=0.1f0)
        second_pixels = _ch43_values(dtype, (24, 8), 900; scale=0.2f0)
        first_grid = reshape(Int[1, 2, 2], 3, 1)
        second_grid = reshape(Int[1, 2, 4], 3, 1)
        forward(pixels, grids) = hf_qwen3_vl_vision_forward(parameters,
            Qwen3VLVisionInput(pixels, grids; spec=parameters.spec))
        first = forward(first_pixels, first_grid)
        second = forward(second_pixels, second_grid)
        combined = forward(hcat(first_pixels, second_pixels), hcat(first_grid, second_grid))
        reversed = forward(hcat(second_pixels, first_pixels), hcat(second_grid, first_grid))
        # Different matrix widths can choose different BLAS kernels. Compare
        # their numerical outputs without claiming cross-shape bitwise parity.
        tolerance = dtype === Float32 ? 1f-6 : 5f-4
        @test Float32.(combined.patch_hidden_state) ≈
            Float32.(hcat(first.patch_hidden_state, second.patch_hidden_state)) atol=tolerance rtol=tolerance
        @test Float32.(combined.visual_embeddings) ≈
            Float32.(hcat(first.visual_embeddings, second.visual_embeddings)) atol=tolerance rtol=tolerance
        @test Float32.(reversed.visual_embeddings) ≈
            Float32.(hcat(second.visual_embeddings, first.visual_embeddings)) atol=tolerance rtol=tolerance
        for index in 1:3
            @test Float32.(combined.deepstack[index]) ≈
                Float32.(hcat(first.deepstack[index], second.deepstack[index])) atol=tolerance rtol=tolerance
        end
    end
end
