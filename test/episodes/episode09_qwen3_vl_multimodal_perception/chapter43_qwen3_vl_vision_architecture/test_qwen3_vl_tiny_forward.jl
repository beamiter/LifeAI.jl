using BFloat16s: BFloat16
using LinearAlgebra: I
using Test
import LifeAI
using LifeAI: Qwen3VLVisionInput,
    Qwen3VLVisionSpec,
    hf_qwen3_vl_vision_forward

const _CH43_TINY_VISION_SPEC = Qwen3VLVisionSpec(
    4,              # depth
    8,              # hidden_size
    32,             # intermediate_size == hidden_size * merge_size^2
    2,              # num_heads
    3,              # in_channels
    2,              # patch_size
    2,              # temporal_patch_size
    2,              # spatial_merge_size
    6,              # out_hidden_size
    16,             # learned 4x4 position grid
    (0, 1, 2),      # exercise all three DeepStack mergers
    "gelu_pytorch_tanh",
)

function _ch43_values(::Type{T}, dimensions::Tuple, offset::Int; scale=0.02f0) where {T}
    values = Float32[
        scale * sin(0.173f0 * Float32(offset + index))
        for index in 1:prod(dimensions)
    ]
    return reshape(T.(values), dimensions)
end

function _ch43_norm(::Type{T}, width::Int, offset::Int) where {T}
    delta = vec(_ch43_values(Float32, (width,), offset; scale=0.01f0))
    return (;
        scale=T.(1.0f0 .+ delta),
        bias=T.(_ch43_values(Float32, (width,), offset + 10_000; scale=0.005f0)),
    )
end

function _ch43_merger(::Type{T}, spec, offset::Int; postshuffle_norm::Bool) where {T}
    merged_width = spec.hidden_size * spec.spatial_merge_size^2
    norm_width = postshuffle_norm ? merged_width : spec.hidden_size
    return (;
        postshuffle_norm,
        norm=_ch43_norm(T, norm_width, offset),
        fc1_weight=_ch43_values(T, (merged_width, merged_width), offset + 100),
        fc1_bias=vec(_ch43_values(T, (merged_width,), offset + 200)),
        fc2_weight=_ch43_values(
            T,
            (spec.out_hidden_size, merged_width),
            offset + 300,
        ),
        fc2_bias=vec(_ch43_values(T, (spec.out_hidden_size,), offset + 400)),
    )
end

function _ch43_tiny_parameters(::Type{T}) where {T}
    spec = _CH43_TINY_VISION_SPEC
    blocks = ntuple(spec.depth) do block_index
        offset = 1_000 * block_index
        (;
            norm1=_ch43_norm(T, spec.hidden_size, offset),
            norm2=_ch43_norm(T, spec.hidden_size, offset + 100),
            qkv_weight=_ch43_values(
                T,
                (3 * spec.hidden_size, spec.hidden_size),
                offset + 200,
            ),
            qkv_bias=vec(_ch43_values(T, (3 * spec.hidden_size,), offset + 300)),
            proj_weight=_ch43_values(
                T,
                (spec.hidden_size, spec.hidden_size),
                offset + 400,
            ),
            proj_bias=vec(_ch43_values(T, (spec.hidden_size,), offset + 500)),
            fc1_weight=_ch43_values(
                T,
                (spec.intermediate_size, spec.hidden_size),
                offset + 600,
            ),
            fc1_bias=vec(_ch43_values(T, (spec.intermediate_size,), offset + 700)),
            fc2_weight=_ch43_values(
                T,
                (spec.hidden_size, spec.intermediate_size),
                offset + 800,
            ),
            fc2_bias=vec(_ch43_values(T, (spec.hidden_size,), offset + 900)),
        )
    end
    return (;
        patch_weight=_ch43_values(T, (spec.hidden_size, 24), 10),
        patch_bias=vec(_ch43_values(T, (spec.hidden_size,), 20)),
        pos_embedding=_ch43_values(
            T,
            (spec.hidden_size, spec.num_position_embeddings),
            30,
        ),
        blocks,
        merger=_ch43_merger(T, spec, 20_000; postshuffle_norm=false),
        deepstack_mergers=ntuple(3) do index
            _ch43_merger(
                T,
                spec,
                30_000 + 1_000 * index;
                postshuffle_norm=true,
            )
        end,
        spec,
        source="deterministic-test-parameters",
    )
end

function _ch43_tiny_input(::Type{T}) where {T}
    pixels = _ch43_values(T, (24, 4), 50; scale=0.1f0)
    return Qwen3VLVisionInput(
        pixels,
        reshape(Int[1, 2, 2], 3, 1);
        spec=_CH43_TINY_VISION_SPEC,
    )
end

function _ch43_assert_tiny_forward(::Type{T}) where {T}
    parameters = _ch43_tiny_parameters(T)
    input = _ch43_tiny_input(T)
    first_run = hf_qwen3_vl_vision_forward(
        parameters,
        input;
        capture_layers=(0, 1, 2, 3),
    )
    second_run = hf_qwen3_vl_vision_forward(
        parameters,
        input;
        capture_layers=(0, 1, 2, 3),
    )

    @test size(first_run.patch_hidden_state) == (8, 4)
    @test size(first_run.visual_embeddings) == (6, 1)
    @test length(first_run.deepstack) == 3
    @test all(feature -> size(feature) == (6, 1), first_run.deepstack)
    @test Set(keys(first_run.checkpoints)) == Set((0, 1, 2, 3))
    @test all(layer -> size(first_run.checkpoints[layer]) == (8, 4), 0:3)
    @test eltype(first_run.patch_hidden_state) == T
    @test eltype(first_run.visual_embeddings) == T
    @test all(feature -> eltype(feature) == T, first_run.deepstack)
    @test all(isfinite, first_run.patch_hidden_state)
    @test all(isfinite, first_run.visual_embeddings)
    @test all(feature -> all(isfinite, feature), first_run.deepstack)
    @test first_run.patch_hidden_state == second_run.patch_hidden_state
    @test first_run.visual_embeddings == second_run.visual_embeddings
    @test first_run.deepstack == second_run.deepstack
    @test all(
        layer -> first_run.checkpoints[layer] == second_run.checkpoints[layer],
        0:3,
    )
    @test first_run.checkpoints[0] != first_run.checkpoints[3]

    @test_throws ArgumentError hf_qwen3_vl_vision_forward(
        parameters,
        input;
        capture_layers=(-1,),
    )
    @test_throws ArgumentError hf_qwen3_vl_vision_forward(
        parameters,
        input;
        capture_layers=(4,),
    )
    return first_run
end

@testset "Qwen3-VL deterministic tiny Float32 and BF16 full forward" begin
    float32_result = _ch43_assert_tiny_forward(Float32)
    bf16_result = _ch43_assert_tiny_forward(BFloat16)
    @test Float32.(bf16_result.visual_embeddings) ≈
        float32_result.visual_embeddings atol=0.08f0 rtol=0.08f0

    @test_throws ArgumentError hf_qwen3_vl_vision_forward(
        _ch43_tiny_parameters(Float32),
        _ch43_tiny_input(BFloat16),
    )
end

const _CH43_BIAS_VISION_SPEC = Qwen3VLVisionSpec(
    4,
    8,
    32,
    2,
    3,
    2,
    2,
    2,
    6,
    4,
    (0, 1, 3),
    "gelu_pytorch_tanh",
)

_ch43_zero_norm(width) = (scale=ones(Float32, width), bias=zeros(Float32, width))

function _ch43_bias_merger(spec, output_bias; postshuffle_norm)
    merged_width = spec.hidden_size * spec.spatial_merge_size^2
    norm_width = postshuffle_norm ? merged_width : spec.hidden_size
    return (;
        postshuffle_norm,
        norm=_ch43_zero_norm(norm_width),
        fc1_weight=zeros(Float32, merged_width, merged_width),
        fc1_bias=zeros(Float32, merged_width),
        fc2_weight=zeros(Float32, spec.out_hidden_size, merged_width),
        fc2_bias=fill(Float32(output_bias), spec.out_hidden_size),
    )
end

function _ch43_bias_oracle_parameters()
    spec = _CH43_BIAS_VISION_SPEC
    blocks = ntuple(spec.depth) do block_index
        (;
            norm1=_ch43_zero_norm(spec.hidden_size),
            norm2=_ch43_zero_norm(spec.hidden_size),
            qkv_weight=zeros(Float32, 3 * spec.hidden_size, spec.hidden_size),
            qkv_bias=zeros(Float32, 3 * spec.hidden_size),
            proj_weight=zeros(Float32, spec.hidden_size, spec.hidden_size),
            proj_bias=zeros(Float32, spec.hidden_size),
            fc1_weight=zeros(Float32, spec.intermediate_size, spec.hidden_size),
            fc1_bias=zeros(Float32, spec.intermediate_size),
            fc2_weight=zeros(Float32, spec.hidden_size, spec.intermediate_size),
            fc2_bias=fill(Float32(block_index), spec.hidden_size),
        )
    end
    return (;
        patch_weight=zeros(Float32, spec.hidden_size, 24),
        patch_bias=zeros(Float32, spec.hidden_size),
        pos_embedding=zeros(Float32, spec.hidden_size, spec.num_position_embeddings),
        blocks,
        merger=_ch43_bias_merger(spec, 9; postshuffle_norm=false),
        deepstack_mergers=ntuple(3) do index
            _ch43_bias_merger(spec, index; postshuffle_norm=true)
        end,
        spec,
        source="bias-only-semantic-oracle",
    )
end

@testset "Qwen3-VL post-block capture and DeepStack order oracle" begin
    spec = _CH43_BIAS_VISION_SPEC
    input = Qwen3VLVisionInput(
        zeros(Float32, 24, 4),
        reshape(Int[1, 2, 2], 3, 1);
        spec,
    )
    result = hf_qwen3_vl_vision_forward(
        _ch43_bias_oracle_parameters(),
        input;
        capture_layers=(0, 2, 3, 2),
    )

    @test Set(keys(result.checkpoints)) == Set((0, 2, 3))
    @test all(==(1.0f0), result.checkpoints[0])
    @test all(==(6.0f0), result.checkpoints[2])
    @test all(==(10.0f0), result.checkpoints[3])
    @test all(==(10.0f0), result.patch_hidden_state)
    @test length(result.deepstack) == 3
    @test all(==(1.0f0), result.deepstack[1])
    @test all(==(2.0f0), result.deepstack[2])
    @test all(==(3.0f0), result.deepstack[3])
    @test all(==(9.0f0), result.visual_embeddings)
end

@testset "Qwen3-VL temporal frames form independent attention chunks" begin
    grid_thw = Int[
        2 1
        2 4
        4 2
    ]
    heights, widths, frame_ranges = LifeAI._qwen3_vl_patch_order(grid_thw, 2)

    @test length(heights) == 24
    @test length(widths) == 24
    @test frame_ranges == UnitRange{Int}[1:8, 9:16, 17:24]
    @test heights[1:8] == heights[9:16]
    @test widths[1:8] == widths[9:16]
end

@testset "Qwen3-VL attention cannot cross temporal frame boundaries" begin
    spec = (hidden_size=2, num_heads=1)
    qkv_weight = zeros(Float32, 6, 2)
    qkv_weight[5:6, :] .= Matrix{Float32}(I, 2, 2)
    block = (;
        qkv_weight,
        qkv_bias=zeros(Float32, 6),
        proj_weight=Matrix{Float32}(I, 2, 2),
        proj_bias=zeros(Float32, 2),
    )
    cos_values = ones(Float32, 2, 4)
    sin_values = zeros(Float32, 2, 4)
    frame_ranges = UnitRange{Int}[1:2, 3:4]
    original = Float32[
        1 3 10 14
        2 4 20 28
    ]
    changed_second_frame = copy(original)
    changed_second_frame[:, 3:4] .= Float32[
        -100 -200
        300 500
    ]

    first = LifeAI._qwen3_vl_vision_attention(
        spec,
        block,
        original,
        cos_values,
        sin_values,
        frame_ranges,
    )
    changed = LifeAI._qwen3_vl_vision_attention(
        spec,
        block,
        changed_second_frame,
        cos_values,
        sin_values,
        frame_ranges,
    )

    @test first[:, 1:2] == changed[:, 1:2]
    @test first[:, 3:4] != changed[:, 3:4]
    @test first[:, 1:2] == Float32[2 2; 3 3]
end

@testset "Qwen3-VL BF16 linear fuses bias before the output cast" begin
    input = reshape(BFloat16[BFloat16(0.42382812f0)], 1, 1)
    weight = reshape(BFloat16[BFloat16(-2.734375f0)], 1, 1)
    bias = BFloat16[BFloat16(0.083496094f0)]
    fused = LifeAI._qwen3_vl_linear(weight, bias, input)
    rounded_product = BFloat16.(Float32.(weight) * Float32.(input))
    split_rounding = BFloat16.(Float32.(rounded_product) .+ Float32.(bias))

    @test only(fused) == BFloat16(-1.078125f0)
    @test fused != split_rounding
end

@testset "Qwen3-VL frozen Float32 numeric constant bits" begin
    frequencies = LifeAI._qwen3_vl_vision_inv_frequency(32)
    bits = reinterpret(UInt32, frequencies)

    @test reinterpret(
        UInt32,
        LifeAI._QWEN3_VL_GELU_TANH_COEFFICIENT,
    ) == 0x3f4c422a
    @test length(frequencies) == 16
    @test bits[[6, 7, 11, 15]] == UInt32[
        0x3d6655c2,
        0x3d0186e3,
        0x3b4f3e38,
        0x39a5cb60,
    ]
    @test_throws ArgumentError LifeAI._qwen3_vl_vision_inv_frequency(0)
end

@testset "Qwen3-VL BF16 position interpolation rounds each corner" begin
    spec = (num_position_embeddings=4, spatial_merge_size=2)
    position_embedding = reshape(
        BFloat16[-10.0f0, -10.0f0, -10.0f0, 10.0f0],
        1,
        4,
    )
    grid_thw = reshape(Int[1, 4, 4], 3, 1)
    interpolated = LifeAI._qwen3_vl_interpolated_positions(
        spec,
        position_embedding,
        grid_thw,
    )

    @test size(interpolated) == (1, 16)
    @test interpolated[1, 4] == BFloat16(-7.75f0)
    @test interpolated[1, 4] != BFloat16(-7.78125f0)
end
