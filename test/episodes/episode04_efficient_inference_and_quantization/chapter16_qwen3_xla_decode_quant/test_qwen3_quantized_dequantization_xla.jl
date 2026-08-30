using BFloat16s: BFloat16
using Reactant
using Test
import LifeAI
using LifeAI: Int4GroupWeight, Int8ChannelWeight

@testset "Qwen3 quantized dequantization compiles on Reactant" begin
    Reactant.set_default_backend(get(ENV, "LIFEAI_XLA_BACKEND", "cpu"))

    int8 = Int8ChannelWeight(
        reshape(Int8[-128, -7, 0, 127, 64, -32], 2, 3),
        Float32[0.25, 0.5],
    )
    int4 = Int4GroupWeight(
        UInt8[
            0x80 0xf1
            0x27 0x9e
        ],
        Float32[
            0.25 0.5
            0.75 1.0
        ],
        2,
        4,
    )
    int8_reference = LifeAI._dequantize_bf16(int8)
    int4_reference = LifeAI._dequantize_bf16(int4)
    int8_slice_reference = LifeAI._dequantize_bf16(
        LifeAI._quant_row_slice(int8, 1:1),
    )
    int4_slice_reference = LifeAI._dequantize_bf16(
        LifeAI._quant_row_slice(int4, 1:1),
    )

    kernel = (int8_weight, int4_weight) -> (
        LifeAI._dequantize_bf16(int8_weight),
        LifeAI._dequantize_bf16(int4_weight),
        LifeAI._dequantize_bf16(
            LifeAI._quant_row_slice(int8_weight, 1:1),
        ),
        LifeAI._dequantize_bf16(
            LifeAI._quant_row_slice(int4_weight, 1:1),
        ),
    )
    int8_device = Reactant.to_rarray(int8)
    int4_device = Reactant.to_rarray(int4)
    compiled = Reactant.@compile kernel(int8_device, int4_device)
    actual = compiled(int8_device, int4_device)

    @test Array(actual[1]) == int8_reference
    @test Array(actual[2]) == int4_reference
    @test Array(actual[3]) == int8_slice_reference
    @test Array(actual[4]) == int4_slice_reference
    @test all(value -> eltype(value) === BFloat16, actual)
end

@testset "BF16 static cache trusted reconstruction traces on Reactant" begin
    Reactant.set_default_backend(get(ENV, "LIFEAI_XLA_BACKEND", "cpu"))
    shape = (2, 2, 4, 1)
    cache = Reactant.to_rarray(LifeAI.BF16AStaticLayerCache(
        zeros(BFloat16, shape),
        zeros(BFloat16, shape),
    ))
    kernel = cache -> LifeAI._bf16a_static_layer_cache_trusted(
        cache.keys .+ BFloat16(1),
        cache.values .+ BFloat16(2),
    )
    compiled = Reactant.@compile kernel(cache)
    actual = compiled(cache)

    @test size(actual.keys) == shape
    @test size(actual.values) == shape
    @test all(Array(actual.keys) .== BFloat16(1))
    @test all(Array(actual.values) .== BFloat16(2))
end
