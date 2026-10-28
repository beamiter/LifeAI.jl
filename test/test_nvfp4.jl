using Test, LifeAI, JSON3, LinearAlgebra

const NVFP4_REFERENCE = JSON3.read(read(joinpath(@__DIR__, "fixtures", "nvfp4", "reference.json"), String))
_nvfp4_reference_matrix(rows, ::Type{T}=Float32) where T = permutedims(hcat((T.(row) for row in rows)...))
_nvfp4_reference_bits(rows) = reinterpret.(Float32, _nvfp4_reference_matrix(rows,UInt32))

@testset "NVFP4 codebooks, RNE and finite saturation" begin
    levels = Float32[0, 0.5, 1, 1.5, 2, 3, 4, 6]
    for code in 0:15
        expected = copysign(levels[(code & 7)+1], code & 8 == 0 ? 1f0 : -1f0)
        @test isequal(LifeAI._nvfp4_decode_e2m1(UInt8(code)), expected)
        @test LifeAI._nvfp4_encode_e2m1(expected) == UInt8(code)
    end
    for (value, code) in zip(Float32[.25, .75, 1.25, 1.75, 2.5, 3.5, 5], UInt8[0,2,2,4,4,6,6])
        @test LifeAI._nvfp4_encode_e2m1(value) == code
        @test LifeAI._nvfp4_encode_e2m1(-value) == code | UInt8(8)
    end
    @test LifeAI._nvfp4_encode_e2m1(100f0) == 0x07
    @test LifeAI._nvfp4_encode_e2m1(-100f0) == 0x0f
    for code in 0:126
        exponent, mantissa = code >> 3, code & 7
        expected = exponent == 0 ? Float32(mantissa)/512f0 : (1f0+Float32(mantissa)/8f0)*Float32(2.0^(exponent-7))
        @test LifeAI._nvfp4_decode_e4m3(UInt8(code)) == expected
        @test LifeAI._nvfp4_encode_e4m3(expected) == UInt8(code)
    end
    @test LifeAI._nvfp4_encode_e4m3(500f0) == 0x7e
    # Midpoint ties go to the representable code with even significand LSB.
    for lower in (0, 1, 6, 7, 8, 15, 55, 56, 62, 63, 64, 119, 125)
        a = LifeAI._nvfp4_decode_e4m3(UInt8(lower))
        b = LifeAI._nvfp4_decode_e4m3(UInt8(lower+1))
        @test LifeAI._nvfp4_encode_e4m3((a+b)/2f0) == UInt8(iseven(lower) ? lower : lower+1)
    end
end

@testset "NVFP4 independent numerical and storage oracle" begin
    for (name, record) in pairs(NVFP4_REFERENCE["matrix_cases"])
        @testset "$name" begin
            input = _nvfp4_reference_bits(record["input_bits"])
            actual = quantize_nvfp4(input)
            @test actual isa NVFP4Weight
            @test actual.packed == _nvfp4_reference_matrix(record["packed"], UInt8)
            @test actual.block_scales == _nvfp4_reference_matrix(record["block_scales"], UInt8)
            @test actual.global_scale == Float32(record["global_scale"])
            @test actual.in_dim == size(input,2)
            @test size(actual.packed) == (size(input,1), cld(size(input,2),16)*8)
            @test size(actual.block_scales) == (size(input,1), cld(size(input,2),16))
            restored = dequantize_nvfp4(actual)
            expected = _nvfp4_reference_bits(record["reconstructed_bits"])
            @test size(restored) == size(input)
            @test all(isfinite, restored)
            @test all(isequal.(restored, expected))
            @test quantized_parameter_bytes(actual) == sizeof(actual.packed)+sizeof(actual.block_scales)+sizeof(Float32)
            slice = LifeAI._quant_row_slice(actual, 1:1)
            @test slice.global_scale == actual.global_scale
            @test all(isequal.(dequantize_nvfp4(slice), expected[1:1,:]))
        end
    end
    @test_throws ArgumentError quantize_nvfp4(fill(NaN32,2,16))
    @test_throws ArgumentError quantize_nvfp4(fill(Inf32,2,16))
    @test_throws ArgumentError quantize_nvfp4(zeros(Float32,0,16))
    @test_throws ArgumentError NVFP4Weight(zeros(UInt8,2,8),zeros(UInt8,2,1),0f0,16)
    @test_throws ArgumentError NVFP4Weight(zeros(UInt8,2,8),zeros(UInt8,2,1),Inf32,16)
    @test_throws DimensionMismatch NVFP4Weight(zeros(UInt8,2,8),zeros(UInt8,3,1),1f0,16)
    @test_throws DimensionMismatch NVFP4Weight(zeros(UInt8,2,7),zeros(UInt8,2,1),1f0,16)
    @test_throws ArgumentError NVFP4Weight(zeros(UInt8,2,8),fill(UInt8(0x7f),2,1),1f0,16)
    @test_throws ArgumentError NVFP4Weight(zeros(UInt8,2,8),fill(UInt8(0x80),2,1),1f0,16)
end

@testset "Flint NVFP4 inference, streaming and portable artifacts" begin
    fixture = joinpath(@__DIR__, "flint", "fixtures", "tiny")
    model = load_hf_flint_model(fixture)
    reference = NVFP4_REFERENCE["flint"]
    tokens = Int.(reference["input_ids"]) .+ 1
    parameters = quantize_flint_parameters(model.parameters; scheme=:nvfp4)
    streamed = load_hf_flint_quantized(fixture; scheme=:nvfp4)
    @test streamed.quantization.group == 16
    @test quantized_parameter_bytes(parameters) < quantized_parameter_bytes(model.parameters)
    for (name, original) in model.parameters
        if haskey(reference["tensors"], name)
            record = reference["tensors"][name]
            weight = parameters[name]
            @test weight isa NVFP4Weight
            @test weight.packed == _nvfp4_reference_matrix(record["packed"], UInt8)
            @test weight.block_scales == _nvfp4_reference_matrix(record["block_scales"], UInt8)
            @test weight.global_scale == Float32(record["global_scale"])
            @test all(isequal.(dequantize_nvfp4(weight), _nvfp4_reference_bits(record["reconstructed_bits"])))
            @test streamed.parameters[name].packed == weight.packed
            @test streamed.parameters[name].block_scales == weight.block_scales
        else
            @test parameters[name] == original
            @test streamed.parameters[name] == original
        end
    end
    expected = permutedims(_nvfp4_reference_matrix(reference["logits"]))
    actual = hf_flint_forward(model.model, parameters, tokens)
    @info "Flint NVFP4 vs independent NumPy" max_absolute_error=maximum(abs.(actual-expected))
    @test actual ≈ expected atol=4f-5 rtol=4f-5
    @test hf_flint_forward(streamed.model,streamed.parameters,tokens) ≈ actual atol=1f-6 rtol=1f-6
    cache = FlintKVCache(streamed.model)
    sequential = hcat((hf_flint_forward(streamed.model,streamed.parameters,[token];cache) for token in tokens)...)
    @test sequential ≈ actual atol=4f-5 rtol=4f-5
    @test generate_flint_tokens(streamed,tokens;max_new_tokens=4,temperature=0,eos_ids=Int[]) == Int.(reference["greedy_tokens"]) .+ 1
    @test_throws ArgumentError quantize_flint_parameters(model.parameters;scheme=:nvfp4,group=32)
    @test_throws ArgumentError quantize_flint_parameters(model.parameters;scheme=:nvfp4,calibration=:mse)
    @test_throws ArgumentError load_hf_flint_quantized(fixture;scheme=:nvfp4,group=8)
    mktempdir() do directory
        artifact = joinpath(directory,"nvfp4")
        save_flint_quantized(artifact,streamed)
        restored = load_flint_quantized(artifact)
        @test quantized_parameter_bytes(restored.parameters) == quantized_parameter_bytes(parameters)
        @test hf_flint_forward(restored.model,restored.parameters,tokens) == actual
        manifest_file = joinpath(artifact,"manifest.json")
        manifest = JSON3.read(read(manifest_file,String),Dict{String,Any})
        @test manifest["version"] == 2
        entry = first(filter(item -> item["kind"] == "nvfp4", manifest["tensors"]))
        @test entry["group"] == 16
        @test entry["encoding"] == "e2m1-e4m3fn-block16"
        manifest["version"] = 1
        write(manifest_file,JSON3.write(manifest))
        @test_throws ArgumentError load_flint_quantized(artifact)
        manifest["version"] = 2
        write(manifest_file,JSON3.write(manifest))
        payload = joinpath(artifact,entry["global_scale"]["file"])
        open(payload,"r+") do io
            byte = read(io,UInt8)
            seekstart(io)
            write(io,byte ⊻ 0x01)
        end
        @test_throws ArgumentError load_flint_quantized(artifact)
    end
    # Existing v1 integer artifacts remain readable after the format bump.
    mktempdir() do directory
        artifact = joinpath(directory,"int8")
        save_flint_quantized(artifact,load_hf_flint_quantized(fixture;scheme=:int8))
        manifest_file = joinpath(artifact,"manifest.json")
        manifest = JSON3.read(read(manifest_file,String),Dict{String,Any})
        manifest["version"] = 1
        write(manifest_file,JSON3.write(manifest))
        @test load_flint_quantized(artifact).parameters["lm_head.weight"] isa Int8ChannelWeight
    end
end
