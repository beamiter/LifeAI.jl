using Test, LifeAI

const FLINT_DEVICE_QUANT_FIXTURE = joinpath(@__DIR__, "flint", "fixtures", "tiny")

function _flint_device_quant_equal(left, right; device=false)
    if left isa NVFP4Weight
        @test right isa NVFP4Weight
        @test left.packed == Array(right.packed)
        @test left.block_scales == Array(right.block_scales)
        @test left.global_scale == right.global_scale
        @test left.in_dim == right.in_dim
    elseif left isa Int4GroupWeight
        @test right isa Int4GroupWeight
        @test left.packed == Array(right.packed)
        @test left.scale ≈ Array(right.scale) rtol=2f-7 atol=0f0
        @test left.group == right.group
        @test left.in_dim == right.in_dim
    elseif left isa Int8ChannelWeight
        @test right isa Int8ChannelWeight
        @test left.q == Array(right.q)
        @test left.scale ≈ Array(right.scale) rtol=2f-7 atol=0f0
    else
        @test left == Array(right)
    end
end

@testset "Flint quantization placement and progress" begin
    for scheme in (:nvfp4, :int4, :int8, :bf16)
        group = scheme === :nvfp4 ? 16 : 4
        expected = load_hf_flint_quantized(FLINT_DEVICE_QUANT_FIXTURE; scheme, group)
        device_calls = Tuple{DataType,Tuple}[]
        progress = Tuple{String,Int,Int}[]
        copy_device = value -> begin
            push!(device_calls, (eltype(value), size(value)))
            copy(value)
        end
        actual = load_hf_flint_quantized(FLINT_DEVICE_QUANT_FIXTURE; scheme, group,
            device=copy_device, quantize_on_device=true,
            on_progress=(name,index,total) -> push!(progress,(name,index,total)))
        # Each tensor is transferred exactly once. In particular, the packed
        # output is not transferred again after on-device quantization.
        @test length(device_calls) == length(expected.parameters)
        @test all(call -> call[1] === Float32, device_calls)
        @test first.(progress) == sort!(collect(keys(expected.parameters)))
        @test getindex.(progress,2) == collect(1:length(expected.parameters))
        @test all(event -> event[3] == length(expected.parameters), progress)
        for name in keys(expected.parameters)
            _flint_device_quant_equal(expected.parameters[name], actual.parameters[name])
        end
        tokens = [1,3,5,7,9]
        @test hf_flint_forward(expected.model,expected.parameters,tokens) ==
            hf_flint_forward(actual.model,actual.parameters,tokens)
    end
    @test_throws ArgumentError load_hf_flint_quantized("nonexistent";
        scheme=:int4, calibration=:mse, quantize_on_device=true)
end

if lowercase(get(ENV,"LIFEAI_TEST_CUDA","false")) in ("1","true","yes")
    using CUDA
    CUDA.functional() || error("CUDA is required for device-quantization verification")
    CUDA.allowscalar(false)
    @testset "Flint CUDA quantization matches CPU" begin
        for scheme in (:nvfp4, :int4, :int8)
            group = scheme === :nvfp4 ? 16 : 4
            expected = load_hf_flint_quantized(FLINT_DEVICE_QUANT_FIXTURE; scheme, group)
            progress = Tuple{String,Int,Int}[]
            actual = load_hf_flint_quantized(FLINT_DEVICE_QUANT_FIXTURE; scheme, group,
                device=CUDA.cu, quantize_on_device=true,
                on_progress=(name,index,total) -> push!(progress,(name,index,total)))
            @test length(progress) == length(expected.parameters)
            for name in keys(expected.parameters)
                _flint_device_quant_equal(expected.parameters[name],actual.parameters[name]; device=true)
                weight = actual.parameters[name]
                payload = weight isa NVFP4Weight ? weight.packed :
                    weight isa Int4GroupWeight ? weight.packed :
                    weight isa Int8ChannelWeight ? weight.q : weight
                @test payload isa CUDA.CuArray
            end
            tokens = [1,3,5,7,9]
            cpu = hf_flint_forward(expected.model,expected.parameters,tokens)
            gpu = Array(hf_flint_forward(actual.model,actual.parameters,tokens))
            @test gpu ≈ cpu atol=5f-5 rtol=5f-5
            @info "Flint on-device quantization" scheme max_absolute_logit_error=maximum(abs.(gpu-cpu))
            CUDA.synchronize()
        end
    end
end
