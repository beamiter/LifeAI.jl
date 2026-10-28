using Test, LifeAI, JSON3
using BFloat16s: BFloat16

function _with_flint_bf16_device_fixture(f)
    source = joinpath(@__DIR__, "flint", "fixtures", "tiny")
    original = load_hf_flint_model(source)
    mktempdir() do directory
        cp(joinpath(source,"config.json"),joinpath(directory,"config.json"))
        payload = IOBuffer()
        header = Dict{String,Any}()
        offset = 0
        for name in sort!(collect(keys(original.parameters)))
            tensor = BFloat16.(original.parameters[name])
            values = ndims(tensor) == 2 ? vec(permutedims(tensor)) : vec(tensor)
            count = sizeof(values)
            header[name] = Dict("dtype"=>"BF16", "shape"=>collect(size(tensor)),
                "data_offsets"=>[offset,offset+count])
            write(payload,values)
            offset += count
        end
        encoded = codeunits(JSON3.write(header))
        open(joinpath(directory,"model.safetensors"),"w") do io
            write(io,UInt64(length(encoded))); write(io,encoded); write(io,take!(payload))
        end
        f(directory)
    end
end

function _flint_bf16_device_compare(expected, actual)
    for name in keys(expected.parameters)
        left, right = expected.parameters[name], actual.parameters[name]
        if left isa NVFP4Weight
            @test right isa NVFP4Weight
            @test left.packed == Array(right.packed)
            @test left.block_scales == Array(right.block_scales)
            @test left.global_scale == right.global_scale
        else
            @test left == Array(right)
        end
    end
end

@testset "BF16 checkpoint keeps raw bits until device quantization" begin
    _with_flint_bf16_device_fixture() do directory
        expected = load_hf_flint_quantized(directory;scheme=:nvfp4)
        names = sort!(collect(keys(expected.parameters)))
        received_types = DataType[]
        progress = Tuple{String,Int,Int}[]
        actual = load_hf_flint_quantized(directory;scheme=:nvfp4,
            quantize_on_device=true,
            device=value -> (push!(received_types,eltype(value)); copy(value)),
            on_progress=(name,index,total)->push!(progress,(name,index,total)))
        @test length(received_types) == length(names)
        @test first.(progress) == names
        for (index,name) in enumerate(names)
            @test received_types[index] === (LifeAI._flint_quantizable(name) ? BFloat16 : Float32)
        end
        _flint_bf16_device_compare(expected,actual)
        @test hf_flint_forward(expected.model,expected.parameters,[1,3,5]) ==
            hf_flint_forward(actual.model,actual.parameters,[1,3,5])
    end
end

if lowercase(get(ENV,"LIFEAI_TEST_CUDA","false")) in ("1","true","yes")
    using CUDA
    CUDA.functional() || error("CUDA required")
    CUDA.allowscalar(false)
    @testset "BF16-to-CUDA NVFP4 matches CPU conversion exactly" begin
        _with_flint_bf16_device_fixture() do directory
            expected = load_hf_flint_quantized(directory;scheme=:nvfp4)
            actual = load_hf_flint_quantized(directory;scheme=:nvfp4,
                quantize_on_device=true,device=CUDA.cu)
            _flint_bf16_device_compare(expected,actual)
            x = [1,3,5]
            cpu = hf_flint_forward(expected.model,expected.parameters,x)
            gpu = Array(hf_flint_forward(actual.model,actual.parameters,x))
            @test gpu ≈ cpu atol=5f-5 rtol=5f-5
            @info "Raw BF16 CUDA quantization" maximum_absolute_logit_error=maximum(abs.(gpu-cpu))
            CUDA.synchronize()
        end
    end
end
