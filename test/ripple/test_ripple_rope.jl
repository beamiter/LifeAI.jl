using Test,LifeAI,JSON3

const RIPPLE_ROPE_FIXTURE=joinpath(@__DIR__,"fixtures","rope","reference.json")

function ripple_rope_reference_checks(;device=identity,reference_path=RIPPLE_ROPE_FIXTURE)
    reference=JSON3.read(read(reference_path,String))
    reader=open_safetensors_reader(joinpath(dirname(reference_path),String(reference["tensor_file"])))
    @test reference["revision"]=="e27e171792f303d74c048184e8259ba663cfa0d5"
    records=Any[]
    for case in reference["cases"]
        dim=Int(case["dim"]);heads=Int(case["heads"]);theta=Float32(case["theta"])
        positions=Int.(case["positions_zero_based"])
        shape=(dim,heads,length(positions))
        prefix=String(case["name"])*"."
        host=read_safetensors_tensor(reader,prefix*"input")
        expected=read_safetensors_tensor(reader,prefix*"rotated")
        @test size(host)==shape && size(expected)==shape
        input=device(host)
        maximum_error=0f0
        # Sparse absolute positions also exercise cached GPU tables far beyond8K.
        for (index,position) in enumerate(positions)
            actual=Array(LifeAI._ripple_rope(input[:,:,index:index],position:position,theta))
            error=maximum(abs.(actual.-expected[:,:,index:index]));maximum_error=max(maximum_error,error)
            @test actual≈expected[:,:,index:index] atol=2f-6 rtol=2f-6
        end
        if all(diff(positions).==1)
            actual=Array(LifeAI._ripple_rope(input,first(positions):last(positions),theta))
            @test actual≈expected atol=2f-6 rtol=2f-6
            # A multi-token range straddling4096 must use the Ripple fallback,
            # never the legacy Flint negative-exponent frequencies.
            @test actual≈cat((Array(LifeAI._ripple_rope(input[:,:,i:i],p:p,theta)) for (i,p) in enumerate(positions))...;dims=3) atol=2f-6 rtol=2f-6
        end
        frequencies=device===identity ? LifeAI._ripple_rope_frequencies(dim,theta) : LifeAI._ripple_rope_frequencies(dim,theta,input)
        expected_frequencies=read_safetensors_tensor(reader,prefix*"inverse_frequencies")
        @test Array(frequencies)==expected_frequencies
        angles=reshape(frequencies,dim÷2,1).*reshape(device(Float32.(positions)),1,:)
        expected_cosine=read_safetensors_tensor(reader,prefix*"cosine")[1:dim÷2,:]
        expected_sine=read_safetensors_tensor(reader,prefix*"sine")[1:dim÷2,:]
        @test Array(cos.(angles))≈expected_cosine atol=3f-7 rtol=3f-7
        @test Array(sin.(angles))≈expected_sine atol=3f-7 rtol=3f-7
        push!(records,(name=String(case["name"]),positions=positions,max_absolute_error=maximum_error))
    end
    return records
end

@testset "Ripple official long-position RoPE CPU" begin
    ripple_rope_reference_checks()
    input=zeros(Float32,128,2,2)
    @test_throws DimensionMismatch LifeAI._ripple_rope(input,0:0,10000f0)
    @test_throws ArgumentError LifeAI._ripple_rope(input,-1:0,10000f0)
    @test_throws ArgumentError LifeAI._ripple_rope(input,0:1,0f0)
    @test_throws ArgumentError LifeAI._ripple_rope(zeros(Float32,127,2,2),0:1,10000f0)
    @test size(LifeAI._ripple_rope(zeros(Float32,128,2,0),0:-1,10000f0))==(128,2,0)
end
