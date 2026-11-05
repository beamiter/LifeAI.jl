using Test,LifeAI,JSON3
using BFloat16s:BFloat16

function ripple_device_reader_checks(device=identity)
    mktempdir() do directory
        for source_type in (Float32,BFloat16)
            matrix=source_type.(reshape(Float32.(1:35)./17,5,7))
            vector=source_type.(Float32[-2.5,0.125,0.3333,4])
            header=Dict{String,Any}();payload=IOBuffer();offset=0
            for (name,value) in (("matrix",matrix),("vector",vector))
                values=ndims(value)==2 ? vec(permutedims(value)) : value
                bytes=sizeof(values)
                header[name]=Dict("dtype"=>source_type===BFloat16 ? "BF16" : "F32",
                    "shape"=>collect(size(value)),"data_offsets"=>[offset,offset+bytes])
                write(payload,values);offset+=bytes
            end
            file=joinpath(directory,"model.safetensors")
            encoded=codeunits(JSON3.write(header))
            open(file,"w") do io
                write(io,UInt64(length(encoded)));write(io,encoded);write(io,take!(payload))
            end
            reader=open_safetensors_reader(file)
            for dtype in (Float32,BFloat16), (name,expected) in (("matrix",matrix),("vector",vector))
                actual=LifeAI._ripple_read_device_tensor(reader,name,dtype,device)
                @test Array(actual)==dtype.(expected)
                LifeAI._ripple_synchronize_weight(actual)
                LifeAI._ripple_release_weight!(actual)
            end
        end
    end
end

@testset "Ripple row-major device reader bit preservation" begin
    ripple_device_reader_checks()
end
