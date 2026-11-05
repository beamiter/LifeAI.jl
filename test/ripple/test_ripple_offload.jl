using Test,LifeAI,JSON3

function ripple_offload_checks(device=identity)
    fixture=joinpath(@__DIR__,"fixtures")
    reference=JSON3.read(read(joinpath(fixture,"reference.json"),String))
    ids=Int.(reference["input_ids"]) .+ 1
    bundle=load_hf_ripple_model(fixture)
    baseline=hf_ripple_forward(bundle.model,bundle.parameters,ids)
    for budget in (0,512,quantized_parameter_bytes(bundle.parameters))
        mixed=offload_ripple(bundle;device,resident_bytes=budget)
        @test Array(hf_ripple_forward(mixed.model,mixed.parameters,ids)) ≈ baseline atol=5f-5 rtol=5f-5
        stats=ripple_offload_stats(mixed)
        @test stats.resident_bytes<=budget
        @test stats.transient_bytes==0
        @test stats.host_bytes+stats.resident_bytes==quantized_parameter_bytes(bundle.parameters)
        @test quantized_parameter_bytes(mixed.parameters)==quantized_parameter_bytes(bundle.parameters)
        @test generate_ripple_tokens(mixed,ids;max_new_tokens=3,temperature=0,eos_ids=Int[]) == Int.(reference["greedy_tokens"][1:3]).+1
    end
    streamed=load_hf_ripple_streamed(fixture;device,dtype=Float32)
    @test Array(hf_ripple_forward(streamed.model,streamed.parameters,ids)) ≈ baseline atol=5f-5 rtol=5f-5
    @test ripple_offload_stats(streamed).resident_bytes==0
    for scheme in (:int4,:int8,:nvfp4)
        mktempdir() do root
            target=joinpath(root,"packed")
            @test quantize_ripple_checkpoint(fixture,target;device,scheme,group=scheme===:nvfp4 ? 16 : 4,dtype=Float32)==target
            loaded=load_ripple_quantized(target;device,resident_bytes=512)
            @test loaded.checksum_backend in (:system,:julia)
            portable=load_ripple_quantized(target;checksum_backend=:julia)
            @test portable.checksum_backend===:julia
            @test_throws ArgumentError load_ripple_quantized(target;checksum_backend=:unknown)
            expected=load_hf_ripple_model(fixture;scheme,group=scheme===:nvfp4 ? 16 : 4)
            @test Array(hf_ripple_forward(loaded.model,loaded.parameters,ids)) ≈ hf_ripple_forward(expected.model,expected.parameters,ids) atol=6f-5 rtol=6f-5
            @test ripple_offload_stats(loaded).transient_bytes==0
            @test encode(load_hf_ripple_tokenizer(target),"hi")==[279]
            @test_throws ArgumentError quantize_ripple_checkpoint(fixture,target)
            manifest=JSON3.read(read(joinpath(target,"manifest.json"),String))
            record=first(manifest["tensors"])["values"]
            file=joinpath(target,String(record["file"]))
            open(file,"r+") do io
                byte=read(io,UInt8);seekstart(io);write(io,byte⊻0x01)
            end
            @test_throws ArgumentError load_ripple_quantized(target)
            @test_throws ArgumentError load_ripple_quantized(target;checksum_backend=:julia)
        end
    end
end

@testset "Ripple portable artifacts and bounded offload" begin
    ripple_offload_checks()
end
