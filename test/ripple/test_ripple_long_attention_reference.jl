# Optional actual-GPU primitive oracle, independent of model weights/prefill.
using Test,LifeAI,CUDA,JSON3,LinearAlgebra

@inline function _ripple_attention_reference_hash(index::UInt32,seed::UInt32)
    state=(index+seed)*UInt32(747796405)+UInt32(2891336453)
    state=xor(state,state>>16)
    state=state*UInt32(277803737)+UInt32(2891336453)
    state=xor(state,state>>16)
    return (Float32(state&UInt32(65535))-32768f0)/32768f0
end
function _ripple_attention_reference_fill_kernel!(output,seed)
    index=(CUDA.blockIdx().x-Int32(1))*CUDA.blockDim().x+CUDA.threadIdx().x
    stride=CUDA.blockDim().x*CUDA.gridDim().x
    while index<=length(output)
        @inbounds output[index]=_ripple_attention_reference_hash(UInt32(index-Int32(1)),seed)
        index+=stride
    end
    return
end
function _ripple_attention_reference_input(shape,seed)
    output=CUDA.CuArray{Float32}(undef,shape)
    CUDA.@cuda threads=256 blocks=min(cld(length(output),256),65535) _ripple_attention_reference_fill_kernel!(output,UInt32(seed))
    return output
end

"""Check128K/256K/512K/1Mi last-token global attention against actual Torch MATH.

Only~1GiB K/V is resident at1Mi. The exporter uses bounded query-head batches for the
public MATH GQA repetition. No checkpoint/model or full NxN score is allocated.
"""
function ripple_long_attention_reference_checks(;reference_path=joinpath(@__DIR__,"fixtures","long_attention","reference.json"),lengths=nothing)
    reference=JSON3.read(read(reference_path,String))
    @test reference["attention_backend"]=="math"
    @test reference["operation"]=="torch.nn.functional.scaled_dot_product_attention, forced SDPBackend.MATH"
    @test reference["algorithm"]["version"]==1
    dim=Int(reference["head_dim"]);heads=Int(reference["query_heads"]);kvheads=Int(reference["kv_heads"])
    @test (dim,heads,kvheads)==(128,10,1)
    records=Any[];algorithm=reference["algorithm"]
    @test all(Int(algorithm[k])==v for (k,v) in (("multiplier_a",747796405),("addend",2891336453),("multiplier_b",277803737),("xor_shift",16),("mask",4294967295),("low_mask",65535),("center",32768),("denominator",32768)))
    for case in reference["cases"]
        count=Int(case["length"])
        lengths===nothing || count in lengths || continue
        query=_ripple_attention_reference_input((dim,heads,1),Int(algorithm["seed_query"]))
        key=_ripple_attention_reference_input((dim,kvheads,count),Int(algorithm["seed_key"]))
        value=_ripple_attention_reference_input((dim,kvheads,count),Int(algorithm["seed_value"]))
        sinks=CUDA.fill(-Inf32,heads)
        try
            probes=Int.(case["probe_indices_zero_based"]).+1
            @test Array(vec(key)[probes])==Float32.(case["key_probes"])
            @test Array(vec(value)[probes])==Float32.(case["value_probes"])
            @test Array(vec(query)[1:16])==Float32.(case["query_probes"])
            CUDA.synchronize()
            started=time()
            output=LifeAI._flint_attention_long(query,key,value,sinks,0,Int(case["query_position"]),0)
            actual=Array(output);seconds=time()-started
            expected=reshape(Float32.(case["output"]),dim,heads,1)
            absolute=maximum(abs.(actual.-expected));relative=norm(actual-expected)/norm(expected)
            @test all(isfinite,actual)
            @test absolute<2f-6
            @test relative<1f-4
            push!(records,(length=count,max_absolute_error=absolute,relative_l2_error=relative,seconds=seconds,
                kv_bytes=sizeof(key)+sizeof(value)))
            println(JSON3.write(last(records)));flush(stdout)
            CUDA.unsafe_free!(output)
        finally
            CUDA.synchronize()
            CUDA.unsafe_free!(query);CUDA.unsafe_free!(key);CUDA.unsafe_free!(value);CUDA.unsafe_free!(sinks)
            GC.gc(false);CUDA.reclaim()
        end
    end
    return records
end
