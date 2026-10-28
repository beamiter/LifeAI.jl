# Run with the LifeAI project and a functional CUDA device. The largest case
# uses 1 GiB FP32 K/V, 64 MiB output, and less than 1 GiB temporary storage.
using LifeAI, CUDA, JSON3, SHA
CUDA.functional() || error("A functional CUDA device is required")
CUDA.allowscalar(false)
ext=Base.get_extension(LifeAI,:LifeAICUDAExt)
function benchmark_flint_long_attention()
    rows=Any[]
    for (keys,queries,window) in ((16,1,0),(128,1,0),(513,1,513),(2048,1,0),
            (8192,1,0),(262144,1,0),(262144,128,0),
            (262144,4096,0),(262144,4096,513))
        q=CUDA.ones(Float32,128,32,queries)
        k=CUDA.zeros(Float32,128,4,keys)
        v=CUDA.ones(Float32,128,4,keys)
        sinks=CUDA.zeros(Float32,32)
        output=LifeAI._flint_attention_long(q,k,v,sinks,0,keys-queries,window)
        CUDA.synchronize()
        samples=Float64[]
        for repeat in 1:3
            push!(samples,CUDA.@elapsed output=LifeAI._flint_attention_long(q,k,v,sinks,0,keys-queries,window))
        end
        actual=Array(output)
        count=window>0 ? min(window,keys) : keys
        @assert all(isfinite,actual)
        @assert abs(actual[1,1,end]-count/(count+1))<5e-5
        plan=queries>=16 ? ext._flint_gemm_attention_plan(128,32,4,queries,keys) :
            ext._flint_long_attention_plan(128,32,queries,keys,window)
        old_context_seconds=nothing
        if queries==1 && keys<=8192
            reference=LifeAI._flint_attention_context(q,k,v,sinks,nothing)
            CUDA.synchronize()
            @assert Array(reference)≈actual atol=3e-5 rtol=3e-5
            old_context_seconds=minimum([CUDA.@elapsed LifeAI._flint_attention_context(q,k,v,sinks,nothing) for repeat in 1:3])
        end
        row=(;keys,queries,window,samples,minimum_seconds=minimum(samples),
            scratch_bytes=plan.scratch_bytes,old_context_seconds)
        push!(rows,row);println(JSON3.write(row));flush(stdout)
        q=k=v=sinks=output=nothing
        GC.gc(true);CUDA.reclaim()
    end
    source=joinpath(pkgdir(LifeAI),"ext","flint_long_attention.jl")
    report=(;device=string(CUDA.device()),source_sha256=bytes2hex(sha256(read(source))),rows)
    path=isempty(ARGS) ? joinpath(@__DIR__,"attention-results.json") : ARGS[1]
    open(path,"w") do file
        JSON3.write(file,report)
    end
end
benchmark_flint_long_attention()
