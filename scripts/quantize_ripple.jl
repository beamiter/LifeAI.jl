#!/usr/bin/env julia
using LifeAI,CUDA,JSON3
using BFloat16s:BFloat16
function main(args)
    length(args) in (2,3,4) || error("Usage: quantize_ripple.jl HF_CHECKPOINT OUTPUT [int4|int8|nvfp4] [GROUP]")
    CUDA.functional() || error("CUDA unavailable")
    CUDA.allowscalar(false)
    scheme=length(args)>=3 ? Symbol(args[3]) : :int4
    group=length(args)>=4 ? parse(Int,args[4]) : nothing
    started=time()
    quantize_ripple_checkpoint(args[1],args[2];device=CUDA.cu,scheme,group,dtype=BFloat16,
        on_progress=(name,index,total)->begin
            if index%10==0 || index==total
                println(JSON3.write((phase="quantize",tensor=name,index=index,total=total,seconds=time()-started)))
                flush(stdout)
            end
        end)
    println(JSON3.write((phase="complete",path=abspath(args[2]),seconds=time()-started)))
end
main(ARGS)
