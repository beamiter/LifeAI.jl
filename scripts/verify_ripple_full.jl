#!/usr/bin/env julia
using LifeAI,CUDA,JSON3,LinearAlgebra,Test
using BFloat16s:BFloat16
function main(args)
    length(args) in (3,4) || error("Usage: verify_ripple_full.jl CHECKPOINT REFERENCE_DIR OUTPUT_JSON [CASE]")
    CUDA.functional() || error("CUDA unavailable")
    CUDA.allowscalar(false)
    checkpoint,reference_dir,destination=args[1:3]
    reference=JSON3.read(read(joinpath(reference_dir,"reference.json"),String))
    reader=open_safetensors_reader(joinpath(reference_dir,"reference.safetensors"))
    bundle=load_hf_ripple_streamed(checkpoint;device=CUDA.cu,dtype=BFloat16)
    tokenizer=load_hf_ripple_tokenizer(checkpoint)
    records=Any[]
    for (number,case) in enumerate(reference["cases"])
        length(args)==4 && case["name"]!=args[4] && continue
        ids=Int.(case["input_ids"]) .+ 1
        haskey(case,"prompt") && @test encode(tokenizer,case["prompt"])==ids
        cache=RippleKVCache(bundle.model)
        started=time()
        actual=Array(hf_ripple_forward(bundle.model,bundle.parameters,ids;cache,last_only=true))[:,1]
        expected=read_safetensors_tensor(reader,"case$(number-1).prefill_logits")
        error=maximum(abs.(actual.-expected));relative=norm(actual-expected)/norm(expected)
        println(JSON3.write((phase="comparison",name=case["name"],max_absolute_error=error,relative_l2_error=relative,
            actual_argmax=argmax(actual),expected_argmax=argmax(expected))));flush(stdout)
        @test error<1f-3
        @test relative<1f-4
        @test argmax(actual)==argmax(expected)
        generated=Int[]
        for (step,token) in enumerate(case["greedy_tokens"])
            expected_step=read_safetensors_tensor(reader,"case$(number-1).step$(step-1)_logits")
            @test maximum(abs.(actual.-expected_step))<1f-3
            chosen=argmax(actual);push!(generated,chosen)
            @test chosen==Int(token)+1
            if step<length(case["greedy_tokens"])
                actual=Array(hf_ripple_forward(bundle.model,bundle.parameters,[chosen];cache,last_only=true))[:,1]
            end
        end
        record=(name=String(case["name"]),prompt_tokens=length(ids),max_absolute_error=error,
            relative_l2_error=relative,argmax_matches=true,greedy_tokens=generated,
            seconds=time()-started,offload=ripple_offload_stats(bundle))
        push!(records,record)
        println(JSON3.write(record));flush(stdout)
        write(destination,JSON3.write((checkpoint=abspath(checkpoint),revision=reference["revision"],
            activation_dtype="float32",weight_storage="original_bf16",cases=records)))
        cache=nothing;GC.gc(true);CUDA.reclaim()
    end
    println("FULL_PRECISION_PARITY_COMPLETE")
end
main(ARGS)
