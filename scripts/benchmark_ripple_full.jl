#!/usr/bin/env julia
using LifeAI,CUDA,JSON3,Random,LinearAlgebra,Test
function main(args)
    length(args) in (2,3,4) || error("Usage: benchmark_ripple_full.jl ARTIFACT_DIR OUTPUT_JSON [RESIDENT_GIB] [REFERENCE_DIR]")
    CUDA.functional() || error("CUDA unavailable")
    CUDA.allowscalar(false)
    path,destination=args[1:2]
    budget=round(Int,(length(args)>=3 ? parse(Float64,args[3]) : 12.0)*2.0^30)
    started=time()
    bundle=load_ripple_quantized(path;device=CUDA.cu,resident_bytes=budget,
        on_progress=(name,index,total)->(index%100==0 && (println("Loaded $index/$total: $name");flush(stdout))))
    load_seconds=time()-started
    tokenizer=load_hf_ripple_tokenizer(path)
    function prompt_ids(text)
        prompt=apply_ripple_chat_template(tokenizer,[(role="user",content=text)])*"<think>\n\n</think>\n"
        return encode(tokenizer,prompt)
    end
    println("WARMUP_START");flush(stdout)
    generate_ripple_tokens(bundle,prompt_ids("Reply OK.");temperature=0,max_new_tokens=2)
    println("WARMUP_COMPLETE");flush(stdout)
    cases=[(name="exact_ok",prompt="Reply with exactly OK.",max_new_tokens=32),
        (name="math",prompt="计算 17 × 23，只回答结果。",max_new_tokens=64),
        (name="code",prompt="Write a Python function add(a, b). Reply only with code.",max_new_tokens=128),
        (name="chinese",prompt="请用一句中文解释什么是二分查找。",max_new_tokens=96),
        (name="throughput",prompt="Explain step by step how binary search works and provide a Python implementation.",max_new_tokens=96)]
    records=Any[]
    for case in cases
        ids=prompt_ids(case.prompt)
        token_times=Float64[];prefill_end=Ref(0.0);started=time()
        generated=generate_ripple_tokens(bundle,ids;temperature=0,max_new_tokens=case.max_new_tokens,
            prefill_chunk_size=256,on_prefill=(done,total)->(done==total && (prefill_end[]=time());true),
            on_token=id->(push!(token_times,time());true))
        CUDA.synchronize()
        duration=time()-started
        text=decode(tokenizer,generated;skip_special_tokens=true,errors=:replace)
        decode_seconds=length(token_times)>1 ? token_times[end]-token_times[1] : 0.0
        record=(name=case.name,prompt=case.prompt,prompt_tokens=length(ids),token_ids=generated,
            text=text,generated_tokens=length(generated),ended_with_eos=!isempty(generated) && last(generated) in bundle.model.eos_token_id.+1,
            duration_seconds=duration,prefill_seconds=prefill_end[]-started,
            time_to_first_token_seconds=isempty(token_times) ? nothing : first(token_times)-started,
            decode_seconds=decode_seconds,decode_tokens_per_second=decode_seconds>0 ? (length(token_times)-1)/decode_seconds : nothing,
            offload=ripple_offload_stats(bundle))
        push!(records,record)
        println(JSON3.write(record));flush(stdout)
        write(destination,JSON3.write((artifact=abspath(path),scheme=bundle.quantization,load_seconds=load_seconds,
            resident_budget_bytes=budget,decode_backend="cuda_packed_int4_fp32_gemv",checksum_backend=bundle.checksum_backend,cases=records)))
    end
    reference_dir=length(args)==4 ? args[4] : joinpath(dirname(destination),"official")
    if isfile(joinpath(reference_dir,"reference.json"))
        reference=JSON3.read(read(joinpath(reference_dir,"reference.json"),String))
        case=only(filter(c->c["name"]=="long_window",reference["cases"]))
        ids=Int.(case["input_ids"]).+1
        started=time()
        full=Array(hf_ripple_forward(bundle.model,bundle.parameters,ids;last_only=true))[:,1]
        cache=RippleKVCache(bundle.model)
        chunked=nothing
        for first in 1:256:length(ids)
            last=min(first+255,length(ids))
            chunked=hf_ripple_forward(bundle.model,bundle.parameters,ids[first:last];cache,
                last_only=true,output_logits=last==length(ids))
        end
        chunked=Array(chunked)[:,1]
        @test maximum(abs.(full.-chunked))<1f-3
        @test argmax(full)==argmax(chunked)
        oracle=open_safetensors_reader(joinpath(reference_dir,"reference.safetensors"))
        expected=read_safetensors_tensor(oracle,"case$(findfirst(c->c["name"]=="long_window",reference["cases"])-1).prefill_logits")
        record=(name="long_context",prompt_tokens=length(ids),chunk_tokens=256,
            chunk_max_absolute_error=maximum(abs.(full.-chunked)),chunk_argmax_matches=true,
            quantized_vs_float32_cosine=dot(full,expected)/(norm(full)*norm(expected)),
            quantized_vs_float32_relative_error=norm(full-expected)/norm(expected),
            quantized_argmax_matches_float32=argmax(full)==argmax(expected),
            duration_seconds=time()-started,offload=ripple_offload_stats(bundle))
        push!(records,record)
        println(JSON3.write(record));flush(stdout)
        write(destination,JSON3.write((artifact=abspath(path),scheme=bundle.quantization,load_seconds=load_seconds,
            resident_budget_bytes=budget,decode_backend="cuda_packed_int4_fp32_gemv",checksum_backend=bundle.checksum_backend,cases=records)))
    end
    println("REAL_GENERATION_COMPLETE")
end
main(ARGS)
