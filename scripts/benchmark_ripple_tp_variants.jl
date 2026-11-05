# Included by the TP server after loading the complete original checkpoint.
using Statistics,Serialization

function tp8_variant_run(bundle,ids;name,schedule,limit=96)
    times=Float64[];start=time();gc_before=bundle.backend.gc_runs
    tokens=generate_ripple_tp_tokens(bundle,ids;temperature=0,max_new_tokens=limit,
        eos_ids=Int[],schedule,on_token=id->(push!(times,time());true))
    elapsed=time()-start;decode=times[end]-times[1]
    return (name=name,schedule=string(schedule),tokens=tokens,generated_tokens=length(tokens),
        duration_seconds=elapsed,gc_runs=bundle.backend.gc_runs-gc_before,first_token_seconds=times[1]-start,decode_seconds=decode,
        decode_tokens_per_second=(length(tokens)-1)/decode)
end

function tp8_capture_decode(bundle,ids;steps=16,schedule=:staged)
    cache=RippleTPKVCache(bundle);capacity=length(ids)+steps
    logits=hf_ripple_tp_forward(bundle,ids;cache,last_only=true,cache_capacity=capacity,schedule)
    outputs=Matrix{Float32}[];tokens=Int[]
    for step in 1:steps
        host=Array(logits);push!(outputs,host);token=argmax(host[:,1]);push!(tokens,token)
        step==steps || (logits=hf_ripple_tp_forward(bundle,[token];cache,last_only=true,cache_capacity=capacity,schedule))
    end
    return (;tokens,outputs)
end

function tp8_check_decode(bundle,ids,reference;schedule)
    cache=RippleTPKVCache(bundle);capacity=length(ids)+length(reference.tokens)
    logits=hf_ripple_tp_forward(bundle,ids;cache,last_only=true,cache_capacity=capacity,schedule)
    records=Any[]
    for step in eachindex(reference.tokens)
        host=Array(logits);expected=reference.outputs[step]
        absolute=maximum(abs.(host.-expected));relative=norm(host-expected)/norm(expected)
        @test absolute<1f-3
        @test relative<1f-4
        @test argmax(host[:,1])==reference.tokens[step]
        push!(records,(step=step,max_absolute_error=absolute,relative_l2=relative))
        step==length(reference.tokens) || (logits=hf_ripple_tp_forward(bundle,[reference.tokens[step]];cache,last_only=true,cache_capacity=capacity,schedule))
    end
    return records
end

function tp8_profile_forward(bundle,ids;schedule)
    cache=RippleTPKVCache(bundle);capacity=length(ids)+16
    logits=hf_ripple_tp_forward(bundle,ids;cache,last_only=true,schedule,cache_capacity=capacity)
    times=Dict{Symbol,Float64}();counts=Dict{Symbol,Int}()
    callback=(layer,stage,seconds)->begin
        times[stage]=get(times,stage,0.0)+seconds;counts[stage]=get(counts,stage,0)+1
    end
    token=argmax(Array(logits)[:,1])
    logits=hf_ripple_tp_forward(bundle,[token];cache,last_only=true,schedule,cache_capacity=capacity,on_stage=callback)
    empty!(times);empty!(counts)
    for _ in 1:3
        token=argmax(Array(logits)[:,1])
        logits=hf_ripple_tp_forward(bundle,[token];cache,last_only=true,schedule,cache_capacity=capacity,on_stage=callback)
    end
    return (sampled_decode_steps=3,seconds=times,calls=counts)
end

function tp8_benchmark_variants(bundle,tokenizer,reference_dir,result_dir)
    prompt=apply_ripple_chat_template(tokenizer,[(role="user",content="Explain step by step how binary search works and provide a Python implementation.")])*"<think>\n\n</think>\n"
    ids=encode(tokenizer,prompt)
    variants=[(name="reference",projection=:reference,schedule=:staged,gc_interval=1),
        (name="merged_reference",projection=:reference,schedule=:merged,gc_interval=1),
        (name="bf16_gemv",projection=:bf16_gemv,schedule=:merged,gc_interval=1),
        (name="fp32",projection=:fp32,schedule=:merged,gc_interval=1),
        (name="bf16_gemv_gc16",projection=:bf16_gemv,schedule=:merged,gc_interval=16)]
    reference=nothing;records=Dict{String,Any}();measurements=Dict{String,Vector{Any}}()
    for variant in variants
        bundle.backend.gc_interval=variant.gc_interval;bundle.backend.decode_since_gc=variant.gc_interval
        prepare_ripple_tp_projections!(bundle;mode=variant.projection)
        path=joinpath(result_dir,"variants",variant.name);mkpath(path)
        parity=tp8_validate(bundle,tokenizer,reference_dir,path;schedule=variant.schedule)
        if reference===nothing
            reference=tp8_capture_decode(bundle,ids;schedule=variant.schedule)
            serialize(joinpath(result_dir,"decode-reference.jls"),reference)
        end
        decode_parity=tp8_check_decode(bundle,ids,reference;schedule=variant.schedule)
        profile=tp8_profile_forward(bundle,ids;schedule=variant.schedule)
        tp8_variant_run(bundle,ids;name=variant.name,schedule=variant.schedule,limit=8)
        run=tp8_variant_run(bundle,ids;name=variant.name,schedule=variant.schedule)
        measurements[variant.name]=Any[run]
        records[variant.name]=(projection_mode=string(variant.projection),schedule=string(variant.schedule),gc_interval=variant.gc_interval,
            parity=parity,decode_parity=decode_parity,profile=profile,weights_per_rank=tp8_stats(bundle).weights_per_rank)
        println(JSON3.write((phase="variant_pilot",name=variant.name,run=run)));flush(stdout)
        tp8_write(joinpath(result_dir,"variants.json"),(records=records,measurements=measurements))
    end
    winner=argmax(name->measurements[name][1].decode_tokens_per_second,collect(keys(measurements)))
    # Confirm the original baseline and selected candidate with three runs each,
    # alternating configurations on these same GPUs. Pilot timings remain visible.
    for _ in 1:2, name in unique(["reference",winner])
        variant=only(filter(v->v.name==name,variants))
        bundle.backend.gc_interval=variant.gc_interval;bundle.backend.decode_since_gc=variant.gc_interval
        prepare_ripple_tp_projections!(bundle;mode=variant.projection)
        tp8_variant_run(bundle,ids;name,schedule=variant.schedule,limit=8)
        run=tp8_variant_run(bundle,ids;name,schedule=variant.schedule)
        push!(measurements[name],run)
        println(JSON3.write((phase="variant_repeat",name=name,run=run)));flush(stdout)
        tp8_write(joinpath(result_dir,"variants.json"),(records=records,measurements=measurements))
    end
    speeds=Dict(name=>median([run.decode_tokens_per_second for run in runs]) for (name,runs) in measurements)
    selected=only(filter(v->v.name==winner,variants))
    bundle.backend.gc_interval=selected.gc_interval;bundle.backend.decode_since_gc=selected.gc_interval
    prepare_ripple_tp_projections!(bundle;mode=selected.projection)
    GC.gc(true)
    LifeAI._ripple_tp_compute_ranks(bundle.backend) do rank
        CUDA.reclaim()
    end
    result=(selected=winner,projection_mode=string(selected.projection),schedule=string(selected.schedule),gc_interval=selected.gc_interval,
        median_decode_tokens_per_second=speeds,speedup=speeds[winner]/speeds["reference"],records=records,measurements=measurements)
    tp8_write(joinpath(result_dir,"variants.json"),result)
    return selected,result
end
