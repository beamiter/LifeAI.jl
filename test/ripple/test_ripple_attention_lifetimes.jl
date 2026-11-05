using Test,LifeAI,CUDA,JSON3,SHA
if !isdefined(@__MODULE__,:_ripple_attention_reference_input)
    include(joinpath(@__DIR__,"test_ripple_long_attention_reference.jl"))
end

function ripple_attention_lifetime_references(directory;queries=2048,key_counts=(8193,16385),query_batch=2048)
    mkpath(directory);records=Any[]
    for keys in key_counts
        query=_ripple_attention_reference_input((128,10,queries),97)
        key=_ripple_attention_reference_input((128,1,keys),17)
        value=_ripple_attention_reference_input((128,1,keys),43);sinks=CUDA.fill(-Inf32,10)
        output=LifeAI._flint_attention_long(query,key,value,sinks,0,keys-queries,0;max_query_batch=query_batch)
        host=Array(output);name="queries$(queries)-keys$(keys).f32";write(joinpath(directory,name),host)
        push!(records,(queries=queries,keys=keys,query_batch=query_batch,file=name,sha256=bytes2hex(sha256(reinterpret(UInt8,vec(host))))))
        for array in (output,query,key,value,sinks);CUDA.unsafe_free!(array);end
        GC.gc(true);CUDA.synchronize();CUDA.reclaim()
    end
    metadata=(source_sha256=bytes2hex(sha256(read(joinpath(@__DIR__,"..","..","ext","flint_long_attention.jl")))),cases=records)
    write(joinpath(directory,"reference.json"),JSON3.write(metadata));return metadata
end

function _ripple_lifetime_pool_used()
    CUDA.synchronize()
    return CUDA.used_memory()
end

"""Stress large GEMM attention with CPU GC disabled and verify old-output parity.

Reference files must be exported from the pre-lifetime-change implementation.
Scores are640MiB per call at2048 queries, so repeated calls expose retained
scratch independently of allocator-reserved cache or CPU finalizer scheduling.
"""
function ripple_attention_lifetime_checks(directory;repeats=12)
    metadata=JSON3.read(read(joinpath(directory,"reference.json"),String));records=Any[]
    for case in metadata["cases"]
        queries,keys,query_batch=Int(case["queries"]),Int(case["keys"]),Int(case["query_batch"])
        bytes=read(joinpath(directory,String(case["file"])));@test bytes2hex(sha256(bytes))==case["sha256"]
        expected=reshape(copy(reinterpret(Float32,bytes)),128,10,queries)
        query=_ripple_attention_reference_input((128,10,queries),97)
        key=_ripple_attention_reference_input((128,1,keys),17)
        value=_ripple_attention_reference_input((128,1,keys),43);sinks=CUDA.fill(-Inf32,10)
        try
            warm=LifeAI._flint_attention_long(query,key,value,sinks,0,keys-queries,0;max_query_batch=query_batch)
            @test Array(warm)≈expected atol=2f-6 rtol=2f-6
            CUDA.unsafe_free!(warm);GC.gc(true);CUDA.synchronize();CUDA.reclaim()
            baseline=_ripple_lifetime_pool_used();samples=Int[];errors=Float32[]
            enabled=GC.enable(false)
            try
                for _ in 1:repeats
                    output=LifeAI._flint_attention_long(query,key,value,sinks,0,keys-queries,0;max_query_batch=query_batch)
                    actual=Array(output);err=maximum(abs.(actual.-expected));push!(errors,err)
                    @test actual≈expected atol=2f-6 rtol=2f-6
                    CUDA.unsafe_free!(output)
                    push!(samples,_ripple_lifetime_pool_used())
                    @test samples[end]<=baseline+64*1024^2
                end
            finally
                GC.enable(enabled)
            end
            # Force mixed finalizer schedules after the no-GC stress.
            for full in (false,true,false)
                GC.gc(full)
                output=LifeAI._flint_attention_long(query,key,value,sinks,0,keys-queries,0;max_query_batch=query_batch)
                @test Array(output)≈expected atol=2f-6 rtol=2f-6
                CUDA.unsafe_free!(output)
            end
            # Host argument conversion fails after allocating/enqueuing scratch;
            # finally must release it without freeing caller tensors or output.
            before=_ripple_lifetime_pool_used()
            extension=Base.get_extension(LifeAI,:LifeAICUDAExt)
            @test_throws InexactError extension._flint_attention_gemm(query,key,value,sinks,0,Int64(typemax(Int32))+1,0;max_query_batch=query_batch)
            @test _ripple_lifetime_pool_used()<=before+64*1024^2
            @test Array(vec(query)[1:16])==[_ripple_attention_reference_hash(UInt32(i),UInt32(97)) for i in 0:15]
            push!(records,(queries=queries,keys=keys,repeats=repeats,baseline_live_bytes=baseline,
                post_call_live_bytes=samples,max_absolute_error=maximum(errors)))
            println(JSON3.write(last(records)));flush(stdout)
        finally
            CUDA.synchronize()
            for array in (query,key,value,sinks);CUDA.unsafe_free!(array);end
            GC.gc(true);CUDA.synchronize();CUDA.reclaim()
        end
    end
    return records
end

function ripple_attention_lifetime_layout_checks()
    extension=Base.get_extension(LifeAI,:LifeAICUDAExt);records=Any[]
    for (dim,heads,kvheads,queries,keys,window,batch) in ((16,4,2,17,37,0,16),(32,8,2,33,71,9,16),(128,10,1,1025,2049,0,512))
        query=_ripple_attention_reference_input((dim,heads,queries),97)
        key=_ripple_attention_reference_input((dim,kvheads,keys),17)
        value=_ripple_attention_reference_input((dim,kvheads,keys),43);sinks=CUDA.fill(-Inf32,heads)
        expected_device=extension._flint_attention_splitk(query,key,value,sinks,0,keys-queries,window)
        expected=Array(expected_device);CUDA.unsafe_free!(expected_device)
        try
            actual_device=LifeAI._flint_attention_long(query,key,value,sinks,0,keys-queries,window;max_query_batch=batch)
            actual=Array(actual_device)
            @test actual≈expected atol=2f-6 rtol=2f-6
            push!(records,(dim=dim,heads=heads,kvheads=kvheads,queries=queries,keys=keys,window=window,
                query_batch=batch,max_absolute_error=maximum(abs.(actual.-expected))))
            CUDA.unsafe_free!(actual_device)
        finally
            CUDA.synchronize()
            for array in (query,key,value,sinks);CUDA.unsafe_free!(array);end
            GC.gc(true);CUDA.synchronize();CUDA.reclaim()
        end
    end
    return records
end
