#!/usr/bin/env julia
using LifeAI,CUDA,NCCL,JSON3,HTTP,LinearAlgebra,Sockets,UUIDs,Dates,Test
using BFloat16s:BFloat16
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_tensor_parallel.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_nccl_transport.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_projection.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_prefix.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_rope.jl"))
include(joinpath(@__DIR__,"..","test","ripple","test_ripple_long_attention_reference.jl"))
include(joinpath(@__DIR__,"benchmark_ripple_tp_variants.jl"))
include(joinpath(@__DIR__,"ripple_tp8_jobs.jl"))

function tp8_options(args)
    options=Dict("model"=>"/jfs/application/user/bgl/models/ricursiveAI/Ripple-65B-Preview",
        "result-dir"=>"benchmark_results/ripple/dlc8/runtime","host"=>"0.0.0.0","port"=>"18085",
        "gpus"=>"8","max-context"=>get(ENV,"LIFEAI_MAX_CONTEXT","8192"),"max-output"=>"512",
        "prefill-chunk-size"=>get(ENV,"LIFEAI_PREFILL_CHUNK_SIZE","256"),
        "prefill-query-batch"=>get(ENV,"LIFEAI_PREFILL_QUERY_BATCH","512"),
        "prefill-gc-layers"=>get(ENV,"LIFEAI_PREFILL_GC_LAYERS","8"),
        "max-body-bytes"=>get(ENV,"LIFEAI_MAX_BODY_BYTES",string(128*1024^2)),
        "request-timeout"=>get(ENV,"LIFEAI_REQUEST_TIMEOUT","21600"),
        "prefix-cache"=>get(ENV,"LIFEAI_PREFIX_CACHE","true"),
        "reference"=>"/jfs/application/user/bgl/LifeAI.jl/benchmark_results/ripple/full/official",
        "api-key-file"=>get(ENV,"LIFEAI_API_KEY_FILE",""),
        "projection-mode"=>get(ENV,"LIFEAI_RIPPLE_PROJECTION_MODE","bf16_gemv"),
        "schedule"=>get(ENV,"LIFEAI_RIPPLE_SCHEDULE","merged"),
        "gc-interval"=>get(ENV,"LIFEAI_RIPPLE_GC_INTERVAL","16"),
        "benchmark-variants"=>get(ENV,"LIFEAI_RIPPLE_BENCHMARK_VARIANTS","false"))
    mode="serve";i=1
    while i<=length(args)
        if args[i]=="--test-only";mode="test";i+=1;continue;end
        startswith(args[i],"--") && i<length(args) || error("invalid TP server argument")
        key=args[i][3:end];haskey(options,key) || error("unknown TP option --$key")
        options[key]=args[i+1];i+=2
    end
    return options,mode
end

function tp8_write(path,value)
    temporary=path*".tmp"
    write(temporary,JSON3.write(value));mv(temporary,path;force=true)
end
function tp8_stats(bundle)
    backend=bundle.backend
    return (world_size=length(bundle.ranks),weights_per_rank=[quantized_parameter_bytes(r.parameters) for r in bundle.ranks],
        allreduce_rank_calls=backend.reductions[],allgather_rank_calls=backend.gathers[],
        gc_interval=backend.gc_interval,gc_runs=backend.gc_runs,
        logical_allreduces=backend.reductions[]÷length(bundle.ranks),logical_allgathers=backend.gathers[]÷length(bundle.ranks))
end

function tp8_validate(bundle,tokenizer,reference_dir,result_dir;schedule=:merged)
    reference=JSON3.read(read(joinpath(reference_dir,"reference.json"),String))
    get(reference,"attention_backend",nothing)=="math" || error("TP validation requires the mathematical PyTorch reference")
    oracle=open_safetensors_reader(joinpath(reference_dir,"reference.safetensors"));records=Any[]
    for (number,case) in enumerate(reference["cases"])
        ids=Int.(case["input_ids"]).+1;cache=RippleTPKVCache(bundle);started=time()
        haskey(case,"prompt") && @test encode(tokenizer,case["prompt"])==ids
        logits=Array(hf_ripple_tp_forward(bundle,ids;cache,last_only=true,schedule))[:,1]
        expected=read_safetensors_tensor(oracle,"case$(number-1).prefill_logits")
        maximum_error=maximum(abs.(logits.-expected));relative=norm(logits-expected)/norm(expected)
        @test maximum_error<1f-3
        @test relative<1f-4
        @test argmax(logits)==argmax(expected)
        generated=Int[]
        for (step,expected_token) in enumerate(case["greedy_tokens"])
            token=argmax(logits);push!(generated,token);@test token==Int(expected_token)+1
            if step<length(case["greedy_tokens"])
                logits=Array(hf_ripple_tp_forward(bundle,[token];cache,last_only=true,schedule))[:,1]
            end
        end
        record=(name=String(case["name"]),prompt_tokens=length(ids),max_absolute_error=maximum_error,
            relative_l2_error=relative,argmax_matches=true,greedy_tokens=generated,seconds=time()-started)
        push!(records,record);println(JSON3.write((phase="parity",record=record)));flush(stdout)
        tp8_write(joinpath(result_dir,"parity.json"),(attention_backend="PyTorch MATH",cases=records,stats=tp8_stats(bundle)))
    end
    return records
end

function tp8_smoke(bundle,tokenizer,result_dir,chunk_size;schedule=:merged,projection_mode=:bf16_gemv)
    cases=[(name="exact_ok",text="Reply with exactly OK.",limit=32),
        (name="math",text="计算 17 × 23，只回答结果。",limit=64),
        (name="code",text="Write a Python function add(a, b). Reply only with code.",limit=128),
        (name="chinese",text="请用一句中文解释什么是二分查找。",limit=96),
        (name="throughput",text="Explain step by step how binary search works and provide a Python implementation.",limit=96)]
    records=Any[]
    for case in cases
        prompt=apply_ripple_chat_template(tokenizer,[(role="user",content=case.text)])*"<think>\n\n</think>\n"
        ids=encode(tokenizer,prompt);times=Float64[];prefill_end=Ref(0.0);started=time()
        tokens=generate_ripple_tp_tokens(bundle,ids;temperature=0,max_new_tokens=case.limit,prefill_chunk_size=chunk_size,
            on_prefill=(done,total)->(done==total && (prefill_end[]=time());true),on_token=id->(push!(times,time());true),schedule)
        decode_seconds=length(times)>1 ? times[end]-times[1] : 0.0
        record=(name=case.name,prompt=case.text,prompt_tokens=length(ids),token_ids=tokens,
            text=decode(tokenizer,tokens;skip_special_tokens=true,errors=:replace),generated_tokens=length(tokens),
            ended_with_eos=!isempty(tokens) && last(tokens) in bundle.model.eos_token_id.+1,
            duration_seconds=time()-started,prefill_seconds=prefill_end[]-started,
            decode_seconds=decode_seconds,decode_tokens_per_second=decode_seconds>0 ? (length(tokens)-1)/decode_seconds : nothing)
        push!(records,record);println(JSON3.write((phase="smoke",record=record)));flush(stdout)
        tp8_write(joinpath(result_dir,"generation.json"),(tensor_parallel=length(bundle.ranks),weights="original BF16 checkpoint values",projection_mode=string(projection_mode),activations="Float32",cases=records,stats=tp8_stats(bundle)))
    end
    @test records[1].text=="OK"
    @test occursin(r"(?<!\d)391(?!\d)",records[2].text)
    @test records[5].generated_tokens>=64
    return records
end

function tp8_main(args)
    options,mode=tp8_options(args);count=parse(Int,options["gpus"]);port=parse(Int,options["port"])
    max_context=parse(Int,options["max-context"]);max_output=parse(Int,options["max-output"])
    chunk_size=parse(Int,options["prefill-chunk-size"])
    query_batch=parse(Int,options["prefill-query-batch"]);prefill_gc_layers=parse(Int,options["prefill-gc-layers"])
    max_body=parse(Int,options["max-body-bytes"]);request_timeout=parse(Float64,options["request-timeout"])
    prefix_enabled=options["prefix-cache"]=="true"
    1<=count<=length(CUDA.devices()) || error("Requested $count GPUs, found $(length(CUDA.devices()))")
    0<max_output<max_context && chunk_size>0 || error("invalid TP serving limits")
    CUDA.allowscalar(false);BLAS.set_num_threads(1)
    result_dir=abspath(options["result-dir"]);mkpath(result_dir)
    devices=collect(CUDA.devices())[1:count]
    hardware=Any[]
    for dev in devices
        CUDA.device!(dev) do
            push!(hardware,(device=CUDA.deviceid(dev),name=CUDA.name(dev),memory_bytes=CUDA.total_memory()))
        end
    end
    println(JSON3.write((phase="hardware",devices=hardware,threads=Threads.nthreads(),nccl=string(NCCL.version()))));flush(stdout)
    # Prove real multi-GPU collectives and the complete partition equations
    # before spending time loading 130GB of source tensors.
    @testset "Physical NCCL tensor parallelism" begin
        ripple_rope_reference_checks(device=CUDA.cu,reference_path=joinpath(@__DIR__,"..","test","ripple","fixtures","rope","rope-cuda.json"))
        ripple_long_attention_reference_checks()
        ripple_nccl_transport_checks(devices)
        ripple_tp_projection_checks(devices)
        ripple_tp_prefix_checks(n->ripple_nccl_backend(devices[1:n]);world_sizes=(count,))
        ripple_tp_numerical_checks(n->ripple_nccl_backend(devices[1:n]);world_sizes=(count,))
    end
    gc_interval=parse(Int,options["gc-interval"])
    backend=ripple_nccl_backend(devices;gc_interval);bundle=nothing;job_manager=nothing
    try
        started=time()
        bundle=load_hf_ripple_tensor_parallel(options["model"];backend,dtype=BFloat16,
            on_progress=(name,index,total)->begin
                if index%50==0 || index==total
                    println(JSON3.write((phase="loading",index=index,total=total,tensor=name,seconds=time()-started)));flush(stdout)
                end
            end)
        load_seconds=time()-started;tokenizer=load_hf_ripple_tokenizer(options["model"])
        projection_mode=Symbol(options["projection-mode"]);schedule=Symbol(options["schedule"])
        options["benchmark-variants"] in ("true","false") || error("--benchmark-variants must be true or false")
        if options["benchmark-variants"]=="true"
            selected,benchmark=tp8_benchmark_variants(bundle,tokenizer,options["reference"],result_dir)
            projection_mode=selected.projection;schedule=selected.schedule;gc_interval=selected.gc_interval
            println(JSON3.write((phase="selected_variant",projection_mode=string(projection_mode),schedule=string(schedule),speedup=benchmark.speedup)));flush(stdout)
        else
            prepare_ripple_tp_projections!(bundle;mode=projection_mode)
        end
        projection_dtype=projection_mode===:fp32 ? "Float32" : "BF16"
        parity=tp8_validate(bundle,tokenizer,options["reference"],result_dir;schedule)
        smokes=tp8_smoke(bundle,tokenizer,result_dir,chunk_size;schedule,projection_mode)
        receipt=(phase="validated",world_size=count,model="ricursiveAI/Ripple-65B-Preview",weights="original BF16 checkpoint values",projection_dtype=projection_dtype,projection_mode=string(projection_mode),schedule=string(schedule),gc_interval=gc_interval,activations="Float32",
            hardware=hardware,nccl=string(NCCL.version()),load_seconds=load_seconds,parity=parity,stats=tp8_stats(bundle),
            code_revision=get(ENV,"LIFEAI_CODE_REVISION","unknown"),validated_at=string(now(UTC)))
        tp8_write(joinpath(result_dir,"validation.json"),receipt)
        mode=="test" && return
        key=isempty(options["api-key-file"]) ? "" : strip(read(options["api-key-file"],String))
        model_id="lifeai-ripple65b-tp$count";model_lock=ReentrantLock();requests=Threads.Atomic{Int}(0)
        prefix_store=prefix_enabled ? RippleTPPrefixCache(bundle;max_tokens=max_context) : nothing
        function validate_payload(payload)
            get(payload,"stream",false)===false || throw(ArgumentError("This endpoint supports non-streaming text completions"))
            get(payload,"n",1)==1 || throw(ArgumentError("n must be 1"))
            model=get(payload,"model",model_id)
            model in (model_id,"ricursiveAI/Ripple-65B-Preview") || throw(ArgumentError("unknown model"))
            messages=get(payload,"messages",nothing)
            messages isa AbstractVector && 1<=length(messages)<=64 || throw(ArgumentError("messages must contain 1..64 entries"))
            total_chars=0
            for message in messages
                message isa AbstractDict && get(message,"role",nothing) in ("system","user","assistant") && get(message,"content",nothing) isa AbstractString || throw(ArgumentError("messages require supported roles and text content"))
                total_chars+=length(message["content"])
            end
            total_chars<=64*1024^2 || throw(ArgumentError("chat exceeds the64Mi-character text limit"))
            tools=get(payload,"tools",nothing)
            (tools===nothing || tools isa AbstractVector && isempty(tools)) || throw(ArgumentError("This text endpoint does not expose structured tool calls"))
            limit=get(payload,"max_tokens",128)
            limit isa Integer && !(limit isa Bool) && 0<limit<=max_output || throw(ArgumentError("invalid max_tokens"))
            temperature=get(payload,"temperature",0.0)
            temperature isa Real && !(temperature isa Bool) && isfinite(temperature) && 0<=temperature<=2 || throw(ArgumentError("invalid temperature"))
            context_limit=get(payload,"context_limit",max_context)
            context_limit isa Integer && !(context_limit isa Bool) && limit<context_limit<=max_context || throw(ArgumentError("invalid context_limit"))
            return (messages=messages,limit=Int(limit),temperature=Float64(temperature),context_limit=Int(context_limit))
        end
        function execute(input,progress,cancelled)
            lock(model_lock) do
                started=time();times=Float64[];prefill_end=Ref(started)
                progress("tokenizing",0,0)
                prompt=apply_ripple_chat_template(tokenizer,input.messages);ids=encode(tokenizer,prompt)
                length(ids)+input.limit<=input.context_limit || throw(ArgumentError("request exceeds context limit"))
                cancelled() && throw(InterruptException())
                progress("prefill",0,length(ids))
                last_report=Ref(0)
                on_prefill=(done,total)->begin
                    progress("prefill",done,total)
                    done==total && (prefill_end[]=time())
                    if done==total || done-last_report[]>=16384
                        println(JSON3.write((phase="prefill",processed=done,total=total,seconds=time()-started)));flush(stdout)
                        last_report[]=done
                    end
                    !cancelled()
                end
                on_token=id->begin
                    push!(times,time());progress("decode",length(ids),length(ids));!cancelled()
                end
                try
                    tokens=generate_ripple_tp_tokens(bundle,ids;max_new_tokens=input.limit,temperature=input.temperature,
                        prefill_chunk_size=chunk_size,on_token,schedule,on_prefill,prefix_cache=prefix_store,
                        prefill_query_batch=query_batch,prefill_gc_layers)
                    cancelled() && throw(InterruptException())
                    text=decode(tokenizer,tokens;skip_special_tokens=true,errors=:replace);Threads.atomic_add!(requests,1)
                    ended=!isempty(tokens) && last(tokens) in bundle.model.eos_token_id.+1
                    prefix=prefix_store===nothing ? nothing : ripple_tp_prefix_stats(prefix_store)
                    return (id="chatcmpl-"*string(uuid4()),object="chat.completion",created=floor(Int,time()),model=model_id,
                        choices=[(index=0,message=(role="assistant",content=text),finish_reason=ended ? "stop" : "length")],
                        usage=(prompt_tokens=length(ids),completion_tokens=length(tokens),total_tokens=length(ids)+length(tokens)),
                        lifeai=(tensor_parallel=count,weight_dtype=projection_dtype,activation_dtype="Float32",projection_mode=string(projection_mode),schedule=string(schedule),gc_interval=gc_interval,
                            duration_seconds=time()-started,prefill_seconds=prefill_end[]-started,prefix_cache=prefix,
                            decode_tokens_per_second=length(times)>1 ? (length(times)-1)/(times[end]-times[1]) : nothing))
                catch error
                    prefix_store===nothing || reset_ripple_tp_prefix!(prefix_store)
                    if backend.closed
                        tp8_write(joinpath(result_dir,"failure.json"),(phase="failed",error=sprint(showerror,error)))
                        @async exit(1)
                    end
                    rethrow()
                end
            end
        end
        health=()->(status="ready",model=model_id,tensor_parallel=count,requests=requests[],max_context=max_context,max_output=max_output)
        handler,job_manager=ripple_tp_http_handler(execute,validate_payload;key,model_id,health,max_body,timeout=request_timeout)
        stream_handler=ripple_tp_http_stream_handler(handler;key,max_body)
        server=HTTP.serve!(stream_handler,options["host"],port;stream=true,verbose=false)
        addresses=string.(filter(ip->ip isa IPv4 && !startswith(string(ip),"169.254."),getipaddrs()))
        isempty(addresses) && error("No routable IPv4 address available for the TP service")
        preferred=get(ENV,"POD_IP","")
        advertised=preferred in addresses ? preferred : first(addresses)
        endpoint="http://"*advertised*":$port"
        ready=merge(receipt,(phase="ready",endpoint=endpoint,model_id=model_id,max_context=max_context,max_output=max_output,
            addresses=addresses,port=port,streaming=false,async_jobs=true,prefix_cache=prefix_enabled,prefill_chunk_size=chunk_size,prefill_query_batch=query_batch,prefill_gc_layers=prefill_gc_layers,max_request_bytes=max_body,request_timeout=request_timeout,authentication=isempty(key) ? "none" : "bearer",api_key_file=options["api-key-file"]))
        tp8_write(joinpath(result_dir,"ready.json"),ready)
        println(JSON3.write((phase="ready",endpoint=endpoint,model=model_id,tensor_parallel=count)));flush(stdout)
        wait(server)
    finally
        job_manager===nothing || close_ripple_http_jobs!(job_manager)
        close_ripple_tp_backend!(backend)
    end
end

tp8_main(ARGS)
