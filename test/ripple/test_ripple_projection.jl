using BFloat16s:BFloat16
# Caller provides CUDA and NCCL; compare exactly the same BF16-valued model.
function ripple_tp_projection_checks(devices)
    cpu_backend=RippleTPReferenceBackend(1);backend=ripple_nccl_backend(devices)
    try
        cpu=load_hf_ripple_tensor_parallel(RIPPLE_TP_FIXTURE;backend=cpu_backend,dtype=BFloat16)
        gpu=load_hf_ripple_tensor_parallel(RIPPLE_TP_FIXTURE;backend,dtype=BFloat16)
        ids=Int.(JSON3.read(read(joinpath(RIPPLE_TP_FIXTURE,"reference.json"),String))["input_ids"]).+1
        expected=Array(hf_ripple_tp_forward(cpu,ids))
        greedy=generate_ripple_tp_tokens(cpu,ids;max_new_tokens=4,eos_ids=Int[],temperature=0)
        baseline=quantized_parameter_bytes(gpu.ranks[1].parameters)
        for mode in (:reference,:bf16_gemv,:fp32,:reference)
            @test prepare_ripple_tp_projections!(gpu;mode)===gpu
            @test eltype(gpu.ranks[1].parameters["model.embed_tokens.weight"])===BFloat16
            for schedule in (:staged,:merged)
                @test Array(hf_ripple_tp_forward(gpu,ids;schedule))≈expected atol=8f-5 rtol=8f-5
                cache=RippleTPKVCache(gpu)
                hf_ripple_tp_forward(gpu,ids[1:end-1];cache,output_logits=false,schedule)
                @test Array(hf_ripple_tp_forward(gpu,ids[end:end];cache,schedule))≈expected[:,end:end] atol=8f-5 rtol=8f-5
                @test generate_ripple_tp_tokens(gpu,ids;max_new_tokens=4,eos_ids=Int[],temperature=0,schedule)==greedy
            end
            bytes=quantized_parameter_bytes(gpu.ranks[1].parameters)
            @test mode===:fp32 ? bytes>baseline : bytes==baseline
        end
        @test_throws ArgumentError prepare_ripple_tp_projections!(gpu;mode=:unknown)
    finally
        close_ripple_tp_backend!(backend);close_ripple_tp_backend!(cpu_backend)
    end
end
