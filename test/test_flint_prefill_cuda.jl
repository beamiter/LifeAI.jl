using Test, LifeAI, CUDA

@testset "CUDA chunked Flint forward preserves full-prompt logits and committed KV" begin
    CUDA.functional() || error("Flint prefill tests require CUDA")
    CUDA.allowscalar(false)
    fixture=joinpath(@__DIR__,"flint","fixtures","tiny")
    host=load_hf_flint_model(fixture)
    device=load_hf_flint_model(fixture;device=CUDA.cu)
    ids=[1,3,5,7,9,11,13,15,17]
    expected=hf_flint_forward(host.model,host.parameters,ids;last_only=true)
    for chunk in (1,2,4,9)
        cache=FlintKVCache(device.model)
        result=nothing
        for first in 1:chunk:length(ids)
            last=min(length(ids),first+chunk-1)
            result=hf_flint_forward(device.model,device.parameters,ids[first:last];cache,
                cache_capacity=32,last_only=true,output_logits=(last==length(ids)))
        end
        @test Array(result)≈expected atol=3f-5 rtol=3f-5
        @test cache.tokens==length(ids)
        reference=hf_flint_forward(host.model,host.parameters,[ids;19];last_only=true)
        @test Array(hf_flint_forward(device.model,device.parameters,[19];cache,
            cache_capacity=32,last_only=true))≈reference atol=3f-5 rtol=3f-5
        previous=[(key=Array(layer.key),value=Array(layer.value)) for layer in cache.layers]
        without_head=copy(device.parameters)
        delete!(without_head,"lm_head.weight")
        @test_throws KeyError hf_flint_forward(device.model,without_head,[21];cache,
            cache_capacity=32,last_only=true)
        @test cache.tokens==length(ids)+1
        @test all(Array(layer.key)==old.key && Array(layer.value)==old.value
            for (layer,old) in zip(cache.layers,previous))
        reference=hf_flint_forward(host.model,host.parameters,[ids;19;21];last_only=true)
        @test Array(hf_flint_forward(device.model,device.parameters,[21];cache,
            cache_capacity=32,last_only=true))≈reference atol=3f-5 rtol=3f-5
    end
end
