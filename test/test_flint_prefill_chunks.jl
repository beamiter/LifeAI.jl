using Test, LifeAI, Random

@testset "Flint bounded prefill and reserved cache interface" begin
    fixture=joinpath(@__DIR__,"flint","fixtures","tiny")
    bundle=load_hf_flint_model(fixture)
    ids=[1,3,5,7,9,11,13,15,17]
    expected=generate_flint_tokens(bundle,ids;max_new_tokens=4,temperature=0,eos_ids=Int[])
    for chunk in (1,2,4,9,4096,typemax(Int))
        progress=Tuple{Int,Int}[]
        observed=generate_flint_tokens(bundle,ids;max_new_tokens=4,temperature=0,
            eos_ids=Int[],prefill_chunk_size=chunk,on_prefill=(n,total)->push!(progress,(n,total)))
        @test observed==expected
        @test last(progress)==(length(ids),length(ids))
        @test issorted(first.(progress))
        @test length(progress)==cld(length(ids),min(chunk,length(ids)))
    end
    # Intermediate chunks need no vocabulary projection or final norm.
    without_head=copy(bundle.parameters)
    delete!(without_head,"lm_head.weight")
    delete!(without_head,"model.norm.weight")
    cache=FlintKVCache(bundle.model)
    @test hf_flint_forward(bundle.model,without_head,ids[1:4];cache,
        cache_capacity=length(ids)+4,output_logits=false)===nothing
    @test cache.tokens==4
    final=hf_flint_forward(bundle.model,bundle.parameters,ids[5:end];cache,last_only=true,
        cache_capacity=length(ids)+4)
    @test final≈hf_flint_forward(bundle.model,bundle.parameters,ids;last_only=true) atol=3f-5 rtol=3f-5
    failed_cache=FlintKVCache(bundle.model)
    @test_throws KeyError hf_flint_forward(bundle.model,without_head,ids[1:2];cache=failed_cache)
    @test failed_cache.tokens==0 && all(isnothing,failed_cache.layers)

    for invalid in (true,-1,bundle.model.max_position_embeddings+1)
        @test_throws ArgumentError hf_flint_forward(bundle.model,bundle.parameters,ids;cache_capacity=invalid)
    end
    for invalid in (true,false,0,-3)
        @test_throws ArgumentError generate_flint_tokens(bundle,ids;max_new_tokens=4,prefill_chunk_size=invalid)
    end
    for invalid in (typemax(Int),big(typemax(Int))+1)
        @test_throws ArgumentError generate_flint_tokens(bundle,ids;max_new_tokens=invalid)
    end
    sampled=Int[]
    progress=Tuple{Int,Int}[]
    canceled=generate_flint_tokens(bundle,ids;prefill_chunk_size=2,max_new_tokens=4,
        on_prefill=(n,total)->(push!(progress,(n,total));false),on_token=n->push!(sampled,n))
    @test isempty(canceled) && isempty(sampled)
    @test progress==[(2,length(ids))]
    @test_throws ErrorException generate_flint_tokens(bundle,ids;max_new_tokens=4,prefill_chunk_size=2,
        on_prefill=(n,total)->error("disconnected during prefill"))
    @test_throws ArgumentError generate_flint_tokens(bundle,[1,0];max_new_tokens=1,prefill_chunk_size=1)
    @test_throws ArgumentError generate_flint_tokens(bundle,[true];max_new_tokens=1,prefill_chunk_size=1)
end
