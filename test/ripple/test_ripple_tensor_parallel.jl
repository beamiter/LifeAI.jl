using Test,LifeAI,JSON3,LinearAlgebra

const RIPPLE_TP_FIXTURE=joinpath(@__DIR__,"fixtures","tp8")

@testset "Ripple TP partition contract" begin
    c=RippleConfig();plan=ripple_tensor_parallel_plan(c,8)
    @test plan.query_heads_per_rank==10 && plan.kv_heads_per_rank==1
    @test plan.ffn_width_per_rank==5120 && plan.vocabulary_per_rank==12544
    @test ripple_tensor_partition(c,"model.layers.0.self_attn.q_proj.weight",1,8).shape==(1280,5120)
    @test ripple_tensor_partition(c,"model.layers.0.self_attn.o_proj.weight",1,8).shape==(5120,1280)
    @test ripple_tensor_partition(c,"model.layers.0.mlp.down_proj.weight",7,8).range==35841:40960
    @test ripple_tensor_partition(c,"model.embed_tokens.weight",7,8).axis==0
    @test_throws ArgumentError ripple_tensor_parallel_plan(c,3)
    @test_throws ArgumentError ripple_tensor_partition(c,"lm_head.weight",8,8)
end

function ripple_tp_numerical_checks(backend_factory;world_sizes=(1,2,4,8),schedules=(:staged,:merged))
    reference=JSON3.read(read(joinpath(RIPPLE_TP_FIXTURE,"reference.json"),String))
    ids=Int.(reference["input_ids"]).+1
    expected=hcat((Float32.(row) for row in reference["logits"])...)
    for n in world_sizes, schedule in schedules
        backend=backend_factory(n)
        try
            bundle=load_hf_ripple_tensor_parallel(RIPPLE_TP_FIXTURE;backend,dtype=Float32)
            @test size(bundle.ranks[1].parameters["model.layers.0.self_attn.q_proj.weight"],1)==64÷n
            actual=Array(hf_ripple_tp_forward(bundle,ids;schedule))
            @test actual ≈ expected atol=7f-5 rtol=7f-5
            @test Array(hf_ripple_tp_forward(bundle,ids;last_only=true,schedule)) ≈ expected[:,end:end] atol=7f-5 rtol=7f-5
            cache=RippleTPKVCache(bundle);outputs=Matrix{Float32}[];offset=0
            for count in (2,3,length(ids)-5)
                push!(outputs,Array(hf_ripple_tp_forward(bundle,ids[offset+1:offset+count];cache,schedule)))
                offset+=count
            end
            @test hcat(outputs...) ≈ expected atol=7f-5 rtol=7f-5
            @test cache.tokens==length(ids)
            @test all(c->c.tokens==length(ids),cache.ranks)
            @test size(cache.ranks[1].layers[1].key,3)==2
            @test reset_ripple_tp_cache!(cache)===cache
            @test cache.tokens==0
            @test generate_ripple_tp_tokens(bundle,ids;max_new_tokens=4,temperature=0,eos_ids=Int[],prefill_chunk_size=3,schedule)==Int.(reference["greedy_tokens"]).+1
            @test generate_ripple_tp_tokens(bundle,ids;max_new_tokens=0,schedule)==Int[]
            @test_throws ArgumentError hf_ripple_tp_forward(bundle,[0])
            @test_throws ArgumentError hf_ripple_tp_forward(bundle,[true])
            @test_throws ArgumentError generate_ripple_tp_tokens(bundle,ids;max_new_tokens=true)
        finally
            close_ripple_tp_backend!(backend)
        end
    end
end

@testset "Ripple TP1/2/4/8 vs official PyTorch and cache" begin
    ripple_tp_numerical_checks(RippleTPReferenceBackend)
end
