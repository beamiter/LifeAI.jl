using Test,LifeAI,JSON3

function ripple_tp_prefix_checks(backend_factory;world_sizes=(1,2))
    ids=Int.(JSON3.read(read(joinpath(RIPPLE_TP_FIXTURE,"reference.json"),String))["input_ids"]).+1
    for n in world_sizes
        backend=backend_factory(n)
        try
            bundle=load_hf_ripple_tensor_parallel(RIPPLE_TP_FIXTURE;backend,dtype=Float32)
            store=RippleTPPrefixCache(bundle;max_tokens=64)
            generate=(prompt;kwargs...)->generate_ripple_tp_tokens(bundle,prompt;max_new_tokens=8,eos_ids=Int[],temperature=0,prefill_chunk_size=4,kwargs...)
            # A one-token final prefill chunk creates a sliding ring. Decode
            # wraps it repeatedly; restoring must use the independent snapshot.
            short=ids[1:9];long=ids
            expected_short=generate(short)
            @test generate(short;prefix_cache=store)==expected_short
            @test ripple_tp_prefix_stats(store).reused_tokens==0
            @test generate(short;prefix_cache=store)==expected_short
            @test ripple_tp_prefix_stats(store).reused_tokens==length(short)
            @test ripple_tp_prefix_stats(store).prefilled_tokens==0
            @test generate(long;prefix_cache=store)==generate(long)
            @test ripple_tp_prefix_stats(store).reused_tokens==length(short)
            @test ripple_tp_prefix_stats(store).prefilled_tokens==length(long)-length(short)
            changed=copy(long);changed[3]=mod(changed[3],bundle.model.vocab_size)+1
            @test generate(changed;prefix_cache=store)==generate(changed)
            @test ripple_tp_prefix_stats(store).reused_tokens==0
            @test generate(changed;prefix_cache=store,schedule=:staged)==generate(changed;schedule=:staged)
            @test ripple_tp_prefix_stats(store).reused_tokens==0
            @test generate(short;prefix_cache=store)==expected_short
            @test ripple_tp_prefix_stats(store).reused_tokens==0
            @test isempty(generate(long;prefix_cache=store,on_prefill=(done,total)->false))
            @test ripple_tp_prefix_stats(store).cached_tokens==0
            @test length(generate(short;prefix_cache=store,on_token=id->false))==1
            @test ripple_tp_prefix_stats(store).cached_tokens==0
            @test generate(short;prefix_cache=store)==expected_short
            larger=generate_ripple_tp_tokens(bundle,short;max_new_tokens=16,eos_ids=Int[],temperature=0,prefill_chunk_size=4,prefix_cache=store)
            fresh=generate_ripple_tp_tokens(bundle,short;max_new_tokens=16,eos_ids=Int[],temperature=0,prefill_chunk_size=4)
            @test larger==fresh
            full=findfirst(==("full_attention"),bundle.model.layer_types)
            if full!==nothing && hasproperty(store.checkpoint[1][full],:storage)
                @test store.checkpoint[1][full].storage===store.cache.ranks[1].layers[full].storage
            end
            store.needs_full_gc=true
            @test reset_ripple_tp_prefix!(store)===store
            @test store.needs_full_gc
            store.needs_full_gc=false;store.prefilled_tokens=32768
            reset_ripple_tp_prefix!(store)
            @test store.needs_full_gc
            @test ripple_tp_prefix_stats(store).cached_tokens==0
        finally
            close_ripple_tp_backend!(backend)
        end
    end
end

@testset "Ripple prompt checkpoint, ring wrap and exact prefix reuse" begin
    ripple_tp_prefix_checks(RippleTPReferenceBackend)
end
