using Test
using Random: MersenneTwister

# Independent reference: no production filtering, softmax, or sampling helper.
function _ch148_oracle(logits, uniform; temperature=1.0f0, top_k=length(logits), top_p=1.0f0)
    scores = Float32.(logits) ./ Float32(temperature)
    ranked = sortperm(collect(eachindex(scores)); by=i -> (-scores[i], i))
    kept = ranked[1:min(top_k, length(ranked))]
    masses = exp.(Float64.(scores[kept]) .- maximum(scores[kept]))
    masses ./= sum(masses)
    cutoff = Float32(top_p) == 1.0f0 ? length(kept) :
        something(findfirst(>=(Float64(Float32(top_p))), cumsum(masses)), length(kept))
    kept = sort(kept[1:cutoff])
    masses = exp.(Float64.(scores[kept]) .- maximum(scores[kept]))
    masses ./= sum(masses)
    index = something(findfirst(>(Float64(uniform)), cumsum(masses)), length(kept))
    return (; token=kept[index], kept, masses, scores=scores[kept])
end

@testset "Chapter 148 — sampled VL generation and independent oracle" begin
    for bf16 in (false, true)
        base = _ch46_tiny_text_parameters()
        parameters = bf16 ? _ch47_bf16_parameters(base) : base
        draws = [0.173, 0.637, 0.913]
        for options in ((;), (; temperature=0.7, top_k=7, top_p=0.83),
                        (; temperature=1.3, top_k=1, top_p=0.1))
            dynamic = _ch147_generate(parameters, :dynamic;
                strategy=:sample, sample_uniforms=draws, capture_distribution=true, options...)
            static = _ch147_generate(parameters, :static; static_capacity=12,
                strategy=:sample, sample_uniforms=draws, capture_distribution=true, options...)
            @test dynamic.generated_ids == static.generated_ids
            @test dynamic.trace == static.trace
            @test static.cache.position == 10
            for (step, draw) in zip(dynamic.trace, draws)
                oracle = _ch148_oracle(step.logits, draw; options...)
                @test step.token_id == oracle.token
                @test step.sample_uniform === draw
                @test step.candidate_count == length(oracle.kept)
                @test step.distribution.token_ids == oracle.kept
                @test step.distribution.hf_token_ids == oracle.kept .- 1
                @test step.distribution.logits == oracle.scores
                @test step.distribution.probabilities ≈ oracle.masses rtol=2e-6 atol=2e-7
                @test step.sampled_probability ≈ oracle.masses[findfirst(==(oracle.token), oracle.kept)] rtol=2e-6
            end
            reused = _ch147_generate(parameters, static.cache;
                strategy=:sample, sample_uniforms=draws, capture_distribution=true, options...)
            @test reused.trace == dynamic.trace
            @test reused.generated_ids == dynamic.generated_ids
            _ch145_check_prefix(reused.cache, dynamic.cache)
        end
        baseline = _ch147_generate(parameters, :dynamic)
        explicit = _ch147_generate(parameters, :dynamic; strategy=:greedy)
        @test baseline.trace == explicit.trace
        @test baseline.generated_ids == explicit.generated_ids
        a = _ch147_generate(parameters, :dynamic; strategy=:sample, rng=MersenneTwister(148))
        b = _ch147_generate(parameters, :static; strategy=:sample, rng=MersenneTwister(148))
        @test a.trace == b.trace
        @test all(step -> step.distribution === nothing, a.trace)
        replay = _ch147_generate(parameters, :dynamic; strategy=:sample,
            sample_uniforms=[step.sample_uniform for step in a.trace])
        @test a.trace == replay.trace
        explicit_rng = MersenneTwister(148)
        untouched_rng = copy(explicit_rng)
        _ch147_generate(parameters, :dynamic; strategy=:sample,
            rng=explicit_rng, sample_uniforms=draws)
        @test rand(explicit_rng) == rand(untouched_rng)
        first = _ch147_generate(parameters, :dynamic; strategy=:sample,
            sample_uniforms=draws, max_new_tokens=1)
        eos = _ch147_generate(parameters, :static; strategy=:sample,
            sample_uniforms=draws, stop_token_ids=first.generated_ids)
        @test eos.stop_reason === :eos
        @test eos.generated_ids == first.generated_ids
        @test eos.cache.position == 8
        zero = _ch147_generate(parameters, eos.cache; strategy=:sample,
            sample_uniforms=Float64[], max_new_tokens=0)
        @test isempty(zero.generated_ids)
        @test isempty(zero.trace)
        @test zero.prefill === nothing
        @test zero.cache.position == 0
    end
end

@testset "Chapter 148 — ties and Float64 uniforms" begin
    base = _ch46_tiny_text_parameters()
    parameters = merge(base, (; final_norm=zero.(base.final_norm)))
    # The Float64 draw lies below a Float32 rounding boundary at 1/2.
    draw = prevfloat(0.5)
    result = _ch147_generate(parameters, :static; strategy=:sample,
        top_k=2, sample_uniforms=[draw, 0.5, 0.99], capture_distribution=true)
    @test result.generated_ids == [1, 2, 2]
    @test result.trace[1].sample_uniform === draw
    @test all(step -> step.distribution.token_ids == [1, 2], result.trace)
    nucleus = _ch147_generate(parameters, :dynamic; strategy=:sample,
        top_k=2, top_p=0.5, sample_uniforms=[0.99, 0.1, 0.8])
    @test nucleus.generated_ids == [1, 1, 1]
end

function _ch148_invalid_options()
    return (
        (; strategy=:unknown), (; temperature=1.0), (; top_k=2), (; top_p=0.5),
        (; sample_uniforms=[0.1, 0.2, 0.3]), (; capture_distribution=true),
        ((; strategy=:sample, temperature=value) for value in (true, 0, -1, Inf, NaN, 1e-100, 1e100))...,
        ((; strategy=:sample, top_k=value) for value in (true, 0, -1, 1.5, big(typemax(Int))+1))...,
        ((; strategy=:sample, top_p=value) for value in (true, 0, -1, 1.1, Inf, NaN, 1e-100))...,
        ((; strategy=:sample, sample_uniforms=value) for value in (
            [0.1], Any[0.1, 0.2, false], [0.1, 0.2, -0.1], [0.1, 0.2, 1.0],
            [0.1, 0.2, Inf], [0.1, 0.2, NaN], [0.1, 0.2, 0.3, NaN]))...,
    )
end

@testset "Chapter 148 — validation preserves request and precedes vision" begin
    parameters = _ch47_bf16_parameters(_ch46_tiny_text_parameters())
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    _ch147_generate(parameters, cache)
    before = _ch147_snapshot(cache)
    for options in _ch148_invalid_options()
        @test_throws ArgumentError _ch147_generate(parameters, cache; options...)
        _ch147_unchanged(cache, before)
    end
    fixture = _ch147_raw_fixture()
    (; parameters, vision, tokenizer, processor) = fixture
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=1024)
    _ch147_seed_text(parameters, cache)
    before = _ch147_snapshot(cache)
    messages = [(role="user", content=Any[
        (type="image", image=zeros(UInt8, 32, 32, 3)),
        (type="text", text="Describe."),
    ])]
    for options in _ch148_invalid_options()
        @test_throws ArgumentError generate_hf_qwen3_vl(vision, parameters, tokenizer, messages;
            cache, max_new_tokens=3, stop_token_ids=Int[], processor_spec=processor, options...)
        @test !vision.touched[]
        _ch147_unchanged(cache, before)
    end
end

# Success-path tower stand-in: keep raw preprocessing/template/mRoPE real while
# giving the tiny decoder correctly shaped image and DeepStack features.
struct _Ch148VisionStub{S,A}
    spec::S
    patch_weight::A
    touched::Base.RefValue{Bool}
    features::Base.RefValue{Any}
end
function LifeAI._qwen3_vl_generation_vision_features(
    parameters::_Ch148VisionStub, processed, requested::Int,
)
    parameters.touched[] = true
    count = sum(prod(processed.grid_thw[:, i]) for i in axes(processed.grid_thw, 2)) ÷
        parameters.spec.spatial_merge_size^2
    features = (;
        visual_embeddings=zeros(Float32, parameters.spec.out_hidden_size, count),
        deepstack=ntuple(_ -> zeros(Float32, parameters.spec.out_hidden_size, count), 3),
    )
    parameters.features[] = features
    return features
end

@testset "Chapter 148 — raw sampled success forwards options and supplied cache" begin
    (; parameters, vision, tokenizer, processor) = _ch147_raw_fixture()
    tower = _Ch148VisionStub(vision.spec, vision.patch_weight, Ref(false), Ref{Any}(nothing))
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=1024)
    _ch147_seed_text(parameters, cache)
    messages = [(role="user", content=Any[
        (type="image", image=zeros(UInt8, 32, 32, 3)),
        (type="text", text="Describe."),
    ])]
    draws = [0.173, 0.637, 0.913]
    rng = MersenneTwister(148)
    untouched = copy(rng)
    options = (;
        strategy=:sample, temperature=0.7, top_k=7, top_p=0.83,
        sample_uniforms=draws, capture_logits=true, capture_distribution=true,
        max_new_tokens=3, stop_token_ids=Int[], rng,
    )
    raw = generate_hf_qwen3_vl(tower, parameters, tokenizer, messages;
        cache, processor_spec=processor, options...)
    @test tower.touched[]
    @test raw.cache === cache
    @test raw.strategy === :sample
    @test length(raw.generated_ids) == 3
    @test length(raw.trace) == 3
    @test [step.sample_uniform for step in raw.trace] == draws
    @test all(step -> step.logits !== nothing && step.distribution !== nothing, raw.trace)
    @test all(step -> step.temperature == 0.7f0 && step.top_k == 7 && step.top_p == 0.83f0, raw.trace)
    @test rand(rng) == rand(untouched)
    expected_kv = [(copy(layer.keys), copy(layer.values)) for layer in cache.layers]
    expected_position = cache.position
    direct = generate_hf_qwen3_vl_tokens(parameters, raw.prompt_ids, raw.rope_layout;
        cache, vision_features=tower.features[], options...)
    @test direct.cache === cache
    @test direct.strategy === raw.strategy
    @test direct.generated_ids == raw.generated_ids
    @test direct.trace == raw.trace
    @test cache.position == expected_position
    for (layer, (keys, values)) in zip(cache.layers, expected_kv)
        @test layer.keys[:, :, 1:cache.position, :] == keys[:, :, 1:cache.position, :]
        @test layer.values[:, :, 1:cache.position, :] == values[:, :, 1:cache.position, :]
    end
    @test rand(rng) == rand(untouched)
end
