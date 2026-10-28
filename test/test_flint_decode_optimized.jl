using Test, LifeAI, Random, LinearAlgebra

# Keep a simple per-head implementation independent of the grouped matrix
# layout, so GQA head/query ordering and learned sink normalization are tested.
function flint_scalar_attention_reference(q, k, v, sinks, mask)
    d, heads, n = size(q)
    output = similar(q)
    for h in 1:heads
        kh = (h-1) ÷ (heads ÷ size(k, 2)) + 1
        scores = transpose(k[:, kh, :]) * q[:, h, :] ./ sqrt(Float32(d))
        mask === nothing || (scores .+= mask)
        sink = reshape(sinks[h:h], 1, 1)
        maxima = max.(maximum(scores; dims=1), sink)
        weights = exp.(scores .- maxima)
        weights ./= sum(weights; dims=1) .+ exp.(sink .- maxima)
        output[:, h, :] = v[:, kh, :] * weights
    end
    return output
end

function flint_scattered_moe_reference(model, parameters, prefix, x)
    logits = LifeAI._flint_linear(parameters[prefix * "gate.weight"], x)
    routes = flint_topk_routing(logits, model.num_experts_per_tok; normalize=model.norm_topk_prob)
    result = zero(x)
    for expert in sort!(unique(vec(routes.expert_ids)))
        matches = findall(==(expert), routes.expert_ids)
        columns = [index[2] for index in matches]
        factors = LifeAI._flint_like(x, reshape([routes.weights[index] for index in matches], 1, :))
        values = LifeAI._flint_mlp(parameters, prefix * "experts.$(expert-1).", x[:, columns]) .* factors
        result[:, columns] .+= values
    end
    return result .+ LifeAI._flint_mlp(parameters, prefix * "shared_expert.", x)
end

function flint_grouped_attention_checks(device)
    rng = Xoshiro(208049)
    for (d, heads, kvheads, n, keys) in ((4, 4, 2, 1, 1), (8, 8, 2, 1, 17),
            (8, 8, 2, 3, 7), (4, 4, 4, 4, 4), (16, 32, 4, 1, 128))
        q = randn(rng, Float32, d, heads, n)
        k = randn(rng, Float32, d, kvheads, keys)
        v = randn(rng, Float32, d, kvheads, keys)
        sinks = randn(rng, Float32, heads) .* 5f0
        for mask in (nothing, ifelse.([i <= keys-n+j for i in 1:keys, j in 1:n], 0f0, -Inf32))
            expected = flint_scalar_attention_reference(q, k, v, sinks, mask)
            actual = LifeAI._flint_attention_context(device(q), device(k), device(v),
                device(sinks), mask === nothing ? nothing : device(mask))
            @test size(actual) == size(expected)
            @test Array(actual) ≈ expected atol=3f-6 rtol=3f-5
        end
    end
end

function flint_decode_moe_checks(device)
    fixture = joinpath(@__DIR__, "flint", "fixtures", "tiny")
    bundle = load_hf_flint_model(fixture; device)
    rng = Xoshiro(29409)
    for layer in (3, 4), n in (1, 3)
        prefix = "model.layers.$(layer-1).mlp."
        x = device(randn(rng, Float32, bundle.model.hidden_size, n))
        expected = flint_scattered_moe_reference(bundle.model, bundle.parameters, prefix, x)
        actual = LifeAI._flint_feedforward(bundle.model, bundle.parameters, prefix, x, layer)
        @test Array(actual) ≈ Array(expected) atol=1f-6 rtol=1f-6
    end
end

@testset "Flint grouped attention and single-token expert dispatch" begin
    flint_grouped_attention_checks(identity)
    flint_decode_moe_checks(identity)
end

if get(ENV, "LIFEAI_FLINT_OPT_CUDA_TESTS", "false") == "true"
    @eval using CUDA
    @testset "Flint grouped attention and expert dispatch on CUDA" begin
        @test CUDA.functional()
        CUDA.allowscalar(false)
        flint_grouped_attention_checks(CUDA.cu)
        flint_decode_moe_checks(CUDA.cu)
    end
end
