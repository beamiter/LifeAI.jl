using Test
using Random
using Reactant
using LifeAI: _device_sample_next_index, device_sample_token

# The traced policy and the host reference are the same code, so this test
# checks the one thing that cannot be checked on ordinary arrays: that the
# policy lowers to a StableHLO program at all, and that the compiled program
# still selects the reference token for every uniform.
@testset "device sampling policy compiles on Reactant" begin
    Reactant.set_default_backend(get(ENV, "LIFEAI_XLA_BACKEND", "cpu"))
    rng = MersenneTwister(2306)
    vocab = 96
    top_k = 5
    logits = randn(rng, Float32, vocab) .* 3.0f0
    temperature = 0.7f0
    top_p = 0.9f0

    kernel = (token, values, uniform, scale, nucleus) ->
        token .* 0 .+ _device_sample_next_index(
            values,
            uniform,
            scale,
            nucleus,
            top_k,
        )

    token_state = Reactant.to_rarray(ones(Int, 1))
    logits_state = Reactant.to_rarray(logits)
    temperature_state = Reactant.to_rarray(Float32[temperature])
    top_p_state = Reactant.to_rarray(Float32[top_p])
    compiled = Reactant.@compile kernel(
        token_state,
        logits_state,
        Reactant.to_rarray(Float32[0.5]),
        temperature_state,
        top_p_state,
    )

    for uniform in Float32[0.0, 0.05, 0.31, 0.5, 0.72, 0.9, 0.999]
        device = compiled(
            token_state,
            logits_state,
            Reactant.to_rarray(Float32[uniform]),
            temperature_state,
            top_p_state,
        )
        @test Int(Array(device)[1]) ==
            device_sample_token(logits, uniform; temperature, top_k, top_p)
    end

    # temperature and top-p are runtime inputs: a second nucleus must change
    # the outcome without recompiling.
    tight_top_p = Reactant.to_rarray(Float32[0.05])
    for uniform in Float32[0.1, 0.6, 0.95]
        device = compiled(
            token_state,
            logits_state,
            Reactant.to_rarray(Float32[uniform]),
            temperature_state,
            tight_top_p,
        )
        @test Int(Array(device)[1]) == device_sample_token(
            logits, uniform; temperature, top_k, top_p=0.05)
    end
end
