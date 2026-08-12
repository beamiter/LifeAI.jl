using Test
using Lux
using NNlib: softmax, swish
using Random: Xoshiro
using LifeAI: Qwen3SparseMoE

function _qwen3_moe_manual_forward(layer, x, parameters)
    tokens = reshape(x, layer.d_model, :)
    logits = parameters.gate.weight * tokens
    probabilities = softmax(Float32.(logits); dims=1)
    output = zeros(Float32, size(tokens))
    for token in axes(tokens, 2)
        selected = partialsortperm(
            view(probabilities, :, token),
            1:layer.experts_per_token;
            rev=true,
        )
        weights = probabilities[selected, token]
        layer.normalize_routing && (weights ./= sum(weights))
        for (route, expert) in enumerate(selected)
            value = view(tokens, :, token)
            gate = view(parameters.experts.gate_proj, :, :, expert) * value
            up = view(parameters.experts.up_proj, :, :, expert) * value
            expert_output = view(parameters.experts.down_proj, :, :, expert) *
                (swish.(gate) .* up)
            output[:, token] .+= weights[route] .* expert_output
        end
    end
    return reshape(output, size(x))
end

@testset "Qwen3 MoE selected expert mixture output" begin
    layer = Qwen3SparseMoE(4, 3, 4, 2)
    parameters, states = Lux.setup(Xoshiro(20260807), layer)
    x = reshape(Float32[
        0.2, -0.3, 0.7, 0.5,
       -0.4,  0.8, 0.1, 0.6,
        0.9,  0.2, -0.5, 0.3,
    ], 4, 3, 1)

    actual, next_states = layer(x, parameters, states)
    expected = _qwen3_moe_manual_forward(layer, x, parameters)
    @test actual ≈ expected atol = 2.0f-7 rtol = 2.0f-6
    @test next_states == states == (;)
    @test Lux.parameterlength(layer) == 4 * 4 + 4 * 3 * 4 * 3
    @test Lux.parameterlength(parameters) == Lux.parameterlength(layer)

    @test_throws DimensionMismatch layer(randn(Float32, 5, 2, 1), parameters, states)
    @test_throws DimensionMismatch layer(randn(Float32, 4, 2), parameters, states)
end
