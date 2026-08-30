using Test
using Lux
using NNlib: softmax, swish
using Random: Xoshiro
using LifeAI: Qwen3SparseMoE, qwen3_cuda_indexed_workspace_bytes

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


@testset "Qwen3 MoE size calculations reject Int overflow" begin
    @test Lux.parameterlength(Qwen3SparseMoE(4, 3, 4, 2)) == 160
    @test_throws ArgumentError Lux.parameterlength(
        Qwen3SparseMoE(typemax(Int), 1, 1, 1),
    )

    add_overflow_experts = typemax(Int) ÷ 4 + 1
    @test_throws ArgumentError Lux.parameterlength(
        Qwen3SparseMoE(1, 1, add_overflow_experts, 1),
    )

    @test qwen3_cuda_indexed_workspace_bytes(128, 64, 64, 8) == 425_984
    @test_throws ArgumentError qwen3_cuda_indexed_workspace_bytes(
        typemax(Int),
        1,
        2,
        2,
    )
    @test_throws ArgumentError qwen3_cuda_indexed_workspace_bytes(
        1,
        1,
        typemax(Int),
        2,
    )
    @test_throws ArgumentError qwen3_cuda_indexed_workspace_bytes(
        1,
        1,
        1,
        1;
        element_bytes=typemax(Int),
    )
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
