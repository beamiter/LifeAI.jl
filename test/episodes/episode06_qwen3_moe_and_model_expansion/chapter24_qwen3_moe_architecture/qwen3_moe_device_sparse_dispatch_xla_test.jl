using Test
using Lux
using Random: Xoshiro
using Reactant
using LifeAI: Qwen3SparseMoE, qwen3_moe_device_forward

@testset "Qwen3 MoE compact sparse dispatch compiles on Reactant" begin
    Reactant.set_default_backend(get(ENV, "LIFEAI_XLA_BACKEND", "cpu"))
    layer = Qwen3SparseMoE(6, 5, 8, 2)
    parameters, _ = Lux.setup(Xoshiro(20260815), layer)
    x = randn(Xoshiro(20260816), Float32, 6, 4, 1)
    reference = qwen3_moe_device_forward(layer, x, parameters)

    kernel = (tokens, router, gate, up, down) -> first(layer(
        tokens,
        (;
            gate=(; weight=router),
            experts=(; gate_proj=gate, up_proj=up, down_proj=down),
        ),
        (;),
    ))
    x_device = Reactant.to_rarray(x)
    router_device = Reactant.to_rarray(parameters.gate.weight)
    gate_device = Reactant.to_rarray(parameters.experts.gate_proj)
    up_device = Reactant.to_rarray(parameters.experts.up_proj)
    down_device = Reactant.to_rarray(parameters.experts.down_proj)
    compiled = Reactant.@compile kernel(
        x_device,
        router_device,
        gate_device,
        up_device,
        down_device,
    )
    actual = Array(compiled(
        x_device,
        router_device,
        gate_device,
        up_device,
        down_device,
    ))

    @test actual ≈ reference.output atol = 3.0f-6 rtol = 3.0f-5
    @test size(actual) == size(x)
    @test all(isfinite, actual)
end
