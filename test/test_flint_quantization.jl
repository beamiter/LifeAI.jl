using Test
using Random
using LifeAI

@testset "Flint weight-only quantization and portable artifacts" begin
    c = FlintConfig(vocab_size=32, hidden_size=8, intermediate_size=12,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=4, max_position_embeddings=32, sliding_window=3,
        mlp_only_layers=[0], moe_intermediate_size=4,
        shared_expert_intermediate_size=8, num_experts=3, num_experts_per_tok=2,
        bos_token_id=0, eos_token_id=[1], pad_token_id=1)
    rng = Xoshiro(107)
    parameters = Dict{String,Any}(name => randn(rng, Float32, shape) .* 0.1f0
        for (name, shape) in flint_expected_tensor_shapes(c))
    quant8 = quantize_flint_parameters(parameters; scheme=:int8)
    quant4 = quantize_flint_parameters(parameters; scheme=:int4, group=4)
    for name in keys(parameters)
        if LifeAI._flint_quantizable(name)
            @test quant8[name] isa Int8ChannelWeight
            @test quant4[name] isa Int4GroupWeight
            reconstructed8 = Float32.(LifeAI._dequantize_bf16(quant8[name]))
            @test maximum(abs.(reconstructed8 .- parameters[name])) < 0.003f0
        else
            @test quant8[name] == parameters[name]
            @test quant4[name] == parameters[name]
            @test quant4[name] !== parameters[name]
        end
    end
    @test quantized_parameter_bytes(quant4) < quantized_parameter_bytes(parameters)
    @test quantized_parameter_bytes(quant8) < quantized_parameter_bytes(parameters)
    @test_throws ArgumentError quantize_flint_parameters(parameters; group=3)
    @test_throws ArgumentError quantize_flint_parameters(parameters; calibration=:activation_mse)
    @test_throws ArgumentError quantize_flint_parameters(parameters; scheme=:float8)
    @test_throws ArgumentError quantize_flint_parameters(parameters; group=16)
    invalid = copy(parameters)
    invalid["model.norm.weight"] = fill(Float32(NaN), c.hidden_size)
    @test_throws ArgumentError quantize_flint_parameters(invalid)

    for quantized in (quant4, quant8, parameters)
        mktempdir() do directory
            destination = joinpath(directory, "artifact")
            @test save_flint_quantized(destination, (model=c, parameters=quantized)) == destination
            restored = load_flint_quantized(destination)
            @test LifeAI.flint_config_dict(restored.model) == LifeAI.flint_config_dict(c)
            @test quantized_parameter_bytes(restored.parameters) == quantized_parameter_bytes(quantized)
            for name in keys(quantized)
                left, right = quantized[name], restored.parameters[name]
                if left isa Int4GroupWeight
                    @test left.packed == right.packed
                    @test left.scale == right.scale
                    @test left.group == right.group
                elseif left isa Int8ChannelWeight
                    @test left.q == right.q
                    @test left.scale == right.scale
                else
                    @test left == right
                end
            end
            @test_throws ArgumentError save_flint_quantized(destination, (model=c, parameters=quantized))
            manifest = LifeAI.JSON3.read(read(joinpath(destination, "manifest.json"), String))
            payload = joinpath(destination, String(manifest["tensors"][1]["values"]["file"]))
            open(payload, "r+") do io
                first_byte = read(io, UInt8)
                seekstart(io)
                write(io, first_byte ⊻ 0x01)
            end
            @test_throws ArgumentError load_flint_quantized(destination)
        end
    end
end
