using Test, LifeAI, JSON3, LinearAlgebra, Random

const FLINT_FIXTURE = joinpath(@__DIR__, "fixtures", "tiny")
const FLINT_ORACLE = JSON3.read(read(joinpath(FLINT_FIXTURE, "reference.json"), String))
flint_reference_matrix(rows) = hcat((Float32.(row) for row in rows)...)

@testset "Flint independent numerical reference" begin
    bundle = load_hf_flint_model(FLINT_FIXTURE)
    config = bundle.model
    parameters = bundle.parameters
    tokens = Int.(FLINT_ORACLE["input_ids"]) .+ 1
    expected = flint_reference_matrix(FLINT_ORACLE["logits"])
    @test config.num_attention_heads * config.head_dim != config.hidden_size
    @test Set(keys(parameters)) == Set(keys(flint_expected_tensor_shapes(config)))
    @test size(parameters["model.layers.0.self_attn.q_proj.weight"]) == (16, 8)
    actual = hf_flint_forward(config, parameters, tokens)
    @info "Flint Float32 vs NumPy" max_absolute_error=maximum(abs.(actual .- expected))
    @test size(actual) == (config.vocab_size, length(tokens))
    @test actual ≈ expected atol=3f-5 rtol=3f-5
    @test hf_flint_forward(config, parameters, tokens; last_only=true) ≈ expected[:, end:end] atol=3f-5 rtol=3f-5

    # The prompt is three times the local window, and chunks cross its boundary.
    for chunks in ([1, 1, 1, 1, 1, 1, 1, 1, 1], [2, 4, 3], [5, 4])
        cache = FlintKVCache(config)
        offset = 0
        output = Matrix{Float32}[]
        for count in chunks
            push!(output, hf_flint_forward(config, parameters, tokens[offset+1:offset+count]; cache))
            offset += count
        end
        @test hcat(output...) ≈ expected atol=3f-5 rtol=3f-5
        @test cache.tokens == length(tokens)
        @test size(cache.layers[3].key, 3) == config.sliding_window - 1
        @test size(cache.layers[1].key, 3) == length(tokens)
        @test reset_flint_cache!(cache) === cache
        @test cache.tokens == 0 && all(isnothing, cache.layers)
        shifted = Int.(FLINT_ORACLE["shifted_input_ids"]) .+ 1
        @test hf_flint_forward(config, parameters, shifted; cache) ≈ flint_reference_matrix(FLINT_ORACLE["shifted_logits"]) atol=3f-5 rtol=3f-5
    end

    # A learned sink changes normalization even for a single valid key.
    without_sinks = copy(parameters)
    for layer in 0:config.num_hidden_layers-1
        without_sinks["model.layers.$layer.self_attn.softmax_offset"] = fill(-100f0, config.num_attention_heads)
    end
    @test norm(hf_flint_forward(config, without_sinks, tokens) - expected) > 0.1f0

    logits = Float32[-2 0 2; 1 3 -3; 0 -1 1; 4 2 0]
    routed = flint_topk_routing(logits, 2)
    @test routed.expert_ids == [4 2 1; 2 4 3]
    for col in axes(logits, 2)
        selected = 1f0 ./ (1f0 .+ exp.(-logits[routed.expert_ids[:, col], col]))
        @test routed.weights[:, col] ≈ selected ./ sum(selected)
    end
    @test vec(sum(routed.weights; dims=1)) ≈ ones(Float32, 3)
    @test_throws ArgumentError flint_topk_routing(logits, 5)
    @test_throws ArgumentError flint_topk_routing(fill(NaN32, 4, 2), 2)
    @test_throws ArgumentError hf_flint_forward(config, parameters, Int[])
    @test_throws ArgumentError hf_flint_forward(config, parameters, [0])
    @test_throws ArgumentError hf_flint_forward(config, parameters, [config.vocab_size + 1])
    @test_throws ArgumentError hf_flint_forward(config, parameters, [true])
    @test_throws ArgumentError hf_flint_forward(config, parameters, fill(1, config.max_position_embeddings + 1))
    @test generate_flint_tokens(bundle, tokens; temperature=0, max_new_tokens=4, eos_ids=Int[]) == Int.(FLINT_ORACLE["greedy_tokens"]) .+ 1
    @test generate_flint_tokens(bundle, tokens; temperature=0, max_new_tokens=0) == Int[]
end

@testset "Flint INT8/INT4 inference and portable artifact" begin
    bundle = load_hf_flint_model(FLINT_FIXTURE)
    tokens = Int.(FLINT_ORACLE["input_ids"]) .+ 1
    baseline = hf_flint_forward(bundle.model, bundle.parameters, tokens)
    for scheme in (:int8, :int4)
        quantized = quantize_flint_parameters(bundle.parameters; scheme, group=4)
        streamed = load_hf_flint_quantized(FLINT_FIXTURE; scheme, group=4)
        @test quantized_parameter_bytes(quantized) < quantized_parameter_bytes(bundle.parameters)
        @test quantized["model.layers.2.mlp.gate.weight"] == bundle.parameters["model.layers.2.mlp.gate.weight"]
        @test quantized["model.embed_tokens.weight"] == bundle.parameters["model.embed_tokens.weight"]
        @test quantized["model.layers.0.self_attn.softmax_offset"] == bundle.parameters["model.layers.0.self_attn.softmax_offset"]
        @test quantized["model.layers.2.mlp.experts.1.down_proj.weight"] isa (scheme == :int8 ? Int8ChannelWeight : Int4GroupWeight)
        result = hf_flint_forward(bundle.model, quantized, tokens)
        @test all(isfinite, result)
        quantized_reference = flint_reference_matrix(FLINT_ORACLE["quantized_logits"][scheme == :int8 ? "8" : "4"])
        @info "Flint quantization vs NumPy" scheme max_absolute_error=maximum(abs.(result .- quantized_reference)) relative_l2_error=norm(result - baseline) / norm(baseline)
        @test result ≈ quantized_reference atol=3f-5 rtol=3f-5
        @test norm(result - baseline) / norm(baseline) < (scheme == :int8 ? 0.04 : 0.35)
        @test result ≈ hf_flint_forward(streamed.model, streamed.parameters, tokens) atol=1f-6 rtol=1f-6
        cache = FlintKVCache(streamed.model)
        sequential = hcat((hf_flint_forward(streamed.model, streamed.parameters, [token]; cache) for token in tokens)...)
        @test sequential ≈ result atol=3f-5 rtol=3f-5

        mktempdir() do temporary
            destination = joinpath(temporary, "quantized")
            @test save_flint_quantized(destination, streamed) == destination
            reloaded = load_flint_quantized(destination)
            @test hf_flint_forward(reloaded.model, reloaded.parameters, tokens) == result
            @test_throws ArgumentError save_flint_quantized(destination, streamed)
            manifest = JSON3.read(read(joinpath(destination, "manifest.json"), String))
            first = manifest["tensors"][1]
            payload = first["kind"] == "float" ? first["values"]["file"] : first["scale"]["file"]
            open(joinpath(destination, payload), "r+") do io
                byte = read(io, UInt8)
                seekstart(io)
                write(io, byte ⊻ 0x01)
            end
            @test_throws ArgumentError load_flint_quantized(destination)
        end
    end
    @test_throws ArgumentError load_hf_flint_quantized(FLINT_FIXTURE; group=6)
    @test_throws ArgumentError quantize_flint_parameters(bundle.parameters; scheme=:int2)
end
