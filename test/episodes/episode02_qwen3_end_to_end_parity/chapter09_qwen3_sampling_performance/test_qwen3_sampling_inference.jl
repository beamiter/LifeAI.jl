using Test
using Random: Xoshiro
using Lux
using JSON3
import LifeAI
using LifeAI:
    GPTModel,
    HFQwen3GenerationConfig,
    RoPE,
    apply_rope,
    generate_hf_text,
    hf_generation_config,
    load_hf_qwen3_tokenizer,
    load_hf_qwen3_bundle,
    load_safetensors,
    vocab_size

if !isdefined(@__MODULE__, :qwen3_tokenizer_fixture_payloads)
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tokenizer_fixture.jl"))
end

const _QWEN3_SAMPLING_MODEL_DIR = get(ENV, "LIFEAI_QWEN3_MODEL_DIR", "")
const _QWEN3_SAMPLING_REFERENCE_DIR = get(ENV, "LIFEAI_QWEN3_SAMPLING_REFERENCE_DIR", "")

if !isempty(_QWEN3_SAMPLING_MODEL_DIR) && !isempty(_QWEN3_SAMPLING_REFERENCE_DIR)
    @testset "Qwen3-0.6B official sampling integration" begin
        reference = JSON3.read(read(joinpath(_QWEN3_SAMPLING_REFERENCE_DIR, "reference.json"), String))
        tensors = load_safetensors(joinpath(_QWEN3_SAMPLING_REFERENCE_DIR, "reference.safetensors"))
        uniforms = Float32.(reference["uniforms"])
        bundle = load_hf_qwen3_bundle(
            _QWEN3_SAMPLING_MODEL_DIR;
            max_seq_len=length(reference["prompt_ids_0_based"]) + length(uniforms),
            revision=String(reference["revision"]),
        )
        result = generate_hf_text(
            bundle,
            String(reference["prompt"]);
            cache=:dynamic,
            max_new_tokens=length(uniforms),
            strategy=:config,
            sample_uniforms=uniforms,
            capture_logits=true,
            capture_distribution=true,
        )
        @test bundle.tokenizer.generation_config_sha256 ==
              String(reference["generation_config_sha256"])
        @test result.prompt_ids .- 1 == Int.(reference["prompt_ids_0_based"])
        @test result.generated_ids .- 1 == Int.(reference["generated_ids_0_based"])
        @test result.completion == String(reference["completion"])
        @test String(result.stop_reason) == String(reference["stop_reason"])
        @test any(step -> step.candidate_count > 1, result.trace)

        for (actual, expected) in zip(result.trace, reference["steps"])
            @test actual.hf_token_id == Int(expected["token_id_0_based"])
            @test actual.distribution.hf_token_ids == Int.(expected["candidate_ids_0_based"])
            @test isapprox(
                actual.logits,
                tensors[String(expected["logits_key"])];
                atol=5.0f-3,
                rtol=5.0f-4,
            )
            @test isapprox(
                actual.distribution.logits,
                tensors[String(expected["filtered_logits_key"])];
                atol=5.0f-3,
                rtol=5.0f-4,
            )
            @test isapprox(
                actual.distribution.probabilities,
                tensors[String(expected["probabilities_key"])];
                atol=1.0f-5,
                rtol=1.0f-4,
            )
        end
    end
else
    @info "Skipping Qwen3 sampling and inference integration; set LIFEAI_QWEN3_MODEL_DIR and LIFEAI_QWEN3_SAMPLING_REFERENCE_DIR"
end

@testset "Qwen3 generation config contract" begin
    source_eos_ids = Int32[261, 259]
    direct = HFQwen3GenerationConfig(
        Int32(259),
        source_eos_ids,
        big(259),
        true,
        Float64(0.6),
        Int128(20),
        Float64(0.95),
        SubString("v4.51.0", 2),
    )
    @test direct.bos_id === 259
    @test direct.eos_ids == [261, 259]
    @test direct.eos_ids isa Vector{Int}
    @test direct.pad_id === 259
    @test direct.do_sample === true
    @test direct.temperature === 0.6f0
    @test direct.top_k === 20
    @test direct.top_p === 0.95f0
    @test direct.transformers_version == "4.51.0"
    source_eos_ids[1] = 1
    @test direct.eos_ids == [261, 259]
    empty!(direct.eos_ids)
    @test direct.eos_ids == [261, 259]
    @test getfield(direct, :eos_ids) == (261, 259)

    minimum_temperature = HFQwen3GenerationConfig(
        1,
        [1],
        1,
        false,
        nextfloat(0.0f0),
        1,
        1.0f0,
        "boundary",
    )
    @test minimum_temperature.temperature == nextfloat(0.0f0)
    maximum_temperature = HFQwen3GenerationConfig(
        1,
        [1],
        1,
        false,
        floatmax(Float32),
        1,
        1.0f0,
        "boundary",
    )
    @test maximum_temperature.temperature == floatmax(Float32)

    too_large = big(typemax(Int)) + 1
    huge_real = big(10)^1_000
    valid_fields = (259, [261, 259], 259, true, 0.6, 20, 0.95, "4.51.0")
    for field in (1, 3), invalid_id in (true, 1.0, 0, -1, too_large)
        invalid = Base.setindex(valid_fields, invalid_id, field)
        @test_throws ArgumentError HFQwen3GenerationConfig(invalid...)
    end
    for invalid_eos in (
        (),
        Int[],
        reshape([261, 259], 1, :),
        [261, 261],
        Any[261, true],
        Any[261, 1.0],
        Any[261, 0],
        Any[261, -1],
        Any[261, too_large],
    )
        invalid = Base.setindex(valid_fields, invalid_eos, 2)
        @test_throws ArgumentError HFQwen3GenerationConfig(invalid...)
    end
    for invalid_do_sample in (0, 1, :yes, nothing)
        invalid = Base.setindex(valid_fields, invalid_do_sample, 4)
        @test_throws ArgumentError HFQwen3GenerationConfig(invalid...)
    end
    for invalid_temperature in (
        true,
        0,
        -1,
        NaN,
        Inf,
        BigFloat("1e-1000"),
        huge_real,
    )
        invalid = Base.setindex(valid_fields, invalid_temperature, 5)
        @test_throws ArgumentError HFQwen3GenerationConfig(invalid...)
    end
    for invalid_top_k in (true, 1.0, 0, -1, too_large)
        invalid = Base.setindex(valid_fields, invalid_top_k, 6)
        @test_throws ArgumentError HFQwen3GenerationConfig(invalid...)
    end
    for invalid_top_p in (
        true,
        0,
        -1,
        nextfloat(1.0),
        BigFloat(1) + eps(BigFloat),
        1.1,
        NaN,
        Inf,
    )
        invalid = Base.setindex(valid_fields, invalid_top_p, 7)
        @test_throws ArgumentError HFQwen3GenerationConfig(invalid...)
    end
    for invalid_version in ("", :version, nothing)
        invalid = Base.setindex(valid_fields, invalid_version, 8)
        @test_throws ArgumentError HFQwen3GenerationConfig(invalid...)
    end

    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_qwen3_tokenizer_fixture(directory))
        config = hf_generation_config(tokenizer)
        @test config.bos_id == 259
        @test config.eos_ids == [261, 259]
        @test config.pad_id == 259
        @test config.do_sample
        @test config.temperature == 0.6f0
        @test config.top_k == 20
        @test config.top_p == 0.95f0
        @test config.transformers_version == "4.51.0"

        empty!(tokenizer.eos_ids)
        empty!(tokenizer.generation.eos_ids)
        @test tokenizer.eos_ids == [261, 259]
        @test tokenizer.generation.eos_ids == [261, 259]
        push!(config.eos_ids, 1)
        @test hf_generation_config(tokenizer).eos_ids == [261, 259]
    end

    mutations = [
        payloads -> (payloads.generation_config["do_sample"] = "true"),
        payloads -> (payloads.generation_config["temperature"] = 0.0),
        payloads -> (payloads.generation_config["top_k"] = 0),
        payloads -> (payloads.generation_config["top_p"] = 1.1),
        payloads -> (payloads.generation_config["top_p"] = nextfloat(1.0)),
        payloads -> (payloads.generation_config["transformers_version"] = ""),
        payloads -> (payloads.generation_config["min_p"] = 0.1),
    ]
    for mutate! in mutations
        mktempdir() do directory
            payloads = qwen3_tokenizer_fixture_payloads()
            mutate!(payloads)
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test_throws ArgumentError load_hf_qwen3_tokenizer(directory)
        end
    end
end

@testset "temperature, top-k, and top-p semantics" begin
    logits = Float32[4, 3, 2, 1]
    filtered, probabilities = LifeAI._sampling_distribution(
        logits;
        temperature=1.0f0,
        top_k=3,
        top_p=0.7f0,
    )
    @test findall(isfinite, filtered) == [1, 2]
    expected = exp.(Float32[4, 3])
    expected ./= sum(expected)
    @test probabilities[1:2] ≈ expected atol = 1.0f-7
    @test probabilities[3:4] == [0.0f0, 0.0f0]

    rng = Xoshiro(9)
    @test LifeAI._sample_token(
        logits,
        rng;
        top_k=3,
        top_p=0.7f0,
        sample_uniform=0.2f0,
    ) == 1
    @test LifeAI._sample_token(
        logits,
        rng;
        top_k=3,
        top_p=0.7f0,
        sample_uniform=0.9f0,
    ) == 2
    @test LifeAI._sample_categorical(Float32[0.6, 0.3999999, 0], 0.9999999) == 2
    @test_throws ArgumentError LifeAI._sample_categorical(zeros(Float32, 3), 0.5)
    @test_throws ArgumentError LifeAI._sample_categorical(Float32[], 0.5)
    @test_throws ArgumentError LifeAI._sample_categorical(Float32[0.5, NaN], 0.5)
    @test_throws ArgumentError LifeAI._sample_categorical(Float32[1.1, -0.1], 0.5)
    @test_throws ArgumentError LifeAI._sample_categorical(Float32[0.2, 0.2], 0.5)

    tied, _ = LifeAI._sampling_distribution(
        Float32[2, 1, 1, 0];
        top_k=2,
        top_p=1.0f0,
    )
    @test findall(isfinite, tied) == [1, 2]
    @test_throws ArgumentError LifeAI._sampling_distribution(logits; top_p=0)
    @test_throws ArgumentError LifeAI._sampling_distribution(logits; top_k=0)
    @test_throws ArgumentError LifeAI._sample_token(
        logits,
        rng;
        sample_uniform=1.0,
    )
    @test_throws ArgumentError LifeAI._sample_token(
        logits,
        rng;
        sample_uniform=false,
    )
end

@testset "Qwen3 generation length arithmetic is checked" begin
    model = GPTModel(
        17,
        8,
        2,
        1;
        max_seq_len=8,
        use_rope=true,
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
    )
    @test LifeAI._hf_generation_limits(model, 2, 7) === nothing
    @test_throws ArgumentError LifeAI._hf_generation_limits(model, 2, 8)
    overflow_failure = try
        LifeAI._hf_generation_limits(model, 2, typemax(Int))
        nothing
    catch error
        error
    end
    @test overflow_failure isa ArgumentError
    @test sprint(showerror, overflow_failure) ==
        "ArgumentError: prompt plus generated context exceeds the host integer range"
end

@testset "Qwen3 long-position rotate-half RoPE" begin
    fixture_dir = joinpath(@__DIR__, "fixtures", "qwen3_rope_long_context")
    reference = JSON3.read(read(joinpath(fixture_dir, "reference.json"), String))
    tensors = load_safetensors(joinpath(fixture_dir, "reference.safetensors"))
    positions = Int.(reference["positions_0_based"])
    head_dim = Int(reference["head_dim"])
    rope = RoPE(
        head_dim;
        max_seq_len=Int(reference["max_position_embeddings"]),
        theta=Float64(reference["rope_theta"]),
        style=:rotate_half,
    )
    input = tensors["input"]
    expected_cos = tensors["cos"]
    expected_sin = tensors["sin"]
    expected_rotated = tensors["rotated"]

    @test String(reference["revision"]) ==
          "c1899de289a04d12100db370d81485cdf75e47ca"
    @test String(reference["transformers_version"]) == "4.51.0"
    @test String(reference["rope_style"]) == "rotate_half"
    @test positions == [0, 2_048, 32_767, 40_959]
    @test size(input) == (length(positions), head_dim)

    for (column, position) in enumerate(positions)
        actual = apply_rope(
            reshape(input[column, :], head_dim, 1, 1, 1),
            rope;
            start_pos=position + 1,
        )
        @test vec(actual) ≈ expected_rotated[column, :] atol = 2.0f-5 rtol = 2.0f-5
        @test rope.cos_cache[:, position + 1] ≈
              expected_cos[column, 1:(head_dim ÷ 2)] atol = 2.0f-5 rtol = 2.0f-5
        @test rope.sin_cache[:, position + 1] ≈
              expected_sin[column, 1:(head_dim ÷ 2)] atol = 2.0f-5 rtol = 2.0f-5
        @test sum(abs2, actual) ≈
              sum(abs2, input[column, :]) atol = 1.0f-5 rtol = 1.0f-6
    end

    formula_input = reshape(Float32.(range(-1, 1; length=head_dim)), head_dim, 1, 1, 1)
    for start_pos in (1, 2_049, 32_768, 40_960)
        actual = apply_rope(formula_input, rope; start_pos)
        expected = similar(formula_input)
        position = Float32(start_pos - 1)
        for pair in 1:(head_dim ÷ 2)
            angle = position * rope.inv_freq[pair]
            first = formula_input[pair, 1, 1, 1]
            second = formula_input[pair + head_dim ÷ 2, 1, 1, 1]
            expected[pair, 1, 1, 1] = first * cos(angle) - second * sin(angle)
            expected[pair + head_dim ÷ 2, 1, 1, 1] =
                first * sin(angle) + second * cos(angle)
        end
        @test actual ≈ expected atol = 1.0f-6 rtol = 1.0f-6
        @test sum(abs2, actual) ≈
              sum(abs2, formula_input) atol = 1.0f-5 rtol = 1.0f-6
    end
    @test_throws AssertionError apply_rope(formula_input, rope; start_pos=40_961)
end

@testset "sampled Qwen3 cache matrix with frozen uniforms" begin
    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_qwen3_tokenizer_fixture(directory))
        model = GPTModel(
            vocab_size(tokenizer),
            8,
            2,
            1;
            max_seq_len=8,
            use_rope=true,
            norm_type=:rmsnorm,
            mlp_type=:swiglu,
            tie_embeddings=true,
        )
        parameters, states = Lux.setup(Xoshiro(909), model)
        bundle = (; model, parameters, states, tokenizer)
        uniforms = Float32[0.15, 0.85, 0.4]
        results = Dict(
            mode => generate_hf_text(
                bundle,
                "hi";
                cache=mode,
                max_new_tokens=3,
                strategy=:config,
                sample_uniforms=uniforms,
                stop_token_ids=Int[],
                capture_distribution=true,
            ) for mode in (:full, :dynamic, :static)
        )

        @test results[:full].strategy === :sample
        @test results[:full].generated_ids == results[:dynamic].generated_ids ==
              results[:static].generated_ids
        @test [step.sample_uniform for step in results[:dynamic].trace] == uniforms
        @test all(step -> step.top_k == 20, results[:dynamic].trace)
        @test all(step -> step.top_p == 0.95f0, results[:dynamic].trace)
        @test all(step -> step.candidate_count <= 20, results[:dynamic].trace)
        @test all(step -> step.distribution !== nothing, results[:dynamic].trace)
        @test all(
            step -> isapprox(sum(step.distribution.probabilities), 1.0f0; atol=1.0f-6),
            results[:dynamic].trace,
        )

        @test_throws ArgumentError generate_hf_text(
            bundle,
            "hi";
            strategy=:sample,
            max_new_tokens=2,
            sample_uniforms=[0.5],
        )
        @test_throws ArgumentError generate_hf_text(
            bundle,
            "hi";
            strategy=:sample,
            max_new_tokens=1,
            sample_uniforms=[NaN],
        )
        @test_throws ArgumentError generate_hf_text(
            bundle,
            "hi";
            strategy=:sample,
            max_new_tokens=1,
            sample_uniforms=[false],
        )
        @test_throws ArgumentError generate_hf_text(
            bundle,
            "hi";
            strategy=:greedy,
            temperature=0.6,
        )
    end
end
