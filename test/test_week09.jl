using Test
using Random: Xoshiro
using Lux
using JSON3
import LifeAI
using LifeAI:
    GPTModel,
    RoPE,
    apply_rope,
    generate_hf_text,
    hf_generation_config,
    load_hf_qwen3_tokenizer,
    load_hf_qwen3_bundle,
    load_safetensors,
    vocab_size

if !isdefined(@__MODULE__, :week08_tokenizer_payloads)
    include("week08_fixture.jl")
end

const _WEEK09_MODEL_DIR = get(ENV, "LIFEAI_QWEN3_MODEL_DIR", "")
const _WEEK09_REFERENCE_DIR = get(ENV, "LIFEAI_QWEN3_SAMPLING_REFERENCE_DIR", "")

if !isempty(_WEEK09_MODEL_DIR) && !isempty(_WEEK09_REFERENCE_DIR)
    @testset "Qwen3-0.6B official sampling integration" begin
        reference = JSON3.read(read(joinpath(_WEEK09_REFERENCE_DIR, "reference.json"), String))
        tensors = load_safetensors(joinpath(_WEEK09_REFERENCE_DIR, "reference.safetensors"))
        uniforms = Float32.(reference["uniforms"])
        bundle = load_hf_qwen3_bundle(
            _WEEK09_MODEL_DIR;
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
    @info "Skipping Qwen3 Week 09 integration; set LIFEAI_QWEN3_MODEL_DIR and LIFEAI_QWEN3_SAMPLING_REFERENCE_DIR"
end

@testset "Qwen3 generation config contract" begin
    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_week08_tokenizer_fixture(directory))
        config = hf_generation_config(tokenizer)
        @test config.bos_id == 259
        @test config.eos_ids == [261, 259]
        @test config.pad_id == 259
        @test config.do_sample
        @test config.temperature == 0.6f0
        @test config.top_k == 20
        @test config.top_p == 0.95f0
        @test config.transformers_version == "4.51.0"

        push!(config.eos_ids, 1)
        @test hf_generation_config(tokenizer).eos_ids == [261, 259]
    end

    mutations = [
        payloads -> (payloads.generation_config["do_sample"] = "true"),
        payloads -> (payloads.generation_config["temperature"] = 0.0),
        payloads -> (payloads.generation_config["top_k"] = 0),
        payloads -> (payloads.generation_config["top_p"] = 1.1),
        payloads -> (payloads.generation_config["transformers_version"] = ""),
        payloads -> (payloads.generation_config["min_p"] = 0.1),
    ]
    for mutate! in mutations
        mktempdir() do directory
            payloads = week08_tokenizer_payloads()
            mutate!(payloads)
            write_week08_tokenizer_fixture(directory; payloads)
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

    tied, _ = LifeAI._sampling_distribution(
        Float32[2, 1, 1, 0];
        top_k=2,
        top_p=1.0f0,
    )
    @test findall(isfinite, tied) == [1, 2, 3]
    @test_throws ArgumentError LifeAI._sampling_distribution(logits; top_p=0)
    @test_throws ArgumentError LifeAI._sampling_distribution(logits; top_k=0)
    @test_throws ArgumentError LifeAI._sample_token(
        logits,
        rng;
        sample_uniform=1.0,
    )
end

@testset "Qwen3 long-position rotate-half RoPE" begin
    rope = RoPE(128; max_seq_len=40_960, theta=1_000_000.0, style=:rotate_half)
    input = reshape(Float32.(range(-1, 1; length=128)), 128, 1, 1, 1)
    for start_pos in (1, 2_049, 32_768, 40_960)
        actual = apply_rope(input, rope; start_pos)
        expected = similar(input)
        position = Float32(start_pos - 1)
        for pair in 1:64
            angle = position * rope.inv_freq[pair]
            first = input[pair, 1, 1, 1]
            second = input[pair + 64, 1, 1, 1]
            expected[pair, 1, 1, 1] = first * cos(angle) - second * sin(angle)
            expected[pair + 64, 1, 1, 1] = first * sin(angle) + second * cos(angle)
        end
        @test actual ≈ expected atol = 1.0f-6 rtol = 1.0f-6
        @test sum(abs2, actual) ≈ sum(abs2, input) atol = 1.0f-5 rtol = 1.0f-6
    end
    @test_throws AssertionError apply_rope(input, rope; start_pos=40_961)
end

@testset "sampled Qwen3 cache matrix with frozen uniforms" begin
    mktempdir() do directory
        tokenizer = load_hf_qwen3_tokenizer(write_week08_tokenizer_fixture(directory))
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
            strategy=:greedy,
            temperature=0.6,
        )
    end
end
