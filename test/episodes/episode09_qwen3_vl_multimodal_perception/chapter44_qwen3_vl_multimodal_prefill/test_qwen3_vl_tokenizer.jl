using Test
using LifeAI: HFQwen3Tokenizer,
    hf_generation_config,
    load_hf_qwen3_vl_tokenizer,
    load_tokenizer,
    save_tokenizer,
    tokenizer_fingerprint

isdefined(@__MODULE__, :qwen3_tokenizer_fixture_payloads) || include(joinpath(
    @__DIR__,
    "..",
    "..",
    "..",
    "support",
    "qwen3_tokenizer_fixture.jl",
))

function _ch44_vl_tokenizer_payloads()
    payloads = qwen3_tokenizer_fixture_payloads()
    delete!(payloads.tokenizer["model"], "ignore_merges")
    payloads.tokenizer["model"]["merges"] = [
        join(String.(pair), " ") for pair in payloads.tokenizer["model"]["merges"]
    ]
    payloads.generation_config["repetition_penalty"] = 1.0
    return payloads
end

@testset "Chapter 44 — Qwen3-VL tokenizer profile and artifact" begin
    mktempdir() do directory
        tokenizer_dir = joinpath(directory, "tokenizer")
        write_qwen3_tokenizer_fixture(
            tokenizer_dir;
            payloads=_ch44_vl_tokenizer_payloads(),
        )
        tokenizer = load_hf_qwen3_vl_tokenizer(
            tokenizer_dir;
            revision="qwen3-vl-tiny-test",
        )
        @test tokenizer isa HFQwen3Tokenizer
        @test tokenizer.profile == :qwen3_vl_generation
        @test hf_generation_config(tokenizer).bos_id == 259

        artifact = joinpath(directory, "tokenizer.toml")
        save_tokenizer(artifact, tokenizer)
        restored = load_tokenizer(artifact)
        @test restored.profile == :qwen3_vl_generation
        @test tokenizer_fingerprint(restored) == tokenizer_fingerprint(tokenizer)
    end
end

@testset "Chapter 44 — Qwen3-VL tokenizer fields fail closed" begin
    mutations = [
        payloads -> (payloads.tokenizer["model"]["ignore_merges"] = false),
        payloads -> (payloads.tokenizer["model"]["merges"] = [["h", "i"]]),
        payloads -> delete!(payloads.generation_config, "repetition_penalty"),
        payloads -> (payloads.generation_config["repetition_penalty"] = 1.1),
    ]
    for mutate! in mutations
        mktempdir() do directory
            payloads = _ch44_vl_tokenizer_payloads()
            mutate!(payloads)
            write_qwen3_tokenizer_fixture(directory; payloads)
            @test_throws ArgumentError load_hf_qwen3_vl_tokenizer(directory)
        end
    end
end
