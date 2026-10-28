using Test, LifeAI

@testset "Flint token callback is incremental and cancellable" begin
    fixture = joinpath(@__DIR__, "flint", "fixtures", "tiny")
    bundle = load_hf_flint_model(fixture)
    prompt = [1, 3, 5]
    expected = generate_flint_tokens(bundle,prompt;max_new_tokens=5,temperature=0,eos_ids=Int[])
    observed = Int[]
    actual = generate_flint_tokens(bundle,prompt;max_new_tokens=5,temperature=0,eos_ids=Int[],
        on_token=token -> push!(observed,token))
    @test observed == actual == expected
    observed = Int[]
    short = generate_flint_tokens(bundle,prompt;max_new_tokens=5,temperature=0,eos_ids=Int[],
        on_token=token -> (push!(observed,token); length(observed) < 2))
    @test short == observed == expected[1:2]
    observed = Int[]
    stopped = generate_flint_tokens(bundle,prompt;max_new_tokens=5,temperature=0,eos_ids=[expected[1]],
        on_token=token -> push!(observed,token))
    @test stopped == observed == expected[1:1]
    @test_throws ErrorException generate_flint_tokens(bundle,prompt;max_new_tokens=5,temperature=0,
        on_token=token -> error("client disconnected"))
    @test isempty(generate_flint_tokens(bundle,prompt;max_new_tokens=0,
        on_token=token -> error("must not be called")))
    progress = Any[]
    quantized = load_hf_flint_quantized(fixture;scheme=:nvfp4,
        on_progress=(name,index,total)->push!(progress,(name,index,total)))
    @test length(progress) == length(quantized.parameters)
    @test [entry[2] for entry in progress] == collect(1:length(progress))
    @test all(entry[3] == length(progress) for entry in progress)
end
