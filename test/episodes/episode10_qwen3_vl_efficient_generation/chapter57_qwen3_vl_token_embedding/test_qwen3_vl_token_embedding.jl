using BFloat16s: BFloat16
using NNlib: gather
using Test
using LifeAI

@testset "Qwen3-VL token embedding scratch" begin
    prototype = zeros(BFloat16, 1)
    scratch = LifeAI.init_qwen3_vl_token_embedding_scratch(
        prototype;
        hidden_size=8,
    )
    @test LifeAI.qwen3_vl_token_embedding_scratch_bytes(scratch) ==
        LifeAI.qwen3_vl_token_embedding_scratch_bytes(; hidden_size=8)
    embedding = randn(BFloat16, 8, 5)
    got = LifeAI._qwen3_vl_decode_token_embedding!(scratch, embedding, 3)
    @test got === scratch.output
    @test got == reshape(gather(embedding, fill(3, 1, 1)), 8, 1, 1)
    @test_throws ArgumentError LifeAI._qwen3_vl_decode_token_embedding!(
        scratch,
        embedding,
        0,
    )
    @test_throws ArgumentError LifeAI.qwen3_vl_token_embedding_scratch_bytes(;
        hidden_size=0,
    )
end

@testset "Qwen3-VL static decode token embedding scratch ownership" begin
    isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || return
    to_bf16(value) = value isa AbstractArray{<:AbstractFloat} ? BFloat16.(value) :
        value isa NamedTuple ? NamedTuple{keys(value)}(map(to_bf16, values(value))) :
        value isa Tuple ? map(to_bf16, value) : value
    inputs = _ch46_tiny_prefill_inputs()
    parameters = to_bf16(_ch46_tiny_text_parameters())
    features = to_bf16(inputs.vision_features)
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=features,
        cache,
        logits_to_keep=1,
    )
    @test cache.token_embedding_scratch === nothing
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.token_embedding_scratch
    @test scratch isa LifeAI.Qwen3VLTokenEmbeddingScratch
    output = scratch.output
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.token_embedding_scratch === scratch
    @test scratch.output === output
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.token_embedding_scratch === scratch
end
