using BFloat16s: BFloat16
using Test
using LifeAI

@testset "Qwen3-VL tied logits scratch" begin
    prototype = zeros(BFloat16, 1)
    for (hidden_size, vocab_size) in ((8, 16), (16, 32))
        scratch = LifeAI.init_qwen3_vl_tied_logits_scratch(
            prototype;
            hidden_size,
            vocab_size,
        )
        @test LifeAI.qwen3_vl_tied_logits_scratch_bytes(scratch) ==
            LifeAI.qwen3_vl_tied_logits_scratch_bytes(; vocab_size)
        embedding = randn(BFloat16, hidden_size, vocab_size)
        hidden = randn(BFloat16, hidden_size, 1, 1)
        got = LifeAI._qwen3_vl_decode_tied_logits!(scratch, embedding, hidden)
        @test got === scratch.output
        @test got == LifeAI._qwen3_vl_project_tied(embedding, hidden)
        saved = copy(got)
        hidden2 = randn(BFloat16, hidden_size, 1, 1)
        LifeAI._qwen3_vl_decode_tied_logits!(scratch, embedding, hidden2)
        @test got == LifeAI._qwen3_vl_project_tied(embedding, hidden2)
        @test saved == LifeAI._qwen3_vl_project_tied(embedding, hidden)
    end
    @test_throws ArgumentError LifeAI.qwen3_vl_tied_logits_scratch_bytes(;
        vocab_size=0,
    )
end

@testset "Qwen3-VL static decode tied logits scratch ownership" begin
    isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || return
    to_bf16(value) = value isa AbstractArray{<:AbstractFloat} ? BFloat16.(value) :
        value isa NamedTuple ? NamedTuple{keys(value)}(map(to_bf16, values(value))) :
        value isa Tuple ? map(to_bf16, value) : value
    inputs = _ch46_tiny_prefill_inputs()
    float_parameters = _ch46_tiny_text_parameters()
    float_cache = init_qwen3_vl_static_kv_cache(float_parameters; capacity=12)
    hf_qwen3_vl_text_prefill_static(
        float_parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=float_cache,
        logits_to_keep=1,
    )
    hf_qwen3_vl_text_decode_step_static(float_parameters, 2, float_cache)
    @test float_cache.tied_logits_scratch === nothing

    parameters = to_bf16(float_parameters)
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
    @test cache.tied_logits_scratch === nothing
    first, _ = hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    scratch = cache.tied_logits_scratch
    @test scratch isa LifeAI.Qwen3VLTiedLogitsScratch
    @test first === scratch.output
    kept = copy(first)
    second, _ = hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test second === first
    @test kept !== first
    @test scratch.output === first
    reset_qwen3_vl_static_kv_cache!(cache)
    @test cache.tied_logits_scratch === scratch
end
