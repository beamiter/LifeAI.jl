using Test
using LifeAI

isdefined(@__MODULE__, :_ch46_tiny_text_parameters) || error(
    "Chapter 47 profiling tests require the Chapter 46 tiny decode fixture",
)

const _CH47_BLOCK_STAGES = Symbol[
    :pre_attention_norm,
    :qkv_projection,
    :qk_norm,
    :qk_rope,
    :kv_write,
    :attention,
    :attention_output_projection_residual,
    :post_attention_norm,
    :mlp_gate_up_projection,
    :mlp_activation,
    :mlp_down_projection_residual,
]

function _ch47_expected_stages(layer_count::Int)
    stages = Tuple{Symbol,Int}[
        (:token_embedding, 0),
        (:mrope_prepare, 0),
    ]
    for layer in 1:layer_count, stage in _CH47_BLOCK_STAGES
        push!(stages, (stage, layer))
    end
    append!(stages, [(:final_norm, 0), (:vocab_logits, 0)])
    return stages
end

@testset "Chapter 47 — decode profiling hook is numerically transparent" begin
    parameters = _ch46_tiny_text_parameters()
    inputs = _ch46_tiny_prefill_inputs()
    baseline_cache = init_qwen3_vl_static_kv_cache(parameters; capacity=10)
    profiled_cache = init_qwen3_vl_static_kv_cache(parameters; capacity=10)

    baseline_prefill, _ = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=baseline_cache,
    )
    profiled_prefill, _ = hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=inputs.vision_features,
        cache=profiled_cache,
    )
    token = _ch46_top_two(baseline_prefill.logits).ids[1]
    @test profiled_prefill.logits == baseline_prefill.logits

    baseline_logits, returned_baseline = hf_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        baseline_cache,
    )
    observed = Tuple{Symbol,Int}[]
    runner = function (stage, layer, thunk)
        push!(observed, (stage, layer))
        return thunk()
    end
    profiled_logits, returned_profiled =
        LifeAI._profile_qwen3_vl_text_decode_step_static(
            parameters,
            token,
            profiled_cache,
            runner,
        )

    @test returned_baseline === baseline_cache
    @test returned_profiled === profiled_cache
    @test profiled_logits == baseline_logits
    @test profiled_cache.position == baseline_cache.position == 9
    @test profiled_cache.rope_delta == baseline_cache.rope_delta == -2
    @test observed == _ch47_expected_stages(parameters.spec.num_hidden_layers)
    @test length(observed) == 48
    @test count(==((:kv_write, 1)), observed) == 1
    for layer in eachindex(profiled_cache.layers)
        @test profiled_cache.layers[layer].keys ==
            baseline_cache.layers[layer].keys
        @test profiled_cache.layers[layer].values ==
            baseline_cache.layers[layer].values
    end

    @test_throws ArgumentError LifeAI._profile_qwen3_vl_text_decode_step_static(
        parameters,
        token,
        profiled_cache,
        nothing,
    )
end
