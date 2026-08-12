using Test
using BFloat16s: BFloat16
using Reactant
using LifeAI: load_hf_qwen3_compact_model

isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
    include(joinpath(@__DIR__, "..", "..", "..", "support", "qwen3_tiny_model_fixture.jl"))

@testset "compact fixed-chunk prefill compiles on Reactant CPU" begin
    Reactant.set_default_backend(get(ENV, "LIFEAI_XLA_BACKEND", "cpu"))
    mktempdir() do directory
        _qwen3_tiny_model_fixture_dir(directory; tie=false)
        loaded = load_hf_qwen3_compact_model(
            directory;
            max_seq_len=16,
            weight_dtype=BFloat16,
        )
        model = loaded.model
        parameters_host = loaded.parameters
        parameters = Reactant.to_rarray(parameters_host)
        rope = first(values(model.blocks.layers)).attn.rope
        cos_table = Reactant.to_rarray(BFloat16.(rope.cos_cache))
        sin_table = Reactant.to_rarray(BFloat16.(rope.sin_cache))
        cache_shape = (
            model.head_dim,
            model.num_kv_heads,
            model.max_seq_len,
            1,
        )
        key_caches = Tuple(
            Reactant.to_rarray(zeros(BFloat16, cache_shape))
            for _ in 1:model.num_layers
        )
        value_caches = Tuple(
            Reactant.to_rarray(zeros(BFloat16, cache_shape))
            for _ in 1:model.num_layers
        )
        token_state = Reactant.to_rarray(reshape(collect(2:5), :, 1))
        position_state = Reactant.to_rarray(zeros(Int32, 1))
        key_positions =
            Reactant.to_rarray(Int32.(collect(1:model.max_seq_len)))

        prefill = (
            ps,
            tokens,
            kc,
            vc,
            position,
            cos_t,
            sin_t,
            kp,
        ) -> LifeAI._bf16a_static_prefill_chunk_greedy(
            model,
            ps,
            tokens,
            kc,
            vc,
            position,
            cos_t,
            sin_t,
            kp,
        )
        compiled = Reactant.@compile prefill(
            parameters,
            token_state,
            key_caches,
            value_caches,
            position_state,
            cos_table,
            sin_table,
            key_positions,
        )
        first_token, first_position = compiled(
            parameters,
            token_state,
            key_caches,
            value_caches,
            position_state,
            cos_table,
            sin_table,
            key_positions,
        )
        @test length(Array(first_token)) == 1
        @test Array(first_position) == Int32[4]

        second_tokens =
            Reactant.to_rarray(reshape(collect(6:9), :, 1))
        second_token, second_position = compiled(
            parameters,
            second_tokens,
            key_caches,
            value_caches,
            first_position,
            cos_table,
            sin_table,
            key_positions,
        )
        @test length(Array(second_token)) == 1
        @test Array(second_position) == Int32[8]

        host_keys = Tuple(
            zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
        )
        host_values = Tuple(
            zeros(BFloat16, cache_shape) for _ in 1:model.num_layers
        )
        host_logits = LifeAI._bf16a_static_prefill(
            model,
            parameters_host,
            reshape(collect(2:9), :, 1),
            host_keys,
            host_values,
            BFloat16.(rope.cos_cache),
            BFloat16.(rope.sin_cache),
            LifeAI._bf16a_causal_mask(8, 8),
        )
        @test only(Array(second_token)) ==
            argmax(vec(Float32.(host_logits[:, end, 1])))
    end
end
