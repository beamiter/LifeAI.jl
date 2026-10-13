using Test
using LifeAI

function _ch145_spec_with(spec; kwargs...)
    fields = NamedTuple{fieldnames(typeof(spec))}(Tuple(
        getfield(spec, name) for name in fieldnames(typeof(spec))
    ))
    return Qwen3VLTextSpec(values(merge(fields, (; kwargs...)))...)
end

function _ch145_prefill(parameters, cache)
    inputs = _ch46_tiny_prefill_inputs()
    return hf_qwen3_vl_text_prefill_static(
        parameters,
        inputs.input_ids,
        inputs.rope_layout;
        vision_features=_ch47_bf16_parameters(inputs.vision_features),
        cache,
        logits_to_keep=1,
    )
end

function _ch145_check_prefix(actual, expected)
    @test actual.position == expected.position
    @test actual.rope_delta == expected.rope_delta
    for (a, b) in zip(actual.layers, expected.layers)
        @test a.keys[:, :, 1:actual.position, :] ==
            b.keys[:, :, 1:expected.position, :]
        @test a.values[:, :, 1:actual.position, :] ==
            b.values[:, :, 1:expected.position, :]
    end
end

@testset "Chapter 145 — static decode binds the complete parameter tree" begin
    parameters = _ch47_bf16_parameters(_ch46_tiny_text_parameters())
    changed_blocks = Base.setindex(
        parameters.blocks,
        merge(first(parameters.blocks), (;
            v_weight=zero.(first(parameters.blocks).v_weight),
        )),
        1,
    )
    replacements = (
        zero_final_norm=merge(parameters, (; final_norm=zero.(parameters.final_norm))),
        changed_block=merge(parameters, (; blocks=changed_blocks)),
        changed_spec=merge(parameters, (;
            spec=_ch145_spec_with(parameters.spec; rms_norm_eps=1.0),
        )),
    )
    for (name, replacement) in pairs(replacements), installed_ready in (false, true)
        @testset "$name / installed Ready=$installed_ready" begin
            @test replacement.embedding === parameters.embedding
            reused = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
            if installed_ready
                _ch145_prefill(parameters, reused)
                hf_qwen3_vl_text_decode_step_static(parameters, 2, reused)
                @test reused.decode_pack isa LifeAI._Qwen3VLDecodeReady
                reset_qwen3_vl_static_kv_cache!(reused)
            else
                @test reused.decode_pack isa LifeAI._Qwen3VLDecodePack
            end
            fresh = init_qwen3_vl_static_kv_cache(replacement; capacity=12)
            reused_prefill, _ = _ch145_prefill(replacement, reused)
            fresh_prefill, _ = _ch145_prefill(replacement, fresh)
            @test reused_prefill.logits == fresh_prefill.logits
            _ch145_check_prefix(reused, fresh)
            for token in (2, 3)
                actual, _ = hf_qwen3_vl_text_decode_step_static(replacement, token, reused)
                expected, _ = hf_qwen3_vl_text_decode_step_static(replacement, token, fresh)
                @test actual == expected
                name === :zero_final_norm && @test all(iszero, actual)
                _ch145_check_prefix(reused, fresh)
                @test reused.decode_pack isa LifeAI._Qwen3VLDecodeReady
                pack = reused.decode_pack.pack
                @test pack.embedding === replacement.embedding
                @test pack.blocks === replacement.blocks
                @test pack.final_norm === replacement.final_norm
                @test pack.spec === replacement.spec
            end
        end
    end
end

@testset "Chapter 145 — unchanged parameters preserve Ready and scratch" begin
    parameters = _ch47_bf16_parameters(_ch46_tiny_text_parameters())
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    _ch145_prefill(parameters, cache)
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    ready = cache.decode_pack
    @test ready isa LifeAI._Qwen3VLDecodeReady
    scratch_names = filter(!=(:pack), fieldnames(typeof(ready)))
    scratch = map(name -> getfield(ready, name), scratch_names)
    hf_qwen3_vl_text_decode_step_static(parameters, 3, cache)
    @test cache.decode_pack === ready
    reset_qwen3_vl_static_kv_cache!(cache)
    _ch145_prefill(parameters, cache)
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    @test cache.decode_pack === ready
    for (name, original) in zip(scratch_names, scratch)
        @test getfield(cache.decode_pack, name) === original
    end
end

@testset "Chapter 145 — replacement validation precedes cache mutation" begin
    parameters = _ch47_bf16_parameters(_ch46_tiny_text_parameters())
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    _ch145_prefill(parameters, cache)
    hf_qwen3_vl_text_decode_step_static(parameters, 2, cache)
    ready = cache.decode_pack
    position, delta = cache.position, cache.rope_delta
    stored = map(layer -> (copy(layer.keys), copy(layer.values)), cache.layers)
    invalid = merge(parameters, (;
        spec=_ch145_spec_with(parameters.spec; max_position_embeddings=8),
    ))
    @test_throws ArgumentError hf_qwen3_vl_text_decode_step_static(invalid, 3, cache)
    @test cache.decode_pack === ready
    @test cache.position == position
    @test cache.rope_delta == delta
    for (layer, (keys, values)) in zip(cache.layers, stored)
        @test layer.keys == keys
        @test layer.values == values
    end
end
