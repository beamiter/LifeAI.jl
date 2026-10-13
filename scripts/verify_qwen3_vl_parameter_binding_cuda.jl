#!/usr/bin/env julia

# Synthetic BF16 CUDA regression; no checkpoint or download is required.
using CUDA
using BFloat16s: BFloat16
using LifeAI
using Test

CUDA.functional() || error("a functional CUDA device is required")
CUDA.allowscalar(false)

function binding_values(rows, columns, offset)
    return BFloat16.(reshape(
        Float32[0.02f0 * sin(0.173f0 * Float32(offset + i))
            for i in 1:(rows * columns)], rows, columns,
    ))
end

function binding_parameters()
    spec = LifeAI.Qwen3VLTextSpec(
        32, 16, 32, 4, 2, 1, 8, 1.0e-6, 10_000.0, 64,
        true, (2, 1, 1), true, "silu",
    )
    blocks = ntuple(4) do index
        offset = 10_000 * index
        return (;
            norm1=CUDA.ones(BFloat16, 16),
            q_weight=CuArray(binding_values(16, 16, offset + 100)),
            k_weight=CuArray(binding_values(8, 16, offset + 200)),
            v_weight=CuArray(binding_values(8, 16, offset + 300)),
            o_weight=CuArray(binding_values(16, 16, offset + 400)),
            q_norm=CUDA.ones(BFloat16, 8),
            k_norm=CUDA.ones(BFloat16, 8),
            norm2=CUDA.ones(BFloat16, 16),
            gate_weight=CuArray(binding_values(32, 16, offset + 500)),
            up_weight=CuArray(binding_values(32, 16, offset + 600)),
            down_weight=CuArray(binding_values(16, 32, offset + 700)),
        )
    end
    return (;
        embedding=CuArray(binding_values(16, 32, 10)),
        blocks,
        final_norm=CUDA.ones(BFloat16, 16),
        spec,
    )
end

function binding_spec(spec; kwargs...)
    fields = NamedTuple{fieldnames(typeof(spec))}(Tuple(
        getfield(spec, name) for name in fieldnames(typeof(spec))
    ))
    return LifeAI.Qwen3VLTextSpec(values(merge(fields, (; kwargs...)))...)
end

function binding_prefill(parameters, cache)
    positions = reshape(repeat(collect(0:3); inner=3), 3, 4, 1)
    layout = LifeAI.Qwen3VLRopeLayout(
        positions, zeros(Int, 1, 1), falses(4, 1), trues(4, 1),
    )
    return LifeAI.hf_qwen3_vl_text_prefill_static(
        parameters, collect(1:4), layout; cache, logits_to_keep=1,
    )
end

function binding_prefix_equal(a, b)
    @test a.position == b.position
    @test a.rope_delta == b.rope_delta
    for (left, right) in zip(a.layers, b.layers)
        @test Array(left.keys[:, :, 1:a.position, :]) ==
            Array(right.keys[:, :, 1:b.position, :])
        @test Array(left.values[:, :, 1:a.position, :]) ==
            Array(right.values[:, :, 1:b.position, :])
    end
end

function binding_check_pack(cache, parameters)
    @test cache.decode_pack isa LifeAI._Qwen3VLDecodeReady
    pack = cache.decode_pack.pack
    @test pack.embedding === parameters.embedding
    @test pack.blocks === parameters.blocks
    @test pack.final_norm === parameters.final_norm
    @test pack.spec === parameters.spec
end

function binding_compare(parameters, reused; zero_logits=false)
    fresh = LifeAI.init_qwen3_vl_static_kv_cache(parameters; capacity=8)
    a, _ = binding_prefill(parameters, reused)
    b, _ = binding_prefill(parameters, fresh)
    @test Array(a.logits) == Array(b.logits)
    binding_prefix_equal(reused, fresh)
    for token in (5, 6)
        actual, _ = LifeAI.hf_qwen3_vl_text_decode_step_static(parameters, token, reused)
        # Copy before running the other cache: global CUDA launch state is reused.
        actual_host = Array(actual)
        expected, _ = LifeAI.hf_qwen3_vl_text_decode_step_static(parameters, token, fresh)
        @test actual_host == Array(expected)
        zero_logits && @test all(iszero, actual_host)
        binding_prefix_equal(reused, fresh)
        binding_check_pack(reused, parameters)
        binding_check_pack(fresh, parameters)
    end
end

@testset "Chapter 145 — CUDA parameter binding" begin
    parameters = binding_parameters()
    replacements = (
        zero_final_norm=merge(parameters, (; final_norm=CUDA.zeros(BFloat16, 16))),
        changed_block=merge(parameters, (; blocks=Base.setindex(
            parameters.blocks,
            merge(first(parameters.blocks), (;
                v_weight=CUDA.zeros(BFloat16, size(first(parameters.blocks).v_weight)),
            )), 1,
        ))),
        changed_spec=merge(parameters, (;
            spec=binding_spec(parameters.spec; rms_norm_eps=1.0),
        )),
    )
    for (name, replacement) in pairs(replacements), ready in (false, true)
        @testset "$name / installed Ready=$ready" begin
            @test replacement.embedding === parameters.embedding
            cache = LifeAI.init_qwen3_vl_static_kv_cache(parameters; capacity=8)
            if ready
                binding_prefill(parameters, cache)
                LifeAI.hf_qwen3_vl_text_decode_step_static(parameters, 5, cache)
                binding_check_pack(cache, parameters)
                LifeAI.reset_qwen3_vl_static_kv_cache!(cache)
            end
            binding_compare(replacement, cache; zero_logits=name === :zero_final_norm)
        end
    end

    @testset "unchanged Ready and global shell parameter rebinding" begin
        first_cache = LifeAI.init_qwen3_vl_static_kv_cache(parameters; capacity=8)
        binding_prefill(parameters, first_cache)
        LifeAI.hf_qwen3_vl_text_decode_step_static(parameters, 5, first_cache)
        ready = first_cache.decode_pack
        workspace = first_cache.decode_workspace
        LifeAI.reset_qwen3_vl_static_kv_cache!(first_cache)
        binding_prefill(parameters, first_cache)
        LifeAI.hf_qwen3_vl_text_decode_step_static(parameters, 5, first_cache)
        @test first_cache.decode_pack === ready
        @test first_cache.decode_workspace === workspace
        binding_check_pack(first_cache, parameters)
        # A different parameter tree replaces the process-wide CUDA shells.
        alternate = replacements.changed_spec
        other_cache = LifeAI.init_qwen3_vl_static_kv_cache(alternate; capacity=8)
        binding_compare(alternate, other_cache)
        # A newly created A cache must not pick up B's captured spec or weights.
        again = LifeAI.init_qwen3_vl_static_kv_cache(parameters; capacity=8)
        binding_compare(parameters, again)
    end
end
CUDA.synchronize()
println("CUDA device: ", CUDA.name(CUDA.device()))
println("Synthetic tiny BF16 parameter-binding validation passed; no real-model parity or allocation claim.")
