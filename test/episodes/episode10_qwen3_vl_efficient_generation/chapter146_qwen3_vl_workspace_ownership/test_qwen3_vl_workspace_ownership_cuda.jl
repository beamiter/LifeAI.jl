using BFloat16s: BFloat16
using CUDA
using LifeAI
using Test

CUDA.functional() || error(
    "Qwen3-VL workspace ownership CUDA test requires functional CUDA",
)
CUDA.allowscalar(false)

# Workspace construction only consumes this metadata. No checkpoint, decoder
# forward, or synthetic decoder weights are needed to exercise ownership.
function _ch146_parameters()
    spec = LifeAI.Qwen3VLTextSpec(
        32, 16, 32, 1, 2, 1, 8, 1.0e-6, 10_000.0, 64,
        true, (2, 1, 1), true, "silu",
    )
    return (;
        spec,
        embedding=CUDA.zeros(BFloat16, 16, 32),
        final_norm=CUDA.ones(BFloat16, 16),
        blocks=(nothing,),
    )
end

function _ch146_cache(parameters, capacity)
    cache = LifeAI.init_qwen3_vl_static_kv_cache(parameters; capacity)
    LifeAI._qwen3_vl_device_decode_workspace!(
        cache, parameters.spec, parameters.embedding,
    )
    @test cache.decode_pack isa LifeAI._Qwen3VLDecodeReady
    return cache
end

function _ch146_arrays(ready)
    arrays = Any[]
    for field in (
        :token_embedding, :mrope, :attention, :mlp, :rope, :qk_norm,
        :hidden, :gate_up, :residual, :tied, :qkv,
    )
        scratch = getfield(ready, field)
        for member in fieldnames(typeof(scratch))
            value = getfield(scratch, member)
            value isa AbstractArray && push!(arrays, value)
        end
    end
    return arrays
end

@testset "Qwen3-VL CUDA live cache workspace ownership" begin
    parameters = _ch146_parameters()
    first = _ch146_cache(parameters, 8)
    ready = first.decode_pack
    arrays = _ch146_arrays(ready)
    shapes = map(size, arrays)
    pointers = map(pointer, arrays)
    workspace = first.decode_workspace
    output = ready.tied.output
    fill!(output, BFloat16(3))
    CUDA.synchronize()
    expected = Array(output)

    second = _ch146_cache(parameters, 16)
    other = second.decode_pack
    other_arrays = _ch146_arrays(other)
    @test other !== ready
    @test length(other_arrays) == length(arrays)
    @test all(pair -> pair[1] !== pair[2], zip(arrays, other_arrays))
    @test all(pair -> pointer(pair[1]) != pointer(pair[2]), zip(arrays, other_arrays))
    @test all(pair -> pair[1] !== pair[2], zip(workspace, second.decode_workspace))
    @test ready.attention.capacity == 8
    @test map(size, arrays) == shapes
    @test map(pointer, arrays) == pointers
    fill!(other.tied.output, BFloat16(7))
    CUDA.synchronize()
    @test Array(output) == expected

    # Reset retains this request's allocations, including its installed Ready.
    first.position = 3
    first.rope_delta = 2
    @test LifeAI.reset_qwen3_vl_static_kv_cache!(first) === first
    @test first.position == 0
    @test first.rope_delta == 0
    @test first.decode_pack === ready
    @test first.decode_workspace === workspace
    LifeAI._qwen3_vl_device_decode_workspace!(
        first, parameters.spec, parameters.embedding,
    )
    @test first.decode_pack === ready
    @test map(size, arrays) == shapes
    @test map(pointer, arrays) == pointers
    @test Array(output) == expected
end

# Let the request die while only its returned logits buffer escapes. A dead
# weak owner does not mean its scratch arrays are safe to lend to another cache.
@noinline function _ch146_retained_output(parameters)
    cache = _ch146_cache(parameters, 8)
    output = cache.decode_pack.tied.output
    fill!(output, BFloat16(5))
    CUDA.synchronize()
    return WeakRef(cache), output, Array(output), pointer(output)
end

@testset "Qwen3-VL CUDA retained output survives owner collection" begin
    parameters = _ch146_parameters()
    owner, output, expected, output_pointer = _ch146_retained_output(parameters)
    GC.gc(true)
    GC.gc(true)
    @test owner.value === nothing

    third = _ch146_cache(parameters, 24)
    @test third.decode_pack.tied.output !== output
    fill!(third.decode_pack.tied.output, BFloat16(9))
    CUDA.synchronize()
    @test pointer(output) == output_pointer
    @test Array(output) == expected
end
