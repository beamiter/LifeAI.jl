using Test
using JSON3
using SHA: sha256
using Base64: base64decode
using BFloat16s: BFloat16
using Random: MersenneTwister, randn
import LifeAI

const _CH149_CONTEXT_FIXTURE_PATH = joinpath(@__DIR__, "fixtures", "attention_context_stride.json")
const _CH149_CONTEXT_FIXTURE_SHA256 = "b044c1173e03b04c091f4129861caa9f6c6e8f1e77e867d5b9b7d5ed7e5d795c"

function _ch149_context_array(entry)
    entry["dtype"] == "BF16" || error("context fixture must use BF16")
    entry["layout"] == "julia_column_major" || error("context fixture layout changed")
    entry["byte_order"] == "little" || error("context fixture byte order changed")
    ENDIAN_BOM == 0x04030201 || error("context fixture currently requires a little-endian host")
    bytes = base64decode(String(entry["base64"]))
    shape = Tuple(Int.(entry["shape"]))
    length(bytes) == 2 * prod(shape) || error("context fixture payload length mismatch")
    return reshape(copy(reinterpret(BFloat16, bytes)), shape)
end

const _CH149_CONTEXT_FIXTURE = JSON3.read(read(_CH149_CONTEXT_FIXTURE_PATH, String))
const _CH149_CONTEXT_V = _ch149_context_array(_CH149_CONTEXT_FIXTURE["V"])
const _CH149_CONTEXT_W = _ch149_context_array(_CH149_CONTEXT_FIXTURE["W"])
const _CH149_CONTEXT_EXPECTED = _ch149_context_array(_CH149_CONTEXT_FIXTURE["expected_context"])
const _CH149_CONTEXT_LEGACY = _ch149_context_array(_CH149_CONTEXT_FIXTURE["legacy_unaligned_context"])

_ch149_context_bits(array) = collect(reinterpret(UInt16, vec(Array(array))))

@testset "Chapter 149 — frozen BF16 context leading-dimension regression" begin
    @test bytes2hex(sha256(read(_CH149_CONTEXT_FIXTURE_PATH))) == _CH149_CONTEXT_FIXTURE_SHA256
    @test _CH149_CONTEXT_FIXTURE["schema_version"] == 1
    @test _CH149_CONTEXT_FIXTURE["kind"] == "qwen3_vl_bf16_context_gemm_alignment_regression"
    @test _CH149_CONTEXT_FIXTURE["layer"] == 3
    @test _CH149_CONTEXT_FIXTURE["head_1_based"] == 2
    @test _CH149_CONTEXT_FIXTURE["num_kv_heads"] == 8
    @test _CH149_CONTEXT_FIXTURE["groups"] == 2
    @test _CH149_CONTEXT_FIXTURE["sequence_length"] == 152
    @test _CH149_CONTEXT_FIXTURE["capacity"] == 157
    @test size(_CH149_CONTEXT_V) == (128, 152)
    @test size(_CH149_CONTEXT_W) == (152, 2)
    @test size(_CH149_CONTEXT_EXPECTED) == size(_CH149_CONTEXT_LEGACY) == (128, 2)
    @test all(isfinite, _CH149_CONTEXT_V)
    @test all(isfinite, _CH149_CONTEXT_W)
    @test all(isfinite, _CH149_CONTEXT_EXPECTED)
    difference = Float32.(_CH149_CONTEXT_EXPECTED) .- Float32.(_CH149_CONTEXT_LEGACY)
    @test findall(!iszero, difference) == [CartesianIndex(25, 2)]
    @test maximum(abs, difference) == 1.4901161193847656f-8
    @test _CH149_CONTEXT_FIXTURE["observed_legacy_mismatch"]["unequal_elements"] == 1
end

if lowercase(get(ENV, "LIFEAI_TEST_CUDA", "false")) in ("1", "true", "yes")
    using CUDA
    CUDA.functional() || error("Chapter 149 context-stride regression requires functional CUDA")
    CUDA.allowscalar(false)

    function _ch149_prefix_allocated(weights, sequence)
        return @allocated LifeAI._qwen3_vl_attention_weight_prefix!(weights, sequence)
    end

    @testset "Chapter 149 — CUDA dense weight prefix preserves storage and context" begin
        fixture = _CH149_CONTEXT_FIXTURE
        head = Int(fixture["head_1_based"])
        heads, groups = Int(fixture["num_kv_heads"]), Int(fixture["groups"])
        sequence, capacity = Int(fixture["sequence_length"]), Int(fixture["capacity"])
        host_values = zeros(BFloat16, 128, sequence, heads)
        host_weights = zeros(BFloat16, sequence, groups, heads)
        host_values[:, :, head] .= _CH149_CONTEXT_V
        host_weights[:, :, head] .= _CH149_CONTEXT_W
        values, weights = CuArray(host_values), CuArray(host_weights)
        dense = LifeAI._bf16a_batched_mul(values, weights)
        @test all(isfinite, Array(dense))
        @test all(iszero, Array(dense)[:, :, setdiff(1:heads, [head])])

        padded_values = CUDA.zeros(BFloat16, 128, capacity, heads)
        padded_weights = CUDA.zeros(BFloat16, capacity, groups, heads)
        copyto!(view(padded_values, :, 1:sequence, :), values)
        copyto!(view(padded_weights, 1:sequence, :, :), weights)
        legacy = similar(dense)
        LifeAI._qwen3_vl_batched_mul!(legacy, view(padded_values, :, 1:sequence, :),
            view(padded_weights, 1:sequence, :, :))
        legacy_head = Array(legacy)[:, :, head]
        # The legacy cuBLAS choice is hardware-sensitive. Record whether this
        # runtime reproduces the negative control; repaired equality against
        # the current dense baseline is mandatory on every device.
        @test size(legacy) == size(dense)
        @test all(isfinite, legacy_head)
        @info "Chapter 149 legacy context layout" differs_from_dense=(_ch149_context_bits(legacy) != _ch149_context_bits(dense)) matches_frozen_legacy=(_ch149_context_bits(legacy_head) == _ch149_context_bits(_CH149_CONTEXT_LEGACY))
        storage, pointer_before = padded_weights.data, pointer(padded_weights)
        offset_before, allocation_before = padded_weights.offset, padded_weights.maxsize
        object_before = objectid(padded_weights)
        for current in (sequence, sequence - 1, sequence)
            LifeAI._qwen3_vl_attention_weight_prefix!(padded_weights, current)
            @test size(padded_weights) == (current, groups, heads)
            @test padded_weights.data === storage
            @test pointer(padded_weights) == pointer_before
            @test objectid(padded_weights) == object_before
            @test padded_weights.offset == offset_before
            @test padded_weights.maxsize == allocation_before
            current_weights = CuArray(host_weights[1:current, :, :])
            copyto!(padded_weights, current_weights)
            expected = LifeAI._bf16a_batched_mul(view(values, :, 1:current, :), current_weights)
            actual = similar(expected)
            LifeAI._qwen3_vl_batched_mul!(actual, view(padded_values, :, 1:current, :), padded_weights)
            @test _ch149_context_bits(actual) == _ch149_context_bits(expected)
        end
        _ch149_prefix_allocated(padded_weights, sequence - 1) # compile before measuring
        @test _ch149_prefix_allocated(padded_weights, sequence) == 0
    end

    @testset "Chapter 149 — CUDA complete attention refreshes changing weight shape" begin
        rng = MersenneTwister(149)
        queries_host = BFloat16.(0.05f0 .* randn(rng, Float32, 128, 16, 1, 1))
        queries = CuArray(queries_host)
        for capacity in (157, 160, 256)
            keys = CuArray(BFloat16.(0.05f0 .* randn(rng, Float32, 128, 8, capacity, 1)))
            values = CuArray(BFloat16.(0.05f0 .* randn(rng, Float32, 128, 8, capacity, 1)))
            scratch = LifeAI.init_qwen3_vl_decode_attention_scratch(queries;
                head_dim=128, num_attention_heads=16, num_kv_heads=8, capacity)
            physical_bytes = LifeAI.qwen3_vl_decode_attention_scratch_bytes(scratch)
            storage, ptr = scratch.weight_perm.data, pointer(scratch.weight_perm)
            allocation = scratch.weight_perm.maxsize
            for sequence in (152, 151, 152)
                key_prefix = view(keys, :, :, 1:sequence, :)
                value_prefix = view(values, :, :, 1:sequence, :)
                expected = LifeAI._bf16a_attention(queries, key_prefix, value_prefix;
                    scaling=1f0 / sqrt(128f0), mask=nothing)
                actual = LifeAI._qwen3_vl_decode_attention!(scratch, queries, key_prefix, value_prefix;
                    scaling=1f0 / sqrt(128f0))
                @test _ch149_context_bits(actual) == _ch149_context_bits(expected)
                @test size(scratch.weight_perm) == (sequence, 2, 8)
                @test scratch.weight_perm.data === storage
                @test pointer(scratch.weight_perm) == ptr
                @test scratch.weight_perm.maxsize == allocation
                @test LifeAI.qwen3_vl_decode_attention_scratch_bytes(scratch) == physical_bytes
            end
        end
    end
end
