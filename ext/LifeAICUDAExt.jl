module LifeAICUDAExt

using BFloat16s: BFloat16
import CUDA
import LifeAI
import LifeAI: qwen3_device_sparse_expert_dispatch

mutable struct _Qwen3CUDAPinnedExpertUpload
    stream
    max_inflight::Int
    pinned_arrays::Vector{Any}
    wait_seconds::Float64
    pinned_bytes::Int
end

LifeAI._qwen3_moe_pinned_upload_supported(::CUDA.CuArray) = true
LifeAI._qwen3_moe_host_buffer_reuse_supported(::CUDA.CuArray) = true

function LifeAI._qwen3_moe_begin_expert_upload(
    prototype::CUDA.CuArray,
    pinned::Bool,
    max_inflight::Int,
)
    pinned || return nothing
    return _Qwen3CUDAPinnedExpertUpload(
        CUDA.CuStream(),
        max_inflight,
        Any[],
        0.0,
        0,
    )
end

function _qwen3_cuda_flush_pinned_expert_upload!(
    state::_Qwen3CUDAPinnedExpertUpload,
)
    isempty(state.pinned_arrays) && return state
    started = time_ns()
    CUDA.synchronize(state.stream)
    state.wait_seconds += (time_ns() - started) / 1.0e9
    for array in state.pinned_arrays
        finalize(array)
    end
    empty!(state.pinned_arrays)
    return state
end

function LifeAI._qwen3_moe_upload_host_expert(
    prototype::CUDA.CuArray,
    host_parameters,
    to_device,
    state::_Qwen3CUDAPinnedExpertUpload,
)
    length(state.pinned_arrays) >= 3 * state.max_inflight &&
        _qwen3_cuda_flush_pinned_expert_upload!(state)
    for array in values(host_parameters)
        CUDA.pin(array)
        push!(state.pinned_arrays, array)
        state.pinned_bytes = Base.checked_add(
            state.pinned_bytes,
            sizeof(eltype(array)) * length(array),
        )
    end
    return CUDA.stream!(state.stream) do
        gate_proj = similar(
            prototype,
            eltype(host_parameters.gate_proj),
            size(host_parameters.gate_proj),
        )
        up_proj = similar(
            prototype,
            eltype(host_parameters.up_proj),
            size(host_parameters.up_proj),
        )
        down_proj = similar(
            prototype,
            eltype(host_parameters.down_proj),
            size(host_parameters.down_proj),
        )
        copyto!(gate_proj, host_parameters.gate_proj)
        copyto!(up_proj, host_parameters.up_proj)
        copyto!(down_proj, host_parameters.down_proj)
        return (; gate_proj, up_proj, down_proj)
    end
end

function LifeAI._qwen3_moe_finish_expert_upload(
    prototype::CUDA.CuArray,
    state::_Qwen3CUDAPinnedExpertUpload,
)
    _qwen3_cuda_flush_pinned_expert_upload!(state)
    return (;
        wait_seconds=state.wait_seconds,
        pinned_bytes=state.pinned_bytes,
    )
end

# CUDA specialization for Qwen3 MoE compact dispatch. The portable
# route-major fallback gathers and replicates all three expert matrices for
# every route. These kernels instead index the original expert tensor in place:
# one kernel forms routed SwiGLU hidden states, one applies the down projection,
# and one combines the fixed top-k routes back into token order.

function _qwen3_cuda_initialize_route_buckets_kernel!(
    route_permutation,
    expert_counts,
    expert_indices,
    pair_count,
)
    pair_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if pair_index <= pair_count
        expert_index = Int(expert_indices[pair_index])
        @inbounds route_permutation[pair_index] = Int32(pair_index)
        CUDA.@atomic expert_counts[expert_index] += Int32(1)
    end
    return
end

function _qwen3_cuda_finalize_route_offsets_kernel!(
    expert_offsets,
    inclusive_counts,
    num_experts,
)
    expert_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if expert_index == 1
        @inbounds expert_offsets[1] = Int32(1)
    end
    if expert_index <= num_experts
        @inbounds expert_offsets[expert_index + 1] =
            inclusive_counts[expert_index] + Int32(1)
    end
    return
end

function _qwen3_cuda_pad_expert_counts_kernel!(
    padded_counts,
    expert_counts,
    num_experts,
    route_tile,
)
    expert_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if expert_index <= num_experts
        count = expert_counts[expert_index]
        tile = Int32(route_tile)
        @inbounds padded_counts[expert_index] =
            ((count + tile - Int32(1)) ÷ tile) * tile
    end
    return
end

function _qwen3_cuda_pack_grouped_bf16_inputs_kernel!(
    padded_inputs,
    inputs,
    sorted_experts,
    expert_offsets,
    padded_offsets,
    input_dim,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= input_dim * pair_count
        input_index = (linear_index - 1) % input_dim + 1
        pair_index = (linear_index - 1) ÷ input_dim + 1
        expert_index = Int(sorted_experts[pair_index])
        within_expert = pair_index - Int(expert_offsets[expert_index])
        padded_pair = Int(padded_offsets[expert_index]) + within_expert
        @inbounds padded_inputs[input_index, padded_pair] =
            inputs[input_index, pair_index]
    end
    return
end

function _qwen3_cuda_grouped_bf16_wmma_m16n16k16_kernel!(
    padded_output,
    weights,
    padded_inputs,
    padded_offsets,
    output_dim,
    input_dim,
    num_experts,
    output_tiles,
)
    block_index = CUDA.blockIdx().x
    output_tile = (block_index - 1) % output_tiles + 1
    route_tile = (block_index - 1) ÷ output_tiles + 1
    output_start = (output_tile - 1) * 16 + 1
    route_start = (route_tile - 1) * 16 + 1
    padded_stop = Int(padded_offsets[num_experts + 1])
    if route_start < padded_stop
        expert_index = 1
        while expert_index <= num_experts &&
                route_start >= Int(padded_offsets[expert_index + 1])
            expert_index += 1
        end
        config = CUDA.WMMA.Config{16,16,16,Float32}
        accumulator = CUDA.WMMA.fill_c(0.0f0, config)
        input_start = 1
        while input_start <= input_dim
            weight_linear =
                output_start +
                (input_start - 1) * output_dim +
                (expert_index - 1) * output_dim * input_dim
            input_linear = input_start + (route_start - 1) * input_dim
            weight_fragment = CUDA.WMMA.load_a(
                pointer(weights, weight_linear),
                output_dim,
                CUDA.WMMA.ColMajor,
                config,
            )
            input_fragment = CUDA.WMMA.load_b(
                pointer(padded_inputs, input_linear),
                input_dim,
                CUDA.WMMA.ColMajor,
                config,
            )
            accumulator = CUDA.WMMA.mma(
                weight_fragment,
                input_fragment,
                accumulator,
                config,
            )
            input_start += 16
        end
        output_linear = output_start + (route_start - 1) * output_dim
        CUDA.WMMA.store_d(
            pointer(padded_output, output_linear),
            accumulator,
            output_dim,
            CUDA.WMMA.ColMajor,
            config,
        )
    end
    return
end

function _qwen3_cuda_grouped_bf16_wmma_m32n8k16_kernel!(
    padded_output,
    weights,
    padded_inputs,
    padded_offsets,
    output_dim,
    input_dim,
    num_experts,
    output_tiles,
)
    block_index = CUDA.blockIdx().x
    output_tile = (block_index - 1) % output_tiles + 1
    route_tile = (block_index - 1) ÷ output_tiles + 1
    output_start = (output_tile - 1) * 32 + 1
    route_start = (route_tile - 1) * 8 + 1
    padded_stop = Int(padded_offsets[num_experts + 1])
    if route_start < padded_stop
        expert_index = 1
        while expert_index <= num_experts &&
                route_start >= Int(padded_offsets[expert_index + 1])
            expert_index += 1
        end
        accumulator = (
            0.0f0, 0.0f0, 0.0f0, 0.0f0,
            0.0f0, 0.0f0, 0.0f0, 0.0f0,
        )
        input_start = 1
        while input_start <= input_dim
            weight_linear =
                output_start +
                (input_start - 1) * output_dim +
                (expert_index - 1) * output_dim * input_dim
            input_linear = input_start + (route_start - 1) * input_dim
            weight_fragment =
                CUDA.WMMA.llvm_wmma_load_a_col_m32n8k16_global_stride_bf16(
                    pointer(weights, weight_linear),
                    output_dim,
                )
            input_fragment =
                CUDA.WMMA.llvm_wmma_load_b_col_m32n8k16_global_stride_bf16(
                    pointer(padded_inputs, input_linear),
                    input_dim,
                )
            accumulator =
                CUDA.WMMA.llvm_wmma_mma_col_col_m32n8k16_bf16(
                    weight_fragment,
                    input_fragment,
                    accumulator,
                )
            input_start += 16
        end
        output_linear = output_start + (route_start - 1) * output_dim
        CUDA.WMMA.llvm_wmma_store_d_col_m32n8k16_global_stride_f32(
            pointer(padded_output, output_linear),
            accumulator,
            output_dim,
        )
    end
    return
end

@inline function _qwen3_cuda_scattered_projection_pointer(
    pointers,
    expert_index,
    linear_index,
)
    raw_pointer = @inbounds pointers[expert_index]
    base_pointer = reinterpret(
        Core.LLVMPtr{BFloat16,CUDA.AS.Global},
        raw_pointer,
    )
    return base_pointer + (linear_index - 1) * sizeof(BFloat16)
end

function _qwen3_cuda_grouped_scattered_bf16_wmma_m16n16k16_kernel!(
    padded_output,
    weight_pointers,
    padded_inputs,
    padded_offsets,
    output_dim,
    input_dim,
    num_experts,
    output_tiles,
)
    block_index = CUDA.blockIdx().x
    output_tile = (block_index - 1) % output_tiles + 1
    route_tile = (block_index - 1) ÷ output_tiles + 1
    output_start = (output_tile - 1) * 16 + 1
    route_start = (route_tile - 1) * 16 + 1
    padded_stop = Int(padded_offsets[num_experts + 1])
    if route_start < padded_stop
        expert_index = 1
        while expert_index <= num_experts &&
                route_start >= Int(padded_offsets[expert_index + 1])
            expert_index += 1
        end
        config = CUDA.WMMA.Config{16,16,16,Float32}
        accumulator = CUDA.WMMA.fill_c(0.0f0, config)
        input_start = 1
        while input_start <= input_dim
            weight_linear = output_start + (input_start - 1) * output_dim
            input_linear = input_start + (route_start - 1) * input_dim
            weight_fragment = CUDA.WMMA.load_a(
                _qwen3_cuda_scattered_projection_pointer(
                    weight_pointers,
                    expert_index,
                    weight_linear,
                ),
                output_dim,
                CUDA.WMMA.ColMajor,
                config,
            )
            input_fragment = CUDA.WMMA.load_b(
                pointer(padded_inputs, input_linear),
                input_dim,
                CUDA.WMMA.ColMajor,
                config,
            )
            accumulator = CUDA.WMMA.mma(
                weight_fragment,
                input_fragment,
                accumulator,
                config,
            )
            input_start += 16
        end
        output_linear = output_start + (route_start - 1) * output_dim
        CUDA.WMMA.store_d(
            pointer(padded_output, output_linear),
            accumulator,
            output_dim,
            CUDA.WMMA.ColMajor,
            config,
        )
    end
    return
end

function _qwen3_cuda_grouped_scattered_bf16_wmma_m32n8k16_kernel!(
    padded_output,
    weight_pointers,
    padded_inputs,
    padded_offsets,
    output_dim,
    input_dim,
    num_experts,
    output_tiles,
)
    block_index = CUDA.blockIdx().x
    output_tile = (block_index - 1) % output_tiles + 1
    route_tile = (block_index - 1) ÷ output_tiles + 1
    output_start = (output_tile - 1) * 32 + 1
    route_start = (route_tile - 1) * 8 + 1
    padded_stop = Int(padded_offsets[num_experts + 1])
    if route_start < padded_stop
        expert_index = 1
        while expert_index <= num_experts &&
                route_start >= Int(padded_offsets[expert_index + 1])
            expert_index += 1
        end
        accumulator = (
            0.0f0, 0.0f0, 0.0f0, 0.0f0,
            0.0f0, 0.0f0, 0.0f0, 0.0f0,
        )
        input_start = 1
        while input_start <= input_dim
            weight_linear = output_start + (input_start - 1) * output_dim
            input_linear = input_start + (route_start - 1) * input_dim
            weight_fragment =
                CUDA.WMMA.llvm_wmma_load_a_col_m32n8k16_global_stride_bf16(
                    _qwen3_cuda_scattered_projection_pointer(
                        weight_pointers,
                        expert_index,
                        weight_linear,
                    ),
                    output_dim,
                )
            input_fragment =
                CUDA.WMMA.llvm_wmma_load_b_col_m32n8k16_global_stride_bf16(
                    pointer(padded_inputs, input_linear),
                    input_dim,
                )
            accumulator =
                CUDA.WMMA.llvm_wmma_mma_col_col_m32n8k16_bf16(
                    weight_fragment,
                    input_fragment,
                    accumulator,
                )
            input_start += 16
        end
        output_linear = output_start + (route_start - 1) * output_dim
        CUDA.WMMA.llvm_wmma_store_d_col_m32n8k16_global_stride_f32(
            pointer(padded_output, output_linear),
            accumulator,
            output_dim,
        )
    end
    return
end

function _qwen3_cuda_unpack_grouped_f32_output_kernel!(
    output,
    padded_output,
    sorted_experts,
    expert_offsets,
    padded_offsets,
    output_dim,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= output_dim * pair_count
        output_index = (linear_index - 1) % output_dim + 1
        pair_index = (linear_index - 1) ÷ output_dim + 1
        expert_index = Int(sorted_experts[pair_index])
        within_expert = pair_index - Int(expert_offsets[expert_index])
        padded_pair = Int(padded_offsets[expert_index]) + within_expert
        @inbounds output[output_index, pair_index] =
            padded_output[output_index, padded_pair]
    end
    return
end

function _qwen3_cuda_pack_padded_tokens_kernel!(
    padded_tokens,
    tokens,
    route_permutation,
    sorted_experts,
    expert_offsets,
    padded_offsets,
    d_model,
    experts_per_token,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * pair_count
        model_index = (linear_index - 1) % d_model + 1
        bucket_pair = (linear_index - 1) ÷ d_model + 1
        original_pair = Int(route_permutation[bucket_pair])
        expert_index = Int(sorted_experts[bucket_pair])
        within_expert = bucket_pair - Int(expert_offsets[expert_index])
        padded_pair = Int(padded_offsets[expert_index]) + within_expert
        token_index = (original_pair - 1) ÷ experts_per_token + 1
        @inbounds padded_tokens[model_index, padded_pair] =
            BFloat16(tokens[model_index, token_index])
    end
    return
end

function _qwen3_cuda_grouped_bf16_layout(
    expert_counts,
    num_experts,
    pair_count,
    route_tile,
)
    padded_capacity = LifeAI._qwen3_cuda_route_layout_capacity(
        pair_count,
        num_experts;
        route_tile,
    )
    threads = 256
    expert_blocks = cld(num_experts, threads)
    padded_counts = CUDA.zeros(Int32, num_experts)
    CUDA.@cuda threads=threads blocks=expert_blocks _qwen3_cuda_pad_expert_counts_kernel!(
        padded_counts,
        expert_counts,
        num_experts,
        route_tile,
    )
    padded_inclusive = cumsum(padded_counts)
    padded_offsets = CUDA.zeros(Int32, num_experts + 1)
    CUDA.@cuda threads=threads blocks=expert_blocks _qwen3_cuda_finalize_route_offsets_kernel!(
        padded_offsets,
        padded_inclusive,
        num_experts,
    )
    return (;
        padded_offsets,
        padded_capacity,
        route_tile,
    )
end

function _qwen3_cuda_pack_grouped_bf16_inputs(
    inputs,
    sorted_experts,
    expert_offsets,
    padded_offsets,
    padded_capacity,
)
    input_dim, pair_count = size(inputs)
    padded_inputs = CUDA.zeros(BFloat16, input_dim, padded_capacity)
    threads = 256
    input_blocks = cld(input_dim * pair_count, threads)
    CUDA.@cuda threads=threads blocks=input_blocks _qwen3_cuda_pack_grouped_bf16_inputs_kernel!(
        padded_inputs,
        inputs,
        sorted_experts,
        expert_offsets,
        padded_offsets,
        input_dim,
        pair_count,
    )
    return padded_inputs
end

function _qwen3_cuda_grouped_bf16_padded_matmul(
    weights,
    padded_inputs,
    padded_offsets,
    padded_capacity,
    route_tile,
)
    output_dim, input_dim, num_experts = size(weights)
    padded_output = CUDA.zeros(Float32, output_dim, padded_capacity)
    if route_tile == 8
        output_tiles = output_dim ÷ 32
        route_tiles = cld(padded_capacity, 8)
        CUDA.@cuda threads=32 blocks=(output_tiles * route_tiles) _qwen3_cuda_grouped_bf16_wmma_m32n8k16_kernel!(
            padded_output,
            weights,
            padded_inputs,
            padded_offsets,
            output_dim,
            input_dim,
            num_experts,
            output_tiles,
        )
    else
        output_tiles = output_dim ÷ 16
        route_tiles = cld(padded_capacity, 16)
        CUDA.@cuda threads=32 blocks=(output_tiles * route_tiles) _qwen3_cuda_grouped_bf16_wmma_m16n16k16_kernel!(
            padded_output,
            weights,
            padded_inputs,
            padded_offsets,
            output_dim,
            input_dim,
            num_experts,
            output_tiles,
        )
    end
    return padded_output
end

function _qwen3_cuda_grouped_scattered_bf16_padded_matmul!(
    padded_output,
    weight_pointers,
    padded_inputs,
    padded_offsets,
    output_dim::Int,
    input_dim::Int,
    num_experts::Int,
    padded_capacity::Int,
    route_tile::Int,
)
    size(padded_output, 1) == output_dim &&
        size(padded_output, 2) >= padded_capacity ||
        throw(DimensionMismatch(
            "grouped scattered output workspace has the wrong shape",
        ))
    size(padded_inputs, 1) == input_dim &&
        size(padded_inputs, 2) >= padded_capacity ||
        throw(DimensionMismatch(
            "grouped scattered input workspace has the wrong shape",
        ))
    length(weight_pointers) == num_experts || throw(DimensionMismatch(
        "grouped scattered pointer count does not match active experts",
    ))
    if route_tile == 8
        output_tiles = output_dim ÷ 32
        route_tiles = cld(padded_capacity, 8)
        CUDA.@cuda threads=32 blocks=(output_tiles * route_tiles) _qwen3_cuda_grouped_scattered_bf16_wmma_m32n8k16_kernel!(
            padded_output,
            weight_pointers,
            padded_inputs,
            padded_offsets,
            output_dim,
            input_dim,
            num_experts,
            output_tiles,
        )
    else
        route_tile == 16 || throw(ArgumentError(
            "grouped scattered route tile must be 8 or 16",
        ))
        output_tiles = output_dim ÷ 16
        route_tiles = cld(padded_capacity, 16)
        CUDA.@cuda threads=32 blocks=(output_tiles * route_tiles) _qwen3_cuda_grouped_scattered_bf16_wmma_m16n16k16_kernel!(
            padded_output,
            weight_pointers,
            padded_inputs,
            padded_offsets,
            output_dim,
            input_dim,
            num_experts,
            output_tiles,
        )
    end
    return padded_output
end

function _qwen3_cuda_unpack_grouped_f32_output(
    padded_output,
    sorted_experts,
    expert_offsets,
    padded_offsets,
    pair_count,
)
    output_dim = size(padded_output, 1)
    output = CUDA.zeros(Float32, output_dim, pair_count)
    threads = 256
    output_blocks = cld(output_dim * pair_count, threads)
    CUDA.@cuda threads=threads blocks=output_blocks _qwen3_cuda_unpack_grouped_f32_output_kernel!(
        output,
        padded_output,
        sorted_experts,
        expert_offsets,
        padded_offsets,
        output_dim,
        pair_count,
    )
    return output
end

function _qwen3_cuda_scatter_expert_major_output_kernel!(
    routed_output,
    grouped_output,
    route_permutation,
    d_model,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * pair_count
        model_index = (linear_index - 1) % d_model + 1
        bucket_pair = (linear_index - 1) ÷ d_model + 1
        original_pair = Int(route_permutation[bucket_pair])
        @inbounds routed_output[model_index, original_pair] =
            grouped_output[model_index, bucket_pair]
    end
    return
end

function _qwen3_cuda_invert_route_permutation_kernel!(
    inverse_route_permutation,
    route_permutation,
    pair_count,
)
    bucket_pair = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if bucket_pair <= pair_count
        original_pair = Int(route_permutation[bucket_pair])
        @inbounds inverse_route_permutation[original_pair] = Int32(bucket_pair)
    end
    return
end

function _qwen3_cuda_combine_padded_grouped_routes_kernel!(
    output,
    padded_output,
    inverse_route_permutation,
    sorted_experts,
    expert_offsets,
    padded_offsets,
    routing_weights,
    d_model,
    num_tokens,
    experts_per_token,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * num_tokens
        model_index = (linear_index - 1) % d_model + 1
        token_index = (linear_index - 1) ÷ d_model + 1
        value = 0.0f0
        @inbounds for slot in 1:experts_per_token
            original_pair = (token_index - 1) * experts_per_token + slot
            bucket_pair = Int(inverse_route_permutation[original_pair])
            expert_index = Int(sorted_experts[bucket_pair])
            within_expert = bucket_pair - Int(expert_offsets[expert_index])
            padded_pair = Int(padded_offsets[expert_index]) + within_expert
            weight = Float32(routing_weights[slot, token_index])
            if !iszero(weight)
                value = muladd(
                    padded_output[model_index, padded_pair],
                    weight,
                    value,
                )
            end
        end
        output[model_index, token_index] = value
    end
    return
end

"""
    qwen3_cuda_grouped_bf16_matmul(
        weights, inputs, sorted_experts, expert_counts, expert_offsets)

Multiply BF16 expert matrices by expert-major BF16 route inputs using CUDA
WMMA with Float32 accumulation. Dynamic route counts stay on device: experts
are padded to 8-column tiles when the output supports `m32n8k16`, otherwise
to 16-column tiles, and the returned output preserves expert-major route order.
"""
function qwen3_cuda_grouped_bf16_matmul(
    weights::CUDA.CuArray{BFloat16,3},
    inputs::CUDA.CuArray{BFloat16,2},
    sorted_experts::CUDA.CuArray{I,1},
    expert_counts::CUDA.CuArray{Int32,1},
    expert_offsets::CUDA.CuArray{Int32,1},
) where {I<:Integer}
    CUDA.capability(CUDA.device()) >= v"8.0" || throw(ArgumentError(
        "grouped BF16 WMMA requires an Ampere-or-newer CUDA device",
    ))
    output_dim, input_dim, num_experts = size(weights)
    size(inputs, 1) == input_dim || throw(DimensionMismatch(
        "grouped BF16 input width must match expert weight input width",
    ))
    pair_count = size(inputs, 2)
    length(sorted_experts) == pair_count || throw(DimensionMismatch(
        "sorted expert count must match grouped BF16 input routes",
    ))
    length(expert_counts) == num_experts || throw(DimensionMismatch(
        "expert count vector must match grouped BF16 expert weights",
    ))
    length(expert_offsets) == num_experts + 1 || throw(DimensionMismatch(
        "expert offsets must contain one boundary beyond the expert count",
    ))
    output_dim % 16 == 0 || throw(ArgumentError(
        "grouped BF16 WMMA output dimension must be divisible by 16",
    ))
    input_dim % 16 == 0 || throw(ArgumentError(
        "grouped BF16 WMMA input dimension must be divisible by 16",
    ))
    pair_count > 0 || throw(ArgumentError(
        "grouped BF16 WMMA requires at least one route",
    ))

    route_tile = output_dim % 32 == 0 ? 8 : 16
    layout = _qwen3_cuda_grouped_bf16_layout(
        expert_counts,
        num_experts,
        pair_count,
        route_tile,
    )
    padded_inputs = _qwen3_cuda_pack_grouped_bf16_inputs(
        inputs,
        sorted_experts,
        expert_offsets,
        layout.padded_offsets,
        layout.padded_capacity,
    )
    padded_output = _qwen3_cuda_grouped_bf16_padded_matmul(
        weights,
        padded_inputs,
        layout.padded_offsets,
        layout.padded_capacity,
        layout.route_tile,
    )
    return _qwen3_cuda_unpack_grouped_f32_output(
        padded_output,
        sorted_experts,
        expert_offsets,
        layout.padded_offsets,
        pair_count,
    )
end

"""
    qwen3_cuda_grouped_bf16_sparse_expert_dispatch(
        tokens, expert_indices, routing_weights, expert_parameters)

Experimental tensor-core Qwen3 MoE dispatch. Routes and expert offsets remain
on device. Tokens and the post-SwiGLU hidden state are rounded to BF16 at the
two WMMA projection boundaries; WMMA accumulation and route combination use
Float32. This explicit numerical contract is intentionally separate from the
production indexed/bucketed Float32-activation dispatch.
"""
function qwen3_cuda_grouped_bf16_sparse_expert_dispatch(
    tokens::CUDA.CuArray{Float32,2},
    expert_indices::CUDA.CuArray{I,2},
    routing_weights::CUDA.CuArray{R,2},
    expert_parameters,
) where {I<:Integer,R<:AbstractFloat}
    CUDA.capability(CUDA.device()) >= v"8.0" || throw(ArgumentError(
        "grouped BF16 dispatch requires an Ampere-or-newer CUDA device",
    ))
    num_experts = size(expert_parameters.gate_proj, 3)
    d_model, num_tokens, num_experts = LifeAI._validate_qwen3_expert_parameters(
        tokens,
        num_experts,
        expert_parameters,
    )
    size(expert_indices) == size(routing_weights) || throw(DimensionMismatch(
        "compact expert indices and routing weights must have matching shapes",
    ))
    size(expert_indices, 2) == num_tokens || throw(DimensionMismatch(
        "compact routing token count does not match expert input token count",
    ))
    experts_per_token = size(expert_indices, 1)
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "compact routing route count must be in 1:num_experts",
    ))
    all(parameter -> parameter isa CUDA.CuArray{BFloat16,3}, (
        expert_parameters.gate_proj,
        expert_parameters.up_proj,
        expert_parameters.down_proj,
    )) || throw(ArgumentError(
        "grouped BF16 dispatch requires BF16 CUDA expert projections",
    ))

    hidden_dim = size(expert_parameters.gate_proj, 1)
    d_model % 16 == 0 || throw(ArgumentError(
        "grouped BF16 dispatch model dimension must be divisible by 16",
    ))
    hidden_dim % 16 == 0 || throw(ArgumentError(
        "grouped BF16 dispatch hidden dimension must be divisible by 16",
    ))
    pair_count = length(expert_indices)
    buckets = qwen3_cuda_bucket_routes(expert_indices, num_experts)
    route_tile = d_model % 32 == 0 && hidden_dim % 32 == 0 ? 8 : 16
    layout = _qwen3_cuda_grouped_bf16_layout(
        buckets.expert_counts,
        num_experts,
        pair_count,
        route_tile,
    )
    padded_tokens = CUDA.zeros(
        BFloat16,
        d_model,
        layout.padded_capacity,
    )
    threads = 256
    token_blocks = cld(d_model * pair_count, threads)
    CUDA.@cuda threads=threads blocks=token_blocks _qwen3_cuda_pack_padded_tokens_kernel!(
        padded_tokens,
        tokens,
        buckets.route_permutation,
        buckets.sorted_experts,
        buckets.expert_offsets,
        layout.padded_offsets,
        d_model,
        experts_per_token,
        pair_count,
    )
    padded_gate = _qwen3_cuda_grouped_bf16_padded_matmul(
        expert_parameters.gate_proj,
        padded_tokens,
        layout.padded_offsets,
        layout.padded_capacity,
        layout.route_tile,
    )
    padded_up = _qwen3_cuda_grouped_bf16_padded_matmul(
        expert_parameters.up_proj,
        padded_tokens,
        layout.padded_offsets,
        layout.padded_capacity,
        layout.route_tile,
    )
    padded_hidden =
        BFloat16.((padded_gate ./ (1.0f0 .+ exp.(-padded_gate))) .* padded_up)
    padded_output = _qwen3_cuda_grouped_bf16_padded_matmul(
        expert_parameters.down_proj,
        padded_hidden,
        layout.padded_offsets,
        layout.padded_capacity,
        layout.route_tile,
    )
    inverse_route_permutation = CUDA.zeros(Int32, pair_count)
    inverse_blocks = cld(pair_count, threads)
    CUDA.@cuda threads=threads blocks=inverse_blocks _qwen3_cuda_invert_route_permutation_kernel!(
        inverse_route_permutation,
        buckets.route_permutation,
        pair_count,
    )
    output = CUDA.zeros(Float32, d_model, num_tokens)
    combine_blocks = cld(length(output), threads)
    CUDA.@cuda threads=threads blocks=combine_blocks _qwen3_cuda_combine_padded_grouped_routes_kernel!(
        output,
        padded_output,
        inverse_route_permutation,
        buckets.sorted_experts,
        buckets.expert_offsets,
        layout.padded_offsets,
        routing_weights,
        d_model,
        num_tokens,
        experts_per_token,
    )
    return output
end

function LifeAI._qwen3_grouped_bf16_expert_dispatch(
    tokens::CUDA.CuArray{Float32,2},
    expert_indices::CUDA.CuArray{I,2},
    routing_weights::CUDA.CuArray{R,2},
    expert_parameters,
) where {I<:Integer,R<:AbstractFloat}
    return qwen3_cuda_grouped_bf16_sparse_expert_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters,
    )
end

"""
    qwen3_cuda_bucket_routes(expert_indices, num_experts)

Build a stable, expert-major route permutation entirely on CUDA. `expert_counts`
contains the number of selected routes per expert and `expert_offsets` contains
1-based half-open boundaries, so expert `i` owns
`expert_offsets[i]:(expert_offsets[i + 1] - 1)` in `route_permutation`.
"""
function qwen3_cuda_bucket_routes(
    expert_indices::CUDA.CuArray{I,2},
    num_experts::Int,
) where {I<:Integer}
    num_experts > 0 || throw(ArgumentError("expert count must be positive"))
    pair_count = length(expert_indices)
    pair_count > 0 || throw(ArgumentError("route table must not be empty"))
    LifeAI._qwen3_cuda_route_layout_capacity(pair_count, num_experts)
    sorted_experts = copy(vec(expert_indices))
    route_permutation = CUDA.zeros(Int32, pair_count)
    expert_counts = CUDA.zeros(Int32, num_experts)
    threads = 256
    blocks = cld(pair_count, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_initialize_route_buckets_kernel!(
        route_permutation,
        expert_counts,
        expert_indices,
        pair_count,
    )
    sortperm!(
        route_permutation,
        sorted_experts;
        initialized=true,
    )
    sorted_experts = sorted_experts[route_permutation]
    inclusive_counts = cumsum(expert_counts)
    expert_offsets = CUDA.zeros(Int32, num_experts + 1)
    offset_blocks = cld(num_experts, threads)
    CUDA.@cuda threads=threads blocks=offset_blocks _qwen3_cuda_finalize_route_offsets_kernel!(
        expert_offsets,
        inclusive_counts,
        num_experts,
    )
    return (;
        sorted_experts,
        route_permutation,
        expert_counts,
        expert_offsets,
    )
end

function _qwen3_cuda_swiglu_routes_kernel!(
    hidden,
    tokens,
    expert_indices,
    gate_proj,
    up_proj,
    d_model,
    hidden_dim,
    experts_per_token,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= hidden_dim * pair_count
        hidden_index = (linear_index - 1) % hidden_dim + 1
        pair_index = (linear_index - 1) ÷ hidden_dim + 1
        token_index = (pair_index - 1) ÷ experts_per_token + 1
        expert_index = Int(expert_indices[pair_index])
        gate = 0.0f0
        up = 0.0f0
        @inbounds for input_index in 1:d_model
            token = Float32(tokens[input_index, token_index])
            gate = muladd(
                Float32(gate_proj[hidden_index, input_index, expert_index]),
                token,
                gate,
            )
            up = muladd(
                Float32(up_proj[hidden_index, input_index, expert_index]),
                token,
                up,
            )
        end
        @inbounds hidden[hidden_index, pair_index] =
            (gate / (1.0f0 + exp(-gate))) * up
    end
    return
end

@inline function _qwen3_cuda_scattered_weight(
    pointers,
    expert_index,
    row_index,
    column_index,
    row_count,
)
    raw_pointer = @inbounds pointers[expert_index]
    pointer = reinterpret(Core.LLVMPtr{BFloat16,CUDA.AS.Global}, raw_pointer)
    linear_index = row_index + (column_index - 1) * row_count
    return @inbounds unsafe_load(pointer, linear_index)
end

function _qwen3_cuda_scattered_swiglu_routes_kernel!(
    hidden,
    tokens,
    expert_indices,
    gate_pointers,
    up_pointers,
    d_model,
    hidden_dim,
    experts_per_token,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= hidden_dim * pair_count
        hidden_index = (linear_index - 1) % hidden_dim + 1
        pair_index = (linear_index - 1) ÷ hidden_dim + 1
        token_index = (pair_index - 1) ÷ experts_per_token + 1
        expert_index = Int(expert_indices[pair_index])
        gate = 0.0f0
        up = 0.0f0
        @inbounds for input_index in 1:d_model
            token = Float32(tokens[input_index, token_index])
            gate = muladd(
                Float32(_qwen3_cuda_scattered_weight(
                    gate_pointers,
                    expert_index,
                    hidden_index,
                    input_index,
                    hidden_dim,
                )),
                token,
                gate,
            )
            up = muladd(
                Float32(_qwen3_cuda_scattered_weight(
                    up_pointers,
                    expert_index,
                    hidden_index,
                    input_index,
                    hidden_dim,
                )),
                token,
                up,
            )
        end
        @inbounds hidden[hidden_index, pair_index] =
            (gate / (1.0f0 + exp(-gate))) * up
    end
    return
end

function _qwen3_cuda_scattered_down_routes_kernel!(
    routed_output,
    hidden,
    expert_indices,
    down_pointers,
    d_model,
    hidden_dim,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * pair_count
        output_index = (linear_index - 1) % d_model + 1
        pair_index = (linear_index - 1) ÷ d_model + 1
        expert_index = Int(expert_indices[pair_index])
        value = 0.0f0
        @inbounds for hidden_index in 1:hidden_dim
            value = muladd(
                Float32(_qwen3_cuda_scattered_weight(
                    down_pointers,
                    expert_index,
                    output_index,
                    hidden_index,
                    d_model,
                )),
                hidden[hidden_index, pair_index],
                value,
            )
        end
        @inbounds routed_output[output_index, pair_index] = value
    end
    return
end

function _qwen3_cuda_down_routes_kernel!(
    routed_output,
    hidden,
    expert_indices,
    down_proj,
    d_model,
    hidden_dim,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * pair_count
        output_index = (linear_index - 1) % d_model + 1
        pair_index = (linear_index - 1) ÷ d_model + 1
        expert_index = Int(expert_indices[pair_index])
        value = 0.0f0
        @inbounds for hidden_index in 1:hidden_dim
            value = muladd(
                Float32(down_proj[output_index, hidden_index, expert_index]),
                hidden[hidden_index, pair_index],
                value,
            )
        end
        @inbounds routed_output[output_index, pair_index] = value
    end
    return
end

function _qwen3_cuda_bucketed_swiglu_routes_kernel!(
    hidden,
    tokens,
    route_permutation,
    sorted_experts,
    gate_proj,
    up_proj,
    d_model,
    hidden_dim,
    experts_per_token,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= hidden_dim * pair_count
        hidden_index = (linear_index - 1) % hidden_dim + 1
        bucket_pair = (linear_index - 1) ÷ hidden_dim + 1
        original_pair = Int(route_permutation[bucket_pair])
        token_index = (original_pair - 1) ÷ experts_per_token + 1
        expert_index = Int(sorted_experts[bucket_pair])
        gate = 0.0f0
        up = 0.0f0
        @inbounds for input_index in 1:d_model
            token = Float32(tokens[input_index, token_index])
            gate = muladd(
                Float32(gate_proj[hidden_index, input_index, expert_index]),
                token,
                gate,
            )
            up = muladd(
                Float32(up_proj[hidden_index, input_index, expert_index]),
                token,
                up,
            )
        end
        @inbounds hidden[hidden_index, bucket_pair] =
            (gate / (1.0f0 + exp(-gate))) * up
    end
    return
end

function _qwen3_cuda_scattered_bucketed_swiglu_routes_kernel!(
    hidden,
    tokens,
    route_permutation,
    sorted_experts,
    gate_pointers,
    up_pointers,
    d_model,
    hidden_dim,
    experts_per_token,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= hidden_dim * pair_count
        hidden_index = (linear_index - 1) % hidden_dim + 1
        bucket_pair = (linear_index - 1) ÷ hidden_dim + 1
        original_pair = Int(route_permutation[bucket_pair])
        token_index = (original_pair - 1) ÷ experts_per_token + 1
        expert_index = Int(sorted_experts[bucket_pair])
        gate = 0.0f0
        up = 0.0f0
        @inbounds for input_index in 1:d_model
            token = Float32(tokens[input_index, token_index])
            gate = muladd(
                Float32(_qwen3_cuda_scattered_weight(
                    gate_pointers,
                    expert_index,
                    hidden_index,
                    input_index,
                    hidden_dim,
                )),
                token,
                gate,
            )
            up = muladd(
                Float32(_qwen3_cuda_scattered_weight(
                    up_pointers,
                    expert_index,
                    hidden_index,
                    input_index,
                    hidden_dim,
                )),
                token,
                up,
            )
        end
        @inbounds hidden[hidden_index, bucket_pair] =
            (gate / (1.0f0 + exp(-gate))) * up
    end
    return
end

function _qwen3_cuda_bucketed_down_routes_kernel!(
    routed_output,
    hidden,
    route_permutation,
    sorted_experts,
    down_proj,
    d_model,
    hidden_dim,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * pair_count
        output_index = (linear_index - 1) % d_model + 1
        bucket_pair = (linear_index - 1) ÷ d_model + 1
        original_pair = Int(route_permutation[bucket_pair])
        expert_index = Int(sorted_experts[bucket_pair])
        value = 0.0f0
        @inbounds for hidden_index in 1:hidden_dim
            value = muladd(
                Float32(down_proj[output_index, hidden_index, expert_index]),
                hidden[hidden_index, bucket_pair],
                value,
            )
        end
        @inbounds routed_output[output_index, original_pair] = value
    end
    return
end

function _qwen3_cuda_scattered_bucketed_down_routes_kernel!(
    routed_output,
    hidden,
    route_permutation,
    sorted_experts,
    down_pointers,
    d_model,
    hidden_dim,
    pair_count,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * pair_count
        output_index = (linear_index - 1) % d_model + 1
        bucket_pair = (linear_index - 1) ÷ d_model + 1
        original_pair = Int(route_permutation[bucket_pair])
        expert_index = Int(sorted_experts[bucket_pair])
        value = 0.0f0
        @inbounds for hidden_index in 1:hidden_dim
            value = muladd(
                Float32(_qwen3_cuda_scattered_weight(
                    down_pointers,
                    expert_index,
                    output_index,
                    hidden_index,
                    d_model,
                )),
                hidden[hidden_index, bucket_pair],
                value,
            )
        end
        @inbounds routed_output[output_index, original_pair] = value
    end
    return
end

function _qwen3_cuda_combine_routes_kernel!(
    output,
    routed_output,
    routing_weights,
    d_model,
    num_tokens,
    experts_per_token,
)
    linear_index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x +
        CUDA.threadIdx().x
    if linear_index <= d_model * num_tokens
        output_index = (linear_index - 1) % d_model + 1
        token_index = (linear_index - 1) ÷ d_model + 1
        value = 0.0f0
        @inbounds for slot in 1:experts_per_token
            pair_index = (token_index - 1) * experts_per_token + slot
            weight = Float32(routing_weights[slot, token_index])
            if !iszero(weight)
                value = muladd(
                    routed_output[output_index, pair_index],
                    weight,
                    value,
                )
            end
        end
        @inbounds output[output_index, token_index] = value
    end
    return
end

function qwen3_cuda_bucketed_sparse_expert_dispatch(
    tokens::CUDA.CuArray{Float32,2},
    expert_indices::CUDA.CuArray{I,2},
    routing_weights::CUDA.CuArray{R,2},
    expert_parameters,
) where {I<:Integer,R<:AbstractFloat}
    num_experts = size(expert_parameters.gate_proj, 3)
    d_model, num_tokens, num_experts = LifeAI._validate_qwen3_expert_parameters(
        tokens,
        num_experts,
        expert_parameters,
    )
    size(expert_indices) == size(routing_weights) || throw(DimensionMismatch(
        "compact expert indices and routing weights must have matching shapes",
    ))
    size(expert_indices, 2) == num_tokens || throw(DimensionMismatch(
        "compact routing token count does not match expert input token count",
    ))
    experts_per_token = size(expert_indices, 1)
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "compact routing route count must be in 1:num_experts",
    ))
    all(parameter -> parameter isa CUDA.CuArray, (
        expert_parameters.gate_proj,
        expert_parameters.up_proj,
        expert_parameters.down_proj,
    )) || throw(ArgumentError(
        "CUDA sparse dispatch requires CUDA expert projection arrays",
    ))

    buckets = qwen3_cuda_bucket_routes(expert_indices, num_experts)
    hidden_dim = size(expert_parameters.gate_proj, 1)
    pair_count = length(expert_indices)
    hidden = similar(tokens, Float32, hidden_dim, pair_count)
    routed_output = similar(tokens, Float32, d_model, pair_count)
    output = similar(tokens, Float32, d_model, num_tokens)
    threads = 256
    hidden_blocks = cld(hidden_dim * pair_count, threads)
    down_blocks = cld(d_model * pair_count, threads)
    combine_blocks = cld(d_model * num_tokens, threads)

    CUDA.@cuda threads=threads blocks=hidden_blocks _qwen3_cuda_bucketed_swiglu_routes_kernel!(
        hidden,
        tokens,
        buckets.route_permutation,
        buckets.sorted_experts,
        expert_parameters.gate_proj,
        expert_parameters.up_proj,
        d_model,
        hidden_dim,
        experts_per_token,
        pair_count,
    )
    CUDA.@cuda threads=threads blocks=down_blocks _qwen3_cuda_bucketed_down_routes_kernel!(
        routed_output,
        hidden,
        buckets.route_permutation,
        buckets.sorted_experts,
        expert_parameters.down_proj,
        d_model,
        hidden_dim,
        pair_count,
    )
    CUDA.@cuda threads=threads blocks=combine_blocks _qwen3_cuda_combine_routes_kernel!(
        output,
        routed_output,
        routing_weights,
        d_model,
        num_tokens,
        experts_per_token,
    )
    return output
end

function qwen3_cuda_indexed_sparse_expert_dispatch(
    tokens::CUDA.CuArray{Float32,2},
    expert_indices::CUDA.CuArray{I,2},
    routing_weights::CUDA.CuArray{R,2},
    expert_parameters,
) where {I<:Integer,R<:AbstractFloat}
    num_experts = size(expert_parameters.gate_proj, 3)
    d_model, num_tokens, num_experts = LifeAI._validate_qwen3_expert_parameters(
        tokens,
        num_experts,
        expert_parameters,
    )
    size(expert_indices) == size(routing_weights) || throw(DimensionMismatch(
        "compact expert indices and routing weights must have matching shapes",
    ))
    size(expert_indices, 2) == num_tokens || throw(DimensionMismatch(
        "compact routing token count does not match expert input token count",
    ))
    experts_per_token = size(expert_indices, 1)
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "compact routing route count must be in 1:num_experts",
    ))
    all(parameter -> parameter isa CUDA.CuArray, (
        expert_parameters.gate_proj,
        expert_parameters.up_proj,
        expert_parameters.down_proj,
    )) || throw(ArgumentError(
        "CUDA sparse dispatch requires CUDA expert projection arrays",
    ))

    hidden_dim = size(expert_parameters.gate_proj, 1)
    pair_count = length(expert_indices)
    hidden = similar(tokens, Float32, hidden_dim, pair_count)
    routed_output = similar(tokens, Float32, d_model, pair_count)
    output = similar(tokens, Float32, d_model, num_tokens)
    threads = 256
    hidden_blocks = cld(hidden_dim * pair_count, threads)
    down_blocks = cld(d_model * pair_count, threads)
    combine_blocks = cld(d_model * num_tokens, threads)

    CUDA.@cuda threads=threads blocks=hidden_blocks _qwen3_cuda_swiglu_routes_kernel!(
        hidden,
        tokens,
        expert_indices,
        expert_parameters.gate_proj,
        expert_parameters.up_proj,
        d_model,
        hidden_dim,
        experts_per_token,
        pair_count,
    )
    CUDA.@cuda threads=threads blocks=down_blocks _qwen3_cuda_down_routes_kernel!(
        routed_output,
        hidden,
        expert_indices,
        expert_parameters.down_proj,
        d_model,
        hidden_dim,
        pair_count,
    )
    CUDA.@cuda threads=threads blocks=combine_blocks _qwen3_cuda_combine_routes_kernel!(
        output,
        routed_output,
        routing_weights,
        d_model,
        num_tokens,
        experts_per_token,
    )
    return output
end

function _qwen3_cuda_use_bucketed_dispatch(
    d_model::Int,
    hidden_dim::Int,
    num_tokens::Int,
)
    return num_tokens >= 32 && d_model * hidden_dim >= 1_048_576
end

struct _Qwen3CUDAScatteredPointerPlan
    expert_ids::Vector{Int}
    generations::Vector{Int}
    gate_host::Vector{CUDA.CuPtr{BFloat16}}
    up_host::Vector{CUDA.CuPtr{BFloat16}}
    down_host::Vector{CUDA.CuPtr{BFloat16}}
    gate_device
    up_device
    down_device
    bytes::Int
end

struct _Qwen3CUDAScatteredWorkspace
    hidden
    routed_output
    output
    bytes::Int
end

struct _Qwen3CUDAGroupedScatteredWorkspace
    padded_tokens
    padded_gate
    padded_up
    padded_hidden
    padded_output
    inverse_route_permutation
    output
    padded_capacity::Int
    bytes::Int
end

mutable struct _Qwen3CUDAScatteredDispatchState
    pointer_plans::Dict{Int,Vector{_Qwen3CUDAScatteredPointerPlan}}
    workspaces::Dict{Tuple{Int,Int},_Qwen3CUDAScatteredWorkspace}
    workspace_last_used::Dict{Tuple{Int,Int},Int}
    grouped_workspaces::Dict{
        NTuple{3,Int},
        _Qwen3CUDAGroupedScatteredWorkspace,
    }
    grouped_workspace_last_used::Dict{NTuple{3,Int},Int}
    clock::Int
    pointer_table_bytes::Int
    workspace_bytes::Int
end

_qwen3_cuda_scattered_state() = _Qwen3CUDAScatteredDispatchState(
    Dict{Int,Vector{_Qwen3CUDAScatteredPointerPlan}}(),
    Dict{Tuple{Int,Int},_Qwen3CUDAScatteredWorkspace}(),
    Dict{Tuple{Int,Int},Int}(),
    Dict{NTuple{3,Int},_Qwen3CUDAGroupedScatteredWorkspace}(),
    Dict{NTuple{3,Int},Int}(),
    0,
    0,
    0,
)

function _qwen3_cuda_scattered_pointer_plan!(
    state::_Qwen3CUDAScatteredDispatchState,
    layer_index::Int,
    expert_ids::Vector{Int},
    generations::Vector{Int},
    entries,
)
    plans = get!(state.pointer_plans, layer_index) do
        _Qwen3CUDAScatteredPointerPlan[]
    end
    plan_index = findfirst(plans) do plan
        plan.expert_ids == expert_ids &&
            plan.generations == generations
    end
    if plan_index !== nothing
        return plans[plan_index], false, true
    end

    gate_host = CUDA.CuPtr{BFloat16}[
        pointer(entry.gate_proj)
        for entry in entries
    ]
    up_host = CUDA.CuPtr{BFloat16}[
        pointer(entry.up_proj)
        for entry in entries
    ]
    down_host = CUDA.CuPtr{BFloat16}[
        pointer(entry.down_proj)
        for entry in entries
    ]
    bytes = 3 * length(entries) * sizeof(CUDA.CuPtr{BFloat16})
    plan = _Qwen3CUDAScatteredPointerPlan(
        copy(expert_ids),
        copy(generations),
        gate_host,
        up_host,
        down_host,
        CUDA.cu(gate_host),
        CUDA.cu(up_host),
        CUDA.cu(down_host),
        bytes,
    )
    if length(plans) >= 4
        evicted = popfirst!(plans)
        state.pointer_table_bytes -= evicted.bytes
    end
    push!(plans, plan)
    state.pointer_table_bytes = Base.checked_add(
        state.pointer_table_bytes,
        bytes,
    )
    return plan, true, false
end

function _qwen3_cuda_scattered_workspace!(
    state::_Qwen3CUDAScatteredDispatchState,
    tokens,
    hidden_dim::Int,
    pair_count::Int,
)
    d_model, num_tokens = size(tokens)
    key = (num_tokens, pair_count)
    state.clock = Base.checked_add(state.clock, 1)
    existing = get(state.workspaces, key, nothing)
    if existing !== nothing
        state.workspace_last_used[key] = state.clock
        return existing, false, true
    end

    if length(state.workspaces) >= 4
        victim = argmin(state.workspace_last_used)
        evicted = pop!(state.workspaces, victim)
        delete!(state.workspace_last_used, victim)
        state.workspace_bytes -= evicted.bytes
    end
    hidden = similar(tokens, Float32, hidden_dim, pair_count)
    routed_output = similar(tokens, Float32, d_model, pair_count)
    output = similar(tokens, Float32, d_model, num_tokens)
    bytes = sizeof(Float32) * (
        hidden_dim * pair_count +
        d_model * pair_count +
        d_model * num_tokens
    )
    workspace = _Qwen3CUDAScatteredWorkspace(
        hidden,
        routed_output,
        output,
        bytes,
    )
    state.workspaces[key] = workspace
    state.workspace_last_used[key] = state.clock
    state.workspace_bytes = Base.checked_add(state.workspace_bytes, bytes)
    return workspace, true, false
end

function _qwen3_cuda_grouped_scattered_workspace!(
    state::_Qwen3CUDAScatteredDispatchState,
    tokens,
    hidden_dim::Int,
    pair_count::Int,
    padded_capacity::Int,
    route_tile::Int,
)
    d_model, num_tokens = size(tokens)
    key = (num_tokens, pair_count, route_tile)
    state.clock = Base.checked_add(state.clock, 1)
    existing = get(state.grouped_workspaces, key, nothing)
    if existing !== nothing &&
            existing.padded_capacity >= padded_capacity
        state.grouped_workspace_last_used[key] = state.clock
        return existing, false, true
    end

    if existing !== nothing
        delete!(state.grouped_workspaces, key)
        delete!(state.grouped_workspace_last_used, key)
        state.workspace_bytes -= existing.bytes
    end

    if length(state.grouped_workspaces) >= 4
        victim = argmin(state.grouped_workspace_last_used)
        evicted = pop!(state.grouped_workspaces, victim)
        delete!(state.grouped_workspace_last_used, victim)
        state.workspace_bytes -= evicted.bytes
    end
    padded_tokens = similar(
        tokens,
        BFloat16,
        d_model,
        padded_capacity,
    )
    padded_gate = similar(
        tokens,
        Float32,
        hidden_dim,
        padded_capacity,
    )
    padded_up = similar(
        tokens,
        Float32,
        hidden_dim,
        padded_capacity,
    )
    padded_hidden = similar(
        tokens,
        BFloat16,
        hidden_dim,
        padded_capacity,
    )
    padded_output = similar(
        tokens,
        Float32,
        d_model,
        padded_capacity,
    )
    inverse_route_permutation = similar(tokens, Int32, pair_count)
    output = similar(tokens, Float32, d_model, num_tokens)
    bytes =
        sizeof(BFloat16) * d_model * padded_capacity +
        2 * sizeof(Float32) * hidden_dim * padded_capacity +
        sizeof(BFloat16) * hidden_dim * padded_capacity +
        sizeof(Float32) * d_model * padded_capacity +
        sizeof(Int32) * pair_count +
        sizeof(Float32) * d_model * num_tokens
    workspace = _Qwen3CUDAGroupedScatteredWorkspace(
        padded_tokens,
        padded_gate,
        padded_up,
        padded_hidden,
        padded_output,
        inverse_route_permutation,
        output,
        padded_capacity,
        bytes,
    )
    state.grouped_workspaces[key] = workspace
    state.grouped_workspace_last_used[key] = state.clock
    state.workspace_bytes = Base.checked_add(state.workspace_bytes, bytes)
    return workspace, true, false
end

function _qwen3_cuda_grouped_scattered_dispatch!(
    state::_Qwen3CUDAScatteredDispatchState,
    tokens,
    expert_indices,
    routing_weights,
    pointer_plan::_Qwen3CUDAScatteredPointerPlan,
    hidden_dim::Int,
)
    CUDA.capability(CUDA.device()) >= v"8.0" || throw(ArgumentError(
        "grouped scattered BF16 dispatch requires an Ampere-or-newer CUDA device",
    ))
    d_model, num_tokens = size(tokens)
    num_experts = length(pointer_plan.expert_ids)
    experts_per_token = size(expert_indices, 1)
    pair_count = length(expert_indices)
    d_model % 16 == 0 || throw(ArgumentError(
        "grouped scattered dispatch model dimension must be divisible by 16",
    ))
    hidden_dim % 16 == 0 || throw(ArgumentError(
        "grouped scattered dispatch hidden dimension must be divisible by 16",
    ))

    buckets = qwen3_cuda_bucket_routes(expert_indices, num_experts)
    route_tile = d_model % 32 == 0 && hidden_dim % 32 == 0 ? 8 : 16
    layout = _qwen3_cuda_grouped_bf16_layout(
        buckets.expert_counts,
        num_experts,
        pair_count,
        route_tile,
    )
    workspace, workspace_allocated, workspace_reused =
        _qwen3_cuda_grouped_scattered_workspace!(
            state,
            tokens,
            hidden_dim,
            pair_count,
            layout.padded_capacity,
            layout.route_tile,
        )

    threads = 256
    token_blocks = cld(d_model * pair_count, threads)
    CUDA.@cuda threads=threads blocks=token_blocks _qwen3_cuda_pack_padded_tokens_kernel!(
        workspace.padded_tokens,
        tokens,
        buckets.route_permutation,
        buckets.sorted_experts,
        buckets.expert_offsets,
        layout.padded_offsets,
        d_model,
        experts_per_token,
        pair_count,
    )
    _qwen3_cuda_grouped_scattered_bf16_padded_matmul!(
        workspace.padded_gate,
        pointer_plan.gate_device,
        workspace.padded_tokens,
        layout.padded_offsets,
        hidden_dim,
        d_model,
        num_experts,
        layout.padded_capacity,
        layout.route_tile,
    )
    _qwen3_cuda_grouped_scattered_bf16_padded_matmul!(
        workspace.padded_up,
        pointer_plan.up_device,
        workspace.padded_tokens,
        layout.padded_offsets,
        hidden_dim,
        d_model,
        num_experts,
        layout.padded_capacity,
        layout.route_tile,
    )
    workspace.padded_hidden .= BFloat16.((
        workspace.padded_gate ./
        (1.0f0 .+ exp.(-workspace.padded_gate))
    ) .* workspace.padded_up)
    _qwen3_cuda_grouped_scattered_bf16_padded_matmul!(
        workspace.padded_output,
        pointer_plan.down_device,
        workspace.padded_hidden,
        layout.padded_offsets,
        d_model,
        hidden_dim,
        num_experts,
        layout.padded_capacity,
        layout.route_tile,
    )

    inverse_blocks = cld(pair_count, threads)
    CUDA.@cuda threads=threads blocks=inverse_blocks _qwen3_cuda_invert_route_permutation_kernel!(
        workspace.inverse_route_permutation,
        buckets.route_permutation,
        pair_count,
    )
    combine_blocks = cld(length(workspace.output), threads)
    CUDA.@cuda threads=threads blocks=combine_blocks _qwen3_cuda_combine_padded_grouped_routes_kernel!(
        workspace.output,
        workspace.padded_output,
        workspace.inverse_route_permutation,
        buckets.sorted_experts,
        buckets.expert_offsets,
        layout.padded_offsets,
        routing_weights,
        d_model,
        num_tokens,
        experts_per_token,
    )
    return workspace.output, workspace_allocated, workspace_reused
end

function LifeAI._qwen3_scattered_expert_dispatch(
    tokens::CUDA.CuArray{Float32,2},
    expert_indices::CUDA.CuArray{I,2},
    routing_weights::CUDA.CuArray{R,2},
    scattered::LifeAI._Qwen3MoEScatteredExperts,
    existing_state,
    layer_index::Int,
    grouped_experts::Bool=false,
) where {I<:Integer,R<:AbstractFloat}
    entries = scattered.entries
    num_experts = length(entries)
    num_experts > 0 || throw(ArgumentError(
        "scattered expert dispatch requires at least one expert",
    ))
    d_model, num_tokens = size(tokens)
    d_model == scattered.d_model || throw(DimensionMismatch(
        "scattered expert d_model does not match tokens",
    ))
    size(expert_indices) == size(routing_weights) || throw(DimensionMismatch(
        "compact expert indices and routing weights must have matching shapes",
    ))
    size(expert_indices, 2) == num_tokens || throw(DimensionMismatch(
        "compact routing token count does not match expert input token count",
    ))
    experts_per_token = size(expert_indices, 1)
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "compact routing route count must be in 1:num_experts",
    ))
    hidden_dim = scattered.hidden_dim
    all(entries) do entry
        entry.gate_proj isa CUDA.CuArray{BFloat16,2} &&
            entry.up_proj isa CUDA.CuArray{BFloat16,2} &&
            entry.down_proj isa CUDA.CuArray{BFloat16,2} &&
            size(entry.gate_proj) == (hidden_dim, d_model) &&
            size(entry.up_proj) == (hidden_dim, d_model) &&
            size(entry.down_proj) == (d_model, hidden_dim)
    end || throw(ArgumentError(
        "scattered CUDA dispatch requires consistently shaped BF16 CuMatrix experts",
    ))

    state = existing_state isa _Qwen3CUDAScatteredDispatchState ?
        existing_state : _qwen3_cuda_scattered_state()
    pointer_plan, pointer_table_built, pointer_table_reused =
        _qwen3_cuda_scattered_pointer_plan!(
            state,
            layer_index,
            scattered.expert_ids,
            scattered.generations,
            entries,
        )
    gate_pointers = pointer_plan.gate_device
    up_pointers = pointer_plan.up_device
    down_pointers = pointer_plan.down_device
    pointer_bytes_uploaded = pointer_table_built ? pointer_plan.bytes : 0
    pair_count = length(expert_indices)

    if grouped_experts
        output, workspace_allocated, workspace_reused =
            _qwen3_cuda_grouped_scattered_dispatch!(
                state,
                tokens,
                expert_indices,
                routing_weights,
                pointer_plan,
                hidden_dim,
            )
        return (;
            output,
            state,
            pointer_bytes_uploaded,
            pointer_table_built,
            pointer_table_reused,
            workspace_allocated,
            workspace_reused,
            pointer_table_bytes=state.pointer_table_bytes,
            workspace_bytes=state.workspace_bytes,
        )
    end

    workspace, workspace_allocated, workspace_reused =
        _qwen3_cuda_scattered_workspace!(
            state,
            tokens,
            hidden_dim,
            pair_count,
        )
    hidden = workspace.hidden
    routed_output = workspace.routed_output
    output = workspace.output
    threads = 256
    hidden_blocks = cld(hidden_dim * pair_count, threads)
    down_blocks = cld(d_model * pair_count, threads)
    combine_blocks = cld(d_model * num_tokens, threads)

    if _qwen3_cuda_use_bucketed_dispatch(d_model, hidden_dim, num_tokens)
        buckets = qwen3_cuda_bucket_routes(expert_indices, num_experts)
        CUDA.@cuda threads=threads blocks=hidden_blocks _qwen3_cuda_scattered_bucketed_swiglu_routes_kernel!(
            hidden,
            tokens,
            buckets.route_permutation,
            buckets.sorted_experts,
            gate_pointers,
            up_pointers,
            d_model,
            hidden_dim,
            experts_per_token,
            pair_count,
        )
        CUDA.@cuda threads=threads blocks=down_blocks _qwen3_cuda_scattered_bucketed_down_routes_kernel!(
            routed_output,
            hidden,
            buckets.route_permutation,
            buckets.sorted_experts,
            down_pointers,
            d_model,
            hidden_dim,
            pair_count,
        )
    else
        CUDA.@cuda threads=threads blocks=hidden_blocks _qwen3_cuda_scattered_swiglu_routes_kernel!(
            hidden,
            tokens,
            expert_indices,
            gate_pointers,
            up_pointers,
            d_model,
            hidden_dim,
            experts_per_token,
            pair_count,
        )
        CUDA.@cuda threads=threads blocks=down_blocks _qwen3_cuda_scattered_down_routes_kernel!(
            routed_output,
            hidden,
            expert_indices,
            down_pointers,
            d_model,
            hidden_dim,
            pair_count,
        )
    end
    CUDA.@cuda threads=threads blocks=combine_blocks _qwen3_cuda_combine_routes_kernel!(
        output,
        routed_output,
        routing_weights,
        d_model,
        num_tokens,
        experts_per_token,
    )
    return (;
        output,
        state,
        pointer_bytes_uploaded,
        pointer_table_built,
        pointer_table_reused,
        workspace_allocated,
        workspace_reused,
        pointer_table_bytes=state.pointer_table_bytes,
        workspace_bytes=state.workspace_bytes,
    )
end

function qwen3_device_sparse_expert_dispatch(
    tokens::CUDA.CuArray{Float32,2},
    expert_indices::CUDA.CuArray{I,2},
    routing_weights::CUDA.CuArray{R,2},
    expert_parameters,
) where {I<:Integer,R<:AbstractFloat}
    dispatch = _qwen3_cuda_use_bucketed_dispatch(
        size(tokens, 1),
        size(expert_parameters.gate_proj, 1),
        size(tokens, 2),
    ) ? qwen3_cuda_bucketed_sparse_expert_dispatch :
        qwen3_cuda_indexed_sparse_expert_dispatch
    return dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters,
    )
end

function _qwen3_cuda_mlp_kernel!(product, gate, up)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index > length(product) && return
    @inbounds begin
        gate_value = Float32(gate[index])
        activated = BFloat16(gate_value / (1.0f0 + exp(-gate_value)))
        product[index] = BFloat16(Float32(activated) * Float32(up[index]))
    end
    return
end

function LifeAI._qwen3_vl_device_mlp!(
    ::CUDA.CuArray{BFloat16},
    product,
    gate,
    up,
)::Bool
    threads = 128
    blocks = cld(length(product), threads)
    return _qwen3_mlp_raw_launch!(product, gate, up, threads, blocks)
end

function _qwen3_cuda_residual_kernel!(output, residual, projected)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index > size(output, 1) && return
    @inbounds output[index, 1, 1] = BFloat16(
        Float32(residual[index, 1, 1]) + Float32(projected[index, 1, 1]),
    )
    return
end

function LifeAI._qwen3_vl_device_residual!(
    ::CUDA.CuArray{BFloat16},
    output,
    residual,
    projected,
)::Bool
    threads = 128
    blocks = cld(size(output, 1), threads)
    if !(_qwen3_array(output, BFloat16, Val(3)) &&
        _qwen3_array(residual, BFloat16, Val(3)) &&
        _qwen3_array(projected, BFloat16, Val(3)))
        CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_residual_kernel!(
            output, residual, projected,
        )
        return true
    end
    raw = if isassigned(_QWEN3_RESIDUAL_RAW)
        existing = _QWEN3_RESIDUAL_RAW[]
        if existing.output_id != objectid(output) ||
                _qwen3_binding_stale(existing.output.val, output)
            existing.output.val = _qwen3_device_array(output)
            existing.projected.val = _qwen3_device_array(projected)
            existing.output_id = objectid(output)
        end
        existing
    else
        compiled = CUDA.@cuda launch=false _qwen3_cuda_residual_kernel!(
            output, residual, projected,
        )
        created = _qwen3_residual_raw(compiled, output, residual, projected)
        _QWEN3_RESIDUAL_RAW[] = created
        created
    end
    raw.residual.val = _qwen3_device_array(residual)
    return _qwen3_cu_launch!(raw, threads, blocks)
end

function _qwen3_cuda_decode_rope_pair_kernel!(
    query_out,
    queries,
    key_out,
    keys,
    cosine,
    sine,
    half,
)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    query_total = half * size(queries, 2)
    total = query_total + half * size(keys, 2)
    index > total && return
    if index <= query_total
        local_index = index
        x = queries
        output = query_out
    else
        local_index = index - query_total
        x = keys
        output = key_out
    end
    head = (local_index - 1) ÷ half + 1
    lane = (local_index - 1) % half + 1
    @inbounds begin
        x1 = Float32(x[lane, head, 1, 1])
        x2 = Float32(x[lane + half, head, 1, 1])
        cosine_value = Float32(cosine[lane, 1, 1])
        sine_value = Float32(sine[lane, 1, 1])
        upper = Float32(BFloat16(x1 * cosine_value)) +
            Float32(BFloat16(-x2 * sine_value))
        lower = Float32(BFloat16(x2 * cosine_value)) +
            Float32(BFloat16(x1 * sine_value))
        output[lane, head, 1, 1] = BFloat16(upper)
        output[lane + half, head, 1, 1] = BFloat16(lower)
    end
    return
end

function _qwen3_cuda_decode_rope_kernel!(output, x, cosine, sine, half)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    total = half * size(x, 2)
    index > total && return
    head = (index - 1) ÷ half + 1
    lane = (index - 1) % half + 1
    @inbounds begin
        x1 = Float32(x[lane, head, 1, 1])
        x2 = Float32(x[lane + half, head, 1, 1])
        cosine_value = Float32(cosine[lane, 1, 1])
        sine_value = Float32(sine[lane, 1, 1])
        upper = Float32(BFloat16(x1 * cosine_value)) +
            Float32(BFloat16(-x2 * sine_value))
        lower = Float32(BFloat16(x2 * cosine_value)) +
            Float32(BFloat16(x1 * sine_value))
        output[lane, head, 1, 1] = BFloat16(upper)
        output[lane + half, head, 1, 1] = BFloat16(lower)
    end
    return
end

function _qwen3_cuda_rope_table(values, half::Int)
    ndims(values) == 3 && return values
    return reshape(values, half, 1, 1)
end

function _qwen3_cuda_qk_promote_kernel!(dest, src)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(src, 1)
    heads = size(src, 2)
    total = head_dim * heads
    index > total && return
    lane = (index - 1) % head_dim + 1
    head = (index - 1) ÷ head_dim + 1
    @inbounds dest[lane, head, 1, 1] = 1.0f0 * Float32(src[lane, head, 1, 1])
    return
end

function _qwen3_cuda_qk_norm_fused_kernel!(
    query_out,
    queries,
    query_scale,
    key_out,
    keys,
    key_scale,
    mean_square,
    epsilon,
    head_dim,
)
    thread = CUDA.threadIdx().x
    head = CUDA.blockIdx().x
    q_heads = size(queries, 2)
    head > q_heads && return
    val = 0.0f0
    if thread <= head_dim
        promoted = 1.0f0 * Float32(queries[thread, head, 1, 1])
        val += abs2(promoted)
    end
    val = _qwen3_reduce_block(val)
    if thread == 1
        @inbounds mean_square[1, head, 1, 1] = val
    end
    CUDA.sync_threads()
    if thread <= head_dim
        @inbounds begin
            mean = mean_square[1, head, 1, 1] / Float32(head_dim)
            rms = sqrt(mean + epsilon)
            normalized = BFloat16(Float32(queries[thread, head, 1, 1]) / rms)
            query_out[thread, head, 1, 1] = BFloat16(
                Float32(normalized) * Float32(query_scale[thread, 1, 1, 1]),
            )
        end
    end
    head <= size(keys, 2) || return
    val = 0.0f0
    if thread <= head_dim
        promoted = 1.0f0 * Float32(keys[thread, head, 1, 1])
        val += abs2(promoted)
    end
    val = _qwen3_reduce_block(val)
    if thread == 1
        @inbounds mean_square[1, head, 1, 1] = val
    end
    CUDA.sync_threads()
    if thread <= head_dim
        @inbounds begin
            mean = mean_square[1, head, 1, 1] / Float32(head_dim)
            rms = sqrt(mean + epsilon)
            normalized = BFloat16(Float32(keys[thread, head, 1, 1]) / rms)
            key_out[thread, head, 1, 1] = BFloat16(
                Float32(normalized) * Float32(key_scale[thread, 1, 1, 1]),
            )
        end
    end
    return
end

function _qwen3_cuda_qk_norm_kernel!(output, x, scale, mean_square, epsilon, head_dim)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    total = head_dim * size(x, 2)
    index > total && return
    head = (index - 1) ÷ head_dim + 1
    lane = (index - 1) % head_dim + 1
    @inbounds begin
        mean = mean_square[1, head, 1, 1] / Float32(head_dim)
        rms = sqrt(mean + epsilon)
        normalized = BFloat16(Float32(x[lane, head, 1, 1]) / rms)
        output[lane, head, 1, 1] = BFloat16(
            Float32(normalized) * Float32(scale[lane, 1, 1, 1]),
        )
    end
    return
end

const _Qwen3HiddenScale = CUDA.CuArray{
    BFloat16,
    3,
    parentmodule(CUDA.CuArray).DeviceMemory,
}
const _QWEN3_HIDDEN_FUSED = Ref(0)
const _QWEN3_HSCALE_N = Ref(0)
const _QWEN3_HSCALE_IDS = Vector{UInt}(undef, 96)
const _QWEN3_HSCALE_TABLES = Vector{_Qwen3HiddenScale}(undef, 96)

function _qwen3_hidden_scale_table(scale::CUDA.CuArray{BFloat16,1})
    id = objectid(scale)
    n = _QWEN3_HSCALE_N[]
    ids = _QWEN3_HSCALE_IDS
    @inbounds for i in 1:n
        ids[i] == id && return _QWEN3_HSCALE_TABLES[i]
    end
    table = reshape(scale, size(scale, 1), 1, 1)
    if n < length(ids)
        n += 1
        @inbounds ids[n] = id
        @inbounds _QWEN3_HSCALE_TABLES[n] = table
        _QWEN3_HSCALE_N[] = n
    end
    return table
end

_qwen3_hidden_scale_table(scale) = reshape(scale, size(scale, 1), 1, 1)

function _qwen3_cuda_hidden_single_kernel!(
    output,
    x,
    scale,
    mean_square,
    epsilon,
    hidden_size,
)
    thread = CUDA.threadIdx().x
    block = CUDA.blockIdx().x
    partials = CUDA.CuStaticSharedArray(Float32, 4)
    stride = 576 * 4
    for slot in 1:4
        ireduce = thread + (slot - 1) * 576
        val = 0.0f0
        while ireduce <= 2048
            val += abs2(1.0f0 * Float32(x[ireduce, 1, 1]))
            ireduce += stride
        end
        val = _qwen3_reduce_block(val)
        if thread == 1
            @inbounds partials[slot] = val
        end
        CUDA.sync_threads()
    end
    val = 0.0f0
    if thread <= 4
        val += partials[thread]
    end
    val = _qwen3_reduce_block(val)
    broadcast = CUDA.CuStaticSharedArray(Float32, 1)
    if thread == 1
        @inbounds broadcast[1] = val
    end
    CUDA.sync_threads()
    sum_val = broadcast[1]
    if thread == 1 && block == 1
        @inbounds mean_square[1, 1, 1] = sum_val
    end
    mean = sum_val / Float32(hidden_size)
    rms = sqrt(mean + epsilon)
    index = (block - 1) * 512 + thread
    if thread <= 512
        @inbounds begin
            normalized = BFloat16(Float32(x[index, 1, 1]) / rms)
            output[index, 1, 1] = BFloat16(
                Float32(normalized) * Float32(scale[index, 1, 1]),
            )
        end
    end
    return
end

function _qwen3_hidden_fused_ready(mean_square, xf)
    flag = _QWEN3_HIDDEN_FUSED[]
    flag == 0 || return flag == 1
    plan = _qwen3_abs2_plan(mean_square, xf)
    ready = plan.mode === :partial && plan.threads == 576 && plan.blocks == 4
    _QWEN3_HIDDEN_FUSED[] = ready ? 1 : -1
    return ready
end

function LifeAI._qwen3_vl_device_hidden_norm_fused!(
    output::CUDA.CuArray{BFloat16},
    x::CUDA.CuArray{BFloat16},
    scale::CUDA.CuArray{BFloat16,1},
    mean_square::CUDA.CuArray{Float32},
    xf::CUDA.CuArray{Float32},
    epsilon::Float32,
    hidden_size::Int,
)::Bool
    hidden_size == 2048 || return false
    size(x, 1) == 2048 || return false
    _qwen3_hidden_fused_ready(mean_square, xf) || return false
    scale_table = _qwen3_hidden_scale_table(scale)
    return _qwen3_hidden_raw_launch!(
        output, x, scale_table, mean_square, epsilon, hidden_size,
    )
end

function _qwen3_cuda_hidden_promote_kernel!(dest, src)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index > size(src, 1) && return
    @inbounds dest[index, 1, 1] = 1.0f0 * Float32(src[index, 1, 1])
    return
end

function _qwen3_cuda_hidden_norm_kernel!(output, x, scale, mean_square, epsilon, hidden_size)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index > hidden_size && return
    @inbounds begin
        mean = mean_square[1, 1, 1] / Float32(hidden_size)
        rms = sqrt(mean + epsilon)
        normalized = BFloat16(Float32(x[index, 1, 1]) / rms)
        output[index, 1, 1] = BFloat16(
            Float32(normalized) * Float32(scale[index, 1, 1]),
        )
    end
    return
end

function LifeAI._qwen3_vl_device_qk_promote!(
    ::CUDA.CuArray{Float32},
    dest,
    src,
)
    total = size(src, 1) * size(src, 2)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_qk_promote_kernel!(dest, src)
    return dest
end

const _Qwen3ScaleTable = CUDA.CuArray{
    BFloat16,
    4,
    parentmodule(CUDA.CuArray).DeviceMemory,
}
const _QWEN3_QK_SCALE_N = Ref(0)
const _QWEN3_QK_SCALE_IDS = Vector{UInt}(undef, 96)
const _QWEN3_QK_SCALE_TABLES = Vector{_Qwen3ScaleTable}(undef, 96)

function _qwen3_qk_scale_table(scale::CUDA.CuArray{BFloat16,1})
    id = objectid(scale)
    n = _QWEN3_QK_SCALE_N[]
    ids = _QWEN3_QK_SCALE_IDS
    @inbounds for i in 1:n
        ids[i] == id && return _QWEN3_QK_SCALE_TABLES[i]
    end
    table = reshape(scale, size(scale, 1), 1, 1, 1)
    if n < length(ids)
        n += 1
        @inbounds ids[n] = id
        @inbounds _QWEN3_QK_SCALE_TABLES[n] = table
        _QWEN3_QK_SCALE_N[] = n
    end
    return table
end

_qwen3_qk_scale_table(scale) = reshape(scale, size(scale, 1), 1, 1, 1)

function LifeAI._qwen3_vl_device_qk_norm_fused!(
    query_out::CUDA.CuArray{BFloat16},
    queries,
    query_scale,
    key_out,
    keys,
    key_scale,
    mean_square,
    epsilon::Float32,
    head_dim::Int,
)::Bool
    size(queries, 1) == head_dim || return false
    size(keys, 1) == head_dim || return false
    size(query_out) == size(queries) || return false
    size(key_out) == size(keys) || return false
    size(queries, 2) >= size(keys, 2) || return false
    size(queries, 3) == 1 && size(queries, 4) == 1 || return false
    size(keys, 3) == 1 && size(keys, 4) == 1 || return false
    threads = CUDA.nextwarp(CUDA.device(), head_dim)
    threads <= 1024 || return false
    query_table = ndims(query_scale) == 4 ? query_scale : _qwen3_qk_scale_table(query_scale)
    key_table = ndims(key_scale) == 4 ? key_scale : _qwen3_qk_scale_table(key_scale)
    return _qwen3_qk_raw_launch!(
        query_out,
        queries,
        query_table,
        key_out,
        keys,
        key_table,
        mean_square,
        epsilon,
        head_dim,
        threads,
    )
end

function LifeAI._qwen3_vl_device_qk_norm!(
    ::CUDA.CuArray{BFloat16},
    output,
    x,
    scale,
    mean_square,
    epsilon::Float32,
    head_dim::Int,
)
    total = head_dim * size(x, 2)
    threads = 128
    blocks = cld(total, threads)
    scale_table = ndims(scale) == 4 ? scale : _qwen3_qk_scale_table(scale)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_qk_norm_kernel!(
        output,
        x,
        scale_table,
        mean_square,
        epsilon,
        head_dim,
    )
    return output
end

function LifeAI._qwen3_vl_device_hidden_promote!(
    dest::CUDA.CuArray{Float32},
    src,
)
    threads = 128
    blocks = cld(size(src, 1), threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_hidden_promote_kernel!(dest, src)
    return dest
end

function LifeAI._qwen3_vl_device_hidden_norm!(
    ::CUDA.CuArray{BFloat16},
    output,
    x,
    scale,
    mean_square,
    epsilon::Float32,
    hidden_size::Int,
)
    threads = 128
    blocks = cld(hidden_size, threads)
    scale_table = ndims(scale) == 3 ? scale : reshape(scale, size(scale, 1), 1, 1)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_hidden_norm_kernel!(
        output,
        x,
        scale_table,
        mean_square,
        epsilon,
        hidden_size,
    )
    return output
end

function LifeAI._qwen3_vl_device_decode_rope!(
    ::CUDA.CuArray{BFloat16},
    output,
    x,
    cos_values,
    sin_values,
    half::Int,
)
    cosine = _qwen3_cuda_rope_table(cos_values, half)
    sine = _qwen3_cuda_rope_table(sin_values, half)
    total = half * size(x, 2)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_decode_rope_kernel!(
        output,
        x,
        cosine,
        sine,
        half,
    )
    return output
end

function LifeAI._qwen3_vl_device_decode_rope_pair!(
    query_out::CUDA.CuArray{BFloat16},
    queries,
    key_out,
    keys,
    cos_values,
    sin_values,
    half::Int,
)::Bool
    size(queries, 1) == 2 * half || return false
    size(keys, 1) == 2 * half || return false
    size(query_out) == size(queries) || return false
    size(key_out) == size(keys) || return false
    cosine = _qwen3_cuda_rope_table(cos_values, half)
    sine = _qwen3_cuda_rope_table(sin_values, half)
    total = half * (size(queries, 2) + size(keys, 2))
    threads = 128
    blocks = cld(total, threads)
    return _qwen3_rope_raw_launch!(
        query_out, queries, key_out, keys, cosine, sine, half, threads, blocks,
    )
end

struct _Qwen3GemmLaunch{H,A}
    handle::H
    alpha::A
    beta::A
end

const _Qwen3GemmScalar = CUDA.CuArray{
    Float32,
    1,
    parentmodule(CUDA.CuArray).DeviceMemory,
}
const _QWEN3_GEMM_LAUNCH = Ref{_Qwen3GemmLaunch{CUDA.CUBLAS.Handle,_Qwen3GemmScalar}}()
const _QWEN3_BF16_TYPE = convert(CUDA.CUBLAS.cudaDataType_t, BFloat16)

function _qwen3_gemm_launch()
    isassigned(_QWEN3_GEMM_LAUNCH) && return _QWEN3_GEMM_LAUNCH[]
    launch = _Qwen3GemmLaunch(
        CUDA.CUBLAS.handle(),
        CUDA.fill(1.0f0, 1),
        CUDA.fill(0.0f0, 1),
    )
    _QWEN3_GEMM_LAUNCH[] = launch
    return launch
end

# Same address as `pointer`, without the per-call stream bookkeeping in
# `convert(::CuPtr, ::Managed)`.
function _qwen3_raw_ptr(x::CUDA.CuArray{T}) where {T}
    return convert(CUDA.CuPtr{T}, x.data[].mem) + x.offset
end

@inline function _qwen3_index_offset(indices::Tuple, strides::Tuple)
    isempty(indices) && return 0
    return (first(first(indices)) - 1) * first(strides) +
        _qwen3_index_offset(Base.tail(indices), Base.tail(strides))
end

function _qwen3_raw_ptr(x::SubArray{T}) where {T}
    parent_array = parent(x)
    strides = ntuple(dim -> stride(parent_array, dim), Val(length(x.indices)))
    return _qwen3_raw_ptr(parent_array) +
        _qwen3_index_offset(x.indices, strides) * sizeof(T)
end

_qwen3_device_ptr(x::Union{CUDA.CuArray,SubArray}) = _qwen3_raw_ptr(x)
_qwen3_device_ptr(x) = pointer(x)

@noinline function _qwen3_gemm_failed(status)
    error("Qwen3-VL cublasGemmEx failed: $status")
end

@inline function _qwen3_cublas_gemm_ex(
    handle,
    transa::CUDA.CUBLAS.cublasOperation_t,
    transb::CUDA.CUBLAS.cublasOperation_t,
    m::Cint,
    n::Cint,
    k::Cint,
    alpha::CUDA.CuPtr,
    A::CUDA.CuPtr,
    lda::Cint,
    B::CUDA.CuPtr,
    ldb::Cint,
    beta::CUDA.CuPtr,
    C::CUDA.CuPtr,
    ldc::Cint,
)
    return @ccall gc_safe=true CUDA.CUBLAS.libcublas.cublasGemmEx(
        handle::CUDA.CUBLAS.cublasHandle_t,
        transa::CUDA.CUBLAS.cublasOperation_t,
        transb::CUDA.CUBLAS.cublasOperation_t,
        m::Cint,
        n::Cint,
        k::Cint,
        convert(CUDA.CuPtr{Cvoid}, alpha)::CUDA.CuPtr{Cvoid},
        convert(CUDA.CuPtr{Cvoid}, A)::CUDA.CuPtr{Cvoid},
        _QWEN3_BF16_TYPE::CUDA.CUBLAS.cudaDataType_t,
        lda::Cint,
        convert(CUDA.CuPtr{Cvoid}, B)::CUDA.CuPtr{Cvoid},
        _QWEN3_BF16_TYPE::CUDA.CUBLAS.cudaDataType_t,
        ldb::Cint,
        convert(CUDA.CuPtr{Cvoid}, beta)::CUDA.CuPtr{Cvoid},
        convert(CUDA.CuPtr{Cvoid}, C)::CUDA.CuPtr{Cvoid},
        _QWEN3_BF16_TYPE::CUDA.CUBLAS.cudaDataType_t,
        ldc::Cint,
        CUDA.CUBLAS.CUBLAS_COMPUTE_32F::CUDA.CUBLAS.cublasComputeType_t,
        CUDA.CUBLAS.CUBLAS_GEMM_DEFAULT::CUDA.CUBLAS.cublasGemmAlgo_t,
    )::CUDA.CUBLAS.cublasStatus_t
end

function LifeAI._qwen3_vl_device_gemm_storage!(
    ::CUDA.CuArray{BFloat16},
    C::CMat,
    A::AMat,
    B::BMat,
    transA::Char,
) where {
    CMat<:AbstractMatrix{BFloat16},
    AMat<:AbstractMatrix{BFloat16},
    BMat<:AbstractMatrix{BFloat16},
}
    launch = _qwen3_gemm_launch()
    opA = transA == 'N' ? CUDA.CUBLAS.CUBLAS_OP_N : CUDA.CUBLAS.CUBLAS_OP_T
    m = size(A, transA == 'N' ? 1 : 2)
    k = size(A, transA == 'N' ? 2 : 1)
    n = size(B, 2)
    status = _qwen3_cublas_gemm_ex(
        launch.handle,
        opA,
        CUDA.CUBLAS.CUBLAS_OP_N,
        Cint(m),
        Cint(n),
        Cint(k),
        _qwen3_device_ptr(launch.alpha),
        _qwen3_device_ptr(A),
        Cint(max(1, stride(A, 2))),
        _qwen3_device_ptr(B),
        Cint(max(1, stride(B, 2))),
        _qwen3_device_ptr(launch.beta),
        _qwen3_device_ptr(C),
        Cint(max(1, stride(C, 2))),
    )
    status == CUDA.CUBLAS.CUBLAS_STATUS_SUCCESS || _qwen3_gemm_failed(status)
    return C
end

function LifeAI._qwen3_vl_device_gemm_token!(
    output::CUDA.CuArray{BFloat16},
    weight::CUDA.CuArray{BFloat16,2},
    x::CUDA.CuArray{BFloat16},
)
    size(output, 2) == 1 && size(output, 3) == 1 ||
        return _qwen3_vl_device_gemm!(
            reshape(output, size(output, 1), :),
            weight,
            reshape(x, size(x, 1), :),
        )
    size(x, 2) == 1 && size(x, 3) == 1 ||
        return _qwen3_vl_device_gemm!(
            reshape(output, size(output, 1), :),
            weight,
            reshape(x, size(x, 1), :),
        )
    launch = _qwen3_gemm_launch()
    status = _qwen3_cublas_gemm_ex(
        launch.handle,
        CUDA.CUBLAS.CUBLAS_OP_N,
        CUDA.CUBLAS.CUBLAS_OP_N,
        Cint(size(weight, 1)),
        Cint(1),
        Cint(size(weight, 2)),
        _qwen3_device_ptr(launch.alpha),
        _qwen3_device_ptr(weight),
        Cint(max(1, stride(weight, 2))),
        _qwen3_device_ptr(x),
        Cint(max(1, stride(x, 2))),
        _qwen3_device_ptr(launch.beta),
        _qwen3_device_ptr(output),
        Cint(max(1, stride(output, 2))),
    )
    status == CUDA.CUBLAS.CUBLAS_STATUS_SUCCESS || _qwen3_gemm_failed(status)
    return output
end

mutable struct _Qwen3Abs2Plan{K,R,O,P}
    mode::Symbol
    kernel::K
    Rreduce::R
    Rother::O
    threads::Int
    blocks::Int
    shmem::Int
    partial::P
end

const _QWEN3_ABS2_PLANS = Dict{Any,Any}()
const _Qwen3Abs2Core = parentmodule(CUDA.CuArray)
const _Qwen3Abs2Device4 = _Qwen3Abs2Core.CuDeviceArray{Float32,4,1}
const _Qwen3Abs2Index = Tuple{
    Base.Slice{Base.OneTo{Int64}},
    UnitRange{Int64},
    Base.Slice{Base.OneTo{Int64}},
    Base.Slice{Base.OneTo{Int64}},
}
const _Qwen3Abs2View = SubArray{
    Float32,4,_Qwen3Abs2Device4,_Qwen3Abs2Index,false,
}
const _Qwen3Abs2Axes = CartesianIndices{4,NTuple{4,Base.OneTo{Int64}}}
const _Qwen3Abs2Kernel = _Qwen3Abs2Core.HostKernel{
    typeof(_Qwen3Abs2Core.partial_mapreduce_grid),
    Tuple{
        typeof(abs2),
        typeof(Base.add_sum),
        Float32,
        _Qwen3Abs2Axes,
        _Qwen3Abs2Axes,
        Val{true},
        _Qwen3Abs2View,
        _Qwen3Abs2View,
    },
}
const _Qwen3Abs2QK = _Qwen3Abs2Plan{
    _Qwen3Abs2Kernel,
    _Qwen3Abs2Axes,
    _Qwen3Abs2Axes,
    Nothing,
}
const _QWEN3_ABS2_QUERY = Ref{_Qwen3Abs2QK}()
const _QWEN3_ABS2_KEY = Ref{_Qwen3Abs2QK}()

function _qwen3_abs2_threads(dev, wanted_threads, max_threads, shuffle)
    if wanted_threads > max_threads
        return shuffle ? CUDA.prevwarp(dev, max_threads) : max_threads
    end
    return wanted_threads
end

function _qwen3_abs2_plan(dest::AbstractArray, x)
    key = (
        CUDA.device(),
        size(dest),
        size(x),
        eltype(dest),
        typeof(dest),
        typeof(x),
    )
    cached = get(_QWEN3_ABS2_PLANS, key, nothing)
    cached === nothing || return cached
    plan = _qwen3_build_abs2_plan(dest, x)
    _QWEN3_ABS2_PLANS[key] = plan
    return plan
end

function _qwen3_build_abs2_plan(dest, x)
    f = abs2
    op = Base.add_sum
    init = zero(eltype(dest))
    T = eltype(dest)
    dev = CUDA.device()
    shuffle = T <: Union{
        Bool,
        UInt8, UInt16, UInt32, UInt64, UInt128,
        Int8, Int16, Int32, Int64, Int128,
        Float16, Float32, Float64,
        ComplexF16, ComplexF32, ComplexF64,
    }
    CUDACore = parentmodule(CUDA.CuArray)
    R = dest
    A = x
    if ndims(R) < ndims(A)
        R = reshape(R, Base.fill_to_length(size(R), 1, Val(ndims(A))))
    end
    Rother = CartesianIndices(axes(R))
    Rreduce = CartesianIndices(ifelse.(
        axes(A) .== axes(R),
        Ref(Base.OneTo(1)),
        axes(A),
    ))
    wanted_threads = shuffle ? CUDA.nextwarp(dev, length(Rreduce)) : length(Rreduce)
    compute_shmem(threads) = shuffle ? 0 : threads * sizeof(T)
    compute_threads(max_threads) = _qwen3_abs2_threads(
        dev, wanted_threads, max_threads, shuffle,
    )
    kernel = CUDA.@cuda launch=false CUDACore.partial_mapreduce_grid(
        f, op, init, Rreduce, Rother, Val(shuffle), R, A,
    )
    kernel_config = CUDA.launch_configuration(
        kernel.fun;
        shmem=compute_shmem ∘ compute_threads,
    )
    reduce_threads = compute_threads(kernel_config.threads)
    other_blocks = length(Rother)
    reduce_blocks = if other_blocks >= kernel_config.blocks
        1
    else
        min(
            cld(length(Rreduce), reduce_threads),
            cld(kernel_config.blocks, other_blocks),
        )
    end
    if reduce_blocks == 1
        return _Qwen3Abs2Plan(
            :single,
            kernel,
            Rreduce,
            Rother,
            reduce_threads,
            reduce_blocks * other_blocks,
            compute_shmem(reduce_threads),
            nothing,
        )
    end

    empty_partial = similar(R, ntuple(_ -> 0, Val(ndims(R) + 1)))
    partial_kernel = CUDA.@cuda launch=false CUDACore.partial_mapreduce_grid(
        f, op, init, Rreduce, Rother, Val(shuffle), empty_partial, A,
    )
    partial_config = CUDA.launch_configuration(
        partial_kernel.fun;
        shmem=compute_shmem ∘ compute_threads,
    )
    partial_threads = compute_threads(partial_config.threads)
    partial_blocks = if other_blocks >= partial_config.blocks
        1
    else
        min(
            cld(length(Rreduce), partial_threads),
            cld(partial_config.blocks, other_blocks),
        )
    end
    partial = similar(R, (size(R)..., partial_blocks))
    return _Qwen3Abs2Plan(
        :partial,
        partial_kernel,
        Rreduce,
        Rother,
        partial_threads,
        partial_blocks * other_blocks,
        compute_shmem(partial_threads),
        partial,
    )
end

function _qwen3_cuda_abs2_tail_kernel!(dest, src, nreduce)
    thread = CUDA.threadIdx().x
    val = 0.0f0
    if thread <= nreduce
        val += src[1, 1, 1, thread]
    end
    val = _qwen3_reduce_block(val)
    if thread == 1
        @inbounds dest[1, 1, 1] = val
    end
    return
end

const _Qwen3TailCore = parentmodule(CUDA.CuArray)
const _Qwen3TailDestD = _Qwen3TailCore.CuDeviceArray{Float32,3,1}
const _Qwen3TailSrcD = _Qwen3TailCore.CuDeviceArray{Float32,4,1}
const _Qwen3TailKernel = _Qwen3TailCore.HostKernel{
    typeof(_qwen3_cuda_abs2_tail_kernel!),
    Tuple{_Qwen3TailDestD,_Qwen3TailSrcD,Int64},
}
const _QWEN3_ABS2_TAIL = Ref{_Qwen3TailKernel}()
const _Qwen3TailDest = CUDA.CuArray{Float32,3,_Qwen3TailCore.DeviceMemory}
const _Qwen3TailSrc = CUDA.CuArray{Float32,4,_Qwen3TailCore.DeviceMemory}

function _qwen3_abs2_tail!(dest::_Qwen3TailDest, src::_Qwen3TailSrc)::Bool
    nreduce = size(src, 4)
    size(dest) == (1, 1, 1) || return false
    size(src) == (1, 1, 1, nreduce) || return false
    threads = CUDA.nextwarp(CUDA.device(), nreduce)
    threads <= 32 || return false
    if isassigned(_QWEN3_ABS2_TAIL)
        _QWEN3_ABS2_TAIL[](dest, src, nreduce; threads, blocks=1)
        return true
    end
    compiled = CUDA.@cuda launch=false _qwen3_cuda_abs2_tail_kernel!(dest, src, nreduce)
    compiled isa _Qwen3TailKernel || return false
    _QWEN3_ABS2_TAIL[] = compiled
    compiled(dest, src, nreduce; threads, blocks=1)
    return true
end

_qwen3_abs2_tail!(dest, src) = false

function _qwen3_run_abs2!(
    plan::_Qwen3Abs2Plan{K,R,O,P},
    dest,
    x,
) where {K,R,O,P}
    init = zero(eltype(dest))
    if plan.mode === :single
        plan.kernel(
            abs2,
            Base.add_sum,
            init,
            plan.Rreduce,
            plan.Rother,
            Val(true),
            dest,
            x;
            threads=plan.threads,
            blocks=plan.blocks,
            shmem=plan.shmem,
        )
    else
        plan.kernel(
            abs2,
            Base.add_sum,
            init,
            plan.Rreduce,
            plan.Rother,
            Val(true),
            plan.partial,
            x;
            threads=plan.threads,
            blocks=plan.blocks,
            shmem=plan.shmem,
        )
        _qwen3_abs2_tail!(dest, plan.partial) || sum!(identity, dest, plan.partial)
    end
    return dest
end

function _qwen3_abs2_is_qk(dest, x)
    return ndims(x) == 4 && ndims(dest) == 4 &&
        size(x, 3) == 1 && size(x, 4) == 1 &&
        size(dest) == (1, size(x, 2), 1, 1) &&
        dest isa SubArray && x isa SubArray
end

function _qwen3_abs2_stored!(ref::Ref{_Qwen3Abs2QK}, dest, x)::Bool
    heads = size(x, 2)
    threads = CUDA.nextwarp(CUDA.device(), size(x, 1))
    if isassigned(ref)
        plan = ref[]
        if plan.blocks == heads && plan.threads == threads
            _qwen3_run_abs2!(plan, dest, x)
            return true
        end
        return false
    end
    plan = _qwen3_build_abs2_plan(dest, x)
    if !(plan isa _Qwen3Abs2QK) || plan.blocks != heads || plan.threads != threads
        return false
    end
    ref[] = plan
    _qwen3_run_abs2!(plan, dest, x)
    return true
end

function LifeAI._qwen3_vl_device_sum_abs2!(::CUDA.CuArray, dest, x)
    eltype(dest) === Float32 || return sum!(abs2, dest, x)
    if _qwen3_abs2_is_qk(dest, x)
        ref = size(x, 2) == size(parent(x), 2) ? _QWEN3_ABS2_QUERY : _QWEN3_ABS2_KEY
        _qwen3_abs2_stored!(ref, dest, x) && return dest
    end
    plan = _qwen3_abs2_plan(dest, x)
    return _qwen3_run_abs2!(plan, dest, x)
end

function _qwen3_cuda_attention_softmax_kernel!(
    f32_scores,
    scaled,
    scores,
    scaling,
    exponents,
    maxima,
    weights,
    weight_perm,
    seq,
)
    thread = CUDA.threadIdx().x
    groups = size(scores, 1)
    other = CUDA.blockIdx().x
    group = (other - 1) % groups + 1
    head = (other - 1) ÷ groups + 1
    if thread <= seq
        @inbounds begin
            promoted = 1.0f0 * Float32(scores[group, thread, head])
            rounded = BFloat16(promoted * scaling)
            scaled[group, thread, head] = rounded
            f32_scores[group, thread, head] = 1.0f0 * Float32(rounded)
        end
    end
    CUDA.sync_threads()
    if thread == 1
        m = f32_scores[group, 1, head]
        for token in 2:seq
            m = max(m, f32_scores[group, token, head])
        end
        @inbounds maxima[group, 1, head] = m
    end
    CUDA.sync_threads()
    m = maxima[group, 1, head]
    exp_val = 0.0f0
    if thread <= seq
        @inbounds begin
            exp_val = exp(f32_scores[group, thread, head] - m)
            exponents[group, thread, head] = exp_val
        end
    end
    exp_val = _qwen3_reduce_block(exp_val)
    if thread == 1
        @inbounds maxima[group, 1, head] = exp_val
    end
    CUDA.sync_threads()
    if thread <= seq
        @inbounds begin
            value = BFloat16(exponents[group, thread, head] / maxima[group, 1, head])
            weights[group, thread, head] = value
            weight_perm[thread, group, head] = value
        end
    end
    capacity = size(weight_perm, 1)
    token = seq + thread
    while token <= capacity
        @inbounds weight_perm[token, group, head] = weights[group, token, head]
        token += CUDA.blockDim().x
    end
    return
end

function _qwen3_cuda_attention_scale_kernel!(f32_scores, scaled, scores, scaling, seq)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    groups = size(scores, 1)
    total = groups * seq * size(scores, 3)
    index > total && return
    group = (index - 1) % groups + 1
    token = ((index - 1) ÷ groups) % seq + 1
    head = (index - 1) ÷ (groups * seq) + 1
    @inbounds begin
        promoted = 1.0f0 * Float32(scores[group, token, head])
        rounded = BFloat16(promoted * scaling)
        scaled[group, token, head] = rounded
        f32_scores[group, token, head] = 1.0f0 * Float32(rounded)
    end
    return
end

function _qwen3_cuda_attention_exp_kernel!(exponents, f32_scores, maxima, seq)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    groups = size(f32_scores, 1)
    total = groups * seq * size(f32_scores, 3)
    index > total && return
    group = (index - 1) % groups + 1
    token = ((index - 1) ÷ groups) % seq + 1
    head = (index - 1) ÷ (groups * seq) + 1
    @inbounds exponents[group, token, head] = exp(
        f32_scores[group, token, head] - maxima[group, 1, head],
    )
    return
end

function _qwen3_cuda_attention_weight_perm_kernel!(dest, weights, exponents, maxima, seq)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    capacity = size(dest, 1)
    groups = size(dest, 2)
    kv_heads = size(dest, 3)
    total = capacity * groups * kv_heads
    index > total && return
    token = (index - 1) % capacity + 1
    group = ((index - 1) ÷ capacity) % groups + 1
    head = (index - 1) ÷ (capacity * groups) + 1
    if token <= seq
        @inbounds begin
            value = BFloat16(exponents[group, token, head] / maxima[group, 1, head])
            weights[group, token, head] = value
            dest[token, group, head] = value
        end
    else
        @inbounds dest[token, group, head] = weights[group, token, head]
    end
    return
end

function _qwen3_cuda_attention_weight_kernel!(weights, exponents, maxima, seq)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    groups = size(exponents, 1)
    total = groups * seq * size(exponents, 3)
    index > total && return
    group = (index - 1) % groups + 1
    token = ((index - 1) ÷ groups) % seq + 1
    head = (index - 1) ÷ (groups * seq) + 1
    @inbounds weights[group, token, head] = BFloat16(
        exponents[group, token, head] / maxima[group, 1, head],
    )
    return
end

const _Qwen3SoftCore = parentmodule(CUDA.CuArray)
const _Qwen3SoftF32D = _Qwen3SoftCore.CuDeviceArray{Float32,3,1}
const _Qwen3SoftBF16D = _Qwen3SoftCore.CuDeviceArray{BFloat16,3,1}
const _Qwen3SoftIndex = Tuple{
    Base.Slice{Base.OneTo{Int64}},
    UnitRange{Int64},
    Base.Slice{Base.OneTo{Int64}},
}
const _Qwen3SoftF32View = SubArray{
    Float32,3,_Qwen3SoftF32D,_Qwen3SoftIndex,false,
}
const _Qwen3SoftBF16View = SubArray{
    BFloat16,3,_Qwen3SoftBF16D,_Qwen3SoftIndex,false,
}
const _Qwen3SoftKernel = _Qwen3SoftCore.HostKernel{
    typeof(_qwen3_cuda_attention_softmax_kernel!),
    Tuple{
        _Qwen3SoftF32View,
        _Qwen3SoftBF16View,
        _Qwen3SoftBF16View,
        Float32,
        _Qwen3SoftF32D,
        _Qwen3SoftF32D,
        _Qwen3SoftBF16D,
        _Qwen3SoftBF16D,
        Int64,
    },
}
const _QWEN3_SOFT_KERNEL = Ref{_Qwen3SoftKernel}()

mutable struct _Qwen3Param{T}
    val::T
end

function Base.unsafe_convert(::Type{Ptr{Cvoid}}, box::_Qwen3Param)
    return pointer_from_objref(box)
end

function _qwen3_device_array(xs::CUDA.CuArray{T,N}) where {T,N}
    core = parentmodule(CUDA.CuArray)
    ptr = reinterpret(core.LLVMPtr{T,core.AS.Global}, _qwen3_raw_ptr(xs))
    return core.CuDeviceArray{T,N,core.AS.Global}(
        ptr, size(xs), xs.maxsize - xs.offset,
    )
end

# Reused carve shells keep their objectid while data, offset, and dims move.
@inline function _qwen3_ptr_stale(stored_ptr, xs::CUDA.CuArray{T}) where {T}
    core = parentmodule(CUDA.CuArray)
    current = reinterpret(core.LLVMPtr{T,core.AS.Global}, _qwen3_raw_ptr(xs))
    return stored_ptr != current
end

@inline function _qwen3_binding_stale(stored, xs::CUDA.CuArray)
    return _qwen3_ptr_stale(stored.ptr, xs) || stored.dims != size(xs)
end

@inline function _qwen3_alias_current(
    table::CUDA.CuArray{T,N},
    storage::CUDA.CuArray{T},
    dims::NTuple{N,Int},
) where {T,N}
    return table.data === storage.data &&
        table.offset == storage.offset &&
        table.dims == dims
end

_qwen3_dense_parent(x::CUDA.CuArray) = x
_qwen3_dense_parent(x::SubArray) = parent(x)

mutable struct _Qwen3SoftRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    param_ptr::Ptr{Ptr{Cvoid}}
    state::_Qwen3Param{parentmodule(CUDA.CuArray).KernelState}
    f32::_Qwen3Param{_Qwen3SoftF32D}
    scaled::_Qwen3Param{_Qwen3SoftBF16D}
    scores::_Qwen3Param{_Qwen3SoftBF16D}
    scaling::_Qwen3Param{Float32}
    exponents::_Qwen3Param{_Qwen3SoftF32D}
    maxima::_Qwen3Param{_Qwen3SoftF32D}
    weights::_Qwen3Param{_Qwen3SoftBF16D}
    weight_perm::_Qwen3Param{_Qwen3SoftBF16D}
    seq::_Qwen3Param{Int64}
    stream::parentmodule(CUDA.CuArray).CUstream
    scores_id::UInt
end

const _QWEN3_SOFT_RAW = Ref{_Qwen3SoftRaw}()

function _qwen3_soft_raw(compiled, f32, scaled, scores, scaling, exponents, maxima, weights, weight_perm)
    core = parentmodule(CUDA.CuArray)
    state = _Qwen3Param(core.KernelState(compiled.state.exception_info, UInt32(0)))
    f32_box = _Qwen3Param(_qwen3_device_array(f32))
    scaled_box = _Qwen3Param(_qwen3_device_array(scaled))
    scores_box = _Qwen3Param(_qwen3_device_array(scores))
    scaling_box = _Qwen3Param(scaling)
    exponents_box = _Qwen3Param(_qwen3_device_array(exponents))
    maxima_box = _Qwen3Param(_qwen3_device_array(maxima))
    weights_box = _Qwen3Param(_qwen3_device_array(weights))
    perm_box = _Qwen3Param(_qwen3_device_array(weight_perm))
    seq_box = _Qwen3Param(Int64(0))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, f32_box),
        Base.unsafe_convert(Ptr{Cvoid}, scaled_box),
        Base.unsafe_convert(Ptr{Cvoid}, scores_box),
        Base.unsafe_convert(Ptr{Cvoid}, scaling_box),
        Base.unsafe_convert(Ptr{Cvoid}, exponents_box),
        Base.unsafe_convert(Ptr{Cvoid}, maxima_box),
        Base.unsafe_convert(Ptr{Cvoid}, weights_box),
        Base.unsafe_convert(Ptr{Cvoid}, perm_box),
        Base.unsafe_convert(Ptr{Cvoid}, seq_box),
    ]
    return _Qwen3SoftRaw(
        compiled.fun,
        params,
        pointer(params),
        state,
        f32_box,
        scaled_box,
        scores_box,
        scaling_box,
        exponents_box,
        maxima_box,
        weights_box,
        perm_box,
        seq_box,
        CUDA.stream().handle,
        objectid(scores),
    )
end

function _qwen3_soft_raw_launch!(raw::_Qwen3SoftRaw, seq::Int, threads::Int, blocks::Int)
    raw.seq.val = seq
    status = GC.@preserve raw begin
        @ccall gc_safe=true _Qwen3SoftCore.libcuda.cuLaunchKernel(
            raw.fun::_Qwen3SoftCore.CUfunction,
            Cuint(blocks)::Cuint,
            Cuint(1)::Cuint,
            Cuint(1)::Cuint,
            Cuint(threads)::Cuint,
            Cuint(1)::Cuint,
            Cuint(1)::Cuint,
            Cuint(0)::Cuint,
            raw.stream::_Qwen3SoftCore.CUstream,
            pointer(raw.params)::Ptr{Ptr{Cvoid}},
            C_NULL::Ptr{Ptr{Cvoid}},
        )::_Qwen3SoftCore.CUresult
    end
    status == _Qwen3SoftCore.CUDA_SUCCESS ||
        error("Qwen3-VL cuLaunchKernel failed: $status")
    return true
end

function _qwen3_launch_attention_softmax!(
    kernel,
    f32_scores,
    scaled,
    scores,
    scaling,
    exponents,
    maxima,
    weights,
    weight_perm,
    seq,
    threads,
)
    kernel(
        f32_scores,
        scaled,
        scores,
        scaling,
        exponents,
        maxima,
        weights,
        weight_perm,
        seq;
        threads,
        blocks=size(scores, 1) * size(scores, 3),
    )
    return true
end

function LifeAI._qwen3_vl_attention_weight_prefix!(
    weight_perm::CUDA.CuArray{BFloat16,3},
    seq::Int,
)
    # The surrounding attention call checks seq against the reserved capacity.
    # Keep storage, offset, and object identity while packing head columns at
    # the same leading dimension as dynamic attention's weight matrix.
    setfield!(weight_perm, :dims, (seq, size(weight_perm, 2), size(weight_perm, 3)))
    return nothing
end

function LifeAI._qwen3_vl_device_attention_softmax!(
    ::CUDA.CuArray{BFloat16},
    f32_scores,
    scaled,
    scores,
    scaling::Float32,
    exponents,
    maxima,
    weights,
    weight_perm,
    seq::Int,
)::Bool
    seq >= 1 || return false
    threads = CUDA.nextwarp(CUDA.device(), seq)
    threads <= _qwen3_attention_sum_thread_limit(maxima, f32_scores) || return false
    dense_f32 = _qwen3_dense_parent(f32_scores)
    dense_scaled = _qwen3_dense_parent(scaled)
    dense_scores = _qwen3_dense_parent(scores)
    dense_exponents = _qwen3_dense_parent(exponents)
    dense_maxima = _qwen3_dense_parent(maxima)
    dense_weights = _qwen3_dense_parent(weights)
    dense_perm = _qwen3_dense_parent(weight_perm)
    if _qwen3_array(dense_f32, Float32, Val(3)) &&
        _qwen3_array(dense_scaled, BFloat16, Val(3)) &&
        _qwen3_array(dense_scores, BFloat16, Val(3)) &&
        _qwen3_array(dense_exponents, Float32, Val(3)) &&
        _qwen3_array(dense_maxima, Float32, Val(3)) &&
        _qwen3_array(dense_weights, BFloat16, Val(3)) &&
        _qwen3_array(dense_perm, BFloat16, Val(3))
        raw = if isassigned(_QWEN3_SOFT_RAW)
            existing = _QWEN3_SOFT_RAW[]
            if existing.scores_id != objectid(dense_scores) ||
                    _qwen3_binding_stale(existing.scores.val, dense_scores) ||
                    _qwen3_binding_stale(existing.f32.val, dense_f32) ||
                    _qwen3_binding_stale(existing.weight_perm.val, dense_perm)
                existing.f32.val = _qwen3_device_array(dense_f32)
                existing.scaled.val = _qwen3_device_array(dense_scaled)
                existing.scores.val = _qwen3_device_array(dense_scores)
                existing.scaling.val = scaling
                existing.exponents.val = _qwen3_device_array(dense_exponents)
                existing.maxima.val = _qwen3_device_array(dense_maxima)
                existing.weights.val = _qwen3_device_array(dense_weights)
                existing.weight_perm.val = _qwen3_device_array(dense_perm)
                existing.scores_id = objectid(dense_scores)
            end
            existing
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_attention_softmax_kernel!(
                dense_f32,
                dense_scaled,
                dense_scores,
                scaling,
                dense_exponents,
                dense_maxima,
                dense_weights,
                dense_perm,
                seq,
            )
            created = _qwen3_soft_raw(
                compiled,
                dense_f32,
                dense_scaled,
                dense_scores,
                scaling,
                dense_exponents,
                dense_maxima,
                dense_weights,
                dense_perm,
            )
            _QWEN3_SOFT_RAW[] = created
            created
        end
        return _qwen3_soft_raw_launch!(
            raw, seq, threads, size(dense_scores, 1) * size(dense_scores, 3),
        )
    end
    if isassigned(_QWEN3_SOFT_KERNEL)
        return _qwen3_launch_attention_softmax!(
            _QWEN3_SOFT_KERNEL[],
            f32_scores,
            scaled,
            scores,
            scaling,
            exponents,
            maxima,
            weights,
            weight_perm,
            seq,
            threads,
        )
    end
    compiled = CUDA.@cuda launch=false _qwen3_cuda_attention_softmax_kernel!(
        f32_scores,
        scaled,
        scores,
        scaling,
        exponents,
        maxima,
        weights,
        weight_perm,
        seq,
    )
    if !(compiled isa _Qwen3SoftKernel)
        CUDA.@cuda threads=threads blocks=size(scores, 1) * size(scores, 3) _qwen3_cuda_attention_softmax_kernel!(
            f32_scores,
            scaled,
            scores,
            scaling,
            exponents,
            maxima,
            weights,
            weight_perm,
            seq,
        )
        return true
    end
    _QWEN3_SOFT_KERNEL[] = compiled
    return _qwen3_launch_attention_softmax!(
        compiled,
        f32_scores,
        scaled,
        scores,
        scaling,
        exponents,
        maxima,
        weights,
        weight_perm,
        seq,
        threads,
    )
end

function LifeAI._qwen3_vl_device_attention_scale!(
    ::CUDA.CuArray{BFloat16},
    f32_scores,
    scaled,
    scores,
    scaling::Float32,
    seq::Int,
)
    total = size(scores, 1) * seq * size(scores, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_attention_scale_kernel!(
        f32_scores,
        scaled,
        scores,
        scaling,
        seq,
    )
    return f32_scores
end

function LifeAI._qwen3_vl_device_attention_exp!(
    ::CUDA.CuArray{BFloat16},
    exponents,
    f32_scores,
    maxima,
    seq::Int,
)
    total = size(f32_scores, 1) * seq * size(f32_scores, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_attention_exp_kernel!(
        exponents,
        f32_scores,
        maxima,
        seq,
    )
    return exponents
end

function _qwen3_cuda_attention_max_kernel!(dest, src, seq)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    groups = size(src, 1)
    total = groups * size(dest, 3)
    index > total && return
    group = (index - 1) % groups + 1
    head = (index - 1) ÷ groups + 1
    m = src[group, 1, head]
    for token in 2:seq
        value = src[group, token, head]
        m = max(m, value)
    end
    @inbounds dest[group, 1, head] = m
    return
end

function LifeAI._qwen3_vl_device_attention_maximum!(
    dest::CUDA.CuArray{Float32},
    src,
    seq::Int,
)
    total = size(dest, 1) * size(dest, 3)
    threads = 32
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_attention_max_kernel!(
        dest,
        src,
        seq,
    )
    return dest
end

function _qwen3_reduce_warp(val)
    offset = UInt32(1)
    while offset < UInt32(32)
        val += CUDA.shfl_down_sync(0xffffffff, val, Int(offset))
        offset <<= 1
    end
    return val
end

function _qwen3_reduce_block(val)
    shared = CUDA.CuStaticSharedArray(Float32, 32)
    wid, lane = fldmod1(CUDA.threadIdx().x, 32)
    val = _qwen3_reduce_warp(val)
    if lane == 1
        @inbounds shared[wid] = val
    end
    CUDA.sync_threads()
    nwarps = fld1(CUDA.blockDim().x, 32)
    val = if CUDA.threadIdx().x <= nwarps
        @inbounds shared[lane]
    else
        0.0f0
    end
    if wid == 1
        val = _qwen3_reduce_warp(val)
    end
    return val
end

function _qwen3_cuda_attention_sum_kernel!(dest, src, seq)
    thread = CUDA.threadIdx().x
    other = CUDA.blockIdx().x
    groups = size(dest, 1)
    group = (other - 1) % groups + 1
    head = (other - 1) ÷ groups + 1
    val = 0.0f0
    if thread <= seq
        val += src[group, thread, head]
    end
    val = _qwen3_reduce_block(val)
    if thread == 1
        @inbounds dest[group, 1, head] = val
    end
    return
end

const _Qwen3SumCore = parentmodule(CUDA.CuArray)
const _Qwen3SumDevice = _Qwen3SumCore.CuDeviceArray{Float32,3,1}
const _Qwen3SumDest = CUDA.CuArray{Float32,3,_Qwen3SumCore.DeviceMemory}
const _Qwen3SumSrc = SubArray{
    Float32,
    3,
    _Qwen3SumDest,
    Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},
    false,
}
const _Qwen3SumView = SubArray{
    Float32,
    3,
    _Qwen3SumDevice,
    Tuple{Base.Slice{Base.OneTo{Int64}},UnitRange{Int64},Base.Slice{Base.OneTo{Int64}}},
    false,
}
const _Qwen3SumKernel = _Qwen3SumCore.HostKernel{
    typeof(_qwen3_cuda_attention_sum_kernel!),
    Tuple{_Qwen3SumDevice,_Qwen3SumView,Int64},
}
const _QWEN3_SUM_KERNEL = Ref{_Qwen3SumKernel}()
const _QWEN3_SUM_THREAD_LIMIT = Ref(0)

function _qwen3_run_attention_sum!(
    kernel::_Qwen3SumKernel,
    dest,
    src,
    seq,
    threads,
)
    kernel(dest, src, seq; threads, blocks=size(dest, 1) * size(dest, 3))
    return dest
end

# `sum!` uses one block only while `nextwarp(seq)` fits in that kernel's
# launch limit. Above the limit it splits the reduction, and the two orders
# are not bitwise identical.
function _qwen3_attention_sum_thread_limit!(dest, src)::Int
    Rreduce = CartesianIndices(ifelse.(
        axes(src) .== axes(dest),
        Ref(Base.OneTo(1)),
        axes(src),
    ))
    Rother = CartesianIndices(axes(dest))
    kernel = CUDA.@cuda launch=false _Qwen3SumCore.partial_mapreduce_grid(
        identity,
        Base.add_sum,
        0.0f0,
        Rreduce,
        Rother,
        Val(true),
        dest,
        src,
    )
    limit = CUDA.launch_configuration(kernel.fun).threads
    _QWEN3_SUM_THREAD_LIMIT[] = limit
    return limit
end

function _qwen3_attention_sum_thread_limit(dest, src)::Int
    limit = _QWEN3_SUM_THREAD_LIMIT[]
    limit > 0 && return limit
    return _qwen3_attention_sum_thread_limit!(dest, src)
end

function LifeAI._qwen3_vl_device_attention_sum!(
    dest::_Qwen3SumDest,
    src::_Qwen3SumSrc,
    seq::Int,
)
    threads = CUDA.nextwarp(CUDA.device(), seq)
    if threads > _qwen3_attention_sum_thread_limit(dest, src) ||
        threads < 1 || size(src, 2) != seq
        return nothing
    end
    if isassigned(_QWEN3_SUM_KERNEL)
        return _qwen3_run_attention_sum!(
            _QWEN3_SUM_KERNEL[], dest, src, seq, threads,
        )
    end
    compiled = CUDA.@cuda launch=false _qwen3_cuda_attention_sum_kernel!(dest, src, seq)
    _QWEN3_SUM_KERNEL[] = compiled
    return _qwen3_run_attention_sum!(compiled, dest, src, seq, threads)
end

function LifeAI._qwen3_vl_device_attention_weight_perm!(
    ::CUDA.CuArray{BFloat16},
    weight_perm,
    weights,
    exponents,
    maxima,
    seq::Int,
)
    seq <= size(weight_perm, 1) || return nothing
    total = size(weight_perm, 1) * size(weight_perm, 2) * size(weight_perm, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_attention_weight_perm_kernel!(
        weight_perm,
        weights,
        exponents,
        maxima,
        seq,
    )
    return weight_perm
end

function LifeAI._qwen3_vl_device_attention_weight!(
    ::CUDA.CuArray{BFloat16},
    weights,
    exponents,
    maxima,
    seq::Int,
)
    total = size(exponents, 1) * seq * size(exponents, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_attention_weight_kernel!(
        weights,
        exponents,
        maxima,
        seq,
    )
    return weights
end

function _qwen3_cuda_prepare_attention_kernel!(
    query_out,
    queries,
    groups,
    dest_keys,
    dest_values,
    src_keys,
    src_values,
)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(queries, 1)
    kv_heads = size(query_out, 3)
    query_total = groups * head_dim * kv_heads
    capacity = size(dest_keys, 2)
    kv_total = head_dim * capacity * size(dest_keys, 3)
    index > query_total + kv_total && return
    if index <= query_total
        group = (index - 1) % groups + 1
        lane = ((index - 1) ÷ groups) % head_dim + 1
        head = (index - 1) ÷ (groups * head_dim) + 1
        query_head = group + (head - 1) * groups
        @inbounds query_out[group, lane, head] = queries[lane, query_head, 1, 1]
    else
        kv_index = index - query_total
        lane = (kv_index - 1) % head_dim + 1
        token = ((kv_index - 1) ÷ head_dim) % capacity + 1
        head = (kv_index - 1) ÷ (head_dim * capacity) + 1
        @inbounds begin
            dest_keys[lane, token, head] = src_keys[lane, head, token, 1]
            dest_values[lane, token, head] = src_values[lane, head, token, 1]
        end
    end
    return
end

function _qwen3_cuda_load_query_kernel!(output, queries, groups)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(queries, 1)
    kv_heads = size(output, 3)
    total = groups * head_dim * kv_heads
    index > total && return
    group = (index - 1) % groups + 1
    lane = ((index - 1) ÷ groups) % head_dim + 1
    head = (index - 1) ÷ (groups * head_dim) + 1
    query_head = group + (head - 1) * groups
    @inbounds output[group, lane, head] = queries[lane, query_head, 1, 1]
    return
end

function _qwen3_cuda_copy_kv_pair_kernel!(dest_keys, dest_values, src_keys, src_values)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(dest_keys, 1)
    capacity = size(dest_keys, 2)
    kv_heads = size(dest_keys, 3)
    total = head_dim * capacity * kv_heads
    index > total && return
    lane = (index - 1) % head_dim + 1
    token = ((index - 1) ÷ head_dim) % capacity + 1
    head = (index - 1) ÷ (head_dim * capacity) + 1
    @inbounds begin
        dest_keys[lane, token, head] = src_keys[lane, head, token, 1]
        dest_values[lane, token, head] = src_values[lane, head, token, 1]
    end
    return
end

function _qwen3_cuda_copy_kv_kernel!(dest, src)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(dest, 1)
    capacity = size(dest, 2)
    kv_heads = size(dest, 3)
    total = head_dim * capacity * kv_heads
    index > total && return
    lane = (index - 1) % head_dim + 1
    token = ((index - 1) ÷ head_dim) % capacity + 1
    head = (index - 1) ÷ (head_dim * capacity) + 1
    @inbounds dest[lane, token, head] = src[lane, head, token, 1]
    return
end

function _qwen3_cuda_permute_weights_kernel!(dest, src)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    capacity = size(dest, 1)
    groups = size(dest, 2)
    kv_heads = size(dest, 3)
    total = capacity * groups * kv_heads
    index > total && return
    token = (index - 1) % capacity + 1
    group = ((index - 1) ÷ capacity) % groups + 1
    head = (index - 1) ÷ (capacity * groups) + 1
    @inbounds dest[token, group, head] = src[group, token, head]
    return
end

function _qwen3_cuda_store_context_kernel!(output, context, groups)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(context, 1)
    kv_heads = size(context, 3)
    total = head_dim * groups * kv_heads
    index > total && return
    lane = (index - 1) % head_dim + 1
    group = ((index - 1) ÷ head_dim) % groups + 1
    head = (index - 1) ÷ (head_dim * groups) + 1
    query_head = group + (head - 1) * groups
    @inbounds output[lane, query_head, 1, 1] = context[lane, group, head]
    return
end

const _Qwen3QueryMatrix = CUDA.CuArray{
    BFloat16,
    3,
    parentmodule(CUDA.CuArray).DeviceMemory,
}
const _QWEN3_QUERY_MATRIX = Ref{_Qwen3QueryMatrix}()
const _QWEN3_QUERY_ID = Ref(UInt(0))

function LifeAI._qwen3_vl_query_matrix(
    query3::CUDA.CuArray{BFloat16,5},
    groups::Int,
)
    id = objectid(query3)
    dims = (groups, size(query3, 3), size(query3, 4))
    if isassigned(_QWEN3_QUERY_MATRIX)
        table = _QWEN3_QUERY_MATRIX[]
        if id == _QWEN3_QUERY_ID[] && _qwen3_alias_current(table, query3, dims)
            return table
        end
        table = _qwen3_retarget_array!(table, query3, dims)
        _QWEN3_QUERY_ID[] = id
        return table
    end
    table = reshape(query3, dims)
    _QWEN3_QUERY_MATRIX[] = table
    _QWEN3_QUERY_ID[] = id
    return table
end

const _QWEN3_ATTN_PROJ = Ref{_Qwen3QueryMatrix}()
const _QWEN3_ATTN_PROJ_ID = Ref(UInt(0))

function LifeAI._qwen3_vl_attention_projected(
    context::CUDA.CuArray{BFloat16,4},
    hidden_size::Int,
)
    hidden_size == size(context, 1) * size(context, 2) &&
        size(context, 3) == 1 &&
        size(context, 4) == 1 ||
        return reshape(context, hidden_size, size(context, 3), size(context, 4))
    id = objectid(context)
    dims = (hidden_size, 1, 1)
    if isassigned(_QWEN3_ATTN_PROJ)
        table = _QWEN3_ATTN_PROJ[]
        if id == _QWEN3_ATTN_PROJ_ID[] && _qwen3_alias_current(table, context, dims)
            return table
        end
        table = _qwen3_retarget_array!(table, context, dims)
        _QWEN3_ATTN_PROJ_ID[] = id
        return table
    end
    table = reshape(context, dims)
    _QWEN3_ATTN_PROJ[] = table
    _QWEN3_ATTN_PROJ_ID[] = id
    return table
end

const _Qwen3BF16D1 = _Qwen3SoftCore.CuDeviceArray{BFloat16,1,1}

function _qwen3_device_vector(xs::CUDA.CuArray{T}) where {T}
    ptr = reinterpret(
        _Qwen3SoftCore.LLVMPtr{T,_Qwen3SoftCore.AS.Global},
        _qwen3_raw_ptr(xs),
    )
    n = length(xs)
    return _Qwen3SoftCore.CuDeviceArray{T,1,_Qwen3SoftCore.AS.Global}(ptr, (n,), n)
end

function _qwen3_embedding_column(embedding::CUDA.CuArray{T,2}, token_id::Int) where {T}
    ptr = reinterpret(
        _Qwen3SoftCore.LLVMPtr{T,_Qwen3SoftCore.AS.Global},
        _qwen3_raw_ptr(embedding) +
            (token_id - 1) * stride(embedding, 2) * sizeof(T),
    )
    n = size(embedding, 1)
    return _Qwen3SoftCore.CuDeviceArray{T,1,_Qwen3SoftCore.AS.Global}(ptr, (n,), n)
end

function _qwen3_cuda_embedding_copy_kernel!(output, source, n)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index > n && return
    @inbounds output[index] = source[index]
    return
end

mutable struct _Qwen3EmbedRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    output::_Qwen3Param{_Qwen3BF16D1}
    source::_Qwen3Param{_Qwen3BF16D1}
    n::_Qwen3Param{Int64}
    stream::_Qwen3SoftCore.CUstream
    output_id::UInt
end

const _QWEN3_EMBED_RAW = Ref{_Qwen3EmbedRaw}()

function _qwen3_embed_raw(compiled, output, source, n)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(
        compiled.state.exception_info, UInt32(0),
    ))
    output_box = _Qwen3Param(_qwen3_device_vector(output))
    source_box = _Qwen3Param(source)
    n_box = _Qwen3Param(Int64(n))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, output_box),
        Base.unsafe_convert(Ptr{Cvoid}, source_box),
        Base.unsafe_convert(Ptr{Cvoid}, n_box),
    ]
    return _Qwen3EmbedRaw(
        compiled.fun,
        params,
        state,
        output_box,
        source_box,
        n_box,
        CUDA.stream().handle,
        objectid(output),
    )
end

function LifeAI._qwen3_vl_device_embedding_copy!(
    ::CUDA.CuArray{BFloat16},
    output,
    embedding,
    token_id::Int,
)::Bool
    _qwen3_array(output, BFloat16, Val(3)) &&
        embedding isa CUDA.CuArray{BFloat16,2} &&
        size(output, 1) == size(embedding, 1) &&
        size(output, 2) == 1 &&
        size(output, 3) == 1 &&
        stride(embedding, 1) == 1 ||
        return false
    n = size(embedding, 1)
    threads = 256
    blocks = cld(n, threads)
    source = _qwen3_embedding_column(embedding, token_id)
    raw = if isassigned(_QWEN3_EMBED_RAW)
        existing = _QWEN3_EMBED_RAW[]
        if existing.output_id != objectid(output) ||
                existing.n.val != Int64(n) ||
                _qwen3_ptr_stale(existing.output.val.ptr, output)
            existing.output.val = _qwen3_device_vector(output)
            existing.n.val = Int64(n)
            existing.output_id = objectid(output)
        end
        existing
    else
        compiled = CUDA.@cuda launch=false _qwen3_cuda_embedding_copy_kernel!(
            reshape(output, n), view(embedding, :, token_id), n,
        )
        created = _qwen3_embed_raw(compiled, output, source, n)
        _QWEN3_EMBED_RAW[] = created
        created
    end
    raw.source.val = source
    return _qwen3_cu_launch!(raw, threads, blocks)
end

const _Qwen3TiedMatrix = CUDA.CuArray{
    BFloat16,
    2,
    parentmodule(CUDA.CuArray).DeviceMemory,
}
const _QWEN3_TIED_OUT = Ref{_Qwen3TiedMatrix}()
const _QWEN3_TIED_OUT_ID = Ref(UInt(0))
const _QWEN3_TIED_X = Ref{_Qwen3TiedMatrix}()
const _QWEN3_TIED_X_ID = Ref(UInt(0))

function _qwen3_tied_matrix(xs::CUDA.CuArray{BFloat16,3}, slot, idref)
    id = objectid(xs)
    dims = (size(xs, 1), size(xs, 2) * size(xs, 3))
    if isassigned(slot)
        table = slot[]
        if id == idref[] && _qwen3_alias_current(table, xs, dims)
            return table
        end
        table = _qwen3_retarget_array!(table, xs, dims)
        idref[] = id
        return table
    end
    table = reshape(xs, dims)
    slot[] = table
    idref[] = id
    return table
end

function LifeAI._qwen3_vl_device_tied_logits!(
    output::CUDA.CuArray{BFloat16},
    embedding::CUDA.CuArray{BFloat16,2},
    hidden::CUDA.CuArray{BFloat16},
)::Bool
    size(output, 2) == 1 && size(output, 3) == 1 &&
        size(hidden, 2) == 1 && size(hidden, 3) == 1 &&
        size(embedding, 1) == size(hidden, 1) &&
        size(embedding, 2) == size(output, 1) ||
        return false
    LifeAI._qwen3_vl_device_gemm!(
        _qwen3_tied_matrix(output, _QWEN3_TIED_OUT, _QWEN3_TIED_OUT_ID),
        embedding,
        _qwen3_tied_matrix(hidden, _QWEN3_TIED_X, _QWEN3_TIED_X_ID);
        transA='T',
    )
    return true
end

function LifeAI._qwen3_vl_device_copy_host!(
    dest::CUDA.CuArray{T},
    src::Array{T},
)::Bool where {T}
    length(dest) == length(src) && stride(dest, 1) == 1 && stride(src, 1) == 1 ||
        return false
    nbytes = length(dest) * sizeof(T)
    status = GC.@preserve src dest begin
        @ccall gc_safe=true _Qwen3SoftCore.libcuda.cuMemcpyHtoDAsync_v2(
            _qwen3_raw_ptr(dest)::CUDA.CuPtr{T},
            pointer(src)::Ptr{Cvoid},
            nbytes::Csize_t,
            CUDA.stream().handle::_Qwen3SoftCore.CUstream,
        )::_Qwen3SoftCore.CUresult
    end
    status == _Qwen3SoftCore.CUDA_SUCCESS ||
        error("Qwen3-VL host copy failed: $status")
    return true
end

function _qwen3_cu_launch!(owner, threads::Int, blocks::Int)
    status = GC.@preserve owner begin
        @ccall gc_safe=true _Qwen3SoftCore.libcuda.cuLaunchKernel(
            owner.fun::_Qwen3SoftCore.CUfunction,
            Cuint(blocks)::Cuint,
            Cuint(1)::Cuint,
            Cuint(1)::Cuint,
            Cuint(threads)::Cuint,
            Cuint(1)::Cuint,
            Cuint(1)::Cuint,
            Cuint(0)::Cuint,
            owner.stream::_Qwen3SoftCore.CUstream,
            pointer(owner.params)::Ptr{Ptr{Cvoid}},
            C_NULL::Ptr{Ptr{Cvoid}},
        )::_Qwen3SoftCore.CUresult
    end
    status == _Qwen3SoftCore.CUDA_SUCCESS ||
        error("Qwen3-VL cuLaunchKernel failed: $status")
    return true
end

const _Qwen3BF16D3 = _Qwen3SoftCore.CuDeviceArray{BFloat16,3,1}
const _Qwen3BF16D4 = _Qwen3SoftCore.CuDeviceArray{BFloat16,4,1}

mutable struct _Qwen3MlpRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    product::_Qwen3Param{_Qwen3BF16D3}
    gate::_Qwen3Param{_Qwen3BF16D3}
    up::_Qwen3Param{_Qwen3BF16D3}
    stream::_Qwen3SoftCore.CUstream
    product_id::UInt
end

const _QWEN3_MLP_RAW = Ref{_Qwen3MlpRaw}()

function _qwen3_mlp_raw(compiled, product, gate, up)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    product_box = _Qwen3Param(_qwen3_device_array(product))
    gate_box = _Qwen3Param(_qwen3_device_array(gate))
    up_box = _Qwen3Param(_qwen3_device_array(up))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, product_box),
        Base.unsafe_convert(Ptr{Cvoid}, gate_box),
        Base.unsafe_convert(Ptr{Cvoid}, up_box),
    ]
    return _Qwen3MlpRaw(
        compiled.fun,
        params,
        state,
        product_box,
        gate_box,
        up_box,
        CUDA.stream().handle,
        objectid(product),
    )
end

function _qwen3_mlp_raw_launch!(product, gate, up, threads, blocks)
    if _qwen3_array(product, BFloat16, Val(3)) &&
        _qwen3_array(gate, BFloat16, Val(3)) &&
        _qwen3_array(up, BFloat16, Val(3))
        raw = if isassigned(_QWEN3_MLP_RAW)
            existing = _QWEN3_MLP_RAW[]
            if existing.product_id != objectid(product) ||
                    _qwen3_binding_stale(existing.product.val, product)
                existing.product.val = _qwen3_device_array(product)
                existing.product_id = objectid(product)
            end
            existing
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_mlp_kernel!(product, gate, up)
            created = _qwen3_mlp_raw(compiled, product, gate, up)
            _QWEN3_MLP_RAW[] = created
            created
        end
        raw.gate.val = _qwen3_device_array(gate)
        raw.up.val = _qwen3_device_array(up)
        return _qwen3_cu_launch!(raw, threads, blocks)
    end
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_mlp_kernel!(product, gate, up)
    return true
end

mutable struct _Qwen3RopeRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    query_out::_Qwen3Param{_Qwen3BF16D4}
    queries::_Qwen3Param{_Qwen3BF16D4}
    key_out::_Qwen3Param{_Qwen3BF16D4}
    keys::_Qwen3Param{_Qwen3BF16D4}
    cosine::_Qwen3Param{_Qwen3BF16D3}
    sine::_Qwen3Param{_Qwen3BF16D3}
    half::_Qwen3Param{Int64}
    stream::_Qwen3SoftCore.CUstream
    output_id::UInt
end

const _QWEN3_ROPE_RAW = Ref{_Qwen3RopeRaw}()

function _qwen3_rope_raw(compiled, query_out, queries, key_out, keys, cosine, sine, half)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    query_out_box = _Qwen3Param(_qwen3_device_array(query_out))
    queries_box = _Qwen3Param(_qwen3_device_array(queries))
    key_out_box = _Qwen3Param(_qwen3_device_array(key_out))
    keys_box = _Qwen3Param(_qwen3_device_array(keys))
    cosine_box = _Qwen3Param(_qwen3_device_array(cosine))
    sine_box = _Qwen3Param(_qwen3_device_array(sine))
    half_box = _Qwen3Param(Int64(half))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, query_out_box),
        Base.unsafe_convert(Ptr{Cvoid}, queries_box),
        Base.unsafe_convert(Ptr{Cvoid}, key_out_box),
        Base.unsafe_convert(Ptr{Cvoid}, keys_box),
        Base.unsafe_convert(Ptr{Cvoid}, cosine_box),
        Base.unsafe_convert(Ptr{Cvoid}, sine_box),
        Base.unsafe_convert(Ptr{Cvoid}, half_box),
    ]
    return _Qwen3RopeRaw(
        compiled.fun,
        params,
        state,
        query_out_box,
        queries_box,
        key_out_box,
        keys_box,
        cosine_box,
        sine_box,
        half_box,
        CUDA.stream().handle,
        objectid(query_out),
    )
end

function _qwen3_rope_raw_launch!(
    query_out, queries, key_out, keys, cosine, sine, half, threads, blocks,
)
    if _qwen3_array(query_out, BFloat16, Val(4)) &&
        _qwen3_array(queries, BFloat16, Val(4)) &&
        _qwen3_array(key_out, BFloat16, Val(4)) &&
        _qwen3_array(keys, BFloat16, Val(4)) &&
        _qwen3_array(cosine, BFloat16, Val(3)) &&
        _qwen3_array(sine, BFloat16, Val(3))
        raw = if isassigned(_QWEN3_ROPE_RAW)
            existing = _QWEN3_ROPE_RAW[]
            if existing.output_id != objectid(query_out) ||
                    _qwen3_binding_stale(existing.query_out.val, query_out)
                existing.query_out.val = _qwen3_device_array(query_out)
                existing.key_out.val = _qwen3_device_array(key_out)
                existing.half.val = Int64(half)
                existing.output_id = objectid(query_out)
            end
            existing
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_decode_rope_pair_kernel!(
                query_out, queries, key_out, keys, cosine, sine, half,
            )
            created = _qwen3_rope_raw(
                compiled, query_out, queries, key_out, keys, cosine, sine, half,
            )
            _QWEN3_ROPE_RAW[] = created
            created
        end
        raw.queries.val = _qwen3_device_array(queries)
        raw.keys.val = _qwen3_device_array(keys)
        raw.cosine.val = _qwen3_device_array(cosine)
        raw.sine.val = _qwen3_device_array(sine)
        return _qwen3_cu_launch!(raw, threads, blocks)
    end
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_decode_rope_pair_kernel!(
        query_out, queries, key_out, keys, cosine, sine, half,
    )
    return true
end
const _Qwen3F32D4 = _Qwen3SoftCore.CuDeviceArray{Float32,4,1}

mutable struct _Qwen3QKRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    query_out::_Qwen3Param{_Qwen3BF16D4}
    queries::_Qwen3Param{_Qwen3BF16D4}
    query_scale::_Qwen3Param{_Qwen3BF16D4}
    key_out::_Qwen3Param{_Qwen3BF16D4}
    keys::_Qwen3Param{_Qwen3BF16D4}
    key_scale::_Qwen3Param{_Qwen3BF16D4}
    mean_square::_Qwen3Param{_Qwen3F32D4}
    epsilon::_Qwen3Param{Float32}
    head_dim::_Qwen3Param{Int64}
    stream::_Qwen3SoftCore.CUstream
    output_id::UInt
end

const _QWEN3_QK_RAW = Ref{_Qwen3QKRaw}()

function _qwen3_qk_raw(compiled, query_out, queries, query_scale, key_out, keys, key_scale, mean_square, epsilon, head_dim)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    query_out_box = _Qwen3Param(_qwen3_device_array(query_out))
    queries_box = _Qwen3Param(_qwen3_device_array(queries))
    query_scale_box = _Qwen3Param(_qwen3_device_array(query_scale))
    key_out_box = _Qwen3Param(_qwen3_device_array(key_out))
    keys_box = _Qwen3Param(_qwen3_device_array(keys))
    key_scale_box = _Qwen3Param(_qwen3_device_array(key_scale))
    mean_box = _Qwen3Param(_qwen3_device_array(mean_square))
    epsilon_box = _Qwen3Param(epsilon)
    head_box = _Qwen3Param(Int64(head_dim))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, query_out_box),
        Base.unsafe_convert(Ptr{Cvoid}, queries_box),
        Base.unsafe_convert(Ptr{Cvoid}, query_scale_box),
        Base.unsafe_convert(Ptr{Cvoid}, key_out_box),
        Base.unsafe_convert(Ptr{Cvoid}, keys_box),
        Base.unsafe_convert(Ptr{Cvoid}, key_scale_box),
        Base.unsafe_convert(Ptr{Cvoid}, mean_box),
        Base.unsafe_convert(Ptr{Cvoid}, epsilon_box),
        Base.unsafe_convert(Ptr{Cvoid}, head_box),
    ]
    return _Qwen3QKRaw(
        compiled.fun,
        params,
        state,
        query_out_box,
        queries_box,
        query_scale_box,
        key_out_box,
        keys_box,
        key_scale_box,
        mean_box,
        epsilon_box,
        head_box,
        CUDA.stream().handle,
        objectid(query_out),
    )
end

function _qwen3_qk_raw_launch!(
    query_out, queries, query_scale, key_out, keys, key_scale,
    mean_square, epsilon, head_dim, threads,
)
    blocks = size(queries, 2)
    if _qwen3_array(query_out, BFloat16, Val(4)) &&
        _qwen3_array(queries, BFloat16, Val(4)) &&
        _qwen3_array(key_out, BFloat16, Val(4)) &&
        _qwen3_array(keys, BFloat16, Val(4)) &&
        query_scale isa _Qwen3ScaleTable &&
        key_scale isa _Qwen3ScaleTable &&
        _qwen3_array(mean_square, Float32, Val(4))
        raw = if isassigned(_QWEN3_QK_RAW)
            existing = _QWEN3_QK_RAW[]
            if existing.output_id != objectid(query_out) ||
                    _qwen3_binding_stale(existing.query_out.val, query_out) ||
                    _qwen3_binding_stale(existing.mean_square.val, mean_square)
                existing.query_out.val = _qwen3_device_array(query_out)
                existing.key_out.val = _qwen3_device_array(key_out)
                existing.mean_square.val = _qwen3_device_array(mean_square)
                existing.epsilon.val = epsilon
                existing.head_dim.val = Int64(head_dim)
                existing.output_id = objectid(query_out)
            end
            existing
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_qk_norm_fused_kernel!(
                query_out, queries, query_scale, key_out, keys, key_scale,
                mean_square, epsilon, head_dim,
            )
            created = _qwen3_qk_raw(
                compiled, query_out, queries, query_scale, key_out, keys, key_scale,
                mean_square, epsilon, head_dim,
            )
            _QWEN3_QK_RAW[] = created
            created
        end
        raw.queries.val = _qwen3_device_array(queries)
        raw.query_scale.val = _qwen3_device_array(query_scale)
        raw.keys.val = _qwen3_device_array(keys)
        raw.key_scale.val = _qwen3_device_array(key_scale)
        return _qwen3_cu_launch!(raw, threads, blocks)
    end
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_qk_norm_fused_kernel!(
        query_out, queries, query_scale, key_out, keys, key_scale,
        mean_square, epsilon, head_dim,
    )
    return true
end
const _Qwen3QKVMatrix = CUDA.CuArray{
    BFloat16,
    4,
    parentmodule(CUDA.CuArray).DeviceMemory,
}
const _QWEN3_Q_MATRIX = Ref{_Qwen3QKVMatrix}()
const _QWEN3_K_MATRIX = Ref{_Qwen3QKVMatrix}()
const _QWEN3_V_MATRIX = Ref{_Qwen3QKVMatrix}()
const _QWEN3_Q_MATRIX_ID = Ref(UInt(0))
const _QWEN3_K_MATRIX_ID = Ref(UInt(0))
const _QWEN3_V_MATRIX_ID = Ref(UInt(0))

function _qwen3_retarget_array!(
    table::CUDA.CuArray{T,N},
    storage::CUDA.CuArray{T},
    dims::NTuple{N,Int},
) where {T,N}
    table.data = storage.data
    table.offset = storage.offset
    table.maxsize = storage.maxsize
    setfield!(table, :dims, dims)
    return table
end

function _qwen3_qkv_matrix(storage::CUDA.CuArray{BFloat16,3}, matrix, matrix_id, dims)
    id = objectid(storage)
    if isassigned(matrix)
        table = matrix[]
        if id == matrix_id[] && _qwen3_alias_current(table, storage, dims)
            return table
        end
        table = _qwen3_retarget_array!(table, storage, dims)
        matrix_id[] = id
        return table
    end
    table = reshape(storage, dims)
    matrix[] = table
    matrix_id[] = id
    return table
end

function LifeAI._qwen3_vl_qkv_matrices(
    query::CUDA.CuArray{BFloat16,3},
    key::CUDA.CuArray{BFloat16,3},
    value::CUDA.CuArray{BFloat16,3},
    head_dim::Int,
    num_heads::Int,
    num_kv_heads::Int,
)
    return (
        _qwen3_qkv_matrix(
            query, _QWEN3_Q_MATRIX, _QWEN3_Q_MATRIX_ID,
            (head_dim, num_heads, 1, 1),
        ),
        _qwen3_qkv_matrix(
            key, _QWEN3_K_MATRIX, _QWEN3_K_MATRIX_ID,
            (head_dim, num_kv_heads, 1, 1),
        ),
        _qwen3_qkv_matrix(
            value, _QWEN3_V_MATRIX, _QWEN3_V_MATRIX_ID,
            (head_dim, num_kv_heads, 1, 1),
        ),
    )
end

mutable struct _Qwen3ResidualRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    output::_Qwen3Param{_Qwen3BF16D3}
    residual::_Qwen3Param{_Qwen3BF16D3}
    projected::_Qwen3Param{_Qwen3BF16D3}
    stream::_Qwen3SoftCore.CUstream
    output_id::UInt
end

const _QWEN3_RESIDUAL_RAW = Ref{_Qwen3ResidualRaw}()

function _qwen3_residual_raw(compiled, output, residual, projected)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    output_box = _Qwen3Param(_qwen3_device_array(output))
    residual_box = _Qwen3Param(_qwen3_device_array(residual))
    projected_box = _Qwen3Param(_qwen3_device_array(projected))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, output_box),
        Base.unsafe_convert(Ptr{Cvoid}, residual_box),
        Base.unsafe_convert(Ptr{Cvoid}, projected_box),
    ]
    return _Qwen3ResidualRaw(
        compiled.fun,
        params,
        state,
        output_box,
        residual_box,
        projected_box,
        CUDA.stream().handle,
        objectid(output),
    )
end

mutable struct _Qwen3HiddenRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    output::_Qwen3Param{_Qwen3BF16D3}
    x::_Qwen3Param{_Qwen3BF16D3}
    scale::_Qwen3Param{_Qwen3BF16D3}
    mean_square::_Qwen3Param{_Qwen3SoftF32D}
    epsilon::_Qwen3Param{Float32}
    hidden_size::_Qwen3Param{Int64}
    stream::_Qwen3SoftCore.CUstream
    output_id::UInt
end

const _QWEN3_HIDDEN_RAW = Ref{_Qwen3HiddenRaw}()

function _qwen3_hidden_raw(compiled, output, x, scale, mean_square, epsilon, hidden_size)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    output_box = _Qwen3Param(_qwen3_device_array(output))
    x_box = _Qwen3Param(_qwen3_device_array(x))
    scale_box = _Qwen3Param(_qwen3_device_array(scale))
    mean_box = _Qwen3Param(_qwen3_device_array(mean_square))
    epsilon_box = _Qwen3Param(epsilon)
    hidden_box = _Qwen3Param(Int64(hidden_size))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, output_box),
        Base.unsafe_convert(Ptr{Cvoid}, x_box),
        Base.unsafe_convert(Ptr{Cvoid}, scale_box),
        Base.unsafe_convert(Ptr{Cvoid}, mean_box),
        Base.unsafe_convert(Ptr{Cvoid}, epsilon_box),
        Base.unsafe_convert(Ptr{Cvoid}, hidden_box),
    ]
    return _Qwen3HiddenRaw(
        compiled.fun,
        params,
        state,
        output_box,
        x_box,
        scale_box,
        mean_box,
        epsilon_box,
        hidden_box,
        CUDA.stream().handle,
        objectid(output),
    )
end

function _qwen3_hidden_raw_launch!(output, x, scale, mean_square, epsilon, hidden_size)
    if scale isa _Qwen3HiddenScale &&
        _qwen3_array(mean_square, Float32, Val(3))
        raw = if isassigned(_QWEN3_HIDDEN_RAW)
            existing = _QWEN3_HIDDEN_RAW[]
            if existing.output_id != objectid(output) ||
                    _qwen3_binding_stale(existing.output.val, output) ||
                    _qwen3_binding_stale(existing.mean_square.val, mean_square)
                existing.output.val = _qwen3_device_array(output)
                existing.mean_square.val = _qwen3_device_array(mean_square)
                existing.epsilon.val = epsilon
                existing.hidden_size.val = Int64(hidden_size)
                existing.output_id = objectid(output)
            end
            existing
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_hidden_single_kernel!(
                output, x, scale, mean_square, epsilon, hidden_size,
            )
            created = _qwen3_hidden_raw(
                compiled, output, x, scale, mean_square, epsilon, hidden_size,
            )
            _QWEN3_HIDDEN_RAW[] = created
            created
        end
        raw.x.val = _qwen3_device_array(x)
        raw.scale.val = _qwen3_device_array(scale)
        return _qwen3_cu_launch!(raw, 576, 4)
    end
    CUDA.@cuda threads=576 blocks=4 _qwen3_cuda_hidden_single_kernel!(
        output, x, scale, mean_square, epsilon, hidden_size,
    )
    return true
end

mutable struct _Qwen3PrepareRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    query_out::_Qwen3Param{_Qwen3BF16D3}
    queries::_Qwen3Param{_Qwen3BF16D4}
    groups::_Qwen3Param{Int64}
    dest_keys::_Qwen3Param{_Qwen3BF16D3}
    dest_values::_Qwen3Param{_Qwen3BF16D3}
    src_keys::_Qwen3Param{_Qwen3BF16D4}
    src_values::_Qwen3Param{_Qwen3BF16D4}
    stream::_Qwen3SoftCore.CUstream
    dest_id::UInt
end

const _QWEN3_PREPARE_RAW = Ref{_Qwen3PrepareRaw}()

function _qwen3_prepare_raw(compiled, output, queries, groups, key_dest, value_dest, key_parent, value_parent)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    query_out = _Qwen3Param(_qwen3_device_array(output))
    query_box = _Qwen3Param(_qwen3_device_array(queries))
    groups_box = _Qwen3Param(Int64(groups))
    dest_keys = _Qwen3Param(_qwen3_device_array(key_dest))
    dest_values = _Qwen3Param(_qwen3_device_array(value_dest))
    src_keys = _Qwen3Param(_qwen3_device_array(key_parent))
    src_values = _Qwen3Param(_qwen3_device_array(value_parent))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, query_out),
        Base.unsafe_convert(Ptr{Cvoid}, query_box),
        Base.unsafe_convert(Ptr{Cvoid}, groups_box),
        Base.unsafe_convert(Ptr{Cvoid}, dest_keys),
        Base.unsafe_convert(Ptr{Cvoid}, dest_values),
        Base.unsafe_convert(Ptr{Cvoid}, src_keys),
        Base.unsafe_convert(Ptr{Cvoid}, src_values),
    ]
    return _Qwen3PrepareRaw(
        compiled.fun,
        params,
        state,
        query_out,
        query_box,
        groups_box,
        dest_keys,
        dest_values,
        src_keys,
        src_values,
        CUDA.stream().handle,
        objectid(key_dest),
    )
end

function LifeAI._qwen3_vl_device_prepare_attention!(
    query_storage::CUDA.CuArray{BFloat16,5},
    queries,
    groups::Int,
    key_dest,
    value_dest,
    key_parent,
    value_parent,
)::Bool
    size(key_dest) == size(value_dest) || return false
    size(key_parent) == size(value_parent) || return false
    size(queries, 3) == 1 && size(queries, 4) == 1 || return false
    output = LifeAI._qwen3_vl_query_matrix(query_storage, groups)
    query_total = groups * size(queries, 1) * size(key_dest, 3)
    kv_total = size(key_dest, 1) * size(key_dest, 2) * size(key_dest, 3)
    threads = 128
    blocks = cld(query_total + kv_total, threads)
    if _qwen3_array(output, BFloat16, Val(3)) &&
        _qwen3_array(queries, BFloat16, Val(4)) &&
        _qwen3_array(key_dest, BFloat16, Val(3)) &&
        _qwen3_array(value_dest, BFloat16, Val(3)) &&
        _qwen3_array(key_parent, BFloat16, Val(4)) &&
        _qwen3_array(value_parent, BFloat16, Val(4))
        raw = if isassigned(_QWEN3_PREPARE_RAW)
            existing = _QWEN3_PREPARE_RAW[]
            if existing.dest_id != objectid(key_dest) ||
                    _qwen3_binding_stale(existing.dest_keys.val, key_dest) ||
                    _qwen3_binding_stale(existing.dest_values.val, value_dest) ||
                    _qwen3_binding_stale(existing.query_out.val, output)
                existing.query_out.val = _qwen3_device_array(output)
                existing.groups.val = Int64(groups)
                existing.dest_keys.val = _qwen3_device_array(key_dest)
                existing.dest_values.val = _qwen3_device_array(value_dest)
                existing.dest_id = objectid(key_dest)
            end
            existing
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_prepare_attention_kernel!(
                output, queries, groups, key_dest, value_dest, key_parent, value_parent,
            )
            created = _qwen3_prepare_raw(
                compiled, output, queries, groups, key_dest, value_dest, key_parent, value_parent,
            )
            _QWEN3_PREPARE_RAW[] = created
            created
        end
        raw.queries.val = _qwen3_device_array(queries)
        raw.src_keys.val = _qwen3_device_array(key_parent)
        raw.src_values.val = _qwen3_device_array(value_parent)
        return _qwen3_cu_launch!(raw, threads, blocks)
    end
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_prepare_attention_kernel!(
        output,
        queries,
        groups,
        key_dest,
        value_dest,
        key_parent,
        value_parent,
    )
    return true
end

function LifeAI._qwen3_vl_device_load_query!(
    ::CUDA.CuArray{BFloat16},
    output,
    queries,
    groups::Int,
)
    total = groups * size(queries, 1) * size(output, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_load_query_kernel!(
        output,
        queries,
        groups,
    )
    return output
end

function _qwen3_cuda_write_kv_pair_kernel!(
    cache_keys,
    cache_values,
    keys,
    values,
    index,
)
    linear = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(keys, 1)
    total = head_dim * size(keys, 2)
    linear > total && return
    lane = (linear - 1) % head_dim + 1
    head = (linear - 1) ÷ head_dim + 1
    @inbounds begin
        cache_keys[lane, head, index, 1] = keys[lane, head, 1, 1]
        cache_values[lane, head, index, 1] = values[lane, head, 1, 1]
    end
    return
end

function _qwen3_cuda_write_kv_kernel!(dest, src, index)
    linear = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    head_dim = size(src, 1)
    total = head_dim * size(src, 2)
    linear > total && return
    lane = (linear - 1) % head_dim + 1
    head = (linear - 1) ÷ head_dim + 1
    @inbounds dest[lane, head, index, 1] = src[lane, head, 1, 1]
    return
end

mutable struct _Qwen3KVRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    cache_keys::_Qwen3Param{_Qwen3BF16D4}
    cache_values::_Qwen3Param{_Qwen3BF16D4}
    keys::_Qwen3Param{_Qwen3BF16D4}
    values::_Qwen3Param{_Qwen3BF16D4}
    index::_Qwen3Param{Int64}
    stream::_Qwen3SoftCore.CUstream
end

const _QWEN3_KV_RAW = Ref{_Qwen3KVRaw}()

function _qwen3_kv_raw(compiled, cache_keys, cache_values, keys, values, index)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    cache_keys_box = _Qwen3Param(_qwen3_device_array(cache_keys))
    cache_values_box = _Qwen3Param(_qwen3_device_array(cache_values))
    keys_box = _Qwen3Param(_qwen3_device_array(keys))
    values_box = _Qwen3Param(_qwen3_device_array(values))
    index_box = _Qwen3Param(Int64(index))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, cache_keys_box),
        Base.unsafe_convert(Ptr{Cvoid}, cache_values_box),
        Base.unsafe_convert(Ptr{Cvoid}, keys_box),
        Base.unsafe_convert(Ptr{Cvoid}, values_box),
        Base.unsafe_convert(Ptr{Cvoid}, index_box),
    ]
    return _Qwen3KVRaw(
        compiled.fun,
        params,
        state,
        cache_keys_box,
        cache_values_box,
        keys_box,
        values_box,
        index_box,
        CUDA.stream().handle,
    )
end

function LifeAI._qwen3_vl_device_write_kv!(
    cache_keys::CUDA.CuArray{BFloat16},
    cache_values,
    keys,
    values,
    index::Int,
)::Bool
    size(keys) == size(values) || return false
    total = size(keys, 1) * size(keys, 2)
    threads = 128
    blocks = cld(total, threads)
    if cache_keys isa CUDA.CuArray{BFloat16,4} &&
        cache_values isa CUDA.CuArray{BFloat16,4} &&
        _qwen3_array(keys, BFloat16, Val(4)) &&
        _qwen3_array(values, BFloat16, Val(4))
        raw = if isassigned(_QWEN3_KV_RAW)
            _QWEN3_KV_RAW[]
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_write_kv_pair_kernel!(
                cache_keys, cache_values, keys, values, index,
            )
            created = _qwen3_kv_raw(
                compiled, cache_keys, cache_values, keys, values, index,
            )
            _QWEN3_KV_RAW[] = created
            created
        end
        raw.cache_keys.val = _qwen3_device_array(cache_keys)
        raw.cache_values.val = _qwen3_device_array(cache_values)
        raw.keys.val = _qwen3_device_array(keys)
        raw.values.val = _qwen3_device_array(values)
        raw.index.val = index
        return _qwen3_cu_launch!(raw, threads, blocks)
    end
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_write_kv_pair_kernel!(
        cache_keys, cache_values, keys, values, index,
    )
    return true
end

function LifeAI._qwen3_vl_device_copy_kv!(
    dest::CUDA.CuArray{BFloat16},
    src,
)
    total = size(dest, 1) * size(dest, 2) * size(dest, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_copy_kv_kernel!(dest, src)
    return dest
end

function LifeAI._qwen3_vl_device_copy_kv_pair!(
    dest_keys::CUDA.CuArray{BFloat16},
    dest_values,
    src_keys,
    src_values,
)
    size(dest_keys) == size(dest_values) || return nothing
    size(src_keys) == size(src_values) || return nothing
    total = size(dest_keys, 1) * size(dest_keys, 2) * size(dest_keys, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_copy_kv_pair_kernel!(
        dest_keys,
        dest_values,
        src_keys,
        src_values,
    )
    return dest_keys
end

function LifeAI._qwen3_vl_device_permute_weights!(
    dest::CUDA.CuArray{BFloat16},
    src,
)
    total = size(dest, 1) * size(dest, 2) * size(dest, 3)
    threads = 128
    blocks = cld(total, threads)
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_permute_weights_kernel!(
        dest,
        src,
    )
    return dest
end

mutable struct _Qwen3StoreRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    output::_Qwen3Param{_Qwen3BF16D4}
    context::_Qwen3Param{_Qwen3BF16D3}
    groups::_Qwen3Param{Int64}
    stream::_Qwen3SoftCore.CUstream
    output_id::UInt
end

const _QWEN3_STORE_RAW = Ref{_Qwen3StoreRaw}()

function _qwen3_store_raw(compiled, output, context, groups)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(compiled.state.exception_info, UInt32(0)))
    output_box = _Qwen3Param(_qwen3_device_array(output))
    context_box = _Qwen3Param(_qwen3_device_array(context))
    groups_box = _Qwen3Param(Int64(groups))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, output_box),
        Base.unsafe_convert(Ptr{Cvoid}, context_box),
        Base.unsafe_convert(Ptr{Cvoid}, groups_box),
    ]
    return _Qwen3StoreRaw(
        compiled.fun,
        params,
        state,
        output_box,
        context_box,
        groups_box,
        CUDA.stream().handle,
        objectid(output),
    )
end

function LifeAI._qwen3_vl_device_store_context!(
    output::CUDA.CuArray{BFloat16},
    context,
    groups::Int,
)::Bool
    total = size(context, 1) * groups * size(context, 3)
    threads = 128
    blocks = cld(total, threads)
    if _qwen3_array(output, BFloat16, Val(4)) &&
        _qwen3_array(context, BFloat16, Val(3))
        raw = if isassigned(_QWEN3_STORE_RAW)
            existing = _QWEN3_STORE_RAW[]
            if existing.output_id != objectid(output) ||
                    _qwen3_binding_stale(existing.output.val, output)
                existing.output.val = _qwen3_device_array(output)
                existing.context.val = _qwen3_device_array(context)
                existing.groups.val = Int64(groups)
                existing.output_id = objectid(output)
            end
            existing
        else
            compiled = CUDA.@cuda launch=false _qwen3_cuda_store_context_kernel!(
                output, context, groups,
            )
            created = _qwen3_store_raw(compiled, output, context, groups)
            _QWEN3_STORE_RAW[] = created
            created
        end
        return _qwen3_cu_launch!(raw, threads, blocks)
    end
    CUDA.@cuda threads=threads blocks=blocks _qwen3_cuda_store_context_kernel!(
        output,
        context,
        groups,
    )
    return true
end

function _qwen3_batch_ptr(x::AbstractArray{T,3}, index::Int) where {T}
    return _qwen3_device_ptr(x) + (index - 1) * stride(x, 3) * sizeof(T)
end

function LifeAI._qwen3_vl_device_batched_mul!(
    ::CUDA.CuArray{BFloat16},
    dest::AbstractArray{BFloat16,3},
    left::AbstractArray{BFloat16,3},
    right::AbstractArray{BFloat16,3},
)
    launch = _qwen3_gemm_launch()
    batch = size(dest, 3)
    m = size(left, 1)
    k = size(left, 2)
    n = size(right, 2)
    size(dest, 1) == m && size(dest, 2) == n && size(right, 1) == k ||
        _qwen3_gemm_failed(CUDA.CUBLAS.CUBLAS_STATUS_INVALID_VALUE)
    lda = Cint(max(1, stride(left, 2)))
    ldb = Cint(max(1, stride(right, 2)))
    ldc = Cint(max(1, stride(dest, 2)))
    alpha = _qwen3_device_ptr(launch.alpha)
    beta = _qwen3_device_ptr(launch.beta)
    for index in 1:batch
        left_index = size(left, 3) == 1 ? 1 : index
        right_index = size(right, 3) == 1 ? 1 : index
        status = _qwen3_cublas_gemm_ex(
            launch.handle,
            CUDA.CUBLAS.CUBLAS_OP_N,
            CUDA.CUBLAS.CUBLAS_OP_N,
            Cint(m),
            Cint(n),
            Cint(k),
            alpha,
            _qwen3_batch_ptr(left, left_index),
            lda,
            _qwen3_batch_ptr(right, right_index),
            ldb,
            beta,
            _qwen3_batch_ptr(dest, index),
            ldc,
        )
        status == CUDA.CUBLAS.CUBLAS_STATUS_SUCCESS || _qwen3_gemm_failed(status)
    end
    return dest
end

function _qwen3_carve(
    parent::CUDA.CuArray{T},
    dims::Dims,
    element_offset::Int,
) where {T}
    byte_offset = parent.offset + element_offset * sizeof(T)
    return CUDA.CuArray{T,length(dims)}(
        copy(parent.data),
        dims;
        maxsize=parent.maxsize,
        offset=byte_offset,
    )
end

function _qwen3_place!(
    arr::CUDA.CuArray{T,N},
    parent::CUDA.CuArray{T},
    dims::NTuple{N,Int},
    element_offset::Int,
) where {T,N}
    arr.data = parent.data
    arr.offset = parent.offset + element_offset * sizeof(T)
    arr.maxsize = parent.maxsize
    setfield!(arr, :dims, dims)
    return element_offset + prod(dims)
end

# The 56 carve shells stay in the scratch objects built by the first decode.
# The same request may retarget those CuArrays onto a new parent pair. The
# holder is parametric so the retarget function specializes on the real
# scratch types; a Ref{Any} call must not be the body that touches them.
mutable struct _Qwen3ReuseShells{A,R,Q,H,M,RS,QV,G,MR,E,T,B,F,P}
    attention::A
    rope::R
    qk::Q
    hidden::H
    mlp::M
    residual::RS
    qkv::QV
    gate_up::G
    mrope::MR
    embedding::E
    tied::T
    # Slices share this DataRef without a retain. These parents have to
    # stay reachable until the next request retargets every slice.
    bf16::B
    f32::F
    # Parameter pack from the first decode. It does not hold the KV cache.
    pack::P
    # Outputs can outlive their cache, so even an expired owner does not
    # make these mutable arrays available to another request.
    owner::WeakRef
end

const _QWEN3_SHELLS = Ref{Any}()
const _QWEN3_READY = Ref{Any}()

function _qwen3_capture_shells!(
    cache::LifeAI.Qwen3VLStaticKVCache,
    attention, rope, qk, hidden, mlp, residual, qkv, gate_up, mrope, embedding, tied,
    bf16::CUDA.CuArray{BFloat16},
    f32::CUDA.CuArray{Float32},
)
    pack = cache.decode_pack
    pack isa LifeAI._Qwen3VLDecodePack ||
        error("Qwen3-VL reuse requires the decode pack")
    return _Qwen3ReuseShells(
        attention, rope, qk, hidden, mlp, residual, qkv, gate_up, mrope,
        embedding, tied, bf16, f32, pack, WeakRef(cache),
    )
end

function _qwen3_place_workspace!(
    cache::LifeAI.Qwen3VLStaticKVCache,
    bf16::CUDA.CuArray{BFloat16},
    f32::CUDA.CuArray{Float32},
    shells::_Qwen3ReuseShells,
    hidden_size::Int,
    head_dim::Int,
    num_heads::Int,
    num_kv_heads::Int,
    intermediate_size::Int,
    vocab_size::Int,
    capacity::Int,
    groups::Int,
    half::Int,
    query_n::Int,
    kv_n::Int,
    n_bf16::Int,
    n_f32::Int,
)
    attention = shells.attention
    rope = shells.rope
    qk = shells.qk
    hidden = shells.hidden
    mlp = shells.mlp
    residual = shells.residual
    qkv = shells.qkv
    gate_up = shells.gate_up
    mrope = shells.mrope
    embedding = shells.embedding
    tied = shells.tied
    offset = 0
    offset = _qwen3_place!(attention.query5, bf16, (1, head_dim, groups, num_kv_heads, 1), offset)
    offset = _qwen3_place!(attention.query3, bf16, (1, groups, head_dim, num_kv_heads, 1), offset)
    offset = _qwen3_place!(attention.keys, bf16, (head_dim, capacity, num_kv_heads), offset)
    offset = _qwen3_place!(attention.values, bf16, (head_dim, capacity, num_kv_heads), offset)
    offset = _qwen3_place!(attention.scores, bf16, (groups, capacity, num_kv_heads), offset)
    offset = _qwen3_place!(attention.scaled, bf16, (groups, capacity, num_kv_heads), offset)
    offset = _qwen3_place!(attention.weights, bf16, (groups, capacity, num_kv_heads), offset)
    offset = _qwen3_place!(attention.weight_perm, bf16, (capacity, groups, num_kv_heads), offset)
    offset = _qwen3_place!(attention.context, bf16, (head_dim, groups, num_kv_heads), offset)
    offset = _qwen3_place!(attention.output, bf16, (head_dim, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.tmp_bf16, bf16, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.query_out, bf16, (head_dim, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.key_out, bf16, (head_dim, num_kv_heads, 1, 1), offset)
    offset = _qwen3_place!(qk.normalized, bf16, (head_dim, num_heads, 1, 1), offset)
    offset = _qwen3_place!(qk.query_out, bf16, (head_dim, num_heads, 1, 1), offset)
    offset = _qwen3_place!(qk.key_out, bf16, (head_dim, num_kv_heads, 1, 1), offset)
    offset = _qwen3_place!(hidden.normalized, bf16, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(hidden.output, bf16, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(mlp.activated, bf16, (intermediate_size, 1, 1), offset)
    offset = _qwen3_place!(mlp.product, bf16, (intermediate_size, 1, 1), offset)
    offset = _qwen3_place!(residual.projected, bf16, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(residual.output, bf16, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(qkv.query, bf16, (query_n, 1, 1), offset)
    offset = _qwen3_place!(qkv.key, bf16, (kv_n, 1, 1), offset)
    offset = _qwen3_place!(qkv.value, bf16, (kv_n, 1, 1), offset)
    offset = _qwen3_place!(gate_up.gate, bf16, (intermediate_size, 1, 1), offset)
    offset = _qwen3_place!(gate_up.up, bf16, (intermediate_size, 1, 1), offset)
    offset = _qwen3_place!(mrope.cos_values, bf16, (half, 1, 1), offset)
    offset = _qwen3_place!(mrope.sin_values, bf16, (half, 1, 1), offset)
    offset = _qwen3_place!(embedding.output, bf16, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(tied.output, bf16, (vocab_size, 1, 1), offset)
    offset == n_bf16 || error("Qwen3-VL reused workspace BF16 span does not match")
    offset = 0
    offset = _qwen3_place!(attention.scores_f32, f32, (groups, capacity, num_kv_heads), offset)
    offset = _qwen3_place!(attention.maxima, f32, (groups, 1, num_kv_heads), offset)
    offset = _qwen3_place!(attention.exponents, f32, (groups, capacity, num_kv_heads), offset)
    offset = _qwen3_place!(rope.x1, f32, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.x2, f32, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.cos_f32, f32, (half, 1, 1, 1), offset)
    offset = _qwen3_place!(rope.sin_f32, f32, (half, 1, 1, 1), offset)
    offset = _qwen3_place!(rope.tmp_f32, f32, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.term_a, f32, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.term_b, f32, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.upper, f32, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(rope.lower, f32, (half, num_heads, 1, 1), offset)
    offset = _qwen3_place!(qk.xf, f32, (head_dim, num_heads, 1, 1), offset)
    offset = _qwen3_place!(qk.mean_square, f32, (1, num_heads, 1, 1), offset)
    offset = _qwen3_place!(qk.norm_f32, f32, (head_dim, num_heads, 1, 1), offset)
    offset = _qwen3_place!(qk.scale_f32, f32, (head_dim, 1, 1, 1), offset)
    offset = _qwen3_place!(hidden.xf, f32, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(hidden.mean_square, f32, (1, 1, 1), offset)
    offset = _qwen3_place!(hidden.norm_f32, f32, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(hidden.scale_f32, f32, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(mlp.gate_f32, f32, (intermediate_size, 1, 1), offset)
    offset = _qwen3_place!(mlp.activated_f32, f32, (intermediate_size, 1, 1), offset)
    offset = _qwen3_place!(mlp.up_f32, f32, (intermediate_size, 1, 1), offset)
    offset = _qwen3_place!(residual.residual_f32, f32, (hidden_size, 1, 1), offset)
    offset = _qwen3_place!(residual.projected_f32, f32, (hidden_size, 1, 1), offset)
    offset == n_f32 || error("Qwen3-VL reused workspace Float32 span does not match")
    shells.bf16 = bf16
    shells.f32 = f32
    if isassigned(_QWEN3_READY) &&
        _QWEN3_READY[] === cache.decode_pack &&
        LifeAI._qwen3_vl_decode_pack_matches(_QWEN3_READY[].pack, shells.pack)
        _qwen3_reuse_ready!(_QWEN3_READY[], cache, capacity)
        return true
    end
    # The first build still needs one attention object. Later requests only
    # change its capacity; the arrays were already retargeted above.
    placed = LifeAI.Qwen3VLDecodeAttentionScratch(
        attention.query5, attention.query3, attention.keys, attention.values,
        attention.scores, attention.scaled, attention.scores_f32, attention.maxima,
        attention.exponents, attention.weights, attention.weight_perm,
        attention.context, attention.output,
        head_dim, num_heads, num_kv_heads, groups, capacity,
    )
    LifeAI._qwen3_vl_install_ready!(
        cache,
        shells.pack,
        embedding,
        mrope,
        placed,
        mlp,
        rope,
        qk,
        hidden,
        gate_up,
        residual,
        tied,
        qkv,
    )
    _QWEN3_READY[] = cache.decode_pack
    _qwen3_reuse_ready!(_QWEN3_READY[], cache, capacity)
    return true
end

function _qwen3_reuse_ready!(
    ready::LifeAI._Qwen3VLDecodeReady,
    cache::LifeAI.Qwen3VLStaticKVCache,
    capacity::Int,
)
    ready.attention.capacity = capacity
    cache.decode_pack = ready
    return ready
end

function _qwen3_fit_parent(existing::CUDA.CuArray{T}, prototype, ::Type{T}, n::Int) where {T}
    length(existing) >= n && return existing
    return similar(prototype, T, n)
end

function _qwen3_fit_parent(existing, prototype, ::Type{T}, n::Int) where {T}
    return similar(prototype, T, n)
end

function _qwen3_carve_shared(
    parent::CUDA.CuArray{T},
    dims::Dims,
    element_offset::Int,
) where {T}
    byte_offset = parent.offset + element_offset * sizeof(T)
    return CUDA.CuArray{T,length(dims)}(
        parent.data,
        dims;
        maxsize=parent.maxsize,
        offset=byte_offset,
    )
end

@inline function _qwen3_array(x, ::Type{T}, ::Val{N}) where {T,N}
    return x isa CUDA.CuArray{T,N}
end

function LifeAI._qwen3_vl_device_attention_scratch(
    prototype::CUDA.CuArray{BFloat16};
    head_dim::Integer,
    num_attention_heads::Integer,
    num_kv_heads::Integer,
    capacity::Integer,
)
    expected = LifeAI.qwen3_vl_decode_attention_scratch_bytes(;
        head_dim,
        num_attention_heads,
        num_kv_heads,
        capacity,
    )
    head_dim = Int(head_dim)
    num_attention_heads = Int(num_attention_heads)
    num_kv_heads = Int(num_kv_heads)
    capacity = Int(capacity)
    groups = num_attention_heads ÷ num_kv_heads
    n_bf16 = (head_dim * groups * num_kv_heads) * 2 +
        (head_dim * capacity * num_kv_heads) * 2 +
        (groups * capacity * num_kv_heads) * 3 +
        (capacity * groups * num_kv_heads) +
        (head_dim * groups * num_kv_heads) +
        (head_dim * num_attention_heads)
    n_f32 = (groups * capacity * num_kv_heads) * 2 + (groups * num_kv_heads)
    bf16 = similar(prototype, BFloat16, n_bf16)
    f32 = similar(prototype, Float32, n_f32)
    offset = 0
    query5 = _qwen3_carve(bf16, (1, head_dim, groups, num_kv_heads, 1), offset)
    offset += length(query5)
    query3 = _qwen3_carve(bf16, (1, groups, head_dim, num_kv_heads, 1), offset)
    offset += length(query3)
    keys = _qwen3_carve(bf16, (head_dim, capacity, num_kv_heads), offset)
    offset += length(keys)
    values = _qwen3_carve(bf16, (head_dim, capacity, num_kv_heads), offset)
    offset += length(values)
    scores = _qwen3_carve(bf16, (groups, capacity, num_kv_heads), offset)
    offset += length(scores)
    scaled = _qwen3_carve(bf16, (groups, capacity, num_kv_heads), offset)
    offset += length(scaled)
    weights = _qwen3_carve(bf16, (groups, capacity, num_kv_heads), offset)
    offset += length(weights)
    weight_perm = _qwen3_carve(bf16, (capacity, groups, num_kv_heads), offset)
    offset += length(weight_perm)
    context = _qwen3_carve(bf16, (head_dim, groups, num_kv_heads), offset)
    offset += length(context)
    output = _qwen3_carve(bf16, (head_dim, num_attention_heads, 1, 1), offset)
    offset = 0
    scores_f32 = _qwen3_carve(f32, (groups, capacity, num_kv_heads), offset)
    offset += length(scores_f32)
    maxima = _qwen3_carve(f32, (groups, 1, num_kv_heads), offset)
    offset += length(maxima)
    exponents = _qwen3_carve(f32, (groups, capacity, num_kv_heads), offset)
    scratch = LifeAI.Qwen3VLDecodeAttentionScratch(
        query5,
        query3,
        keys,
        values,
        scores,
        scaled,
        scores_f32,
        maxima,
        exponents,
        weights,
        weight_perm,
        context,
        output,
        head_dim,
        num_attention_heads,
        num_kv_heads,
        groups,
        capacity,
    )
    LifeAI.qwen3_vl_decode_attention_scratch_bytes(scratch) == expected ||
        error("Qwen3-VL packed attention scratch bytes do not match the formula")
    return scratch
end

function LifeAI._qwen3_vl_device_qk_rope_scratch(
    prototype::CUDA.CuArray{BFloat16};
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    expected = LifeAI.qwen3_vl_qk_rope_scratch_bytes(;
        head_dim,
        num_heads,
        num_kv_heads,
    )
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    half = head_dim ÷ 2
    spatial = half * num_heads
    n_f32 = 7 * spatial + 2 * half
    n_bf16 = spatial + head_dim * num_heads + head_dim * num_kv_heads
    f32 = similar(prototype, Float32, n_f32)
    bf16 = similar(prototype, BFloat16, n_bf16)
    offset = 0
    x1 = _qwen3_carve(f32, (half, num_heads, 1, 1), offset)
    offset += length(x1)
    x2 = _qwen3_carve(f32, (half, num_heads, 1, 1), offset)
    offset += length(x2)
    cos_f32 = _qwen3_carve(f32, (half, 1, 1, 1), offset)
    offset += length(cos_f32)
    sin_f32 = _qwen3_carve(f32, (half, 1, 1, 1), offset)
    offset += length(sin_f32)
    tmp_f32 = _qwen3_carve(f32, (half, num_heads, 1, 1), offset)
    offset += length(tmp_f32)
    term_a = _qwen3_carve(f32, (half, num_heads, 1, 1), offset)
    offset += length(term_a)
    term_b = _qwen3_carve(f32, (half, num_heads, 1, 1), offset)
    offset += length(term_b)
    upper = _qwen3_carve(f32, (half, num_heads, 1, 1), offset)
    offset += length(upper)
    lower = _qwen3_carve(f32, (half, num_heads, 1, 1), offset)
    offset = 0
    tmp_bf16 = _qwen3_carve(bf16, (half, num_heads, 1, 1), offset)
    offset += length(tmp_bf16)
    query_out = _qwen3_carve(bf16, (head_dim, num_heads, 1, 1), offset)
    offset += length(query_out)
    key_out = _qwen3_carve(bf16, (head_dim, num_kv_heads, 1, 1), offset)
    scratch = LifeAI.Qwen3VLQKRopeScratch(
        x1,
        x2,
        cos_f32,
        sin_f32,
        tmp_f32,
        tmp_bf16,
        term_a,
        term_b,
        upper,
        lower,
        query_out,
        key_out,
        head_dim,
        num_heads,
        num_kv_heads,
    )
    LifeAI.qwen3_vl_qk_rope_scratch_bytes(scratch) == expected ||
        error("Qwen3-VL packed RoPE scratch bytes do not match the formula")
    return scratch
end

function LifeAI._qwen3_vl_device_qk_norm_scratch(
    prototype::CUDA.CuArray{BFloat16};
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    expected = LifeAI.qwen3_vl_qk_norm_scratch_bytes(;
        head_dim,
        num_heads,
        num_kv_heads,
    )
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    spatial = head_dim * num_heads
    n_f32 = 2 * spatial + num_heads + head_dim
    n_bf16 = 2 * spatial + head_dim * num_kv_heads
    f32 = similar(prototype, Float32, n_f32)
    bf16 = similar(prototype, BFloat16, n_bf16)
    offset = 0
    xf = _qwen3_carve(f32, (head_dim, num_heads, 1, 1), offset)
    offset += length(xf)
    mean_square = _qwen3_carve(f32, (1, num_heads, 1, 1), offset)
    offset += length(mean_square)
    norm_f32 = _qwen3_carve(f32, (head_dim, num_heads, 1, 1), offset)
    offset += length(norm_f32)
    scale_f32 = _qwen3_carve(f32, (head_dim, 1, 1, 1), offset)
    offset = 0
    normalized = _qwen3_carve(bf16, (head_dim, num_heads, 1, 1), offset)
    offset += length(normalized)
    query_out = _qwen3_carve(bf16, (head_dim, num_heads, 1, 1), offset)
    offset += length(query_out)
    key_out = _qwen3_carve(bf16, (head_dim, num_kv_heads, 1, 1), offset)
    scratch = LifeAI.Qwen3VLQKNormScratch(
        xf,
        mean_square,
        normalized,
        norm_f32,
        scale_f32,
        query_out,
        key_out,
        head_dim,
        num_heads,
        num_kv_heads,
    )
    LifeAI.qwen3_vl_qk_norm_scratch_bytes(scratch) == expected ||
        error("Qwen3-VL packed QK norm scratch bytes do not match the formula")
    return scratch
end

function LifeAI._qwen3_vl_device_hidden_norm_scratch(
    prototype::CUDA.CuArray{BFloat16};
    hidden_size::Integer,
)
    expected = LifeAI.qwen3_vl_hidden_norm_scratch_bytes(; hidden_size)
    hidden_size = Int(hidden_size)
    n_f32 = 3 * hidden_size + 1
    n_bf16 = 2 * hidden_size
    f32 = similar(prototype, Float32, n_f32)
    bf16 = similar(prototype, BFloat16, n_bf16)
    offset = 0
    xf = _qwen3_carve(f32, (hidden_size, 1, 1), offset)
    offset += length(xf)
    mean_square = _qwen3_carve(f32, (1, 1, 1), offset)
    offset += length(mean_square)
    norm_f32 = _qwen3_carve(f32, (hidden_size, 1, 1), offset)
    offset += length(norm_f32)
    scale_f32 = _qwen3_carve(f32, (hidden_size, 1, 1), offset)
    offset = 0
    normalized = _qwen3_carve(bf16, (hidden_size, 1, 1), offset)
    offset += length(normalized)
    output = _qwen3_carve(bf16, (hidden_size, 1, 1), offset)
    scratch = LifeAI.Qwen3VLHiddenNormScratch(
        xf,
        mean_square,
        normalized,
        norm_f32,
        scale_f32,
        output,
        hidden_size,
    )
    LifeAI.qwen3_vl_hidden_norm_scratch_bytes(scratch) == expected ||
        error("Qwen3-VL packed hidden norm scratch bytes do not match the formula")
    return scratch
end

function LifeAI._qwen3_vl_device_mlp_activation_scratch(
    prototype::CUDA.CuArray{BFloat16};
    intermediate_size::Integer,
)
    expected = LifeAI.qwen3_vl_mlp_activation_scratch_bytes(; intermediate_size)
    intermediate_size = Int(intermediate_size)
    n_f32 = 3 * intermediate_size
    n_bf16 = 2 * intermediate_size
    f32 = similar(prototype, Float32, n_f32)
    bf16 = similar(prototype, BFloat16, n_bf16)
    offset = 0
    gate_f32 = _qwen3_carve(f32, (intermediate_size, 1, 1), offset)
    offset += length(gate_f32)
    activated_f32 = _qwen3_carve(f32, (intermediate_size, 1, 1), offset)
    offset += length(activated_f32)
    up_f32 = _qwen3_carve(f32, (intermediate_size, 1, 1), offset)
    offset = 0
    activated = _qwen3_carve(bf16, (intermediate_size, 1, 1), offset)
    offset += length(activated)
    product = _qwen3_carve(bf16, (intermediate_size, 1, 1), offset)
    scratch = LifeAI.Qwen3VLMlpActivationScratch(
        gate_f32,
        activated,
        activated_f32,
        up_f32,
        product,
        intermediate_size,
    )
    LifeAI.qwen3_vl_mlp_activation_scratch_bytes(scratch) == expected ||
        error("Qwen3-VL packed MLP activation scratch bytes do not match the formula")
    return scratch
end

function LifeAI._qwen3_vl_device_hidden_residual_scratch(
    prototype::CUDA.CuArray{BFloat16};
    hidden_size::Integer,
)
    expected = LifeAI.qwen3_vl_hidden_residual_scratch_bytes(; hidden_size)
    hidden_size = Int(hidden_size)
    n = 2 * hidden_size
    f32 = similar(prototype, Float32, n)
    bf16 = similar(prototype, BFloat16, n)
    offset = 0
    residual_f32 = _qwen3_carve(f32, (hidden_size, 1, 1), offset)
    offset += length(residual_f32)
    projected_f32 = _qwen3_carve(f32, (hidden_size, 1, 1), offset)
    offset = 0
    projected = _qwen3_carve(bf16, (hidden_size, 1, 1), offset)
    offset += length(projected)
    output = _qwen3_carve(bf16, (hidden_size, 1, 1), offset)
    scratch = LifeAI.Qwen3VLHiddenResidualScratch(
        projected,
        residual_f32,
        projected_f32,
        output,
        hidden_size,
    )
    LifeAI.qwen3_vl_hidden_residual_scratch_bytes(scratch) == expected ||
        error("Qwen3-VL packed hidden residual scratch bytes do not match the formula")
    return scratch
end

function LifeAI._qwen3_vl_device_bf16_tails!(
    cache::LifeAI.Qwen3VLStaticKVCache,
    spec::LifeAI.Qwen3VLTextSpec,
    prototype::CUDA.CuArray{BFloat16},
)::Bool
    cache.token_embedding_scratch isa LifeAI.Qwen3VLTokenEmbeddingScratch &&
        return true
    hidden_size = spec.hidden_size
    head_dim = spec.head_dim
    num_heads = spec.num_attention_heads
    num_kv_heads = spec.num_key_value_heads
    intermediate_size = spec.intermediate_size
    vocab_size = spec.vocab_size
    query_n = head_dim * num_heads
    kv_n = head_dim * num_kv_heads
    half = head_dim ÷ 2
    n = query_n + 2 * kv_n + 2 * intermediate_size + 2 * half +
        hidden_size + vocab_size
    parent = similar(prototype, BFloat16, n)
    offset = 0
    query = _qwen3_carve(parent, (query_n, 1, 1), offset)
    offset += length(query)
    key = _qwen3_carve(parent, (kv_n, 1, 1), offset)
    offset += length(key)
    value = _qwen3_carve(parent, (kv_n, 1, 1), offset)
    offset += length(value)
    gate = _qwen3_carve(parent, (intermediate_size, 1, 1), offset)
    offset += length(gate)
    up = _qwen3_carve(parent, (intermediate_size, 1, 1), offset)
    offset += length(up)
    cos_values = _qwen3_carve(parent, (half, 1, 1), offset)
    offset += length(cos_values)
    sin_values = _qwen3_carve(parent, (half, 1, 1), offset)
    offset += length(sin_values)
    embedded = _qwen3_carve(parent, (hidden_size, 1, 1), offset)
    offset += length(embedded)
    logits = _qwen3_carve(parent, (vocab_size, 1, 1), offset)
    qkv = LifeAI.Qwen3VLQKVScratch(
        query, key, value, hidden_size, head_dim, num_heads, num_kv_heads,
    )
    gate_up = LifeAI.Qwen3VLGateUpScratch(
        gate, up, hidden_size, intermediate_size,
    )
    mrope = LifeAI.Qwen3VLMropeScratch(cos_values, sin_values, half)
    embedding = LifeAI.Qwen3VLTokenEmbeddingScratch(embedded, hidden_size)
    tied = LifeAI.Qwen3VLTiedLogitsScratch(logits, hidden_size, vocab_size)
    LifeAI.qwen3_vl_qkv_scratch_bytes(qkv) ==
        LifeAI.qwen3_vl_qkv_scratch_bytes(; head_dim, num_heads, num_kv_heads) ||
        error("Qwen3-VL packed QKV scratch bytes do not match the formula")
    LifeAI.qwen3_vl_gate_up_scratch_bytes(gate_up) ==
        LifeAI.qwen3_vl_gate_up_scratch_bytes(; intermediate_size) ||
        error("Qwen3-VL packed gate/up scratch bytes do not match the formula")
    LifeAI.qwen3_vl_mrope_scratch_bytes(mrope) ==
        LifeAI.qwen3_vl_mrope_scratch_bytes(; head_dim) ||
        error("Qwen3-VL packed mRoPE scratch bytes do not match the formula")
    LifeAI.qwen3_vl_token_embedding_scratch_bytes(embedding) ==
        LifeAI.qwen3_vl_token_embedding_scratch_bytes(; hidden_size) ||
        error("Qwen3-VL packed token embedding scratch bytes do not match the formula")
    LifeAI.qwen3_vl_tied_logits_scratch_bytes(tied) ==
        LifeAI.qwen3_vl_tied_logits_scratch_bytes(; vocab_size) ||
        error("Qwen3-VL packed tied logits scratch bytes do not match the formula")
    cache.qkv_scratch = qkv
    cache.gate_up_scratch = gate_up
    cache.mrope_scratch = mrope
    cache.token_embedding_scratch = embedding
    cache.tied_logits_scratch = tied
    return true
end

function _qwen3_reuse_place!(
    cache::LifeAI.Qwen3VLStaticKVCache,
    prototype::CUDA.CuArray{BFloat16},
    shells::_Qwen3ReuseShells,
)
    spec = shells.pack.spec
    hidden_size = spec.hidden_size
    head_dim = spec.head_dim
    num_heads = spec.num_attention_heads
    num_kv_heads = spec.num_key_value_heads
    intermediate_size = spec.intermediate_size
    vocab_size = spec.vocab_size
    capacity = cache.capacity
    groups = num_heads ÷ num_kv_heads
    half = head_dim ÷ 2
    rope_spatial = half * num_heads
    qk_spatial = head_dim * num_heads
    query_n = head_dim * num_heads
    kv_n = head_dim * num_kv_heads
    n_bf16 = (head_dim * groups * num_kv_heads) * 2 +
        (head_dim * capacity * num_kv_heads) * 2 +
        (groups * capacity * num_kv_heads) * 3 +
        (capacity * groups * num_kv_heads) +
        (head_dim * groups * num_kv_heads) +
        head_dim * num_heads +
        rope_spatial + head_dim * num_heads + kv_n +
        2 * qk_spatial + kv_n +
        2 * hidden_size +
        2 * intermediate_size +
        2 * hidden_size +
        query_n + 2 * kv_n + 2 * intermediate_size + 2 * half +
        hidden_size + vocab_size
    n_f32 = (groups * capacity * num_kv_heads) * 2 + groups * num_kv_heads +
        7 * rope_spatial + 2 * half +
        2 * qk_spatial + num_heads + head_dim +
        3 * hidden_size + 1 +
        3 * intermediate_size +
        2 * hidden_size
    bf16 = _qwen3_fit_parent(shells.bf16, prototype, BFloat16, n_bf16)
    f32 = _qwen3_fit_parent(shells.f32, prototype, Float32, n_f32)
    _qwen3_place_workspace!(
        cache, bf16, f32, shells,
        hidden_size, head_dim, num_heads, num_kv_heads,
        intermediate_size, vocab_size, capacity, groups, half,
        query_n, kv_n, n_bf16, n_f32,
    )
    return nothing
end

function _qwen3_build_workspace!(
    cache::LifeAI.Qwen3VLStaticKVCache,
    spec::LifeAI.Qwen3VLTextSpec,
    prototype::CUDA.CuArray{BFloat16},
)::Bool
    # A reused request already installed Ready. Profiling that step must
    # not allocate another parent pair before the stage runner starts.
    cache.decode_pack isa LifeAI._Qwen3VLDecodeReady && return true
    cache.attention_scratch isa LifeAI.Qwen3VLDecodeAttentionScratch &&
        return true
    hidden_size = spec.hidden_size
    head_dim = spec.head_dim
    num_heads = spec.num_attention_heads
    num_kv_heads = spec.num_key_value_heads
    intermediate_size = spec.intermediate_size
    vocab_size = spec.vocab_size
    capacity = cache.capacity
    groups = num_heads ÷ num_kv_heads
    half = head_dim ÷ 2
    rope_spatial = half * num_heads
    qk_spatial = head_dim * num_heads
    query_n = head_dim * num_heads
    kv_n = head_dim * num_kv_heads
    n_bf16 = (head_dim * groups * num_kv_heads) * 2 +
        (head_dim * capacity * num_kv_heads) * 2 +
        (groups * capacity * num_kv_heads) * 3 +
        (capacity * groups * num_kv_heads) +
        (head_dim * groups * num_kv_heads) +
        head_dim * num_heads +
        rope_spatial + head_dim * num_heads + kv_n +
        2 * qk_spatial + kv_n +
        2 * hidden_size +
        2 * intermediate_size +
        2 * hidden_size +
        query_n + 2 * kv_n + 2 * intermediate_size + 2 * half +
        hidden_size + vocab_size
    n_f32 = (groups * capacity * num_kv_heads) * 2 + groups * num_kv_heads +
        7 * rope_spatial + 2 * half +
        2 * qk_spatial + num_heads + head_dim +
        3 * hidden_size + 1 +
        3 * intermediate_size +
        2 * hidden_size
    if isassigned(_QWEN3_SHELLS)
        shells = _QWEN3_SHELLS[]
        pack = cache.decode_pack
        if shells isa _Qwen3ReuseShells &&
            shells.owner.value === cache &&
            pack isa LifeAI._Qwen3VLDecodePack &&
            LifeAI._qwen3_vl_decode_pack_matches(shells.pack, pack)
            _qwen3_reuse_place!(cache, prototype, shells)
            return true
        end
    end
    bf16 = similar(prototype, BFloat16, n_bf16)
    f32 = similar(prototype, Float32, n_f32)
    offset = 0
    query5 = _qwen3_carve_shared(bf16, (1, head_dim, groups, num_kv_heads, 1), offset)
    offset += length(query5)
    query3 = _qwen3_carve_shared(bf16, (1, groups, head_dim, num_kv_heads, 1), offset)
    offset += length(query3)
    keys = _qwen3_carve_shared(bf16, (head_dim, capacity, num_kv_heads), offset)
    offset += length(keys)
    values = _qwen3_carve_shared(bf16, (head_dim, capacity, num_kv_heads), offset)
    offset += length(values)
    scores = _qwen3_carve_shared(bf16, (groups, capacity, num_kv_heads), offset)
    offset += length(scores)
    scaled = _qwen3_carve_shared(bf16, (groups, capacity, num_kv_heads), offset)
    offset += length(scaled)
    weights = _qwen3_carve_shared(bf16, (groups, capacity, num_kv_heads), offset)
    offset += length(weights)
    weight_perm = _qwen3_carve_shared(bf16, (capacity, groups, num_kv_heads), offset)
    offset += length(weight_perm)
    context = _qwen3_carve_shared(bf16, (head_dim, groups, num_kv_heads), offset)
    offset += length(context)
    attn_output = _qwen3_carve_shared(bf16, (head_dim, num_heads, 1, 1), offset)
    offset += length(attn_output)
    rope_tmp_bf16 = _qwen3_carve_shared(bf16, (half, num_heads, 1, 1), offset)
    offset += length(rope_tmp_bf16)
    rope_query = _qwen3_carve_shared(bf16, (head_dim, num_heads, 1, 1), offset)
    offset += length(rope_query)
    rope_key = _qwen3_carve_shared(bf16, (head_dim, num_kv_heads, 1, 1), offset)
    offset += length(rope_key)
    qk_normalized = _qwen3_carve_shared(bf16, (head_dim, num_heads, 1, 1), offset)
    offset += length(qk_normalized)
    qk_query = _qwen3_carve_shared(bf16, (head_dim, num_heads, 1, 1), offset)
    offset += length(qk_query)
    qk_key = _qwen3_carve_shared(bf16, (head_dim, num_kv_heads, 1, 1), offset)
    offset += length(qk_key)
    hidden_normalized = _qwen3_carve_shared(bf16, (hidden_size, 1, 1), offset)
    offset += length(hidden_normalized)
    hidden_output = _qwen3_carve_shared(bf16, (hidden_size, 1, 1), offset)
    offset += length(hidden_output)
    activated = _qwen3_carve_shared(bf16, (intermediate_size, 1, 1), offset)
    offset += length(activated)
    product = _qwen3_carve_shared(bf16, (intermediate_size, 1, 1), offset)
    offset += length(product)
    projected = _qwen3_carve_shared(bf16, (hidden_size, 1, 1), offset)
    offset += length(projected)
    residual_output = _qwen3_carve_shared(bf16, (hidden_size, 1, 1), offset)
    offset += length(residual_output)
    qkv_query = _qwen3_carve_shared(bf16, (query_n, 1, 1), offset)
    offset += length(qkv_query)
    qkv_key = _qwen3_carve_shared(bf16, (kv_n, 1, 1), offset)
    offset += length(qkv_key)
    qkv_value = _qwen3_carve_shared(bf16, (kv_n, 1, 1), offset)
    offset += length(qkv_value)
    gate = _qwen3_carve_shared(bf16, (intermediate_size, 1, 1), offset)
    offset += length(gate)
    up = _qwen3_carve_shared(bf16, (intermediate_size, 1, 1), offset)
    offset += length(up)
    cos_values = _qwen3_carve_shared(bf16, (half, 1, 1), offset)
    offset += length(cos_values)
    sin_values = _qwen3_carve_shared(bf16, (half, 1, 1), offset)
    offset += length(sin_values)
    embedded = _qwen3_carve_shared(bf16, (hidden_size, 1, 1), offset)
    offset += length(embedded)
    logits = _qwen3_carve_shared(bf16, (vocab_size, 1, 1), offset)
    offset += length(logits)
    offset == n_bf16 || error("Qwen3-VL decode workspace BF16 span does not match")
    offset = 0
    scores_f32 = _qwen3_carve_shared(f32, (groups, capacity, num_kv_heads), offset)
    offset += length(scores_f32)
    maxima = _qwen3_carve_shared(f32, (groups, 1, num_kv_heads), offset)
    offset += length(maxima)
    exponents = _qwen3_carve_shared(f32, (groups, capacity, num_kv_heads), offset)
    offset += length(exponents)
    x1 = _qwen3_carve_shared(f32, (half, num_heads, 1, 1), offset)
    offset += length(x1)
    x2 = _qwen3_carve_shared(f32, (half, num_heads, 1, 1), offset)
    offset += length(x2)
    cos_f32 = _qwen3_carve_shared(f32, (half, 1, 1, 1), offset)
    offset += length(cos_f32)
    sin_f32 = _qwen3_carve_shared(f32, (half, 1, 1, 1), offset)
    offset += length(sin_f32)
    tmp_f32 = _qwen3_carve_shared(f32, (half, num_heads, 1, 1), offset)
    offset += length(tmp_f32)
    term_a = _qwen3_carve_shared(f32, (half, num_heads, 1, 1), offset)
    offset += length(term_a)
    term_b = _qwen3_carve_shared(f32, (half, num_heads, 1, 1), offset)
    offset += length(term_b)
    upper = _qwen3_carve_shared(f32, (half, num_heads, 1, 1), offset)
    offset += length(upper)
    lower = _qwen3_carve_shared(f32, (half, num_heads, 1, 1), offset)
    offset += length(lower)
    qk_xf = _qwen3_carve_shared(f32, (head_dim, num_heads, 1, 1), offset)
    offset += length(qk_xf)
    qk_mean = _qwen3_carve_shared(f32, (1, num_heads, 1, 1), offset)
    offset += length(qk_mean)
    qk_norm_f32 = _qwen3_carve_shared(f32, (head_dim, num_heads, 1, 1), offset)
    offset += length(qk_norm_f32)
    qk_scale = _qwen3_carve_shared(f32, (head_dim, 1, 1, 1), offset)
    offset += length(qk_scale)
    hidden_xf = _qwen3_carve_shared(f32, (hidden_size, 1, 1), offset)
    offset += length(hidden_xf)
    hidden_mean = _qwen3_carve_shared(f32, (1, 1, 1), offset)
    offset += length(hidden_mean)
    hidden_norm_f32 = _qwen3_carve_shared(f32, (hidden_size, 1, 1), offset)
    offset += length(hidden_norm_f32)
    hidden_scale = _qwen3_carve_shared(f32, (hidden_size, 1, 1), offset)
    offset += length(hidden_scale)
    gate_f32 = _qwen3_carve_shared(f32, (intermediate_size, 1, 1), offset)
    offset += length(gate_f32)
    activated_f32 = _qwen3_carve_shared(f32, (intermediate_size, 1, 1), offset)
    offset += length(activated_f32)
    up_f32 = _qwen3_carve_shared(f32, (intermediate_size, 1, 1), offset)
    offset += length(up_f32)
    residual_f32 = _qwen3_carve_shared(f32, (hidden_size, 1, 1), offset)
    offset += length(residual_f32)
    projected_f32 = _qwen3_carve_shared(f32, (hidden_size, 1, 1), offset)
    offset += length(projected_f32)
    offset == n_f32 || error("Qwen3-VL decode workspace Float32 span does not match")
    attention = LifeAI.Qwen3VLDecodeAttentionScratch(
        query5, query3, keys, values, scores, scaled, scores_f32, maxima,
        exponents, weights, weight_perm, context, attn_output,
        head_dim, num_heads, num_kv_heads, groups, capacity,
    )
    rope = LifeAI.Qwen3VLQKRopeScratch(
        x1, x2, cos_f32, sin_f32, tmp_f32, rope_tmp_bf16, term_a, term_b,
        upper, lower, rope_query, rope_key, head_dim, num_heads, num_kv_heads,
    )
    qk_norm = LifeAI.Qwen3VLQKNormScratch(
        qk_xf, qk_mean, qk_normalized, qk_norm_f32, qk_scale, qk_query, qk_key,
        head_dim, num_heads, num_kv_heads,
    )
    hidden = LifeAI.Qwen3VLHiddenNormScratch(
        hidden_xf, hidden_mean, hidden_normalized, hidden_norm_f32, hidden_scale,
        hidden_output, hidden_size,
    )
    mlp = LifeAI.Qwen3VLMlpActivationScratch(
        gate_f32, activated, activated_f32, up_f32, product, intermediate_size,
    )
    residual = LifeAI.Qwen3VLHiddenResidualScratch(
        projected, residual_f32, projected_f32, residual_output, hidden_size,
    )
    qkv = LifeAI.Qwen3VLQKVScratch(
        qkv_query, qkv_key, qkv_value, hidden_size, head_dim, num_heads, num_kv_heads,
    )
    gate_up = LifeAI.Qwen3VLGateUpScratch(
        gate, up, hidden_size, intermediate_size,
    )
    mrope = LifeAI.Qwen3VLMropeScratch(cos_values, sin_values, half)
    embedding = LifeAI.Qwen3VLTokenEmbeddingScratch(embedded, hidden_size)
    tied = LifeAI.Qwen3VLTiedLogitsScratch(logits, hidden_size, vocab_size)
    LifeAI.qwen3_vl_decode_attention_scratch_bytes(attention) ==
        LifeAI.qwen3_vl_decode_attention_scratch_bytes(;
            head_dim, num_attention_heads=num_heads, num_kv_heads, capacity,
        ) || error("Qwen3-VL workspace attention bytes do not match the formula")
    LifeAI.qwen3_vl_qk_rope_scratch_bytes(rope) ==
        LifeAI.qwen3_vl_qk_rope_scratch_bytes(;
            head_dim, num_heads, num_kv_heads,
        ) || error("Qwen3-VL workspace RoPE bytes do not match the formula")
    LifeAI.qwen3_vl_qk_norm_scratch_bytes(qk_norm) ==
        LifeAI.qwen3_vl_qk_norm_scratch_bytes(;
            head_dim, num_heads, num_kv_heads,
        ) || error("Qwen3-VL workspace QK norm bytes do not match the formula")
    LifeAI.qwen3_vl_hidden_norm_scratch_bytes(hidden) ==
        LifeAI.qwen3_vl_hidden_norm_scratch_bytes(; hidden_size) ||
        error("Qwen3-VL workspace hidden norm bytes do not match the formula")
    LifeAI.qwen3_vl_mlp_activation_scratch_bytes(mlp) ==
        LifeAI.qwen3_vl_mlp_activation_scratch_bytes(; intermediate_size) ||
        error("Qwen3-VL workspace MLP bytes do not match the formula")
    LifeAI.qwen3_vl_hidden_residual_scratch_bytes(residual) ==
        LifeAI.qwen3_vl_hidden_residual_scratch_bytes(; hidden_size) ||
        error("Qwen3-VL workspace residual bytes do not match the formula")
    LifeAI.qwen3_vl_qkv_scratch_bytes(qkv) ==
        LifeAI.qwen3_vl_qkv_scratch_bytes(; head_dim, num_heads, num_kv_heads) ||
        error("Qwen3-VL workspace QKV bytes do not match the formula")
    LifeAI.qwen3_vl_gate_up_scratch_bytes(gate_up) ==
        LifeAI.qwen3_vl_gate_up_scratch_bytes(; intermediate_size) ||
        error("Qwen3-VL workspace gate/up bytes do not match the formula")
    LifeAI.qwen3_vl_mrope_scratch_bytes(mrope) ==
        LifeAI.qwen3_vl_mrope_scratch_bytes(; head_dim) ||
        error("Qwen3-VL workspace mRoPE bytes do not match the formula")
    LifeAI.qwen3_vl_token_embedding_scratch_bytes(embedding) ==
        LifeAI.qwen3_vl_token_embedding_scratch_bytes(; hidden_size) ||
        error("Qwen3-VL workspace embedding bytes do not match the formula")
    LifeAI.qwen3_vl_tied_logits_scratch_bytes(tied) ==
        LifeAI.qwen3_vl_tied_logits_scratch_bytes(; vocab_size) ||
        error("Qwen3-VL workspace tied logits bytes do not match the formula")
    cache.attention_scratch = attention
    cache.qk_rope_scratch = rope
    cache.qk_norm_scratch = qk_norm
    cache.hidden_norm_scratch = hidden
    cache.mlp_activation_scratch = mlp
    cache.hidden_residual_scratch = residual
    cache.qkv_scratch = qkv
    cache.gate_up_scratch = gate_up
    cache.mrope_scratch = mrope
    cache.token_embedding_scratch = embedding
    cache.tied_logits_scratch = tied
    # Children share these DataRefs. The parents have to stay reachable
    # for as long as the slices do.
    cache.decode_workspace = (bf16, f32)
    shells = _qwen3_capture_shells!(
        cache, attention, rope, qk_norm, hidden, mlp, residual, qkv, gate_up,
        mrope, embedding, tied, bf16, f32,
    )
    _QWEN3_SHELLS[] = shells
    # Compile the concrete retarget while this build is still the warmup.
    _qwen3_place_workspace!(
        cache, bf16, f32, shells,
        hidden_size, head_dim, num_heads, num_kv_heads,
        intermediate_size, vocab_size, capacity, groups, half,
        query_n, kv_n, n_bf16, n_f32,
    )
    # Compile the fit check while this build is still the warmup.
    _qwen3_fit_parent(bf16, prototype, BFloat16, n_bf16)
    _qwen3_fit_parent(f32, prototype, Float32, n_f32)
    return true
end

function LifeAI._qwen3_vl_after_impl!(parameters, token, cache::LifeAI.Qwen3VLStaticKVCache)
    # The profiled warmup compiles the untyped impl after this method. Refresh
    # it here so the next sample's first decode does not pay that compile.
    precompile(
        LifeAI.hf_qwen3_vl_text_decode_step_static,
        (typeof(parameters), typeof(token), typeof(cache)),
    )
    ready = cache.decode_pack
    if ready isa LifeAI._Qwen3VLDecodeReady
        precompile(
            LifeAI._qwen3_vl_text_decode_ready!,
            (typeof(ready), typeof(token), typeof(cache)),
        )
    end
    if isassigned(_QWEN3_READY)
        reused = _QWEN3_READY[]
        if reused isa LifeAI._Qwen3VLDecodeReady
            precompile(
                _qwen3_reuse_ready!,
                (typeof(reused), typeof(cache), Int),
            )
        end
    end
    if isassigned(_QWEN3_SHELLS)
        shells = _QWEN3_SHELLS[]
        precompile(
            _qwen3_fit_parent,
            (typeof(shells.bf16), typeof(parameters.embedding), Type{BFloat16}, Int),
        )
        precompile(
            _qwen3_fit_parent,
            (typeof(shells.f32), typeof(parameters.embedding), Type{Float32}, Int),
        )
        if shells isa _Qwen3ReuseShells
            precompile(
                _qwen3_reuse_place!,
                (typeof(cache), typeof(parameters.embedding), typeof(shells)),
            )
            precompile(
                LifeAI._qwen3_vl_device_decode_workspace!,
                (typeof(cache), typeof(parameters.spec), typeof(parameters.embedding)),
            )
        end
    end
    return nothing
end

function LifeAI._qwen3_vl_device_decode_workspace!(
    cache::LifeAI.Qwen3VLStaticKVCache,
    spec::LifeAI.Qwen3VLTextSpec,
    prototype::CUDA.CuArray{BFloat16},
)::Bool
    cache.decode_pack isa LifeAI._Qwen3VLDecodeReady && return true
    cache.attention_scratch isa LifeAI.Qwen3VLDecodeAttentionScratch &&
        return true
    if isassigned(_QWEN3_SHELLS)
        shells = _QWEN3_SHELLS[]
        pack = cache.decode_pack
        if shells isa _Qwen3ReuseShells &&
            shells.owner.value === cache &&
            pack isa LifeAI._Qwen3VLDecodePack &&
            LifeAI._qwen3_vl_decode_pack_matches(shells.pack, pack)
            _qwen3_reuse_place!(cache, prototype, shells)
            return true
        end
    end
    return _qwen3_build_workspace!(cache, spec, prototype)
end

const _Qwen3Mem = parentmodule(CUDA.CuArray).DeviceMemory
# Concrete BF16 Ready. An `isa` of this type keeps the steady call static.
const _Qwen3BF16Ready = LifeAI._Qwen3VLDecodeReady{LifeAI._Qwen3VLDecodePack{NTuple{28, @NamedTuple{norm1::CUDA.CuArray{BFloat16, 1, _Qwen3Mem}, q_weight::CUDA.CuArray{BFloat16, 2, _Qwen3Mem}, k_weight::CUDA.CuArray{BFloat16, 2, _Qwen3Mem}, v_weight::CUDA.CuArray{BFloat16, 2, _Qwen3Mem}, o_weight::CUDA.CuArray{BFloat16, 2, _Qwen3Mem}, q_norm::CUDA.CuArray{BFloat16, 1, _Qwen3Mem}, k_norm::CUDA.CuArray{BFloat16, 1, _Qwen3Mem}, norm2::CUDA.CuArray{BFloat16, 1, _Qwen3Mem}, gate_weight::CUDA.CuArray{BFloat16, 2, _Qwen3Mem}, up_weight::CUDA.CuArray{BFloat16, 2, _Qwen3Mem}, down_weight::CUDA.CuArray{BFloat16, 2, _Qwen3Mem}}}, CUDA.CuArray{BFloat16, 2, _Qwen3Mem}, CUDA.CuArray{BFloat16, 1, _Qwen3Mem}, LifeAI.Qwen3VLTextSpec}, LifeAI.Qwen3VLTokenEmbeddingScratch{CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}, LifeAI.Qwen3VLMropeScratch{CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}, LifeAI.Qwen3VLDecodeAttentionScratch{CUDA.CuArray{BFloat16, 5, _Qwen3Mem}, CUDA.CuArray{BFloat16, 5, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 4, _Qwen3Mem}}, LifeAI.Qwen3VLMlpActivationScratch{CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}, LifeAI.Qwen3VLQKRopeScratch{CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{BFloat16, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{BFloat16, 4, _Qwen3Mem}, CUDA.CuArray{BFloat16, 4, _Qwen3Mem}}, LifeAI.Qwen3VLQKNormScratch{CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{BFloat16, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{Float32, 4, _Qwen3Mem}, CUDA.CuArray{BFloat16, 4, _Qwen3Mem}, CUDA.CuArray{BFloat16, 4, _Qwen3Mem}}, LifeAI.Qwen3VLHiddenNormScratch{CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}, LifeAI.Qwen3VLGateUpScratch{CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}, LifeAI.Qwen3VLHiddenResidualScratch{CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{Float32, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}, LifeAI.Qwen3VLTiedLogitsScratch{CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}, LifeAI.Qwen3VLQKVScratch{CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}, CUDA.CuArray{BFloat16, 3, _Qwen3Mem}}}

function _qwen3_cuda_bf16_to_f32_kernel!(dest, src, n)
    index = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
    index > n && return
    @inbounds dest[index] = 1.0f0 * Float32(src[index])
    return
end

function _qwen3_contiguous(x::CUDA.CuArray)
    ndims(x) == 0 && return true
    stride(x, 1) == 1 || return false
    step = 1
    for dim in 1:ndims(x)
        stride(x, dim) == step || return false
        step *= size(x, dim)
    end
    return true
end

function _qwen3_linear_device(xs::CUDA.CuArray{T}) where {T}
    core = parentmodule(CUDA.CuArray)
    ptr = reinterpret(core.LLVMPtr{T,core.AS.Global}, _qwen3_raw_ptr(xs))
    n = length(xs)
    return core.CuDeviceArray{T,1,core.AS.Global}(ptr, (n,))
end

const _Qwen3F32Linear = _Qwen3SoftCore.CuDeviceArray{Float32,1,1}
const _Qwen3BF16Linear = _Qwen3SoftCore.CuDeviceArray{BFloat16,1,1}

mutable struct _Qwen3PromoteRaw
    fun::CUDA.CuFunction
    params::Vector{Ptr{Cvoid}}
    state::_Qwen3Param{_Qwen3SoftCore.KernelState}
    dest::_Qwen3Param{_Qwen3F32Linear}
    src::_Qwen3Param{_Qwen3BF16Linear}
    n::_Qwen3Param{Int64}
    stream::_Qwen3SoftCore.CUstream
end

const _QWEN3_PROMOTE_RAW = Ref{_Qwen3PromoteRaw}()

function _qwen3_promote_raw(compiled, dest, src, n)
    state = _Qwen3Param(_Qwen3SoftCore.KernelState(
        compiled.state.exception_info, UInt32(0),
    ))
    dest_box = _Qwen3Param(_qwen3_linear_device(dest))
    src_box = _Qwen3Param(_qwen3_linear_device(src))
    n_box = _Qwen3Param(Int64(n))
    params = Ptr{Cvoid}[
        Base.unsafe_convert(Ptr{Cvoid}, state),
        Base.unsafe_convert(Ptr{Cvoid}, dest_box),
        Base.unsafe_convert(Ptr{Cvoid}, src_box),
        Base.unsafe_convert(Ptr{Cvoid}, n_box),
    ]
    return _Qwen3PromoteRaw(
        compiled.fun,
        params,
        state,
        dest_box,
        src_box,
        n_box,
        CUDA.stream().handle,
    )
end

function _qwen3_promote_launch!(
    dest::CUDA.CuArray{Float32},
    src::CUDA.CuArray{BFloat16},
)
    n = length(src)
    n == length(dest) || return false
    n == 0 && return true
    threads = 128
    blocks = cld(n, threads)
    raw = if isassigned(_QWEN3_PROMOTE_RAW)
        _QWEN3_PROMOTE_RAW[]
    else
        compiled = CUDA.@cuda launch=false _qwen3_cuda_bf16_to_f32_kernel!(
            reshape(dest, n), reshape(src, n), n,
        )
        created = _qwen3_promote_raw(compiled, dest, src, n)
        _QWEN3_PROMOTE_RAW[] = created
        created
    end
    raw.dest.val = _qwen3_linear_device(dest)
    raw.src.val = _qwen3_linear_device(src)
    raw.n.val = Int64(n)
    raw.stream = CUDA.stream().handle
    _qwen3_cu_launch!(raw, threads, blocks)
    return true
end

function LifeAI._bf16a_f32(x::CUDA.CuArray{BFloat16})
    # `1.0f0 .* x` is an exact BF16-to-Float32 conversion. The reused kernel
    # writes `1.0f0 * Float32(x[i])` into a dense result.
    y = similar(x, Float32)
    if _qwen3_contiguous(x) && _qwen3_promote_launch!(y, x)
        return y
    end
    copyto!(y, x)
    return y
end

function LifeAI._qwen3_vl_bf16_ready_type(
    ::CUDA.CuArray{BFloat16},
)
    return _Qwen3BF16Ready
end

@inline function LifeAI.hf_qwen3_vl_text_decode_step_static(
    parameters::NamedTuple,
    token::Int,
    cache::LifeAI.Qwen3VLStaticKVCache,
)
    ready = cache.decode_pack
    if ready isa _Qwen3BF16Ready &&
            LifeAI._qwen3_vl_decode_pack_matches(ready.pack, parameters)
        return LifeAI._qwen3_vl_text_decode_ready!(ready, token, cache)
    end
    if ready isa LifeAI._Qwen3VLDecodeReady &&
            LifeAI._qwen3_vl_decode_pack_matches(ready.pack, parameters)
        return LifeAI._qwen3_vl_text_decode_ready!(ready, token, cache)
    end
    embedding = parameters.embedding
    if embedding isa CUDA.CuArray{BFloat16} && isassigned(_QWEN3_SHELLS) &&
        _QWEN3_SHELLS[].owner.value === cache &&
        ready isa LifeAI._Qwen3VLDecodePack &&
        LifeAI._qwen3_vl_decode_pack_matches(ready, parameters) &&
        LifeAI._qwen3_vl_decode_pack_matches(_QWEN3_SHELLS[].pack, parameters)
        LifeAI._qwen3_vl_device_decode_workspace!(
            cache, parameters.spec, embedding,
        )
        installed = cache.decode_pack
        if installed isa _Qwen3BF16Ready
            return LifeAI._qwen3_vl_text_decode_ready!(installed, token, cache)
        end
        installed isa LifeAI._Qwen3VLDecodeReady ||
            error("Qwen3-VL reuse did not install ready")
        return LifeAI._qwen3_vl_text_decode_ready!(installed, token, cache)
    end
    return LifeAI._qwen3_vl_text_decode_step_static_impl(
        parameters, token, cache, nothing,
    )
end

include("flint_nvfp4_native.jl")
include("flint_decode_kernels.jl")
include("flint_kv_decode.jl")
include("flint_long_attention.jl")

end # module LifeAICUDAExt
