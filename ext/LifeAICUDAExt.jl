module LifeAICUDAExt

using BFloat16s: BFloat16
import CUDA
import LifeAI
import LifeAI: qwen3_device_sparse_expert_dispatch

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
        padded_capacity=pair_count + (route_tile - 1) * num_experts,
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
            value = muladd(
                padded_output[model_index, padded_pair],
                Float32(routing_weights[slot, token_index]),
                value,
            )
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
    pair_count = experts_per_token * num_tokens
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
            value = muladd(
                routed_output[output_index, pair_index],
                Float32(routing_weights[slot, token_index]),
                value,
            )
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
    pair_count = experts_per_token * num_tokens
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
    pair_count = experts_per_token * num_tokens
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

mutable struct _Qwen3CUDAScatteredDispatchState
    pointer_plans::Dict{Int,Vector{_Qwen3CUDAScatteredPointerPlan}}
    workspaces::Dict{Tuple{Int,Int},_Qwen3CUDAScatteredWorkspace}
    workspace_last_used::Dict{Tuple{Int,Int},Int}
    clock::Int
    pointer_table_bytes::Int
    workspace_bytes::Int
end

_qwen3_cuda_scattered_state() = _Qwen3CUDAScatteredDispatchState(
    Dict{Int,Vector{_Qwen3CUDAScatteredPointerPlan}}(),
    Dict{Tuple{Int,Int},_Qwen3CUDAScatteredWorkspace}(),
    Dict{Tuple{Int,Int},Int}(),
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

function LifeAI._qwen3_scattered_expert_dispatch(
    tokens::CUDA.CuArray{Float32,2},
    expert_indices::CUDA.CuArray{I,2},
    routing_weights::CUDA.CuArray{R,2},
    scattered::LifeAI._Qwen3MoEScatteredExperts,
    existing_state,
    layer_index::Int,
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
    pair_count = experts_per_token * num_tokens
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

end # module LifeAICUDAExt
