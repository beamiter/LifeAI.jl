module LifeAICUDAExt

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
