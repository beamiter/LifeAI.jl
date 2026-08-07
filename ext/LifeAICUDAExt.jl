module LifeAICUDAExt

import CUDA
import LifeAI
import LifeAI: qwen3_device_sparse_expert_dispatch

# CUDA specialization for Qwen3 MoE compact dispatch. The portable
# route-major fallback gathers and replicates all three expert matrices for
# every route. These kernels instead index the original expert tensor in place:
# one kernel forms routed SwiGLU hidden states, one applies the down projection,
# and one combines the fixed top-k routes back into token order.

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

function qwen3_device_sparse_expert_dispatch(
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

end # module LifeAICUDAExt
