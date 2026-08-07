using Lux
using ConcreteStructs
using NNlib: batched_mul, gather, softmax, swish
using Random: AbstractRNG

"""
    SwiGLU(d_model, hidden_dim; use_bias=false)

Gated feed-forward layer:

```text
gate = W_gate * x
up   = W_up * x
y    = W_down * (SiLU(gate) .* up)
```

`hidden_dim` is the shared width of the gate and up projections.
"""
@concrete struct SwiGLU <: AbstractLuxContainerLayer{(
    :gate_proj,
    :up_proj,
    :down_proj,
)}
    gate_proj
    up_proj
    down_proj

    d_model::Int
    hidden_dim::Int
    use_bias::Bool
end

function SwiGLU(
    d_model::Int,
    hidden_dim::Int;
    use_bias::Bool=false,
)
    @assert d_model > 0 "`d_model` must be positive"
    @assert hidden_dim > 0 "`hidden_dim` must be positive"

    return SwiGLU(
        Dense(d_model, hidden_dim; use_bias),
        Dense(d_model, hidden_dim; use_bias),
        Dense(hidden_dim, d_model; use_bias),
        d_model,
        hidden_dim,
        use_bias,
    )
end

function (mlp::SwiGLU)(x, ps, st::NamedTuple)
    gate, st_gate = mlp.gate_proj(x, ps.gate_proj, st.gate_proj)
    up, st_up = mlp.up_proj(x, ps.up_proj, st.up_proj)
    hidden = swish.(gate) .* up
    y, st_down = mlp.down_proj(hidden, ps.down_proj, st.down_proj)

    return (
        y,
        (;
            gate_proj=st_gate,
            up_proj=st_up,
            down_proj=st_down,
        ),
    )
end

"""
    qwen3_topk_routing(router_logits, experts_per_token; normalize=true)

Apply the Qwen3 MoE routing contract to a `(num_experts, num_tokens)` logits
matrix. Softmax is evaluated in Float32, exactly `experts_per_token` routes are
kept for every token, and the selected probabilities are optionally
renormalized. The returned dense routing matrix contains zero for experts that
were not selected.

This is the correctness-first host implementation. It deliberately makes the
top-k boundary explicit; [`qwen3_device_topk_routing`](@ref) provides the
compact accelerator representation without changing the expert or checkpoint
parameter contract.
"""
function qwen3_topk_routing(
    router_logits::AbstractMatrix,
    experts_per_token::Int;
    normalize::Bool=true,
)
    num_experts, num_tokens = size(router_logits)
    num_experts > 0 || throw(ArgumentError("router must contain at least one expert"))
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "experts_per_token must be in 1:num_experts",
    ))

    probabilities = softmax(Float32.(router_logits); dims=1)
    routing = zeros(Float32, num_experts, num_tokens)
    for token in 1:num_tokens
        selected = partialsortperm(
            view(probabilities, :, token),
            1:experts_per_token;
            rev=true,
        )
        @inbounds routing[selected, token] .= probabilities[selected, token]
        if normalize
            @inbounds routing[selected, token] ./= sum(routing[selected, token])
        end
    end
    return routing
end

"""
    qwen3_device_topk_routing(router_logits, experts_per_token; normalize=true)

Device-compatible Qwen3 routing for an `(experts, tokens)` logits matrix.
Instead of constructing a dense routing table, this returns compact
`expert_indices` and `routing_weights`, both shaped `(experts_per_token,
tokens)`. Top-k extraction uses a fixed number of reductions and elementwise
selects, so Reactant/XLA can trace it without data-dependent host control flow.

Exact ties select the highest remaining expert index first. The tie rule is
explicit and deterministic; normal checkpoint logits are not expected to tie.
"""
function qwen3_device_topk_routing(
    router_logits::AbstractMatrix,
    experts_per_token::Int;
    normalize::Bool=true,
)
    num_experts, num_tokens = size(router_logits)
    num_experts > 0 || throw(ArgumentError("router must contain at least one expert"))
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "experts_per_token must be in 1:num_experts",
    ))

    probabilities = softmax(Float32.(router_logits); dims=1)
    # Expert count is a compile-time model constant. Building into `similar`
    # keeps this column on the same CUDA/XLA device as the probabilities.
    index_column = similar(probabilities, Int32, num_experts, 1)
    index_column .= reshape(
        Int32(1):Int32(num_experts),
        num_experts,
        1,
    )
    work = probabilities
    selected_indices = Vector{Any}(undef, experts_per_token)
    selected_weights = Vector{Any}(undef, experts_per_token)
    for slot in 1:experts_per_token
        values = maximum(work; dims=1)
        indices = maximum(
            ifelse.(work .== values, index_column, Int32(0));
            dims=1,
        )
        mask = index_column .== indices
        selected_indices[slot] = indices
        selected_weights[slot] = sum(ifelse.(mask, probabilities, 0.0f0); dims=1)
        slot == experts_per_token && break
        work = ifelse.(mask, -Inf32, work)
    end

    expert_indices = cat(selected_indices...; dims=1)
    routing_weights = cat(selected_weights...; dims=1)
    if normalize
        routing_weights = routing_weights ./ sum(routing_weights; dims=1)
    end
    return (; expert_indices, routing_weights)
end

"""
    Qwen3SparseMoE(d_model, hidden_dim, num_experts, experts_per_token;
                   normalize_routing=true)

Reference Qwen3 sparse Mixture-of-Experts feed-forward layer. Parameters use a
checkpoint-oriented layout:

```text
gate.weight                 (num_experts, d_model)
experts.gate_proj           (hidden_dim, d_model, num_experts)
experts.up_proj             (hidden_dim, d_model, num_experts)
experts.down_proj           (d_model, hidden_dim, num_experts)
```

The default host forward pass gathers the tokens assigned to each expert and
does not evaluate unselected token-expert pairs. [`qwen3_dense_expert_reference`](@ref)
retains the all-expert masked implementation as a correctness oracle. Device
arrays use compact routes: the portable fallback is route-major
gather/matmul/combine, while CUDA activates indexed kernels through a package
extension. Grouped-GEMM/tensor-core tuning remains a separate concern.
"""
struct Qwen3SparseMoE <: AbstractLuxLayer
    d_model::Int
    hidden_dim::Int
    num_experts::Int
    experts_per_token::Int
    normalize_routing::Bool
end

function Qwen3SparseMoE(
    d_model::Int,
    hidden_dim::Int,
    num_experts::Int,
    experts_per_token::Int;
    normalize_routing::Bool=true,
)
    d_model > 0 || throw(ArgumentError("d_model must be positive"))
    hidden_dim > 0 || throw(ArgumentError("hidden_dim must be positive"))
    num_experts > 0 || throw(ArgumentError("num_experts must be positive"))
    1 <= experts_per_token <= num_experts || throw(ArgumentError(
        "experts_per_token must be in 1:num_experts",
    ))
    return Qwen3SparseMoE(
        d_model,
        hidden_dim,
        num_experts,
        experts_per_token,
        normalize_routing,
    )
end

function _stack_dense_weights(
    rng::AbstractRNG,
    input_dim::Int,
    output_dim::Int,
    count::Int,
)
    weights = ntuple(count) do _
        parameters = LuxCore.initialparameters(
            rng,
            Dense(input_dim, output_dim; use_bias=false),
        )
        return reshape(parameters.weight, output_dim, input_dim, 1)
    end
    return cat(weights...; dims=3)
end

function LuxCore.initialparameters(rng::AbstractRNG, moe::Qwen3SparseMoE)
    gate = LuxCore.initialparameters(
        rng,
        Dense(moe.d_model, moe.num_experts; use_bias=false),
    )
    experts = (;
        gate_proj=_stack_dense_weights(
            rng,
            moe.d_model,
            moe.hidden_dim,
            moe.num_experts,
        ),
        up_proj=_stack_dense_weights(
            rng,
            moe.d_model,
            moe.hidden_dim,
            moe.num_experts,
        ),
        down_proj=_stack_dense_weights(
            rng,
            moe.hidden_dim,
            moe.d_model,
            moe.num_experts,
        ),
    )
    return (; gate, experts)
end

LuxCore.initialstates(::AbstractRNG, ::Qwen3SparseMoE) = (;)

function LuxCore.parameterlength(moe::Qwen3SparseMoE)
    router = moe.num_experts * moe.d_model
    experts = moe.num_experts * 3 * moe.d_model * moe.hidden_dim
    return router + experts
end

"""Execution counts reported by [`qwen3_sparse_expert_dispatch`](@ref)."""
struct Qwen3MoEDispatchStats
    token_count::Int
    expert_count::Int
    active_expert_count::Int
    routed_token_expert_pairs::Int
    dense_token_expert_pairs::Int
    expert_token_counts::Vector{Int}
end

function _validate_qwen3_expert_parameters(tokens, num_experts, expert_parameters)
    ndims(tokens) == 2 || throw(DimensionMismatch(
        "expert dispatch tokens must have shape (d_model, num_tokens)",
    ))
    d_model, num_tokens = size(tokens)
    gate_proj = expert_parameters.gate_proj
    up_proj = expert_parameters.up_proj
    down_proj = expert_parameters.down_proj
    ndims(gate_proj) == ndims(up_proj) == ndims(down_proj) == 3 ||
        throw(DimensionMismatch("expert projections must be rank-three tensors"))
    size(gate_proj) == size(up_proj) || throw(DimensionMismatch(
        "expert gate and up projections must have matching shapes",
    ))
    hidden_dim = size(gate_proj, 1)
    size(gate_proj, 2) == d_model || throw(DimensionMismatch(
        "expert projection input width does not match token d_model",
    ))
    size(gate_proj, 3) == num_experts || throw(DimensionMismatch(
        "expert projection count does not match router expert count",
    ))
    size(down_proj) == (d_model, hidden_dim, num_experts) ||
        throw(DimensionMismatch("expert down projection shape is inconsistent"))
    return d_model, num_tokens, num_experts
end

function _validate_qwen3_expert_dispatch(tokens, routing, expert_parameters)
    ndims(routing) == 2 || throw(DimensionMismatch(
        "expert routing must have shape (num_experts, num_tokens)",
    ))
    num_experts, routed_tokens = size(routing)
    size(tokens, 2) == routed_tokens || throw(DimensionMismatch(
        "routing token count does not match expert input token count",
    ))
    return _validate_qwen3_expert_parameters(
        tokens,
        num_experts,
        expert_parameters,
    )
end

"""
    qwen3_dense_expert_reference(tokens, routing, expert_parameters)

Evaluate every expert for every token and mask the results with `routing`.
This intentionally inefficient host implementation is retained as an
independent oracle for sparse dispatch changes.
"""
function qwen3_dense_expert_reference(tokens, routing, expert_parameters)
    _validate_qwen3_expert_dispatch(tokens, routing, expert_parameters)
    output = similar(tokens)
    fill!(output, zero(eltype(output)))
    for expert in axes(routing, 1)
        gate = view(expert_parameters.gate_proj, :, :, expert) * tokens
        up = view(expert_parameters.up_proj, :, :, expert) * tokens
        hidden = swish.(gate) .* up
        expert_output = view(expert_parameters.down_proj, :, :, expert) * hidden
        weights = reshape(
            convert.(eltype(expert_output), routing[expert, :]),
            1,
            :,
        )
        output .+= expert_output .* weights
    end
    return output
end

"""
    qwen3_sparse_expert_dispatch(tokens, routing, expert_parameters)

Dispatch host tokens only to experts with non-zero routing weight. Returns the
combined output and [`Qwen3MoEDispatchStats`](@ref). The executed pair count is
the number of non-zero entries in `routing`; unlike the dense oracle, inactive
experts and unassigned tokens never enter an expert matrix multiplication.
"""
function qwen3_sparse_expert_dispatch(tokens, routing, expert_parameters)
    _, num_tokens, num_experts = _validate_qwen3_expert_dispatch(
        tokens,
        routing,
        expert_parameters,
    )
    routing isa Matrix || throw(ArgumentError(
        "host sparse dispatch requires a CPU Matrix routing table",
    ))
    tokens isa Matrix || throw(ArgumentError(
        "host sparse dispatch requires a CPU Matrix token table",
    ))
    all(parameter -> parameter isa Array, (
        expert_parameters.gate_proj,
        expert_parameters.up_proj,
        expert_parameters.down_proj,
    )) || throw(ArgumentError(
        "host sparse dispatch requires CPU expert projection arrays",
    ))

    output = similar(tokens)
    fill!(output, zero(eltype(output)))
    expert_token_counts = zeros(Int, num_experts)
    for expert in 1:num_experts
        token_indices = findall(!iszero, view(routing, expert, :))
        expert_token_counts[expert] = length(token_indices)
        isempty(token_indices) && continue

        expert_tokens = tokens[:, token_indices]
        gate = view(expert_parameters.gate_proj, :, :, expert) * expert_tokens
        up = view(expert_parameters.up_proj, :, :, expert) * expert_tokens
        hidden = swish.(gate) .* up
        expert_output = view(expert_parameters.down_proj, :, :, expert) * hidden
        weights = reshape(
            convert.(eltype(expert_output), routing[expert, token_indices]),
            1,
            :,
        )
        target = view(output, :, token_indices)
        target .+= expert_output .* weights
    end

    routed_pairs = sum(expert_token_counts)
    stats = Qwen3MoEDispatchStats(
        num_tokens,
        num_experts,
        count(!iszero, expert_token_counts),
        routed_pairs,
        num_tokens * num_experts,
        expert_token_counts,
    )
    return output, stats
end

"""
    qwen3_route_major_expert_dispatch(
        tokens, expert_indices, routing_weights, expert_parameters)

Portable device fallback that evaluates the compact route-major token/expert
pairs produced by
[`qwen3_device_topk_routing`](@ref). Expert weights are gathered for exactly
`experts_per_token * num_tokens` routes, evaluated with batched matrix
multiplication, then combined back into token order on the device.

The implementation contains no `findall`, scalar indexing, or data-dependent
Julia branch. Reactant/XLA uses this path; CUDA overrides
[`qwen3_device_sparse_expert_dispatch`](@ref) with indexed kernels that avoid
materializing selected expert weights.
"""
function qwen3_route_major_expert_dispatch(
    tokens::AbstractMatrix,
    expert_indices::AbstractMatrix,
    routing_weights::AbstractMatrix,
    expert_parameters,
)
    num_experts = size(expert_parameters.gate_proj, 3)
    d_model, num_tokens, num_experts = _validate_qwen3_expert_parameters(
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
    experts_per_token > 0 || throw(ArgumentError(
        "compact routing must select at least one expert per token",
    ))
    experts_per_token <= num_experts || throw(ArgumentError(
        "compact routing cannot select more routes than the expert count",
    ))
    routed_experts = vec(expert_indices)
    pair_count = experts_per_token * num_tokens
    pair_tokens = reshape(
        repeat(reshape(tokens, d_model, 1, num_tokens), 1, experts_per_token, 1),
        d_model,
        1,
        pair_count,
    )
    hidden_dim = size(expert_parameters.gate_proj, 1)
    gate_weights = reshape(
        gather(
            copy(reshape(
                expert_parameters.gate_proj,
                hidden_dim * d_model,
                num_experts,
            )),
            routed_experts,
        ),
        hidden_dim,
        d_model,
        pair_count,
    )
    up_weights = reshape(
        gather(
            copy(reshape(
                expert_parameters.up_proj,
                hidden_dim * d_model,
                num_experts,
            )),
            routed_experts,
        ),
        hidden_dim,
        d_model,
        pair_count,
    )
    down_weights = reshape(
        gather(
            copy(reshape(
                expert_parameters.down_proj,
                d_model * hidden_dim,
                num_experts,
            )),
            routed_experts,
        ),
        d_model,
        hidden_dim,
        pair_count,
    )

    gate = batched_mul(gate_weights, pair_tokens)
    up = batched_mul(up_weights, pair_tokens)
    hidden = swish.(gate) .* up
    routed_output = batched_mul(down_weights, hidden)
    routed_output = reshape(
        routed_output,
        d_model,
        experts_per_token,
        num_tokens,
    )
    weights = reshape(
        convert.(eltype(routed_output), routing_weights),
        1,
        experts_per_token,
        num_tokens,
    )
    return dropdims(sum(routed_output .* weights; dims=2); dims=2)
end

"""
    qwen3_device_sparse_expert_dispatch(
        tokens, expert_indices, routing_weights, expert_parameters)

Dispatch compact token/expert routes on the current device. The generic
fallback is [`qwen3_route_major_expert_dispatch`](@ref); accelerator files can
provide a more specific method without changing the MoE forward contract.
"""
function qwen3_device_sparse_expert_dispatch(
    tokens::AbstractMatrix,
    expert_indices::AbstractMatrix,
    routing_weights::AbstractMatrix,
    expert_parameters,
)
    return qwen3_route_major_expert_dispatch(
        tokens,
        expert_indices,
        routing_weights,
        expert_parameters,
    )
end

"""Return the temporary bytes used by the CUDA indexed MoE route kernels."""
function qwen3_cuda_indexed_workspace_bytes(
    d_model::Int,
    hidden_dim::Int,
    num_tokens::Int,
    experts_per_token::Int;
    element_bytes::Int=sizeof(Float32),
)
    all(>(0), (d_model, hidden_dim, num_tokens, experts_per_token)) ||
        throw(ArgumentError("CUDA indexed workspace dimensions must be positive"))
    element_bytes > 0 || throw(ArgumentError("element_bytes must be positive"))
    pair_count = num_tokens * experts_per_token
    return (
        hidden_dim * pair_count +
        d_model * pair_count +
        d_model * num_tokens
    ) * element_bytes
end

"""
    qwen3_moe_device_forward(moe, x, ps)

Run Qwen3 MoE routing and compact expert dispatch without host routing
fallback. The returned route tables stay device-resident and contain exactly
`experts_per_token * seq_len * batch` entries.
"""
function qwen3_moe_device_forward(moe::Qwen3SparseMoE, x, ps)
    ndims(x) == 3 || throw(DimensionMismatch(
        "Qwen3SparseMoE input must have shape (d_model, seq_len, batch)",
    ))
    size(x, 1) == moe.d_model || throw(DimensionMismatch(
        "Qwen3SparseMoE input d_model does not match the layer",
    ))
    _, seq_len, batch_size = size(x)
    tokens = reshape(x, moe.d_model, :)
    router_logits = ps.gate.weight * tokens
    routed = qwen3_device_topk_routing(
        router_logits,
        moe.experts_per_token;
        normalize=moe.normalize_routing,
    )
    output = qwen3_device_sparse_expert_dispatch(
        tokens,
        routed.expert_indices,
        routed.routing_weights,
        ps.experts,
    )
    return (;
        output=reshape(output, moe.d_model, seq_len, batch_size),
        router_logits,
        routed.expert_indices,
        routed.routing_weights,
    )
end

"""
    qwen3_moe_forward_with_stats(moe, x, ps)

Run router plus host sparse dispatch and expose routing and execution counts for
correctness tests and benchmarks. The normal Lux call returns only the output.
"""
function qwen3_moe_forward_with_stats(moe::Qwen3SparseMoE, x, ps)
    ndims(x) == 3 || throw(DimensionMismatch(
        "Qwen3SparseMoE input must have shape (d_model, seq_len, batch)",
    ))
    size(x, 1) == moe.d_model || throw(DimensionMismatch(
        "Qwen3SparseMoE input d_model does not match the layer",
    ))
    _, seq_len, batch_size = size(x)
    tokens = reshape(x, moe.d_model, :)
    router_logits = ps.gate.weight * tokens
    routing = qwen3_topk_routing(
        router_logits,
        moe.experts_per_token;
        normalize=moe.normalize_routing,
    )
    output, stats = qwen3_sparse_expert_dispatch(tokens, routing, ps.experts)
    return (;
        output=reshape(output, moe.d_model, seq_len, batch_size),
        router_logits,
        routing,
        stats,
    )
end

function (moe::Qwen3SparseMoE)(x, ps, st::NamedTuple)
    result = if x isa Array && ps.gate.weight isa Array
        qwen3_moe_forward_with_stats(moe, x, ps)
    else
        qwen3_moe_device_forward(moe, x, ps)
    end
    return result.output, st
end
