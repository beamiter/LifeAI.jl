import NNlib
import LinearAlgebra

# Request-local workspace for one BF16 decode attention step.
# Query length is one and the attention mask is empty. K/V permutes and the
# score/softmax chain are sized to the caller-supplied capacity and reused
# across the 28 decoder layers. Prefill and masked attention stay on
# `_bf16a_attention`.

struct Qwen3VLDecodeAttentionScratch{
    Q5,
    Q3,
    KV,
    S,
    SF,
    MX,
    EX,
    W,
    WP,
    C,
    O,
}
    query5::Q5
    query3::Q3
    keys::KV
    values::KV
    scores::S
    scaled::S
    scores_f32::SF
    maxima::MX
    exponents::EX
    weights::W
    weight_perm::WP
    context::C
    output::O
    head_dim::Int
    num_heads::Int
    num_kv_heads::Int
    groups::Int
    capacity::Int
end

function qwen3_vl_decode_attention_scratch_bytes(;
    head_dim::Integer,
    num_attention_heads::Integer,
    num_kv_heads::Integer,
    capacity::Integer,
    batch_size::Integer=1,
)
    head_dim > 0 || throw(ArgumentError("head_dim must be positive"))
    num_kv_heads > 0 || throw(ArgumentError("num_kv_heads must be positive"))
    num_attention_heads > 0 || throw(ArgumentError("num_attention_heads must be positive"))
    capacity > 0 || throw(ArgumentError("capacity must be positive"))
    batch_size == 1 || throw(ArgumentError(
        "Qwen3-VL decode attention scratch supports batch size one",
    ))
    num_attention_heads % num_kv_heads == 0 || throw(ArgumentError(
        "num_attention_heads must be divisible by num_kv_heads",
    ))
    groups = num_attention_heads ÷ num_kv_heads
    bf16 = sizeof(BFloat16)
    f32 = sizeof(Float32)
    query5 = head_dim * groups * num_kv_heads * bf16
    query3 = groups * head_dim * num_kv_heads * bf16
    kv = head_dim * capacity * num_kv_heads * bf16
    score = groups * capacity * num_kv_heads * bf16
    score_f32 = groups * capacity * num_kv_heads * f32
    weight_perm = capacity * groups * num_kv_heads * bf16
    context = head_dim * groups * num_kv_heads * bf16
    maxima = groups * num_kv_heads * f32
    output = head_dim * num_attention_heads * bf16
    return query5 + query3 + 2 * kv + 3 * score + 2 * score_f32 +
        weight_perm + context + maxima + output
end

function qwen3_vl_decode_attention_scratch_bytes(
    scratch::Qwen3VLDecodeAttentionScratch,
)
    total = 0
    for array in (
        scratch.query5,
        scratch.query3,
        scratch.keys,
        scratch.values,
        scratch.scores,
        scratch.scaled,
        scratch.scores_f32,
        scratch.maxima,
        scratch.exponents,
        scratch.weights,
        scratch.weight_perm,
        scratch.context,
        scratch.output,
    )
        total += sizeof(array)
    end
    return total
end

function init_qwen3_vl_decode_attention_scratch(
    prototype::AbstractArray{BFloat16};
    head_dim::Integer,
    num_attention_heads::Integer,
    num_kv_heads::Integer,
    capacity::Integer,
)
    expected = qwen3_vl_decode_attention_scratch_bytes(;
        head_dim,
        num_attention_heads,
        num_kv_heads,
        capacity,
    )
    groups = num_attention_heads ÷ num_kv_heads
    query5 = similar(prototype, BFloat16, 1, head_dim, groups, num_kv_heads, 1)
    query3 = similar(prototype, BFloat16, 1, groups, head_dim, num_kv_heads, 1)
    keys = similar(prototype, BFloat16, head_dim, capacity, num_kv_heads)
    values = similar(prototype, BFloat16, head_dim, capacity, num_kv_heads)
    scores = similar(prototype, BFloat16, groups, capacity, num_kv_heads)
    scaled = similar(prototype, BFloat16, groups, capacity, num_kv_heads)
    scores_f32 = similar(prototype, Float32, groups, capacity, num_kv_heads)
    maxima = similar(prototype, Float32, groups, 1, num_kv_heads)
    exponents = similar(prototype, Float32, groups, capacity, num_kv_heads)
    weights = similar(prototype, BFloat16, groups, capacity, num_kv_heads)
    weight_perm = similar(prototype, BFloat16, capacity, groups, num_kv_heads)
    context = similar(prototype, BFloat16, head_dim, groups, num_kv_heads)
    output = similar(prototype, BFloat16, head_dim, num_attention_heads, 1, 1)
    scratch = Qwen3VLDecodeAttentionScratch(
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
        Int(head_dim),
        Int(num_attention_heads),
        Int(num_kv_heads),
        Int(groups),
        Int(capacity),
    )
    qwen3_vl_decode_attention_scratch_bytes(scratch) == expected ||
        throw(ArgumentError(
            "Qwen3-VL decode attention scratch bytes do not match the formula",
        ))
    return scratch
end

function _qwen3_vl_copy_kv_prefix!(dest, src)
    head_dim, num_kv_heads, seq, batch = size(src)
    batch == 1 || throw(ArgumentError(
        "Qwen3-VL decode attention scratch supports batch size one",
    ))
    size(dest, 1) == head_dim || throw(DimensionMismatch(
        "Qwen3-VL K/V permute destination has the wrong head dimension",
    ))
    size(dest, 3) == num_kv_heads || throw(DimensionMismatch(
        "Qwen3-VL K/V permute destination has the wrong KV count",
    ))
    seq <= size(dest, 2) || throw(ArgumentError(
        "Qwen3-VL decode attention sequence exceeds scratch capacity",
    ))
    parent_keys = _qwen3_vl_dense_kv_parent(src, size(dest, 2))
    permutedims!(reshape(dest, head_dim, size(dest, 2), num_kv_heads, 1), parent_keys, (1, 3, 2, 4))
    return view(dest, :, 1:seq, :)
end

function _qwen3_vl_dense_kv_parent(src, capacity::Int)
    if ndims(src) == 4 && size(src, 3) == capacity && size(src, 4) == 1 &&
        !(src isa SubArray)
        return src
    end
    src isa SubArray || throw(ArgumentError(
        "Qwen3-VL decode attention expected a static-cache K/V prefix view",
    ))
    parent_keys = parent(src)
    ndims(parent_keys) == 4 || throw(ArgumentError(
        "Qwen3-VL decode attention K/V parent must be rank 4",
    ))
    size(parent_keys, 3) == capacity || throw(DimensionMismatch(
        "Qwen3-VL decode attention K/V parent capacity does not match scratch",
    ))
    size(parent_keys, 4) == 1 || throw(DimensionMismatch(
        "Qwen3-VL decode attention K/V parent batch is not one",
    ))
    return parent_keys
end

function _qwen3_vl_device_gemm_storage!(::Any, C, A, B, transA::Char)
    if transA == 'N'
        LinearAlgebra.mul!(C, A, B)
    else
        LinearAlgebra.mul!(C, transpose(A), B)
    end
    return C
end

function _qwen3_vl_device_gemm!(C, A, B; transA::Char='N')
    storage = C isa SubArray ? parent(C) : C
    return _qwen3_vl_device_gemm_storage!(storage, C, A, B, transA)
end

# `sum!(abs2, dest, x)` on a GPU reduction longer than one block allocates the
# partial buffer inside CUDA's own kernel. The device method reuses that
# buffer and the same launch configuration. CPU reductions stay on `sum!`.
function _qwen3_vl_device_sum_abs2!(::Any, dest, x)
    sum!(abs2, dest, x)
    return dest
end

function _qwen3_vl_sum_abs2!(dest, x)
    storage = dest isa SubArray ? parent(dest) : dest
    return _qwen3_vl_device_sum_abs2!(storage, dest, x)
end

function _qwen3_vl_device_batched_mul!(::Any, dest, left, right)
    NNlib.batched_mul!(dest, left, right)
    return dest
end

function _qwen3_vl_batched_mul!(dest, left, right)
    storage = dest isa SubArray ? parent(dest) : dest
    if storage isa Array
        computed = _bf16a_batched_mul(Array(left), Array(right))
        dest .= computed
        return dest
    end
    return _qwen3_vl_device_batched_mul!(storage, dest, left, right)
end

function _qwen3_vl_load_decode_query!(scratch, queries)
    head_dim, num_heads, query_tokens, batch = size(queries)
    query_tokens == 1 || throw(ArgumentError(
        "Qwen3-VL reused decode attention requires one query token",
    ))
    batch == 1 || throw(ArgumentError(
        "Qwen3-VL reused decode attention requires batch size one",
    ))
    num_heads == scratch.num_heads || throw(DimensionMismatch(
        "Qwen3-VL reused decode attention query heads do not match scratch",
    ))
    head_dim == scratch.head_dim || throw(DimensionMismatch(
        "Qwen3-VL reused decode attention head_dim does not match scratch",
    ))
    permutedims!(
        reshape(scratch.query5, 1, head_dim, num_heads, 1),
        queries,
        (3, 1, 2, 4),
    )
    permutedims!(scratch.query3, scratch.query5, (1, 3, 2, 4, 5))
    return reshape(scratch.query3, scratch.groups, head_dim, scratch.num_kv_heads)
end

function _qwen3_vl_decode_attention!(
    scratch::Qwen3VLDecodeAttentionScratch,
    queries,
    keys,
    values;
    scaling::Float32,
)
    _, _, query_tokens, _ = size(queries)
    head_dim, num_kv_heads, seq, batch = size(keys)
    query_tokens == 1 || throw(ArgumentError(
        "Qwen3-VL reused decode attention requires one query token",
    ))
    batch == 1 || throw(ArgumentError(
        "Qwen3-VL reused decode attention requires batch size one",
    ))
    size(values) == size(keys) || throw(DimensionMismatch(
        "Qwen3-VL reused decode attention K/V shapes differ",
    ))
    seq <= scratch.capacity || throw(ArgumentError(
        "Qwen3-VL decode attention sequence exceeds scratch capacity",
    ))
    head_dim == scratch.head_dim || throw(DimensionMismatch(
        "Qwen3-VL reused decode attention head_dim does not match scratch",
    ))
    num_kv_heads == scratch.num_kv_heads || throw(DimensionMismatch(
        "Qwen3-VL reused decode attention KV heads do not match scratch",
    ))
    q3 = _qwen3_vl_load_decode_query!(scratch, queries)
    k_prefix = _qwen3_vl_copy_kv_prefix!(scratch.keys, keys)
    v_prefix = _qwen3_vl_copy_kv_prefix!(scratch.values, values)
    k3 = k_prefix
    v3 = v_prefix
    score_view = view(scratch.scores, :, 1:seq, :)
    _qwen3_vl_batched_mul!(score_view, q3, k3)
    scaled_view = view(scratch.scaled, :, 1:seq, :)
    f32_view = view(scratch.scores_f32, :, 1:seq, :)
    f32_view .= 1.0f0 .* score_view
    scaled_view .= BFloat16.(f32_view .* scaling)
    f32_view .= 1.0f0 .* scaled_view
    maximum!(scratch.maxima, f32_view)
    exp_view = view(scratch.exponents, :, 1:seq, :)
    exp_view .= exp.(f32_view .- scratch.maxima)
    sum!(scratch.maxima, exp_view)
    weight_view = view(scratch.weights, :, 1:seq, :)
    weight_view .= BFloat16.(exp_view ./ scratch.maxima)
    permutedims!(scratch.weight_perm, scratch.weights, (2, 1, 3))
    w3 = view(scratch.weight_perm, 1:seq, :, :)
    _qwen3_vl_batched_mul!(scratch.context, v3, w3)
    packed = reshape(
        scratch.output,
        head_dim,
        scratch.groups,
        num_kv_heads,
        1,
        1,
    )
    permutedims!(
        packed,
        reshape(scratch.context, head_dim, 1, scratch.groups, num_kv_heads, 1),
        (1, 3, 4, 2, 5),
    )
    return scratch.output
end

struct Qwen3VLMlpActivationScratch{G,A,AF,U,P}
    gate_f32::G
    activated::A
    activated_f32::AF
    up_f32::U
    product::P
    intermediate_size::Int
end

function qwen3_vl_mlp_activation_scratch_bytes(; intermediate_size::Integer)
    intermediate_size > 0 || throw(ArgumentError(
        "intermediate_size must be positive",
    ))
    return Int(intermediate_size) * (3 * sizeof(Float32) + 2 * sizeof(BFloat16))
end

function qwen3_vl_mlp_activation_scratch_bytes(scratch::Qwen3VLMlpActivationScratch)
    return sizeof(scratch.gate_f32) + sizeof(scratch.activated) +
        sizeof(scratch.activated_f32) + sizeof(scratch.up_f32) +
        sizeof(scratch.product)
end

function init_qwen3_vl_mlp_activation_scratch(
    prototype::AbstractArray{BFloat16};
    intermediate_size::Integer,
)
    expected = qwen3_vl_mlp_activation_scratch_bytes(; intermediate_size)
    shape = (Int(intermediate_size), 1, 1)
    scratch = Qwen3VLMlpActivationScratch(
        similar(prototype, Float32, shape...),
        similar(prototype, BFloat16, shape...),
        similar(prototype, Float32, shape...),
        similar(prototype, Float32, shape...),
        similar(prototype, BFloat16, shape...),
        Int(intermediate_size),
    )
    qwen3_vl_mlp_activation_scratch_bytes(scratch) == expected ||
        throw(ArgumentError(
            "Qwen3-VL MLP activation scratch bytes do not match the formula",
        ))
    return scratch
end

function _qwen3_vl_mlp_activation!(scratch::Qwen3VLMlpActivationScratch, gate, up)
    size(gate) == size(scratch.product) || throw(DimensionMismatch(
        "Qwen3-VL MLP activation gate does not match the scratch",
    ))
    size(up) == size(gate) || throw(DimensionMismatch(
        "Qwen3-VL MLP activation gate and up shapes differ",
    ))
    scratch.gate_f32 .= 1.0f0 .* gate
    scratch.activated .= BFloat16.(
        scratch.gate_f32 ./ (1.0f0 .+ exp.(.-scratch.gate_f32))
    )
    scratch.activated_f32 .= 1.0f0 .* scratch.activated
    scratch.up_f32 .= 1.0f0 .* up
    scratch.product .= BFloat16.(scratch.activated_f32 .* scratch.up_f32)
    return scratch.product
end

function _qwen3_vl_reference_mlp_activation(gate, up)
    gate_f32 = _bf16a_f32(gate)
    activated = BFloat16.(gate_f32 ./ (1.0f0 .+ exp.(.-gate_f32)))
    return BFloat16.(_bf16a_f32(activated) .* _bf16a_f32(up))
end

function _qwen3_vl_ensure_mlp_activation_scratch!(cache, spec, prototype)
    scratch = cache.mlp_activation_scratch
    if scratch isa Qwen3VLMlpActivationScratch &&
        scratch.intermediate_size == spec.intermediate_size &&
        eltype(scratch.product) === eltype(prototype) &&
        get_device(scratch.product) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL MLP activation scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_mlp_activation_scratch(
        prototype;
        intermediate_size=spec.intermediate_size,
    )
    cache.mlp_activation_scratch = scratch
    return scratch
end

function _qwen3_vl_ensure_decode_attention_scratch!(cache, spec, prototype)
    scratch = cache.attention_scratch
    if scratch isa Qwen3VLDecodeAttentionScratch &&
        scratch.capacity == cache.capacity &&
        scratch.head_dim == spec.head_dim &&
        scratch.num_heads == spec.num_attention_heads &&
        scratch.num_kv_heads == spec.num_key_value_heads &&
        eltype(scratch.keys) === eltype(prototype) &&
        get_device(scratch.keys) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL decode attention scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_decode_attention_scratch(
        prototype;
        head_dim=spec.head_dim,
        num_attention_heads=spec.num_attention_heads,
        num_kv_heads=spec.num_key_value_heads,
        capacity=cache.capacity,
    )
    cache.attention_scratch = scratch
    return scratch
end

# One-token BF16 Q/K RoPE workspace. Query and key outputs stay distinct
# because both are live until the KV write and attention. The half-width
# intermediates are sized to the query heads and reused for the smaller key
# prefix, then across decoder layers. The shared `_bf16a_apply_rope` path is
# unchanged and still serves prefill, Float32, and other models.

struct Qwen3VLQKRopeScratch{X1,X2,C,S,TF,TB,A,B,U,L,QO,KO}
    x1::X1
    x2::X2
    cos_f32::C
    sin_f32::S
    tmp_f32::TF
    tmp_bf16::TB
    term_a::A
    term_b::B
    upper::U
    lower::L
    query_out::QO
    key_out::KO
    head_dim::Int
    num_heads::Int
    num_kv_heads::Int
end

function qwen3_vl_qk_rope_scratch_bytes(;
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    head_dim > 0 && iseven(head_dim) || throw(ArgumentError(
        "Qwen3-VL RoPE scratch head_dim must be a positive even integer",
    ))
    num_heads > 0 || throw(ArgumentError(
        "Qwen3-VL RoPE scratch num_heads must be positive",
    ))
    num_kv_heads > 0 || throw(ArgumentError(
        "Qwen3-VL RoPE scratch num_kv_heads must be positive",
    ))
    num_heads >= num_kv_heads || throw(ArgumentError(
        "Qwen3-VL RoPE scratch query heads must cover the KV heads",
    ))
    half = head_dim ÷ 2
    spatial = half * num_heads
    return 7 * spatial * sizeof(Float32) +
        spatial * sizeof(BFloat16) +
        2 * half * sizeof(Float32) +
        head_dim * num_heads * sizeof(BFloat16) +
        head_dim * num_kv_heads * sizeof(BFloat16)
end

function qwen3_vl_qk_rope_scratch_bytes(scratch::Qwen3VLQKRopeScratch)
    return sizeof(scratch.x1) + sizeof(scratch.x2) + sizeof(scratch.cos_f32) +
        sizeof(scratch.sin_f32) + sizeof(scratch.tmp_f32) +
        sizeof(scratch.tmp_bf16) + sizeof(scratch.term_a) +
        sizeof(scratch.term_b) + sizeof(scratch.upper) +
        sizeof(scratch.lower) + sizeof(scratch.query_out) +
        sizeof(scratch.key_out)
end

function init_qwen3_vl_qk_rope_scratch(
    prototype::AbstractArray{BFloat16};
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    expected = qwen3_vl_qk_rope_scratch_bytes(;
        head_dim,
        num_heads,
        num_kv_heads,
    )
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    half = head_dim ÷ 2
    half_shape = (half, num_heads, 1, 1)
    scratch = Qwen3VLQKRopeScratch(
        similar(prototype, Float32, half_shape...),
        similar(prototype, Float32, half_shape...),
        similar(prototype, Float32, half, 1, 1, 1),
        similar(prototype, Float32, half, 1, 1, 1),
        similar(prototype, Float32, half_shape...),
        similar(prototype, BFloat16, half_shape...),
        similar(prototype, Float32, half_shape...),
        similar(prototype, Float32, half_shape...),
        similar(prototype, Float32, half_shape...),
        similar(prototype, Float32, half_shape...),
        similar(prototype, BFloat16, head_dim, num_heads, 1, 1),
        similar(prototype, BFloat16, head_dim, num_kv_heads, 1, 1),
        head_dim,
        num_heads,
        num_kv_heads,
    )
    qwen3_vl_qk_rope_scratch_bytes(scratch) == expected || throw(ArgumentError(
        "Qwen3-VL RoPE scratch bytes do not match the formula",
    ))
    return scratch
end

function _qwen3_vl_qk_rope_heads(buffer, heads::Int)
    size(buffer, 2) == heads && return buffer
    return view(buffer, :, 1:heads, :, :)
end

function _qwen3_vl_qk_rope_table(values, half::Int)
    ndims(values) == 2 || ndims(values) == 3 || throw(DimensionMismatch(
        "Qwen3-VL RoPE table must be 2D or 3D",
    ))
    size(values, 1) == half || throw(DimensionMismatch(
        "Qwen3-VL RoPE table does not match half of head_dim",
    ))
    size(values, 2) == 1 || throw(DimensionMismatch(
        "Qwen3-VL decode RoPE scratch supports one token",
    ))
    if ndims(values) == 3
        size(values, 3) == 1 || throw(DimensionMismatch(
            "Qwen3-VL decode RoPE scratch supports batch size one",
        ))
    end
    return reshape(values, half, 1, 1, 1)
end

function _qwen3_vl_decode_rope_one!(
    scratch::Qwen3VLQKRopeScratch,
    x,
    cos_values,
    sin_values,
    output,
)
    eltype(x) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL decode RoPE scratch requires BFloat16 values",
    ))
    size(x) == size(output) || throw(DimensionMismatch(
        "Qwen3-VL RoPE output does not match the input",
    ))
    size(x, 1) == scratch.head_dim || throw(DimensionMismatch(
        "Qwen3-VL RoPE head_dim does not match the scratch",
    ))
    size(x, 3) == 1 && size(x, 4) == 1 || throw(DimensionMismatch(
        "Qwen3-VL decode RoPE scratch supports one token and batch size one",
    ))
    heads = size(x, 2)
    heads == scratch.num_heads || heads == scratch.num_kv_heads ||
        throw(DimensionMismatch(
            "Qwen3-VL RoPE heads do not match the scratch",
        ))
    half = scratch.head_dim ÷ 2
    x1 = _qwen3_vl_qk_rope_heads(scratch.x1, heads)
    x2 = _qwen3_vl_qk_rope_heads(scratch.x2, heads)
    tmp_f32 = _qwen3_vl_qk_rope_heads(scratch.tmp_f32, heads)
    tmp_bf16 = _qwen3_vl_qk_rope_heads(scratch.tmp_bf16, heads)
    term_a = _qwen3_vl_qk_rope_heads(scratch.term_a, heads)
    term_b = _qwen3_vl_qk_rope_heads(scratch.term_b, heads)
    upper = _qwen3_vl_qk_rope_heads(scratch.upper, heads)
    lower = _qwen3_vl_qk_rope_heads(scratch.lower, heads)
    cosine = _qwen3_vl_qk_rope_table(cos_values, half)
    sine = _qwen3_vl_qk_rope_table(sin_values, half)
    x1 .= 1.0f0 .* view(x, 1:half, :, :, :)
    x2 .= 1.0f0 .* view(x, (half + 1):scratch.head_dim, :, :, :)
    scratch.cos_f32 .= 1.0f0 .* cosine
    scratch.sin_f32 .= 1.0f0 .* sine
    tmp_f32 .= x1 .* scratch.cos_f32
    tmp_bf16 .= BFloat16.(tmp_f32)
    term_a .= 1.0f0 .* tmp_bf16
    tmp_f32 .= .-x2 .* scratch.sin_f32
    tmp_bf16 .= BFloat16.(tmp_f32)
    term_b .= 1.0f0 .* tmp_bf16
    upper .= term_a .+ term_b
    tmp_f32 .= x2 .* scratch.cos_f32
    tmp_bf16 .= BFloat16.(tmp_f32)
    term_a .= 1.0f0 .* tmp_bf16
    tmp_f32 .= x1 .* scratch.sin_f32
    tmp_bf16 .= BFloat16.(tmp_f32)
    term_b .= 1.0f0 .* tmp_bf16
    lower .= term_a .+ term_b
    view(output, 1:half, :, :, :) .= BFloat16.(upper)
    view(output, (half + 1):scratch.head_dim, :, :, :) .= BFloat16.(lower)
    return output
end

function _qwen3_vl_decode_qk_rope!(
    scratch::Qwen3VLQKRopeScratch,
    queries,
    keys,
    cos_values,
    sin_values,
)
    _qwen3_vl_decode_rope_one!(
        scratch,
        queries,
        cos_values,
        sin_values,
        scratch.query_out,
    )
    _qwen3_vl_decode_rope_one!(
        scratch,
        keys,
        cos_values,
        sin_values,
        scratch.key_out,
    )
    return scratch.query_out, scratch.key_out
end

function _qwen3_vl_ensure_qk_rope_scratch!(cache, spec, prototype)
    scratch = cache.qk_rope_scratch
    if scratch isa Qwen3VLQKRopeScratch &&
        scratch.head_dim == spec.head_dim &&
        scratch.num_heads == spec.num_attention_heads &&
        scratch.num_kv_heads == spec.num_key_value_heads &&
        eltype(scratch.query_out) === eltype(prototype) &&
        get_device(scratch.query_out) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL RoPE scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_qk_rope_scratch(
        prototype;
        head_dim=spec.head_dim,
        num_heads=spec.num_attention_heads,
        num_kv_heads=spec.num_key_value_heads,
    )
    cache.qk_rope_scratch = scratch
    return scratch
end

# One-token BF16 Q/K RMSNorm workspace. Query and key outputs stay distinct
# because both are live until RoPE. Half-width is not involved: the reduction
# is over head_dim, and the key path reuses a head prefix of the query-sized
# intermediates. Prefill and Float32 stay on `_qwen3_vl_text_rmsnorm`.

struct Qwen3VLQKNormScratch{XF,MS,N,NF,SC,QO,KO}
    xf::XF
    mean_square::MS
    normalized::N
    norm_f32::NF
    scale_f32::SC
    query_out::QO
    key_out::KO
    head_dim::Int
    num_heads::Int
    num_kv_heads::Int
end

function qwen3_vl_qk_norm_scratch_bytes(;
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    head_dim > 0 || throw(ArgumentError(
        "Qwen3-VL QK norm scratch head_dim must be positive",
    ))
    num_heads > 0 || throw(ArgumentError(
        "Qwen3-VL QK norm scratch num_heads must be positive",
    ))
    num_kv_heads > 0 || throw(ArgumentError(
        "Qwen3-VL QK norm scratch num_kv_heads must be positive",
    ))
    num_heads >= num_kv_heads || throw(ArgumentError(
        "Qwen3-VL QK norm scratch query heads must cover the KV heads",
    ))
    spatial = head_dim * num_heads
    return 2 * spatial * sizeof(Float32) +
        spatial * sizeof(BFloat16) +
        num_heads * sizeof(Float32) +
        head_dim * sizeof(Float32) +
        spatial * sizeof(BFloat16) +
        head_dim * num_kv_heads * sizeof(BFloat16)
end

function qwen3_vl_qk_norm_scratch_bytes(scratch::Qwen3VLQKNormScratch)
    return sizeof(scratch.xf) + sizeof(scratch.mean_square) +
        sizeof(scratch.normalized) + sizeof(scratch.norm_f32) +
        sizeof(scratch.scale_f32) + sizeof(scratch.query_out) +
        sizeof(scratch.key_out)
end

function init_qwen3_vl_qk_norm_scratch(
    prototype::AbstractArray{BFloat16};
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    expected = qwen3_vl_qk_norm_scratch_bytes(;
        head_dim,
        num_heads,
        num_kv_heads,
    )
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    spatial = (head_dim, num_heads, 1, 1)
    scratch = Qwen3VLQKNormScratch(
        similar(prototype, Float32, spatial...),
        similar(prototype, Float32, 1, num_heads, 1, 1),
        similar(prototype, BFloat16, spatial...),
        similar(prototype, Float32, spatial...),
        similar(prototype, Float32, head_dim, 1, 1, 1),
        similar(prototype, BFloat16, spatial...),
        similar(prototype, BFloat16, head_dim, num_kv_heads, 1, 1),
        head_dim,
        num_heads,
        num_kv_heads,
    )
    qwen3_vl_qk_norm_scratch_bytes(scratch) == expected || throw(ArgumentError(
        "Qwen3-VL QK norm scratch bytes do not match the formula",
    ))
    return scratch
end

function _qwen3_vl_decode_qk_norm_one!(
    scratch::Qwen3VLQKNormScratch,
    x,
    scale,
    output,
    epsilon::Float32,
)
    eltype(x) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL decode QK norm scratch requires BFloat16 values",
    ))
    size(x) == size(output) || throw(DimensionMismatch(
        "Qwen3-VL QK norm output does not match the input",
    ))
    size(x, 1) == scratch.head_dim || throw(DimensionMismatch(
        "Qwen3-VL QK norm head_dim does not match the scratch",
    ))
    size(x, 3) == 1 && size(x, 4) == 1 || throw(DimensionMismatch(
        "Qwen3-VL decode QK norm scratch supports one token and batch size one",
    ))
    heads = size(x, 2)
    heads == scratch.num_heads || heads == scratch.num_kv_heads ||
        throw(DimensionMismatch(
            "Qwen3-VL QK norm heads do not match the scratch",
        ))
    length(scale) == scratch.head_dim || throw(DimensionMismatch(
        "Qwen3-VL QK norm scale does not match head_dim",
    ))
    xf = _qwen3_vl_qk_rope_heads(scratch.xf, heads)
    mean_square = _qwen3_vl_qk_rope_heads(scratch.mean_square, heads)
    normalized = _qwen3_vl_qk_rope_heads(scratch.normalized, heads)
    norm_f32 = _qwen3_vl_qk_rope_heads(scratch.norm_f32, heads)
    xf .= 1.0f0 .* x
    sum!(abs2, mean_square, xf)
    mean_square .= mean_square ./ Float32(size(x, 1))
    normalized .= BFloat16.(xf ./ sqrt.(mean_square .+ epsilon))
    scratch.scale_f32 .= 1.0f0 .* reshape(scale, scratch.head_dim, 1, 1, 1)
    norm_f32 .= 1.0f0 .* normalized
    output .= BFloat16.(scratch.scale_f32 .* norm_f32)
    return output
end

function _qwen3_vl_decode_qk_norm!(
    scratch::Qwen3VLQKNormScratch,
    queries,
    keys,
    query_scale,
    key_scale,
    epsilon::Real,
)
    epsilon_f32 = Float32(epsilon)
    _qwen3_vl_decode_qk_norm_one!(
        scratch,
        queries,
        query_scale,
        scratch.query_out,
        epsilon_f32,
    )
    _qwen3_vl_decode_qk_norm_one!(
        scratch,
        keys,
        key_scale,
        scratch.key_out,
        epsilon_f32,
    )
    return scratch.query_out, scratch.key_out
end

function _qwen3_vl_ensure_qk_norm_scratch!(cache, spec, prototype)
    scratch = cache.qk_norm_scratch
    if scratch isa Qwen3VLQKNormScratch &&
        scratch.head_dim == spec.head_dim &&
        scratch.num_heads == spec.num_attention_heads &&
        scratch.num_kv_heads == spec.num_key_value_heads &&
        eltype(scratch.query_out) === eltype(prototype) &&
        get_device(scratch.query_out) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL QK norm scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_qk_norm_scratch(
        prototype;
        head_dim=spec.head_dim,
        num_heads=spec.num_attention_heads,
        num_kv_heads=spec.num_key_value_heads,
    )
    cache.qk_norm_scratch = scratch
    return scratch
end

# One-token BF16 hidden RMSNorm workspace. Prefill, post-attention, and the
# final norm share it because only one result is live at a time. The large
# tensors stay here. A GPU reduction longer than one block reuses CUDA's
# partial buffer without changing that reduction's launch configuration.

struct Qwen3VLHiddenNormScratch{XF,MS,N,NF,SC,O}
    xf::XF
    mean_square::MS
    normalized::N
    norm_f32::NF
    scale_f32::SC
    output::O
    hidden_size::Int
end

function qwen3_vl_hidden_norm_scratch_bytes(; hidden_size::Integer)
    hidden_size = Int(hidden_size)
    hidden_size > 0 || throw(ArgumentError(
        "Qwen3-VL hidden norm scratch hidden_size must be positive",
    ))
    return hidden_size * (3 * sizeof(Float32) + 2 * sizeof(BFloat16)) +
        sizeof(Float32)
end

function qwen3_vl_hidden_norm_scratch_bytes(scratch::Qwen3VLHiddenNormScratch)
    return sizeof(scratch.xf) + sizeof(scratch.mean_square) +
        sizeof(scratch.normalized) + sizeof(scratch.norm_f32) +
        sizeof(scratch.scale_f32) + sizeof(scratch.output)
end

function init_qwen3_vl_hidden_norm_scratch(
    prototype::AbstractArray{BFloat16};
    hidden_size::Integer,
)
    expected = qwen3_vl_hidden_norm_scratch_bytes(; hidden_size)
    hidden_size = Int(hidden_size)
    shape = (hidden_size, 1, 1)
    scratch = Qwen3VLHiddenNormScratch(
        similar(prototype, Float32, shape...),
        similar(prototype, Float32, 1, 1, 1),
        similar(prototype, BFloat16, shape...),
        similar(prototype, Float32, shape...),
        similar(prototype, Float32, shape...),
        similar(prototype, BFloat16, shape...),
        hidden_size,
    )
    qwen3_vl_hidden_norm_scratch_bytes(scratch) == expected ||
        throw(ArgumentError(
            "Qwen3-VL hidden norm scratch bytes do not match the formula",
        ))
    return scratch
end

function _qwen3_vl_decode_hidden_norm!(
    scratch::Qwen3VLHiddenNormScratch,
    x,
    scale,
    epsilon::Real,
)
    eltype(x) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL decode hidden norm scratch requires BFloat16 values",
    ))
    size(x) == size(scratch.output) || throw(DimensionMismatch(
        "Qwen3-VL hidden norm input does not match the scratch",
    ))
    length(scale) == scratch.hidden_size || throw(DimensionMismatch(
        "Qwen3-VL hidden norm scale does not match hidden_size",
    ))
    epsilon_f32 = Float32(epsilon)
    scratch.xf .= 1.0f0 .* x
    _qwen3_vl_sum_abs2!(scratch.mean_square, scratch.xf)
    scratch.mean_square .= scratch.mean_square ./ Float32(size(x, 1))
    scratch.normalized .= BFloat16.(
        scratch.xf ./ sqrt.(scratch.mean_square .+ epsilon_f32)
    )
    scratch.scale_f32 .= 1.0f0 .* reshape(scale, scratch.hidden_size, 1, 1)
    scratch.norm_f32 .= 1.0f0 .* scratch.normalized
    scratch.output .= BFloat16.(scratch.scale_f32 .* scratch.norm_f32)
    return scratch.output
end

function _qwen3_vl_ensure_hidden_norm_scratch!(cache, spec, prototype)
    scratch = cache.hidden_norm_scratch
    if scratch isa Qwen3VLHiddenNormScratch &&
        scratch.hidden_size == spec.hidden_size &&
        eltype(scratch.output) === eltype(prototype) &&
        get_device(scratch.output) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL hidden norm scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_hidden_norm_scratch(
        prototype;
        hidden_size=spec.hidden_size,
    )
    cache.hidden_norm_scratch = scratch
    return scratch
end

# One-token BF16 gate/up outputs. Both stay live until the SiLU product.
# `mul!` on GPU matches the allocating BF16 GEMM and still reports a few
# bytes; the output tensors themselves are the bytes this scratch removes.
# CPU arrays use the same chunked Float32 accumulation as `_bf16_linear`.

struct Qwen3VLGateUpScratch{G,U}
    gate::G
    up::U
    hidden_size::Int
    intermediate_size::Int
end

function qwen3_vl_gate_up_scratch_bytes(; intermediate_size::Integer)
    intermediate_size = Int(intermediate_size)
    intermediate_size > 0 || throw(ArgumentError(
        "Qwen3-VL gate/up scratch intermediate_size must be positive",
    ))
    return 2 * intermediate_size * sizeof(BFloat16)
end

function qwen3_vl_gate_up_scratch_bytes(scratch::Qwen3VLGateUpScratch)
    return sizeof(scratch.gate) + sizeof(scratch.up)
end

function init_qwen3_vl_gate_up_scratch(
    prototype::AbstractArray{BFloat16};
    hidden_size::Integer,
    intermediate_size::Integer,
)
    hidden_size = Int(hidden_size)
    intermediate_size = Int(intermediate_size)
    hidden_size > 0 || throw(ArgumentError(
        "Qwen3-VL gate/up scratch hidden_size must be positive",
    ))
    expected = qwen3_vl_gate_up_scratch_bytes(; intermediate_size)
    shape = (intermediate_size, 1, 1)
    scratch = Qwen3VLGateUpScratch(
        similar(prototype, BFloat16, shape...),
        similar(prototype, BFloat16, shape...),
        hidden_size,
        intermediate_size,
    )
    qwen3_vl_gate_up_scratch_bytes(scratch) == expected || throw(ArgumentError(
        "Qwen3-VL gate/up scratch bytes do not match the formula",
    ))
    return scratch
end

function _qwen3_vl_bf16_linear!(
    output::AbstractArray{BFloat16,3},
    weight::AbstractMatrix{BFloat16},
    x::AbstractArray{BFloat16,3},
)
    in_dim, num_tokens, batch_size = size(x)
    out_dim = size(weight, 1)
    size(output) == (out_dim, num_tokens, batch_size) || throw(DimensionMismatch(
        "Qwen3-VL linear output does not match the weight and input",
    ))
    size(weight, 2) == in_dim || throw(DimensionMismatch(
        "Qwen3-VL linear input dimension does not match the weight",
    ))
    if output isa Array
        x_f32 = reshape(Float32.(x), in_dim, :)
        y_f32 = Matrix{Float32}(undef, out_dim, size(x_f32, 2))
        rows_per_chunk = clamp(16 * 1024 * 1024 ÷ max(in_dim, 1), 128, out_dim)
        for row_start in 1:rows_per_chunk:out_dim
            row_stop = min(row_start + rows_per_chunk - 1, out_dim)
            LinearAlgebra.mul!(
                view(y_f32, row_start:row_stop, :),
                Float32.(view(weight, row_start:row_stop, :)),
                x_f32,
            )
        end
        output .= reshape(BFloat16.(y_f32), out_dim, num_tokens, batch_size)
        return output
    end
    _qwen3_vl_device_gemm!(
        reshape(output, out_dim, :),
        weight,
        reshape(x, in_dim, :),
    )
    return output
end

function _qwen3_vl_decode_gate_up!(scratch::Qwen3VLGateUpScratch, gate_weight, up_weight, x)
    eltype(x) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL gate/up scratch requires BFloat16 values",
    ))
    size(x) == (scratch.hidden_size, 1, 1) || throw(DimensionMismatch(
        "Qwen3-VL gate/up scratch supports one token and batch size one",
    ))
    _qwen3_vl_bf16_linear!(scratch.gate, gate_weight, x)
    _qwen3_vl_bf16_linear!(scratch.up, up_weight, x)
    return scratch.gate, scratch.up
end

function _qwen3_vl_ensure_gate_up_scratch!(cache, spec, prototype)
    scratch = cache.gate_up_scratch
    if scratch isa Qwen3VLGateUpScratch &&
        scratch.hidden_size == spec.hidden_size &&
        scratch.intermediate_size == spec.intermediate_size &&
        eltype(scratch.gate) === eltype(prototype) &&
        get_device(scratch.gate) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL gate/up scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_gate_up_scratch(
        prototype;
        hidden_size=spec.hidden_size,
        intermediate_size=spec.intermediate_size,
    )
    cache.gate_up_scratch = scratch
    return scratch
end

# Shared one-token workspace for the attention output projection and the MLP
# down projection. Each is a hidden-sized BF16 GEMM followed by the same
# BF16 residual rounding. The two calls are sequential, so one projected
# buffer and one residual output are enough.

struct Qwen3VLHiddenResidualScratch{P,X,V,O}
    projected::P
    residual_f32::X
    projected_f32::V
    output::O
    hidden_size::Int
end

function qwen3_vl_hidden_residual_scratch_bytes(; hidden_size::Integer)
    hidden_size = Int(hidden_size)
    hidden_size > 0 || throw(ArgumentError(
        "Qwen3-VL hidden residual scratch hidden_size must be positive",
    ))
    return hidden_size * (2 * sizeof(Float32) + 2 * sizeof(BFloat16))
end

function qwen3_vl_hidden_residual_scratch_bytes(
    scratch::Qwen3VLHiddenResidualScratch,
)
    return sizeof(scratch.projected) + sizeof(scratch.residual_f32) +
        sizeof(scratch.projected_f32) + sizeof(scratch.output)
end

function init_qwen3_vl_hidden_residual_scratch(
    prototype::AbstractArray{BFloat16};
    hidden_size::Integer,
)
    expected = qwen3_vl_hidden_residual_scratch_bytes(; hidden_size)
    hidden_size = Int(hidden_size)
    shape = (hidden_size, 1, 1)
    scratch = Qwen3VLHiddenResidualScratch(
        similar(prototype, BFloat16, shape...),
        similar(prototype, Float32, shape...),
        similar(prototype, Float32, shape...),
        similar(prototype, BFloat16, shape...),
        hidden_size,
    )
    qwen3_vl_hidden_residual_scratch_bytes(scratch) == expected ||
        throw(ArgumentError(
            "Qwen3-VL hidden residual scratch bytes do not match the formula",
        ))
    return scratch
end

function _qwen3_vl_decode_hidden_residual!(
    scratch::Qwen3VLHiddenResidualScratch,
    weight,
    value,
    residual,
)
    eltype(value) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL hidden residual scratch requires BFloat16 values",
    ))
    eltype(residual) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL hidden residual scratch requires a BFloat16 residual",
    ))
    size(residual) == size(scratch.output) || throw(DimensionMismatch(
        "Qwen3-VL hidden residual does not match the scratch",
    ))
    size(value, 2) == 1 && size(value, 3) == 1 || throw(DimensionMismatch(
        "Qwen3-VL hidden residual scratch supports one token and batch size one",
    ))
    _qwen3_vl_bf16_linear!(scratch.projected, weight, value)
    scratch.residual_f32 .= 1.0f0 .* residual
    scratch.projected_f32 .= 1.0f0 .* scratch.projected
    scratch.output .= BFloat16.(scratch.residual_f32 .+ scratch.projected_f32)
    return scratch.output
end

function _qwen3_vl_ensure_hidden_residual_scratch!(cache, spec, prototype)
    scratch = cache.hidden_residual_scratch
    if scratch isa Qwen3VLHiddenResidualScratch &&
        scratch.hidden_size == spec.hidden_size &&
        eltype(scratch.output) === eltype(prototype) &&
        get_device(scratch.output) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL hidden residual scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_hidden_residual_scratch(
        prototype;
        hidden_size=spec.hidden_size,
    )
    cache.hidden_residual_scratch = scratch
    return scratch
end

# One-token tied vocabulary logits. The returned array aliases this buffer and
# is only valid until the next BF16 decode on the same cache. GPU `mul!`
# matches `transpose(embedding) * hidden`; CPU copies the existing projection
# so the numeric path stays the one tests already compare.

struct Qwen3VLTiedLogitsScratch{O}
    output::O
    hidden_size::Int
    vocab_size::Int
end

function qwen3_vl_tied_logits_scratch_bytes(; vocab_size::Integer)
    vocab_size = Int(vocab_size)
    vocab_size > 0 || throw(ArgumentError(
        "Qwen3-VL tied logits scratch vocab_size must be positive",
    ))
    return vocab_size * sizeof(BFloat16)
end

function qwen3_vl_tied_logits_scratch_bytes(scratch::Qwen3VLTiedLogitsScratch)
    return sizeof(scratch.output)
end

function init_qwen3_vl_tied_logits_scratch(
    prototype::AbstractArray{BFloat16};
    hidden_size::Integer,
    vocab_size::Integer,
)
    hidden_size = Int(hidden_size)
    vocab_size = Int(vocab_size)
    hidden_size > 0 || throw(ArgumentError(
        "Qwen3-VL tied logits scratch hidden_size must be positive",
    ))
    expected = qwen3_vl_tied_logits_scratch_bytes(; vocab_size)
    scratch = Qwen3VLTiedLogitsScratch(
        similar(prototype, BFloat16, vocab_size, 1, 1),
        hidden_size,
        vocab_size,
    )
    qwen3_vl_tied_logits_scratch_bytes(scratch) == expected ||
        throw(ArgumentError(
            "Qwen3-VL tied logits scratch bytes do not match the formula",
        ))
    return scratch
end

function _qwen3_vl_decode_tied_logits!(
    scratch::Qwen3VLTiedLogitsScratch,
    embedding,
    hidden,
)
    eltype(hidden) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL tied logits scratch requires BFloat16 hidden states",
    ))
    size(hidden) == (scratch.hidden_size, 1, 1) || throw(DimensionMismatch(
        "Qwen3-VL tied logits scratch supports one token and batch size one",
    ))
    size(embedding) == (scratch.hidden_size, scratch.vocab_size) ||
        throw(DimensionMismatch(
            "Qwen3-VL tied embedding does not match the scratch",
        ))
    if scratch.output isa Array
        scratch.output .= _qwen3_vl_project_tied(embedding, hidden)
        return scratch.output
    end
    _qwen3_vl_device_gemm!(
        reshape(scratch.output, scratch.vocab_size, :),
        embedding,
        reshape(hidden, scratch.hidden_size, :);
        transA='T',
    )
    return scratch.output
end

function _qwen3_vl_ensure_tied_logits_scratch!(cache, spec, prototype)
    scratch = cache.tied_logits_scratch
    if scratch isa Qwen3VLTiedLogitsScratch &&
        scratch.hidden_size == spec.hidden_size &&
        scratch.vocab_size == spec.vocab_size &&
        eltype(scratch.output) === eltype(prototype) &&
        get_device(scratch.output) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL tied logits scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_tied_logits_scratch(
        prototype;
        hidden_size=spec.hidden_size,
        vocab_size=spec.vocab_size,
    )
    cache.tied_logits_scratch = scratch
    return scratch
end

# One-token Q, K and V projection outputs. QK-norm and attention consume them
# before the next layer, so one triple is shared across layers.

struct Qwen3VLQKVScratch{Q,K,V}
    query::Q
    key::K
    value::V
    hidden_size::Int
    head_dim::Int
    num_heads::Int
    num_kv_heads::Int
end

function qwen3_vl_qkv_scratch_bytes(;
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    head_dim > 0 || throw(ArgumentError(
        "Qwen3-VL QKV scratch head_dim must be positive",
    ))
    num_heads > 0 || throw(ArgumentError(
        "Qwen3-VL QKV scratch num_heads must be positive",
    ))
    num_kv_heads > 0 || throw(ArgumentError(
        "Qwen3-VL QKV scratch num_kv_heads must be positive",
    ))
    return (head_dim * num_heads + 2 * head_dim * num_kv_heads) * sizeof(BFloat16)
end

function qwen3_vl_qkv_scratch_bytes(scratch::Qwen3VLQKVScratch)
    return sizeof(scratch.query) + sizeof(scratch.key) + sizeof(scratch.value)
end

function init_qwen3_vl_qkv_scratch(
    prototype::AbstractArray{BFloat16};
    hidden_size::Integer,
    head_dim::Integer,
    num_heads::Integer,
    num_kv_heads::Integer,
)
    hidden_size = Int(hidden_size)
    head_dim = Int(head_dim)
    num_heads = Int(num_heads)
    num_kv_heads = Int(num_kv_heads)
    hidden_size > 0 || throw(ArgumentError(
        "Qwen3-VL QKV scratch hidden_size must be positive",
    ))
    expected = qwen3_vl_qkv_scratch_bytes(;
        head_dim,
        num_heads,
        num_kv_heads,
    )
    scratch = Qwen3VLQKVScratch(
        similar(prototype, BFloat16, head_dim * num_heads, 1, 1),
        similar(prototype, BFloat16, head_dim * num_kv_heads, 1, 1),
        similar(prototype, BFloat16, head_dim * num_kv_heads, 1, 1),
        hidden_size,
        head_dim,
        num_heads,
        num_kv_heads,
    )
    qwen3_vl_qkv_scratch_bytes(scratch) == expected || throw(ArgumentError(
        "Qwen3-VL QKV scratch bytes do not match the formula",
    ))
    return scratch
end

function _qwen3_vl_decode_qkv!(
    scratch::Qwen3VLQKVScratch,
    q_weight,
    k_weight,
    v_weight,
    x,
)
    eltype(x) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL QKV scratch requires BFloat16 values",
    ))
    size(x) == (scratch.hidden_size, 1, 1) || throw(DimensionMismatch(
        "Qwen3-VL QKV scratch supports one token and batch size one",
    ))
    _qwen3_vl_bf16_linear!(scratch.query, q_weight, x)
    _qwen3_vl_bf16_linear!(scratch.key, k_weight, x)
    _qwen3_vl_bf16_linear!(scratch.value, v_weight, x)
    return (
        reshape(scratch.query, scratch.head_dim, scratch.num_heads, 1, 1),
        reshape(scratch.key, scratch.head_dim, scratch.num_kv_heads, 1, 1),
        reshape(scratch.value, scratch.head_dim, scratch.num_kv_heads, 1, 1),
    )
end

function _qwen3_vl_ensure_qkv_scratch!(cache, spec, prototype)
    scratch = cache.qkv_scratch
    if scratch isa Qwen3VLQKVScratch &&
        scratch.hidden_size == spec.hidden_size &&
        scratch.head_dim == spec.head_dim &&
        scratch.num_heads == spec.num_attention_heads &&
        scratch.num_kv_heads == spec.num_key_value_heads &&
        eltype(scratch.query) === eltype(prototype) &&
        get_device(scratch.query) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL QKV scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_qkv_scratch(
        prototype;
        hidden_size=spec.hidden_size,
        head_dim=spec.head_dim,
        num_heads=spec.num_attention_heads,
        num_kv_heads=spec.num_key_value_heads,
    )
    cache.qkv_scratch = scratch
    return scratch
end

# One decode token gathered from the tied embedding. The column write matches
# `NNlib.gather` for a single 1-based id and does not allocate on the GPU.

struct Qwen3VLTokenEmbeddingScratch{O}
    output::O
    hidden_size::Int
end

function qwen3_vl_token_embedding_scratch_bytes(; hidden_size::Integer)
    hidden_size = Int(hidden_size)
    hidden_size > 0 || throw(ArgumentError(
        "Qwen3-VL token embedding scratch hidden_size must be positive",
    ))
    return hidden_size * sizeof(BFloat16)
end

function qwen3_vl_token_embedding_scratch_bytes(scratch::Qwen3VLTokenEmbeddingScratch)
    return sizeof(scratch.output)
end

function init_qwen3_vl_token_embedding_scratch(
    prototype::AbstractArray{BFloat16};
    hidden_size::Integer,
)
    expected = qwen3_vl_token_embedding_scratch_bytes(; hidden_size)
    hidden_size = Int(hidden_size)
    scratch = Qwen3VLTokenEmbeddingScratch(
        similar(prototype, BFloat16, hidden_size, 1, 1),
        hidden_size,
    )
    qwen3_vl_token_embedding_scratch_bytes(scratch) == expected ||
        throw(ArgumentError(
            "Qwen3-VL token embedding scratch bytes do not match the formula",
        ))
    return scratch
end

function _qwen3_vl_decode_token_embedding!(scratch::Qwen3VLTokenEmbeddingScratch, embedding, token_id::Int)
    size(embedding, 1) == scratch.hidden_size || throw(DimensionMismatch(
        "Qwen3-VL token embedding width does not match the scratch",
    ))
    1 <= token_id <= size(embedding, 2) || throw(ArgumentError(
        "Qwen3-VL decode token is outside the vocabulary",
    ))
    scratch.output .= view(embedding, :, token_id)
    return scratch.output
end

function _qwen3_vl_ensure_token_embedding_scratch!(cache, spec, prototype)
    scratch = cache.token_embedding_scratch
    if scratch isa Qwen3VLTokenEmbeddingScratch &&
        scratch.hidden_size == spec.hidden_size &&
        eltype(scratch.output) === eltype(prototype) &&
        get_device(scratch.output) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL token embedding scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_token_embedding_scratch(
        prototype;
        hidden_size=spec.hidden_size,
    )
    cache.token_embedding_scratch = scratch
    return scratch
end

# One-token BF16 mRoPE cos/sin tables. All layers of a decode step share them.
# The host frequencies stay on the same path as `_qwen3_vl_text_mrope`; only
# the device copy is reused.

struct Qwen3VLMropeScratch{C,S}
    cos_values::C
    sin_values::S
    half::Int
end

function qwen3_vl_mrope_scratch_bytes(; head_dim::Integer)
    head_dim = Int(head_dim)
    head_dim > 0 && iseven(head_dim) || throw(ArgumentError(
        "Qwen3-VL mRoPE scratch head_dim must be a positive even integer",
    ))
    return (head_dim ÷ 2) * 2 * sizeof(BFloat16)
end

function qwen3_vl_mrope_scratch_bytes(scratch::Qwen3VLMropeScratch)
    return sizeof(scratch.cos_values) + sizeof(scratch.sin_values)
end

function init_qwen3_vl_mrope_scratch(
    prototype::AbstractArray{BFloat16};
    head_dim::Integer,
)
    expected = qwen3_vl_mrope_scratch_bytes(; head_dim)
    half = Int(head_dim) ÷ 2
    scratch = Qwen3VLMropeScratch(
        similar(prototype, BFloat16, half, 1, 1),
        similar(prototype, BFloat16, half, 1, 1),
        half,
    )
    qwen3_vl_mrope_scratch_bytes(scratch) == expected ||
        throw(ArgumentError(
            "Qwen3-VL mRoPE scratch bytes do not match the formula",
        ))
    return scratch
end

function _qwen3_vl_decode_mrope!(
    scratch::Qwen3VLMropeScratch,
    spec,
    coordinate::Integer,
)
    scratch.half == spec.head_dim ÷ 2 || throw(DimensionMismatch(
        "Qwen3-VL mRoPE scratch does not match head_dim",
    ))
    position_ids = fill(Int(coordinate), 3, 1, 1)
    frequencies = _qwen3_vl_text_mrope_frequencies(spec, position_ids)
    copyto!(scratch.cos_values, BFloat16.(cos.(frequencies)))
    copyto!(scratch.sin_values, BFloat16.(sin.(frequencies)))
    return scratch.cos_values, scratch.sin_values
end

function _qwen3_vl_ensure_mrope_scratch!(cache, spec, prototype)
    scratch = cache.mrope_scratch
    half = spec.head_dim ÷ 2
    if scratch isa Qwen3VLMropeScratch &&
        scratch.half == half &&
        eltype(scratch.cos_values) === eltype(prototype) &&
        get_device(scratch.cos_values) == get_device(prototype)
        return scratch
    end
    eltype(prototype) === BFloat16 || throw(ArgumentError(
        "Qwen3-VL mRoPE scratch requires BFloat16 parameters",
    ))
    scratch = init_qwen3_vl_mrope_scratch(
        prototype;
        head_dim=spec.head_dim,
    )
    cache.mrope_scratch = scratch
    return scratch
end
