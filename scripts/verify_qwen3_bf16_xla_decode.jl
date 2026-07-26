#!/usr/bin/env julia

using Reactant
using BFloat16s: BFloat16
using JSON3
using Statistics: mean
using LifeAI
using LifeAI: hf_token_ids

length(ARGS) == 2 || error(
    "usage: julia --project=. scripts/verify_qwen3_bf16_xla_decode.jl MODEL_DIR REFERENCE_DIR",
)
model_dir, reference_dir = ARGS
Reactant.set_default_backend("gpu")

metadata = JSON3.read(read(joinpath(reference_dir, "reference.json"), String))
String(metadata["compute_dtype"]) == "bfloat16" || error(
    "reference was not exported with --compute-dtype bfloat16",
)

tokens = reshape(
    hf_token_ids(Int.(collect(metadata["token_ids_0_based"])); vocab_size=151936),
    :,
    1,
)
expected_greedy = Int.(collect(metadata["greedy_token_ids_0_based"])) .+ 1
greedy_steps = length(expected_greedy)
prompt_len = size(tokens, 1)
max_len = 64

loaded = load_hf_qwen3_model(model_dir; max_seq_len=max_len, weight_dtype=BFloat16)
model = loaded.model
rope = first(values(model.blocks.layers)).attn.rope
cos_table = BFloat16.(rope.cos_cache)
sin_table = BFloat16.(rope.sin_cache)
mask = LifeAI._bf16a_causal_mask(prompt_len, prompt_len)
key_positions = Int32.(collect(1:max_len))

ps_r = Reactant.to_rarray(loaded.parameters)
cos_r = Reactant.to_rarray(cos_table)
sin_r = Reactant.to_rarray(sin_table)
mask_r = Reactant.to_rarray(mask)
key_positions_r = Reactant.to_rarray(key_positions)
key_caches = Tuple(
    Reactant.to_rarray(zeros(BFloat16, model.head_dim, model.num_kv_heads, max_len, 1))
    for _ in 1:model.num_layers
)
value_caches = Tuple(
    Reactant.to_rarray(zeros(BFloat16, model.head_dim, model.num_kv_heads, max_len, 1))
    for _ in 1:model.num_layers
)

prefill_fn(ps, kc, vc, cos_t, sin_t, mask_t) = LifeAI._bf16a_static_prefill(
    model, ps, tokens, kc, vc, cos_t, sin_t, mask_t,
)
decode_fn(ps, token, kc, vc, position, cos_t, sin_t, kp) =
    LifeAI._bf16a_static_decode_step(
        model, ps, token, kc, vc, position, cos_t, sin_t, kp,
    )

prefill_compile = @timed @compile prefill_fn(
    ps_r, key_caches, value_caches, cos_r, sin_r, mask_r,
)
compiled_prefill = prefill_compile.value
prefill_exec = @timed compiled_prefill(
    ps_r, key_caches, value_caches, cos_r, sin_r, mask_r,
)
prefill_logits = Float32.(Array(prefill_exec.value))

token_buffer = Reactant.to_rarray(fill(1, 1))
position_buffer = Reactant.to_rarray(fill(Int32(prompt_len), 1))
decode_compile = @timed @compile decode_fn(
    ps_r, token_buffer, key_caches, value_caches,
    position_buffer, cos_r, sin_r, key_positions_r,
)
compiled_decode = decode_compile.value

greedy_tokens = Int[]
next_token = argmax(vec(prefill_logits[:, end, 1]))
push!(greedy_tokens, next_token)
first_decode_seconds = 0.0
steady_samples = Float64[]
position = prompt_len
while length(greedy_tokens) < greedy_steps
    copyto!(token_buffer, fill(next_token, 1))
    copyto!(position_buffer, fill(Int32(position), 1))
    step_timing = @timed begin
        logits = compiled_decode(
            ps_r, token_buffer, key_caches, value_caches,
            position_buffer, cos_r, sin_r, key_positions_r,
        )
        Float32.(Array(logits))
    end
    global position += 1
    global next_token = argmax(vec(step_timing.value[:, 1, 1]))
    push!(greedy_tokens, next_token)
    if length(greedy_tokens) == 2
        global first_decode_seconds = step_timing.time
    else
        push!(steady_samples, step_timing.time)
    end
end

steady_seconds = isempty(steady_samples) ? NaN : sum(steady_samples) / length(steady_samples)

# Device-resident greedy fast path: argmax inside the executable, one Int
# fetched per token. Stale cache positions beyond the prefill are masked by
# the valid-prefix check, so no cache reset is needed between runs.
greedy_fn(ps, token, kc, vc, pos, cos_t, sin_t, kp) =
    LifeAI._bf16a_static_decode_greedy_step(
        model, ps, token, kc, vc, pos, cos_t, sin_t, kp,
    )
greedy_compile = @timed @compile greedy_fn(
    ps_r, token_buffer, key_caches, value_caches,
    position_buffer, cos_r, sin_r, key_positions_r,
)
compiled_greedy = greedy_compile.value

fast_tokens = Int[argmax(vec(prefill_logits[:, end, 1]))]
copyto!(token_buffer, fill(fast_tokens[1], 1))
copyto!(position_buffer, fill(Int32(prompt_len), 1))
token_state = token_buffer
position_state = position_buffer
fast_samples = Float64[]
for step in 1:(greedy_steps - 1)
    step_timing = @timed begin
        state = compiled_greedy(
            ps_r, token_state, key_caches, value_caches,
            position_state, cos_r, sin_r, key_positions_r,
        )
        global token_state, position_state = state
        Array(token_state)[1]
    end
    push!(fast_tokens, step_timing.value)
    step > 2 && push!(fast_samples, step_timing.time)
end
fast_steady = isempty(fast_samples) ? NaN : sum(fast_samples) / length(fast_samples)

println("model_revision\t", metadata["revision"])
println(
    "dense_variant\t",
    loaded.variant === nothing ? "custom" : String(loaded.variant.variant),
)
println("prefill_compile_seconds\t", prefill_compile.time)
println("decode_compile_seconds\t", decode_compile.time)
println("prefill_exec_seconds\t", prefill_exec.time)
println("first_decode_seconds\t", first_decode_seconds)
println("steady_decode_seconds\t", steady_seconds)
println("steady_tokens_per_second\t", 1 / steady_seconds)
println(
    "greedy_match\t",
    greedy_tokens == expected_greedy,
    "\t",
    join(greedy_tokens .- 1, ','),
)
println("greedy_compile_seconds\t", greedy_compile.time)
println("fast_greedy_match\t", fast_tokens == expected_greedy)
println("fast_steady_decode_seconds\t", fast_steady)
println("fast_steady_tokens_per_second\t", 1 / fast_steady)
