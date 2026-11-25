#!/usr/bin/env julia
# Verify Qwen3-VL video preprocessing, mRoPE, and generation on CUDA against official Transformers oracle.
using Test
using JSON3
using SHA
using CUDA
using BFloat16s: BFloat16
using LifeAI

function _read_video_oracle_safetensors(path::AbstractString)
    bytes = read(path)
    length(bytes) >= 8 || error("truncated oracle safetensors header")
    header_size = Int(ltoh(reinterpret(UInt64, bytes[1:8])[1]))
    0 < header_size <= length(bytes) - 8 || error("invalid oracle header size")
    header = JSON3.read(String(bytes[9:8 + header_size]))
    base = 8 + header_size
    result = Dict{String,Any}()
    for (raw_name, entry) in pairs(header)
        name = String(raw_name)
        name == "__metadata__" && continue
        dtype = String(entry["dtype"])
        T = dtype == "U8" ? UInt8 : dtype == "I64" ? Int64 :
            dtype == "F32" ? Float32 : dtype == "BF16" ? BFloat16 :
            error("unsupported oracle dtype $dtype")
        shape = Int.(collect(entry["shape"]))
        first, last = Int.(collect(entry["data_offsets"]))
        payload = bytes[base + first + 1:base + last]
        values = copy(reinterpret(T, payload))
        result[name] = LifeAI._semantic_array(values, shape)
    end
    return result, bytes2hex(sha256(bytes))
end

function main(args)
    length(args) >= 2 || error("Usage: verify_qwen3_vl_video_cuda.jl MODEL_DIR ORACLE_DIR [SUMMARY_JSON]")
    model_dir = abspath(args[1])
    oracle_dir = abspath(args[2])
    summary_path = length(args) >= 3 ? abspath(args[3]) : nothing

    CUDA.functional() || error("CUDA is not functional")
    meta = JSON3.read(read(joinpath(oracle_dir, "reference.json"), String))
    tensors, tensor_sha = _read_video_oracle_safetensors(joinpath(oracle_dir, "reference.safetensors"))
    @test tensor_sha == String(meta.tensor_sha256)

    raw_video = tensors["video_frames"]
    expected_pixels = tensors["pixel_values_videos"]
    expected_grid = tensors["video_grid_thw"]
    expected_ids = vec(Int.(tensors["input_ids"])) .+ 1
    expected_pos = tensors["position_ids"]
    expected_deltas = tensors["rope_deltas"]
    expected_greedy = Int.(collect(meta.greedy_ids_zero_based)) .+ 1
    fps = Float64(meta.fps)
    n_steps = length(expected_greedy)

    vspec = load_hf_qwen3_vl_video_processor_config(joinpath(model_dir, "video_preprocessor_config.json"))
    tokenizer = load_hf_qwen3_vl_tokenizer(model_dir)

    # 1. Exact video preprocessing & pixel parity.
    processed = qwen3_vl_process_video(raw_video; fps, layout=:thwc, spec=vspec)
    # Transformers stores pixel_values_videos as (patches, 1536); LifeAI stores (1536, patches).
    pixel_max_err = maximum(abs.(processed.pixel_values .- permutedims(expected_pixels, (2, 1))))
    @test pixel_max_err == 0.0f0
    @test processed.grid_thw == permutedims(Int.(expected_grid), (2, 1))

    # 2. Exact chat template, timestamp placeholder expansion, token IDs, and mRoPE parity.
    messages = Any[(role="user", content=Any[
        (type="video", video=raw_video, fps=fps),
        (type="text", text="Briefly describe the moving object in this video."),
    ])]
    prompt = apply_qwen3_vl_chat_template(tokenizer, messages; add_generation_prompt=true)
    expanded = qwen3_vl_expand_video_placeholders(prompt, processed.grid_thw, [processed.timestamps]; spec=vspec)
    ids = encode(tokenizer, expanded; add_special_tokens=false)
    @test ids == expected_ids

    rope = qwen3_vl_rope_layout(ids, nothing; video_grid_thw=processed.grid_thw)
    # expected_pos from HF is (3, batch=1, seq); LifeAI is (3, seq, batch=1).
    @test rope.position_ids == permutedims(Int.(expected_pos), (1, 3, 2))
    @test rope.rope_deltas == Int.(expected_deltas)

    # 3. Real Qwen3-VL-2B Float32 CUDA strict gate against Transformers oracle.
    vision_f32 = load_hf_qwen3_vl_vision_parameters(model_dir; target_dtype=Float32, to_device=CUDA.cu)
    text_f32 = load_hf_qwen3_vl_text_parameters(model_dir; target_dtype=Float32, to_device=CUDA.cu)

    dyn_f32 = generate_hf_qwen3_vl(
        vision_f32, text_f32, tokenizer, messages;
        max_new_tokens=n_steps, capture_logits=true, cache=:dynamic,
        video_processor_spec=vspec,
    )
    stat_f32 = generate_hf_qwen3_vl(
        vision_f32, text_f32, tokenizer, messages;
        max_new_tokens=n_steps, capture_logits=true, cache=:static,
        video_processor_spec=vspec,
    )
    reusable_f32 = init_qwen3_vl_static_kv_cache(text_f32; capacity=length(ids) + n_steps)
    reuse_f32 = generate_hf_qwen3_vl(
        vision_f32, text_f32, tokenizer, messages;
        max_new_tokens=n_steps, capture_logits=true, cache=reusable_f32,
        video_processor_spec=vspec,
    )

    @test dyn_f32.generated_ids == expected_greedy
    @test stat_f32.generated_ids == expected_greedy
    @test reuse_f32.generated_ids == expected_greedy

    f32_step_max_errors = Float64[]
    for step in 1:n_steps
        ref_logits = tensors["step$(step - 1)_logits"]
        dyn_logits = Array(dyn_f32.trace[step].logits)
        stat_logits = Array(stat_f32.trace[step].logits)
        reuse_logits = Array(reuse_f32.trace[step].logits)
        @test dyn_logits == stat_logits
        @test dyn_logits == reuse_logits
        err = Float64(maximum(abs.(dyn_logits .- ref_logits)))
        push!(f32_step_max_errors, err)
        @test err < 2e-4
    end

    # Release F32 weights before loading BF16 weights.
    vision_f32 = nothing; text_f32 = nothing; dyn_f32 = nothing; stat_f32 = nothing; reuse_f32 = nothing; reusable_f32 = nothing
    GC.gc(true); CUDA.reclaim()

    # 4. Real Qwen3-VL-2B BFloat16 CUDA internal dynamic == static == reused exact parity.
    vision_bf16 = load_hf_qwen3_vl_vision_parameters(model_dir; target_dtype=BFloat16, to_device=CUDA.cu)
    text_bf16 = load_hf_qwen3_vl_text_parameters(model_dir; target_dtype=BFloat16, to_device=CUDA.cu)

    dyn_bf16 = generate_hf_qwen3_vl(
        vision_bf16, text_bf16, tokenizer, messages;
        max_new_tokens=n_steps, capture_logits=true, cache=:dynamic,
        video_processor_spec=vspec,
    )
    stat_bf16 = generate_hf_qwen3_vl(
        vision_bf16, text_bf16, tokenizer, messages;
        max_new_tokens=n_steps, capture_logits=true, cache=:static,
        video_processor_spec=vspec,
    )
    reusable_bf16 = init_qwen3_vl_static_kv_cache(text_bf16; capacity=length(ids) + n_steps)
    reuse_bf16 = generate_hf_qwen3_vl(
        vision_bf16, text_bf16, tokenizer, messages;
        max_new_tokens=n_steps, capture_logits=true, cache=reusable_bf16,
        video_processor_spec=vspec,
    )
    @test dyn_bf16.generated_ids == stat_bf16.generated_ids
    @test dyn_bf16.generated_ids == reuse_bf16.generated_ids
    for step in 1:n_steps
        @test Array(dyn_bf16.trace[step].logits) == Array(stat_bf16.trace[step].logits)
        @test Array(dyn_bf16.trace[step].logits) == Array(reuse_bf16.trace[step].logits)
    end

    summary = (;
        device=CUDA.name(CUDA.device()),
        model_dir,
        prompt_length=length(ids),
        video_grid_thw=collect(vec(processed.grid_thw)),
        rope_delta=only(rope.rope_deltas),
        pixel_max_abs_error=Float64(pixel_max_err),
        f32_greedy_exact=true,
        f32_max_logits_abs_error=maximum(f32_step_max_errors),
        f32_step_max_errors,
        bf16_dynamic_static_reuse_exact=true,
        bf16_generated_ids_zero_based=dyn_bf16.generated_ids .- 1,
        f32_completion=meta.completion,
        bf16_completion=dyn_bf16.completion,
    )
    println(JSON3.write(summary))
    if summary_path !== nothing
        mkpath(dirname(summary_path))
        write(summary_path, JSON3.write(summary))
    end
end

main(ARGS)
