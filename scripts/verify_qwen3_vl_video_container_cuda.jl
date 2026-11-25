#!/usr/bin/env julia
# Chapter 152 real Qwen3-VL-2B-Instruct Blackwell GPU acceptance:
# 1. Real MP4 container encode/probe/decode + direct video path generation (F32 & BF16)
# 2. Reproducible sampled video generation (strategy=:sample, dynamic == static == reused)
# 3. Mixed image + MP4 video generation
# 4. Single preallocated Qwen3VLStaticKVCache pointer-identity reuse across:
#    image -> 8-frame MP4 -> mixed image+MP4 -> 16-frame longer video.

using Test
using JSON3
using SHA
using CUDA
using BFloat16s: BFloat16
using LifeAI

function _make_moving_frames(num_frames::Int; height::Int=96, width::Int=128)
    frames = Vector{Array{UInt8,3}}(undef, num_frames)
    for t in 1:num_frames
        frame = Array{UInt8,3}(undef, height, width, 3)
        @inbounds for r in 1:height, c in 1:width
            frame[r, c, 1] = UInt8(mod((c - 1) * 2 + (t - 1) * 19, 256))
            frame[r, c, 2] = UInt8(mod((r - 1) * 2 + (t - 1) * 29, 256))
            frame[r, c, 3] = UInt8(mod((r + c) + (t - 1) * 37, 256))
        end
        y0 = 10 + clamp((t - 1) * 4, 0, height - 30)
        x0 = 12 + clamp((t - 1) * 6, 0, width - 30)
        frame[y0:(y0 + 21), x0:(x0 + 21), 1] .= UInt8(240)
        frame[y0:(y0 + 21), x0:(x0 + 21), 2] .= UInt8(40)
        frame[y0:(y0 + 21), x0:(x0 + 21), 3] .= UInt8(35)
        frames[t] = frame
    end
    return frames
end

function _cache_pointers(cache::Qwen3VLStaticKVCache)
    return [(pointer(layer.keys), pointer(layer.values)) for layer in cache.layers]
end

function main(args)
    length(args) >= 1 || error("Usage: verify_qwen3_vl_video_container_cuda.jl MODEL_DIR [SUMMARY_JSON]")
    model_dir = abspath(args[1])
    summary_path = length(args) >= 2 ? abspath(args[2]) : nothing

    CUDA.functional() || error("CUDA is not functional")
    vspec = load_hf_qwen3_vl_video_processor_config(joinpath(model_dir, "video_preprocessor_config.json"))
    pspec = load_hf_qwen3_vl_processor_config(joinpath(model_dir, "preprocessor_config.json"))
    tokenizer = load_hf_qwen3_vl_tokenizer(model_dir)

    tmp = mktempdir()
    frames_8 = _make_moving_frames(8; height=96, width=128)
    frames_16 = _make_moving_frames(16; height=96, width=128)
    mp4_8 = joinpath(tmp, "clip_8f_4fps.mp4")
    mp4_16 = joinpath(tmp, "clip_16f_4fps.mp4")
    qwen3_vl_encode_video_file(mp4_8, frames_8; fps=4.0)
    qwen3_vl_encode_video_file(mp4_16, frames_16; fps=4.0)

    meta_8 = qwen3_vl_probe_video_file(mp4_8)
    meta_16 = qwen3_vl_probe_video_file(mp4_16)
    @test meta_8.total_frames == 8 && meta_8.width == 128 && meta_8.height == 96
    @test meta_16.total_frames == 16 && meta_16.width == 128 && meta_16.height == 96

    msg_image = Any[(role="user", content=Any[
        (type="image", image=frames_8[1]),
        (type="text", text="What color is the bright square?"),
    ])]
    msg_mp4 = Any[(role="user", content=Any[
        (type="video", video=mp4_8, target_fps=2.0, min_frames=4, max_frames=8),
        (type="text", text="Describe the motion in this short clip."),
    ])]
    msg_mixed = Any[(role="user", content=Any[
        (type="image", image=frames_8[1]),
        (type="video", video=mp4_8, target_fps=2.0, min_frames=4, max_frames=8),
        (type="text", text="Does the square in the image also appear in the video?"),
    ])]
    msg_16f = Any[(role="user", content=Any[
        (type="video", video=mp4_16, target_fps=4.0, min_frames=4, max_frames=16),
        (type="text", text="Summarize the moving square trajectory."),
    ])]

    uniforms = Float64[0.13, 0.47, 0.81, 0.29, 0.63, 0.91]
    n_steps = length(uniforms)
    results = Dict{String,Any}()

    for (label, dtype) in (("float32", Float32), ("bfloat16", BFloat16))
        vision = load_hf_qwen3_vl_vision_parameters(model_dir; target_dtype=dtype, to_device=CUDA.cu)
        text = load_hf_qwen3_vl_text_parameters(model_dir; target_dtype=dtype, to_device=CUDA.cu)

        # Allocate ONE shared static cache big enough for all four workloads (including 16-frame video).
        shared_cache = init_qwen3_vl_static_kv_cache(text; capacity=256)
        initial_ptrs = _cache_pointers(shared_cache)

        # (a) Image request on shared cache
        img_dyn = generate_hf_qwen3_vl(vision, text, tokenizer, msg_image;
            max_new_tokens=n_steps, capture_logits=true, cache=:dynamic,
            processor_spec=pspec, video_processor_spec=vspec)
        img_reuse = generate_hf_qwen3_vl(vision, text, tokenizer, msg_image;
            max_new_tokens=n_steps, capture_logits=true, cache=shared_cache,
            processor_spec=pspec, video_processor_spec=vspec)
        @test _cache_pointers(shared_cache) == initial_ptrs
        @test img_dyn.generated_ids == img_reuse.generated_ids
        for s in 1:n_steps
            @test Array(img_dyn.trace[s].logits) == Array(img_reuse.trace[s].logits)
        end

        # (b) Real MP4 container greedy + reproducible sampling on shared cache
        mp4_dyn = generate_hf_qwen3_vl(vision, text, tokenizer, msg_mp4;
            max_new_tokens=n_steps, capture_logits=true, cache=:dynamic,
            processor_spec=pspec, video_processor_spec=vspec)
        mp4_reuse = generate_hf_qwen3_vl(vision, text, tokenizer, msg_mp4;
            max_new_tokens=n_steps, capture_logits=true, cache=shared_cache,
            processor_spec=pspec, video_processor_spec=vspec)
        @test _cache_pointers(shared_cache) == initial_ptrs
        @test mp4_dyn.generated_ids == mp4_reuse.generated_ids
        for s in 1:n_steps
            @test Array(mp4_dyn.trace[s].logits) == Array(mp4_reuse.trace[s].logits)
        end

        samp_dyn = generate_hf_qwen3_vl(vision, text, tokenizer, msg_mp4;
            max_new_tokens=n_steps, capture_logits=true, cache=:dynamic,
            strategy=:sample, temperature=0.8f0, top_k=20, top_p=0.9f0,
            sample_uniforms=uniforms, capture_distribution=true,
            processor_spec=pspec, video_processor_spec=vspec)
        samp_reuse = generate_hf_qwen3_vl(vision, text, tokenizer, msg_mp4;
            max_new_tokens=n_steps, capture_logits=true, cache=shared_cache,
            strategy=:sample, temperature=0.8f0, top_k=20, top_p=0.9f0,
            sample_uniforms=uniforms, capture_distribution=true,
            processor_spec=pspec, video_processor_spec=vspec)
        @test _cache_pointers(shared_cache) == initial_ptrs
        @test samp_dyn.generated_ids == samp_reuse.generated_ids
        for s in 1:n_steps
            @test Array(samp_dyn.trace[s].logits) == Array(samp_reuse.trace[s].logits)
        end

        # (c) Mixed image + MP4 video on shared cache
        mix_dyn = generate_hf_qwen3_vl(vision, text, tokenizer, msg_mixed;
            max_new_tokens=n_steps, capture_logits=true, cache=:dynamic,
            processor_spec=pspec, video_processor_spec=vspec)
        mix_reuse = generate_hf_qwen3_vl(vision, text, tokenizer, msg_mixed;
            max_new_tokens=n_steps, capture_logits=true, cache=shared_cache,
            processor_spec=pspec, video_processor_spec=vspec)
        @test _cache_pointers(shared_cache) == initial_ptrs
        @test mix_dyn.generated_ids == mix_reuse.generated_ids
        for s in 1:n_steps
            @test Array(mix_dyn.trace[s].logits) == Array(mix_reuse.trace[s].logits)
        end

        # (d) 16-frame MP4 video (grid_t=8, 96 visual tokens) on the exact same shared cache
        v16_dyn = generate_hf_qwen3_vl(vision, text, tokenizer, msg_16f;
            max_new_tokens=n_steps, capture_logits=true, cache=:dynamic,
            processor_spec=pspec, video_processor_spec=vspec)
        v16_reuse = generate_hf_qwen3_vl(vision, text, tokenizer, msg_16f;
            max_new_tokens=n_steps, capture_logits=true, cache=shared_cache,
            processor_spec=pspec, video_processor_spec=vspec)
        @test _cache_pointers(shared_cache) == initial_ptrs
        @test v16_dyn.generated_ids == v16_reuse.generated_ids
        for s in 1:n_steps
            @test Array(v16_dyn.trace[s].logits) == Array(v16_reuse.trace[s].logits)
        end

        results[label] = (;
            shared_cache_capacity=256,
            kv_pointer_identity_preserved=true,
            image_completion=img_reuse.completion,
            mp4_8f_grid_thw=collect(vec(mp4_reuse.video_grid_thw)),
            mp4_8f_greedy_completion=mp4_reuse.completion,
            mp4_8f_sampled_completion=samp_reuse.completion,
            mixed_image_mp4_prompt_len=length(mix_reuse.prompt_ids),
            mixed_image_mp4_completion=mix_reuse.completion,
            mp4_16f_grid_thw=collect(vec(v16_reuse.video_grid_thw)),
            mp4_16f_prompt_len=length(v16_reuse.prompt_ids),
            mp4_16f_completion=v16_reuse.completion,
            all_dynamic_vs_reused_bitwise_exact=true,
        )

        vision = nothing; text = nothing; shared_cache = nothing
        GC.gc(true); CUDA.reclaim()
    end

    summary = (;
        device=CUDA.name(CUDA.device()),
        model_dir,
        mp4_8f_bytes=filesize(mp4_8),
        mp4_16f_bytes=filesize(mp4_16),
        results,
    )
    rm(tmp; recursive=true)
    payload = JSON3.write(summary)
    println(payload)
    if summary_path !== nothing
        mkpath(dirname(summary_path))
        write(summary_path, payload)
    end
end

main(ARGS)
