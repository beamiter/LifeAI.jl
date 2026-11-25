using Test
using Random
using BFloat16s: BFloat16
using LifeAI

@testset "Chapter 152 — Qwen3-VL video sampling and container demuxing" begin
    # 1. Official frame sampling contract & strict fail-closed validation.
    idx_even = qwen3_vl_sample_frame_indices(16; source_fps=4.0, target_fps=2.0, min_frames=4, max_frames=32)
    @test length(idx_even) == 8
    @test first(idx_even) == 0 && last(idx_even) == 15
    @test issorted(idx_even)

    # Clamped by max_frames and rounded to frame_factor=2.
    idx_cap = qwen3_vl_sample_frame_indices(100; source_fps=10.0, target_fps=5.0, min_frames=4, max_frames=12)
    @test length(idx_cap) == 12
    @test first(idx_cap) == 0 && last(idx_cap) == 99

    # Clamped by min_frames when target_fps would ask for fewer.
    idx_min = qwen3_vl_sample_frame_indices(10; source_fps=10.0, target_fps=1.0, min_frames=4, max_frames=12)
    @test length(idx_min) == 4
    @test first(idx_min) == 0 && last(idx_min) == 9

    # Short video with fewer frames than min_frames still respects available even frame count.
    idx_short = qwen3_vl_sample_frame_indices(2; source_fps=2.0, target_fps=2.0, min_frames=4, max_frames=16)
    @test idx_short == [0, 1]

    # Strict fail-closed rejections.
    @test_throws ArgumentError qwen3_vl_sample_frame_indices(true; source_fps=2.0)
    @test_throws ArgumentError qwen3_vl_sample_frame_indices(1; source_fps=2.0, frame_factor=2)
    @test_throws ArgumentError qwen3_vl_sample_frame_indices(8; source_fps=0.0)
    @test_throws ArgumentError qwen3_vl_sample_frame_indices(8; source_fps=NaN)
    @test_throws ArgumentError qwen3_vl_sample_frame_indices(8; source_fps=2.0, target_fps=-1.0)
    @test_throws ArgumentError qwen3_vl_sample_frame_indices(8; source_fps=2.0, min_frames=6, max_frames=4)
    @test_throws ArgumentError qwen3_vl_sample_frame_indices(8; source_fps=2.0, min_frames=3, max_frames=8)

    # Fail-closed container errors.
    @test_throws ArgumentError qwen3_vl_probe_video_file("/nonexistent/path/video.mp4")
    tmp = mktempdir()
    bad_file = joinpath(tmp, "corrupt.mp4")
    write(bad_file, b"not a valid mp4 container")
    @test_throws ArgumentError qwen3_vl_probe_video_file(bad_file)
    empty_file = joinpath(tmp, "empty.mp4")
    write(empty_file, UInt8[])
    @test_throws ArgumentError qwen3_vl_probe_video_file(empty_file)

    # 2. Deterministic real MP4 encode -> probe -> decode -> process -> generate roundtrip.
    synth_frames = Vector{Array{UInt8,3}}(undef, 8)
    for t in 1:8
        frame = fill(UInt8(24), 64, 64, 3)
        y0 = 6 + (t - 1) * 4
        x0 = 6 + (t - 1) * 4
        frame[y0:(y0 + 15), x0:(x0 + 15), 1] .= UInt8(230)
        frame[y0:(y0 + 15), x0:(x0 + 15), 2] .= UInt8(40 + t * 20)
        frame[y0:(y0 + 15), x0:(x0 + 15), 3] .= UInt8(30)
        synth_frames[t] = frame
    end

    mp4_path = joinpath(tmp, "moving_square.mp4")
    qwen3_vl_encode_video_file(mp4_path, synth_frames; fps=4.0)
    @test isfile(mp4_path)
    @test filesize(mp4_path) > 0

    meta = qwen3_vl_probe_video_file(mp4_path)
    @test meta.width == 64 && meta.height == 64
    @test meta.source_fps ≈ 4.0 atol=1e-3
    @test meta.total_frames == 8

    # Oversized file budget fails closed before spawning ffprobe.
    @test_throws ArgumentError qwen3_vl_probe_video_file(mp4_path; max_file_bytes=16)

    # Oversized decoded byte budget fails closed.
    @test_throws ArgumentError qwen3_vl_decode_video_file(mp4_path; max_decoded_bytes=1024)

    decoded = qwen3_vl_decode_video_file(mp4_path; target_fps=2.0, min_frames=4, max_frames=8)
    @test decoded isa Qwen3VLDecodedVideo
    @test length(decoded.frames) == 4
    @test decoded.total_frames == 8
    @test decoded.frame_indices == [0, 2, 5, 7]
    @test decoded.height == 64 && decoded.width == 64
    @test all(f -> size(f) == (64, 64, 3), decoded.frames)

    # Processing directly from the MP4 path matches processing from the DecodedVideo struct.
    vspec = qwen3_vl_video_processor_spec()
    proc_path = qwen3_vl_process_video(mp4_path; target_fps=2.0, min_frames=4, max_frames=8, spec=vspec)
    proc_dec = qwen3_vl_process_video(decoded; spec=vspec)
    @test proc_path.pixel_values == proc_dec.pixel_values
    @test proc_path.grid_thw == proc_dec.grid_thw
    @test proc_path.timestamps == proc_dec.timestamps
    @test proc_path.grid_thw == reshape([2, 4, 4], 3, 1)

    # 3. End-to-end tiny generation with MP4 file path, sampling, and static cache reuse.
    fixture = _ch151_video_fixture()
    tokenizer = fixture.tokenizer
    vision_f32 = fixture.tower
    text_f32 = fixture.parameters
    vspec_tiny = fixture.video_processor
    pspec_tiny = fixture.processor

    mp4_messages = Any[(role="user", content=Any[
        (type="video", video=mp4_path, target_fps=2.0, min_frames=4, max_frames=8),
        (type="text", text="describe"),
    ])]
    uniforms = Float64[0.25, 0.75, 0.5]
    dyn_sample = generate_hf_qwen3_vl(
        vision_f32, text_f32, tokenizer, mp4_messages;
        max_new_tokens=3, stop_token_ids=Int[], capture_logits=true,
        cache=:dynamic, strategy=:sample, temperature=0.8f0, top_k=10, top_p=0.9f0,
        sample_uniforms=uniforms, capture_distribution=true,
        processor_spec=pspec_tiny, video_processor_spec=vspec_tiny,
    )
    shared_cache = init_qwen3_vl_static_kv_cache(text_f32; capacity=128)
    stat_sample = generate_hf_qwen3_vl(
        vision_f32, text_f32, tokenizer, mp4_messages;
        max_new_tokens=3, stop_token_ids=Int[], capture_logits=true,
        cache=shared_cache, strategy=:sample, temperature=0.8f0, top_k=10, top_p=0.9f0,
        sample_uniforms=uniforms, capture_distribution=true,
        processor_spec=pspec_tiny, video_processor_spec=vspec_tiny,
    )
    @test dyn_sample.generated_ids == stat_sample.generated_ids
    for step in 1:3
        @test dyn_sample.trace[step].logits == stat_sample.trace[step].logits
    end

    # Reuse the exact same static cache for a mixed image + MP4 video request.
    layer1_k_ref = shared_cache.layers[1].keys
    mixed_messages = Any[(role="user", content=Any[
        (type="image", image=synth_frames[1]),
        (type="video", video=mp4_path, target_fps=2.0, min_frames=4, max_frames=8),
        (type="text", text="compare"),
    ])]
    mixed_dyn = generate_hf_qwen3_vl(
        vision_f32, text_f32, tokenizer, mixed_messages;
        max_new_tokens=3, stop_token_ids=Int[], capture_logits=true,
        cache=:dynamic, processor_spec=pspec_tiny, video_processor_spec=vspec_tiny,
    )
    mixed_reuse = generate_hf_qwen3_vl(
        vision_f32, text_f32, tokenizer, mixed_messages;
        max_new_tokens=3, stop_token_ids=Int[], capture_logits=true,
        cache=shared_cache, processor_spec=pspec_tiny, video_processor_spec=vspec_tiny,
    )
    @test shared_cache.layers[1].keys === layer1_k_ref
    @test mixed_dyn.generated_ids == mixed_reuse.generated_ids
    for step in 1:3
        @test mixed_dyn.trace[step].logits == mixed_reuse.trace[step].logits
    end

    rm(tmp; recursive=true)
end
