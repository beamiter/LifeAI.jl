using Test
using BFloat16s: BFloat16
using Random: MersenneTwister
using LifeAI

function _ch151_video_fixture(; bf16=false)
    fixture = _ch149_fixture(; bf16)
    video_processor = _ch147_rebuild(fixture.processor;
        min_pixels=32 * 32, max_pixels=64 * 64 * 4,
    )
    frame1 = fill(UInt8(20), 32, 32, 3)
    frame2 = fill(UInt8(90), 32, 32, 3)
    frame3 = fill(UInt8(170), 32, 32, 3)
    frame4 = fill(UInt8(240), 32, 32, 3)
    return merge(fixture, (; video_processor, frames=[frame1, frame2, frame3, frame4]))
end

@testset "Chapter 151 — video smart resize, timestamps, and patchify" begin
    vspec = qwen3_vl_video_processor_spec()
    @test vspec.min_pixels == 4_096
    @test vspec.max_pixels == 25_165_824
    @test vspec.patch_size == 16
    @test vspec.temporal_patch_size == 2
    @test vspec.merge_size == 2

    # Smart resize exact boundary & fail-closed checks.
    @test qwen3_vl_video_smart_resize(4, 64, 96; spec=vspec) == (64, 96)
    @test qwen3_vl_video_grid(4, 64, 96; spec=vspec) == (2, 4, 6)
    @test qwen3_vl_video_grid(3, 64, 96; spec=vspec) == (2, 4, 6) # odd frames padded to 4
    @test qwen3_vl_video_token_count((2, 4, 6); spec=vspec) == 12
    @test_throws ArgumentError qwen3_vl_video_smart_resize(1, 64, 64; spec=vspec)
    @test_throws ArgumentError qwen3_vl_video_smart_resize(4, 16, 64; spec=vspec)

    # Timestamps match Transformers `_calculate_timestamps` midpoint formula.
    @test qwen3_vl_video_timestamps([0, 1, 2, 3], 2.0; merge_size=2) == [0.25, 1.25]
    # Odd length pads by repeating last index: [0, 1, 2] -> [0, 1, 2, 2].
    @test qwen3_vl_video_timestamps([0, 1, 2], 2.0; merge_size=2) == [0.25, 1.0]
    @test_throws ArgumentError qwen3_vl_video_timestamps(Int[], 2.0)
    @test_throws ArgumentError qwen3_vl_video_timestamps([0, 1], 0.0)
    @test_throws ArgumentError qwen3_vl_video_timestamps([0, 1], true)

    # Video patchify preserves exact temporal & merge-group ordering.
    f0 = fill(1.0f0, 3, 32, 32)
    f1 = fill(2.0f0, 3, 32, 32)
    f2 = fill(3.0f0, 3, 32, 32)
    f3 = fill(4.0f0, 3, 32, 32)
    patches = qwen3_vl_patchify_video([f0, f1, f2, f3]; spec=vspec)
    # 2 temporal groups * (2*2) spatial patches = 8 columns; 3*2*16*16 = 1536 rows
    @test size(patches) == (1536, 8)
    # First temporal group (columns 1:4) has dt=1 from f0 (1.0) and dt=2 from f1 (2.0).
    @test all(==(1.0f0), patches[1:256, 1:4])
    @test all(==(2.0f0), patches[257:512, 1:4])
    # Second temporal group (columns 5:8) has dt=1 from f2 (3.0) and dt=2 from f3 (4.0).
    @test all(==(3.0f0), patches[1:256, 5:8])
    @test all(==(4.0f0), patches[257:512, 5:8])

    # Odd 3-frame video repeats last frame for dt=2 in group 2.
    patches_odd = qwen3_vl_patchify_video([f0, f1, f2]; spec=vspec)
    @test size(patches_odd) == (1536, 8)
    @test all(==(3.0f0), patches_odd[1:256, 5:8])
    @test all(==(3.0f0), patches_odd[257:512, 5:8])
end

@testset "Chapter 151 — video placeholder expansion and mRoPE layout" begin
    vspec = qwen3_vl_video_processor_spec()
    prompt = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n"
    expanded = qwen3_vl_expand_video_placeholders(
        prompt,
        reshape([2, 2, 2], 3, 1),
        [[0.25, 1.25]];
        spec=vspec,
    )
    @test occursin("<0.2 seconds><|vision_start|><|video_pad|><|vision_end|>", expanded)
    @test occursin("<1.2 seconds><|vision_start|><|video_pad|><|vision_end|>", expanded)

    # Check mRoPE splits video_grid_thw (2, 2, 2) into two (1, 2, 2) spans cleanly.
    fixture = _ch151_video_fixture()
    ids = encode(fixture.tokenizer, expanded; add_special_tokens=false)
    # Without video_grid_thw, legacy image-only error is preserved.
    @test_throws ArgumentError qwen3_vl_rope_layout(ids, nothing; checkpoint=fixture.parameters.checkpoint)

    layout = qwen3_vl_rope_layout(
        ids,
        nothing;
        video_grid_thw=reshape([2, 2, 2], 3, 1),
        checkpoint=fixture.parameters.checkpoint,
    )
    @test count(layout.visual_mask) == 2
    @test size(layout.position_ids) == (3, length(ids), 1)
    # Verify exact delta invariant: max(position_ids) + 1 - length(ids).
    @test only(layout.rope_deltas) == maximum(layout.position_ids) + 1 - length(ids)
end

@testset "Chapter 151 — end-to-end video & mixed image+video generation (F32/BF16, dynamic/static/reuse)" begin
    for bf16 in (false, true)
        fixture = _ch151_video_fixture(; bf16)
        video_messages = Any[(role="user", content=Any[
            (type="video", video=fixture.frames, fps=2.0),
            (type="text", text="What changes?"),
        ])]
        # 4D tensor input matches vector-of-frames input bitwise.
        tensor_4d = Array{UInt8,4}(undef, 4, 32, 32, 3)
        for t in 1:4
            tensor_4d[t, :, :, :] .= fixture.frames[t]
        end
        tensor_messages = Any[(role="user", content=Any[
            (type="video", video=tensor_4d, fps=2.0),
            (type="text", text="What changes?"),
        ])]

        dyn = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer, video_messages;
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], capture_logits=true, cache=:dynamic,
        )
        stat = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer, tensor_messages;
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], capture_logits=true, cache=:static,
        )
        @test dyn.video_grid_thw == reshape([2, 2, 2], 3, 1)
        @test dyn.image_grid_thw === nothing
        @test dyn.original_video_sizes == [(32, 32)]
        @test dyn.video_timestamps == [[0.25, 1.25]]
        @test dyn.generated_ids == stat.generated_ids
        for step in 1:3
            @test dyn.trace[step].logits == stat.trace[step].logits
        end

        # Mixed image + video in one request and static cache reuse across image -> video -> mixed.
        mixed_messages = Any[(role="user", content=Any[
            (type="image", image=fixture.first_image),
            (type="text", text="Then video:"),
            (type="video", video=fixture.frames, fps=4.0),
        ])]
        reusable = init_qwen3_vl_static_kv_cache(fixture.parameters; capacity=512)
        r_img = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer,
            Any[(role="user", content=Any[(type="image", image=fixture.first_image), (type="text", text="Hi")])];
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], cache=reusable,
        )
        r_vid = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer, video_messages;
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], capture_logits=true, cache=reusable,
        )
        @test r_vid.generated_ids == dyn.generated_ids
        @test r_vid.cache === reusable

        r_mix_dyn = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer, mixed_messages;
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], capture_logits=true, cache=:dynamic,
        )
        r_mix_reuse = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer, mixed_messages;
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], capture_logits=true, cache=reusable,
        )
        @test r_mix_dyn.image_grid_thw == reshape([1, 2, 2], 3, 1)
        @test r_mix_dyn.video_grid_thw == reshape([2, 2, 2], 3, 1)
        @test r_mix_dyn.generated_ids == r_mix_reuse.generated_ids
        for step in 1:3
            @test r_mix_dyn.trace[step].logits == r_mix_reuse.trace[step].logits
        end

        # Sampled video generation matches dynamic vs static with fixed uniforms.
        u = [0.15, 0.55, 0.85]
        s_dyn = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer, video_messages;
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], strategy=:sample,
            temperature=0.8, top_k=16, top_p=0.95, sample_uniforms=u, cache=:dynamic,
        )
        s_stat = generate_hf_qwen3_vl(
            fixture.tower, fixture.parameters, fixture.tokenizer, video_messages;
            processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
            max_new_tokens=3, stop_token_ids=Int[], strategy=:sample,
            temperature=0.8, top_k=16, top_p=0.95, sample_uniforms=u, cache=reusable,
        )
        @test s_dyn.generated_ids == s_stat.generated_ids

        # Invalid video requests preserve existing static cache atomically.
        before = _ch147_snapshot(reusable)
        calls = fixture.tower.calls[]
        for bad_messages in (
            Any[(role="user", content=Any[(type="video", video="not-a-video")])],
            Any[(role="user", content=Any[(type="video", video=[fixture.frames[1]])])], # < 2 frames
            Any[(role="user", content=Any[(type="video", video=nothing)])],
            Any[(role="assistant", content=Any[(type="video", video=fixture.frames)])],
            Any[(role="user", content=Any[(type="video", video=fixture.frames, fps=-1.0)])],
        )
            @test_throws ArgumentError generate_hf_qwen3_vl(
                fixture.tower, fixture.parameters, fixture.tokenizer, bad_messages;
                processor_spec=fixture.processor, video_processor_spec=fixture.video_processor,
                max_new_tokens=3, stop_token_ids=Int[], cache=reusable,
            )
            @test fixture.tower.calls[] == calls
            _ch147_unchanged(reusable, before)
        end
    end
end
