using SHA: sha256
using Test
import LifeAI
using LifeAI: apply_qwen3_vl_chat_template,
    qwen3_vl_checkpoint_spec,
    qwen3_vl_expand_image_placeholders,
    qwen3_vl_rope_layout

function _ch44_replace_spec_field(value, name::Symbol, replacement)
    names = fieldnames(typeof(value))
    index = findfirst(==(name), names)
    index === nothing && error("unknown specification field: $name")
    fields = Tuple(getfield(value, field) for field in names)
    return typeof(value)(Base.setindex(fields, replacement, index)...)
end

function _ch44_layout_captured_error(thunk)
    try
        thunk()
    catch error
        return error
    end
    error("expected Qwen3-VL layout call to fail")
end

@testset "Chapter 44 — Qwen3-VL token ids require integers" begin
    @test qwen3_vl_rope_layout(Int32[1, 2]).position_ids[:, :, 1] == [
        0 1
        0 1
        0 1
    ]
    @test qwen3_vl_rope_layout(BigInt[1, 2]).rope_deltas == reshape(Int[0], 1, 1)

    overflow = big(typemax(Int)) + 1
    for invalid in (
        Float64[1, 2],
        Bool[true, false],
        Char['\x01', '\x02'],
        BigInt[1, overflow],
    )
        @test_throws ArgumentError qwen3_vl_rope_layout(invalid)
    end
end

@testset "Chapter 44 — mRoPE checkpoint bounds are preflighted" begin
    base = qwen3_vl_checkpoint_spec()
    for name in (
        :image_token_id,
        :video_token_id,
        :vision_start_token_id,
        :vision_end_token_id,
    )
        checkpoint = _ch44_replace_spec_field(
            base,
            name,
            base.text.vocab_size,
        )
        failure = _ch44_layout_captured_error() do
            qwen3_vl_rope_layout(Int[1]; checkpoint)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL checkpoint $name must be in " *
            "0:$(base.text.vocab_size - 1)"
    end

    invalid_checkpoint = _ch44_replace_spec_field(
        base,
        :image_token_id,
        base.text.vocab_size,
    )
    priority_failure = _ch44_layout_captured_error() do
        qwen3_vl_rope_layout(Bool[true]; checkpoint=invalid_checkpoint)
    end
    @test priority_failure isa ArgumentError
    @test occursin("image_token_id", sprint(showerror, priority_failure))

    short_text = _ch44_replace_spec_field(
        base.text,
        :max_position_embeddings,
        1,
    )
    short_checkpoint = _ch44_replace_spec_field(base, :text, short_text)
    context_failure = _ch44_layout_captured_error() do
        qwen3_vl_rope_layout(Int[1, 2]; checkpoint=short_checkpoint)
    end
    @test context_failure isa ArgumentError
    @test sprint(showerror, context_failure) ==
        "ArgumentError: Qwen3-VL input length 2 exceeds checkpoint context 1"
end

function _ch44_occurrences(text::AbstractString, needle::AbstractString)
    return length(split(String(text), String(needle); keepempty=true)) - 1
end

function _ch44_position_sha256(position_ids)
    # HF stores (axis, batch, sequence) in C order. LifeAI stores
    # (axis, sequence, batch), so this permutation gives the same byte stream.
    canonical = permutedims(Int64.(position_ids), (2, 3, 1))
    return bytes2hex(sha256(collect(reinterpret(UInt8, vec(canonical)))))
end

@testset "Chapter 44 — frozen content-list chat rendering" begin
    messages = [(
        role="user",
        content=Any[
            (type="image", image="fixture.png"),
            (type="text", text="Describe."),
        ],
    )]
    rendered = apply_qwen3_vl_chat_template(messages)
    @test rendered ==
        "<|im_start|>user\n" *
        "<|vision_start|><|image_pad|><|vision_end|>Describe." *
        "<|im_end|>\n<|im_start|>assistant\n"
    @test bytes2hex(sha256(codeunits(rendered))) ==
        "7a50d10ccb53359de53e3e9b032c39b15fd3abbfed51ec844117c3b93da07271"

    @test apply_qwen3_vl_chat_template(messages; add_vision_id=true) ==
        "<|im_start|>user\nPicture 1: " *
        "<|vision_start|><|image_pad|><|vision_end|>Describe." *
        "<|im_end|>\n<|im_start|>assistant\n"
    @test apply_qwen3_vl_chat_template([
        (role="system", content="System."),
        (role="user", content="Question."),
    ]; add_generation_prompt=false) ==
        "<|im_start|>system\nSystem.<|im_end|>\n" *
        "<|im_start|>user\nQuestion.<|im_end|>\n"

    # The official Jinja template tests key presence, not payload truthiness.
    # A null image key still denotes an image placeholder, while null text is
    # rejected instead of silently rendering Python/Jinja's `None` spelling.
    @test occursin(
        "<|vision_start|><|image_pad|><|vision_end|>",
        apply_qwen3_vl_chat_template([
            Dict("role" => "user", "content" => [Dict("image" => nothing)]),
        ]),
    )
    @test_throws ArgumentError apply_qwen3_vl_chat_template([
        Dict("role" => "user", "content" => [Dict("text" => nothing)]),
    ])
end

@testset "Chapter 44 — stringified tool arguments are canonical JSON objects" begin
    call(arguments) = [(
        role="assistant",
        content="",
        tool_calls=[(; name="add", arguments)],
    )]

    @test apply_qwen3_vl_chat_template(
        call("{ \"a\": 1, \"b\": 2, \"a\": 3 }");
        add_generation_prompt=false,
    ) == "<|im_start|>assistant\n" *
         "<tool_call>\n{\"name\": \"add\", \"arguments\": {\"a\": 3, \"b\": 2}}\n" *
         "</tool_call><|im_end|>\n"

    @test_throws ArgumentError apply_qwen3_vl_chat_template(
        call("not JSON");
        add_generation_prompt=false,
    )
    @test_throws ArgumentError apply_qwen3_vl_chat_template(
        call("[1, 2]");
        add_generation_prompt=false,
    )
end

@testset "Chapter 44 — per-image placeholder expansion" begin
    image_pad = "<|image_pad|>"
    prompt = "A$(image_pad)B$(image_pad)C"
    grids = Int[1 1; 16 12; 16 24]
    expanded = qwen3_vl_expand_image_placeholders(prompt, grids)
    expected = "A" * repeat(image_pad, 64) * "B" *
        repeat(image_pad, 72) * "C"

    @test expanded == expected
    @test _ch44_occurrences(expanded, image_pad) == 136
    @test startswith(expanded, "A" * repeat(image_pad, 64) * "B")
    @test endswith(expanded, repeat(image_pad, 72) * "C")

    single_tuple = qwen3_vl_expand_image_placeholders(
        "A$(image_pad)B",
        (1, 16, 16),
    )
    @test single_tuple == "A" * repeat(image_pad, 64) * "B"
    single_vector = qwen3_vl_expand_image_placeholders(
        "A$(image_pad)B",
        Int[1, 12, 24],
    )
    @test single_vector == "A" * repeat(image_pad, 72) * "B"
    @test qwen3_vl_expand_image_placeholders(
        prompt,
        ((1, 16, 16), Int[1, 12, 24]),
    ) == expected
    @test qwen3_vl_expand_image_placeholders(
        "no image",
        NTuple{3,Int}[],
    ) == "no image"
    @test qwen3_vl_expand_image_placeholders("no image", nothing) == "no image"
    @test_throws ArgumentError qwen3_vl_expand_image_placeholders(
        image_pad,
        nothing,
    )
    for invalid_grids in (1, :grid, Dict(:grid => (1, 16, 16)))
        failure = _ch44_layout_captured_error() do
            qwen3_vl_expand_image_placeholders("no image", invalid_grids)
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL image grids must be a matrix, one grid, " *
            "or a grid collection"
    end

    # Equal total counts are insufficient: every original sentinel consumes
    # exactly its corresponding grid, in prompt order.
    reversed = qwen3_vl_expand_image_placeholders(prompt, grids[:, end:-1:1])
    @test reversed == "A" * repeat(image_pad, 72) * "B" *
        repeat(image_pad, 64) * "C"
    @test reversed != expanded

    @test_throws ArgumentError qwen3_vl_expand_image_placeholders(
        prompt,
        grids[:, 1:1],
    )
    @test_throws ArgumentError qwen3_vl_expand_image_placeholders(
        "no image",
        grids[:, 1:1],
    )
    @test_throws DimensionMismatch qwen3_vl_expand_image_placeholders(
        image_pad,
        ones(Int, 2, 1),
    )
end

@testset "Chapter 44 — hand mRoPE image oracle and strict sentinels" begin
    checkpoint = qwen3_vl_checkpoint_spec()
    vision_start = checkpoint.vision_start_token_id + 1
    vision_end = checkpoint.vision_end_token_id + 1
    image_token = checkpoint.image_token_id + 1
    grid = reshape(Int[1, 4, 4], 3, 1)

    tokens = vcat(Int[11, vision_start], fill(image_token, 4), Int[vision_end, 12])
    layout = qwen3_vl_rope_layout(tokens, grid)
    @test size(layout.position_ids) == (3, 8, 1)
    @test layout.position_ids[:, :, 1] == Int[
        0 1 2 2 2 2 4 5
        0 1 2 2 3 3 4 5
        0 1 2 3 2 3 4 5
    ]
    @test layout.rope_deltas == reshape(Int[-2], 1, 1)
    @test layout.visual_mask[:, 1] ==
        Bool[false, false, true, true, true, true, false, false]
    @test all(layout.attention_mask)

    tuple_layout = qwen3_vl_rope_layout(tokens, (1, 4, 4))
    @test tuple_layout.position_ids == layout.position_ids
    @test tuple_layout.rope_deltas == layout.rope_deltas
    @test tuple_layout.visual_mask == layout.visual_mask
    @test tuple_layout.attention_mask == layout.attention_mask

    wide_layout = qwen3_vl_rope_layout(
        tokens,
        reshape(BigInt[1, 4, 4], 3, 1),
    )
    @test wide_layout.position_ids == layout.position_ids
    @test wide_layout.rope_deltas == layout.rope_deltas
    @test wide_layout.visual_mask == layout.visual_mask
    @test wide_layout.attention_mask == layout.attention_mask

    overflow_integer = big(typemax(Int)) + 1
    invalid_matrix_grids = (
        reshape(Any[true, 4, 4], 3, 1),
        reshape(Float64[1, 4, 4], 3, 1),
        reshape(BigInt[1, 4, overflow_integer], 3, 1),
        reshape(Int[1, 0, 4], 3, 1),
        reshape(Int[1, -4, 4], 3, 1),
    )
    for invalid_grid in invalid_matrix_grids
        grid_error = try
            qwen3_vl_rope_layout(tokens, invalid_grid)
            nothing
        catch caught
            caught
        end
        @test grid_error isa ArgumentError
        @test grid_error isa Exception &&
            occursin("image grid", sprint(showerror, grid_error))
    end

    @test LifeAI._qwen3_vl_visual_span(1, 2, 2, 3, 8) == (;
        visual_length=4,
        last_visual=6,
    )
    large_even = typemax(Int) - 1
    for merged_dimensions in (
        (large_even ÷ 2, 4),
        (large_even ÷ 2, 1),
    )
        span_error = try
            LifeAI._qwen3_vl_visual_span(
                1,
                merged_dimensions[1],
                merged_dimensions[2],
                3,
                5,
            )
            nothing
        catch caught
            caught
        end
        @test span_error isa ArgumentError
        @test span_error isa Exception && occursin(
            "image placeholder run exceeds the prompt",
            sprint(showerror, span_error),
        )
    end
    oversized_grid_error = try
        qwen3_vl_rope_layout(tokens, reshape(Int[1, 6, 6], 3, 1))
        nothing
    catch caught
        caught
    end
    @test oversized_grid_error isa ArgumentError
    @test oversized_grid_error isa Exception && occursin(
        "image placeholder run exceeds the prompt",
        sprint(showerror, oversized_grid_error),
    )

    missing_end = vcat(Int[11, vision_start], fill(image_token, 4), Int[12])
    extra_pad = vcat(
        Int[11, vision_start],
        fill(image_token, 5),
        Int[vision_end, 12],
    )
    short_run = vcat(
        Int[11, vision_start],
        fill(image_token, 3),
        Int[vision_end, 12],
    )
    @test_throws ArgumentError qwen3_vl_rope_layout(missing_end, grid)
    @test_throws ArgumentError qwen3_vl_rope_layout(extra_pad, grid)
    @test_throws ArgumentError qwen3_vl_rope_layout(short_run, grid)
    @test_throws ArgumentError qwen3_vl_rope_layout(Int[11, image_token, 12])
    @test_throws ArgumentError qwen3_vl_rope_layout(Int[11, 12], grid)
    @test_throws ArgumentError qwen3_vl_rope_layout(tokens)
end

@testset "Chapter 44 — two-image mRoPE base progression" begin
    checkpoint = qwen3_vl_checkpoint_spec()
    vision_start = checkpoint.vision_start_token_id + 1
    vision_end = checkpoint.vision_end_token_id + 1
    image_token = checkpoint.image_token_id + 1
    tokens = vcat(
        Int[11, vision_start],
        fill(image_token, 4),
        Int[vision_end, 12, vision_start],
        fill(image_token, 8),
        Int[vision_end, 13],
    )
    grids = BigInt[
        1 1
        4 4
        4 8
    ]
    layout = qwen3_vl_rope_layout(tokens, grids)

    @test size(layout.position_ids) == (3, 19, 1)
    @test layout.position_ids[:, 3, 1] == [2, 2, 2]
    @test layout.position_ids[:, 6, 1] == [2, 3, 3]
    @test layout.position_ids[:, 7, 1] == [4, 4, 4]
    @test layout.position_ids[:, 8, 1] == [5, 5, 5]
    @test layout.position_ids[:, 9, 1] == [6, 6, 6]
    @test layout.position_ids[:, 10, 1] == [7, 7, 7]
    @test layout.position_ids[:, 17, 1] == [7, 8, 10]
    @test layout.position_ids[:, 18, 1] == [11, 11, 11]
    @test layout.position_ids[:, 19, 1] == [12, 12, 12]
    @test layout.rope_deltas == reshape(Int[-6], 1, 1)
    @test findall(view(layout.visual_mask, :, 1)) == vcat(3:6, 10:17)
    @test all(layout.attention_mask)
end

@testset "Chapter 44 — 256-patch-grid visual mRoPE oracle" begin
    checkpoint = qwen3_vl_checkpoint_spec()
    tokens = fill(11, 76)
    tokens[4] = checkpoint.vision_start_token_id + 1
    tokens[5:68] .= checkpoint.image_token_id + 1
    tokens[69] = checkpoint.vision_end_token_id + 1
    grid = reshape(Int[1, 16, 16], 3, 1)
    layout = qwen3_vl_rope_layout(tokens, grid)

    @test size(layout.position_ids) == (3, 76, 1)
    @test layout.position_ids[:, 1, 1] == [0, 0, 0]
    @test layout.position_ids[:, 4, 1] == [3, 3, 3]
    @test layout.position_ids[:, 5, 1] == [4, 4, 4]
    @test layout.position_ids[:, 68, 1] == [4, 11, 11]
    @test layout.position_ids[:, 69, 1] == [12, 12, 12]
    @test layout.position_ids[:, 76, 1] == [19, 19, 19]
    @test maximum(layout.position_ids) == 19
    @test layout.rope_deltas == reshape(Int[-56], 1, 1)
    @test count(layout.visual_mask) == 64
    @test findall(view(layout.visual_mask, :, 1)) == collect(5:68)
    @test all(layout.attention_mask)
    @test _ch44_position_sha256(layout.position_ids) ==
        "59dddb6103ddacbc299aafe24d49e57880c7d9d2aced135f92b2bb0162d317f4"
end

@testset "Chapter 44 — padding, batch, and video boundary" begin
    checkpoint = qwen3_vl_checkpoint_spec()
    tokens = Int[1, 1, 11, 12, 13, 14, 15, 16]
    attention = Bool[false, false, true, true, true, true, true, true]
    layout = qwen3_vl_rope_layout(tokens; attention_mask=attention)
    expected = Int[1, 1, 0, 1, 2, 3, 4, 5]

    @test layout.position_ids[:, :, 1] == repeat(reshape(expected, 1, :), 3, 1)
    @test layout.rope_deltas == reshape(Int[-2], 1, 1)
    @test layout.attention_mask[:, 1] == attention
    @test !any(layout.visual_mask)

    singleton = qwen3_vl_rope_layout(
        Int[1, 11];
        attention_mask=Bool[false, true],
    )
    @test singleton.position_ids[:, :, 1] ==
        repeat(reshape(Int[1, 0], 1, :), 3, 1)
    @test singleton.rope_deltas == reshape(Int[-1], 1, 1)

    integer_mask = qwen3_vl_rope_layout(
        Int[1, 11];
        attention_mask=Int8[0, 1],
    )
    @test integer_mask.attention_mask == reshape(Bool[false, true], 2, 1)
    for invalid_mask in (
        Float64[0, 1],
        ComplexF64[0, 1],
        Any[missing, 1],
        Int[0, 2],
    )
        failure = _ch44_layout_captured_error() do
            qwen3_vl_rope_layout(
                Int[1, 11];
                attention_mask=invalid_mask,
            )
        end
        @test failure isa ArgumentError
        @test sprint(showerror, failure) ==
            "ArgumentError: Qwen3-VL attention_mask values must be Bool or " *
            "integer zero/one"
    end

    batch = repeat(reshape(Int[11, 12], 2, 1), 1, 2)
    @test_throws ArgumentError qwen3_vl_rope_layout(batch)
    @test_throws ArgumentError qwen3_vl_rope_layout(
        Int[11, checkpoint.video_token_id + 1, 12],
    )
end
