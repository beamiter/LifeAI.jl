using Test
using LifeAI

function _ch147_generate(parameters, cache; prompt_ids=nothing, kwargs...)
    inputs = _ch46_tiny_prefill_inputs()
    features = eltype(parameters.embedding) === BFloat16 ?
        _ch47_bf16_parameters(inputs.vision_features) : inputs.vision_features
    options = merge((;
        vision_features=features,
        max_new_tokens=3,
        stop_token_ids=Int[],
        capture_logits=true,
        cache,
    ), (; kwargs...))
    return generate_hf_qwen3_vl_tokens(
        parameters, prompt_ids === nothing ? inputs.input_ids : prompt_ids,
        inputs.rope_layout; options...,
    )
end

function _ch147_snapshot(cache)
    return (;
        position=cache.position,
        rope_delta=cache.rope_delta,
        ready=cache.decode_pack,
        workspace=cache.decode_workspace,
        seen_storage=cache.seen_storage,
        seen_contents=cache.seen_storage === nothing ? nothing : copy(cache.seen_storage),
        layers=map(layer -> (copy(layer.keys), copy(layer.values)), cache.layers),
    )
end

function _ch147_unchanged(cache, before)
    @test cache.position == before.position
    @test cache.rope_delta == before.rope_delta
    @test cache.decode_pack === before.ready
    @test cache.decode_workspace === before.workspace
    @test cache.seen_storage === before.seen_storage
    if before.seen_contents !== nothing
        @test length(cache.seen_storage) == length(before.seen_contents)
        @test all(a === b for (a, b) in zip(cache.seen_storage, before.seen_contents))
    end
    for (layer, (keys, values)) in zip(cache.layers, before.layers)
        @test layer.keys == keys
        @test layer.values == values
    end
end

function _ch147_same_result(actual, expected)
    @test actual.generated_ids == expected.generated_ids
    @test actual.token_ids == expected.token_ids
    @test actual.stop_reason == expected.stop_reason
    @test actual.trace == expected.trace
    @test actual.prefill.logits == expected.prefill.logits
    @test actual.cache_mode === :static
    _ch145_check_prefix(actual.cache, expected.cache)
end

@testset "Chapter 147 — explicit generation cache reuse" begin
    for bf16 in (false, true)
        @testset "BF16=$bf16" begin
            float_parameters = _ch46_tiny_text_parameters()
            parameters = bf16 ? _ch47_bf16_parameters(float_parameters) : float_parameters
            cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
            storage = map(layer -> (layer.keys, layer.values), cache.layers)
            first = _ch147_generate(parameters, cache)
            @test first.cache === cache
            expected = _ch147_generate(parameters, :static; static_capacity=12)
            _ch147_same_result(first, expected)
            installed = cache.decode_pack
            workspace = cache.decode_workspace

            second = _ch147_generate(parameters, cache;
                max_new_tokens=2, prompt_ids=reverse(collect(1:8)))
            second_expected = _ch147_generate(parameters, :static;
                static_capacity=12, max_new_tokens=2, prompt_ids=reverse(collect(1:8)))
            @test second.cache === cache
            _ch147_same_result(second, second_expected)
            for (layer, (keys, values)) in zip(cache.layers, storage)
                @test layer.keys === keys
                @test layer.values === values
            end
            if bf16
                @test cache.decode_pack === installed
                @test cache.decode_workspace === workspace
            end

            replacement = merge(parameters, (; final_norm=zero.(parameters.final_norm)))
            replaced = _ch147_generate(replacement, cache)
            replaced_expected = _ch147_generate(replacement, :static; static_capacity=12)
            _ch147_same_result(replaced, replaced_expected)
            @test all(step -> all(iszero, step.logits), replaced.trace)

            one = _ch147_generate(parameters, cache; max_new_tokens=1)
            @test one.cache === cache
            @test one.cache.position == 8
            eos = _ch147_generate(parameters, cache;
                stop_token_ids=[only(one.generated_ids)])
            @test length(eos.generated_ids) == 1
            @test eos.stop_reason === :eos
            @test eos.cache.position == 8
            zero_result = _ch147_generate(parameters, cache; max_new_tokens=0)
            @test zero_result.cache === cache
            @test zero_result.cache_mode === :static
            @test isempty(cache)
            @test cache.rope_delta == 0
            @test zero_result.prefill === nothing
            @test isempty(zero_result.generated_ids)
        end
    end
end

@testset "Chapter 147 — invalid requests preserve the supplied cache" begin
    parameters = _ch47_bf16_parameters(_ch46_tiny_text_parameters())
    cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
    _ch147_generate(parameters, cache)
    before = _ch147_snapshot(cache)
    for (error_type, options) in (
        (ArgumentError, (; static_capacity=12)),
        (ArgumentError, (; max_new_tokens=-1)),
        (ArgumentError, (; max_new_tokens=6)), # prompt 8 + append 5 exceeds capacity 12
        (ArgumentError, (; stop_token_ids=[0])),
        (ArgumentError, (; vision_features=nothing)),
        (DimensionMismatch, (; vision_features=(;
            visual_embeddings=zeros(BFloat16, 16, 3),
            deepstack=ntuple(_ -> zeros(BFloat16, 16, 4), 3),
        ))),
    )
        @test_throws error_type _ch147_generate(parameters, cache; options...)
        _ch147_unchanged(cache, before)
    end
    inputs = _ch46_tiny_prefill_inputs()
    features = _ch47_bf16_parameters(inputs.vision_features)
    for requested in (0, 3)
        @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
            parameters, [0; inputs.input_ids[2:end]], inputs.rope_layout;
            cache, vision_features=features, max_new_tokens=requested,
        )
        _ch147_unchanged(cache, before)
        broken_positions = copy(inputs.rope_layout.position_ids)
        broken_positions[1, 1, 1] = -1
        layout = Qwen3VLRopeLayout(
            broken_positions, inputs.rope_layout.rope_deltas,
            inputs.rope_layout.visual_mask, inputs.rope_layout.attention_mask,
        )
        @test_throws ArgumentError generate_hf_qwen3_vl_tokens(
            parameters, inputs.input_ids, layout;
            cache, vision_features=features, max_new_tokens=requested,
        )
        _ch147_unchanged(cache, before)
    end
    # Compatibility must be checked before resetting an existing request.
    float_parameters = _ch46_tiny_text_parameters()
    @test_throws ArgumentError _ch147_generate(float_parameters, cache)
    _ch147_unchanged(cache, before)
end

@testset "Chapter 147 — mutated cache geometry fails without bookkeeping changes" begin
    parameters = _ch47_bf16_parameters(_ch46_tiny_text_parameters())
    for (error_type, mutate) in (
        (DimensionMismatch, cache -> (cache.capacity += 1)),
        (ArgumentError, cache -> (cache.batch_size = 2)),
        (ArgumentError, cache -> (cache.position = cache.capacity + 1)),
    )
        cache = init_qwen3_vl_static_kv_cache(parameters; capacity=12)
        _ch147_generate(parameters, cache)
        mutate(cache)
        before = _ch147_snapshot(cache)
        @test_throws error_type _ch147_generate(parameters, cache)
        _ch147_unchanged(cache, before)
    end
end

function _ch147_rebuild(value; kwargs...)
    fields = NamedTuple{fieldnames(typeof(value))}(Tuple(
        getfield(value, name) for name in fieldnames(typeof(value))
    ))
    return typeof(value)(values(merge(fields, (; kwargs...)))...)
end

struct _Ch147VisionFailure{S,A}
    spec::S
    patch_weight::A
    touched::Base.RefValue{Bool}
end
function LifeAI._qwen3_vl_generation_vision_features(
    parameters::_Ch147VisionFailure, processed, requested::Int,
)
    parameters.touched[] = true
    error("Chapter 147 vision preparation sentinel")
end

function _ch147_raw_fixture()
    payloads = _ch44_vl_tokenizer_payloads()
    official_template = read(joinpath(@__DIR__, "fixtures", "official_chat_template.jinja"), String)
    @test bytes2hex(sha256(official_template)) == LifeAI._QWEN3_VL_CHAT_TEMPLATE_SHA256
    payloads.tokenizer_config["chat_template"] = official_template
    for (id, content) in (
        (263, "<|vision_start|>"), (264, "<|image_pad|>"),
        (265, "<|vision_end|>"), (266, "<|video_pad|>"),
    )
        entry = deepcopy(first(payloads.tokenizer["added_tokens"]))
        entry["id"], entry["content"] = id, content
        push!(payloads.tokenizer["added_tokens"], entry)
        push!(payloads.tokenizer_config["additional_special_tokens"], content)
        config_entry = deepcopy(first(values(payloads.tokenizer_config["added_tokens_decoder"])))
        config_entry["content"] = content
        payloads.tokenizer_config["added_tokens_decoder"][string(id)] = config_entry
    end
    tokenizer = mktempdir() do directory
        write_qwen3_tokenizer_fixture(directory; payloads)
        load_hf_qwen3_vl_tokenizer(directory; revision="chapter147-raw-boundary")
    end
    base = _ch46_tiny_text_parameters()
    spec = _ch147_rebuild(base.spec; vocab_size=267, max_position_embeddings=2048)
    checkpoint_base = qwen3_vl_checkpoint_spec()
    vision_spec = _ch147_rebuild(checkpoint_base.vision; out_hidden_size=16)
    checkpoint = _ch147_rebuild(checkpoint_base;
        image_token_id=264, video_token_id=266,
        vision_start_token_id=263, vision_end_token_id=265,
        bos_token_id=258, eos_token_id=260, text=spec, vision=vision_spec,
    )
    parameters = merge(base, (;
        spec, checkpoint, embedding=repeat(base.embedding, 1, 9)[:, 1:267],
    ))
    vision = _Ch147VisionFailure(vision_spec, zeros(Float32, 1, 1), Ref(false))
    processor = _ch147_rebuild(LifeAI.qwen3_vl_processor_spec();
        min_pixels=32 * 32, max_pixels=32 * 32,
    )
    return (; parameters, vision, tokenizer, processor)
end

function _ch147_seed_text(parameters, cache)
    positions = reshape(repeat(collect(0:3); inner=3), 3, 4, 1)
    layout = Qwen3VLRopeLayout(positions, zeros(Int, 1, 1), falses(4, 1), trues(4, 1))
    return generate_hf_qwen3_vl_tokens(parameters, collect(1:4), layout;
        cache, max_new_tokens=2, stop_token_ids=Int[])
end

@testset "Chapter 147 — raw chat failures preserve the supplied request" begin
    fixture = _ch147_raw_fixture()
    (; parameters, vision, tokenizer, processor) = fixture
    valid_messages = [(role="user", content=Any[
        (type="image", image=zeros(UInt8, 32, 32, 3)),
        (type="text", text="Describe."),
    ])]
    invalid_messages = [(role="user", content=Any[(type="image", image=42)])]
    large = init_qwen3_vl_static_kv_cache(parameters; capacity=1024)
    _ch147_seed_text(parameters, large)
    before = _ch147_snapshot(large)
    image_error = _ch45_captured_error() do
        generate_hf_qwen3_vl(vision, parameters, tokenizer, invalid_messages;
            cache=large, max_new_tokens=2, stop_token_ids=Int[], processor_spec=processor)
    end
    @test image_error isa ArgumentError
    @test occursin("image payload", sprint(showerror, image_error))
    @test !vision.touched[]
    _ch147_unchanged(large, before)

    small = init_qwen3_vl_static_kv_cache(parameters; capacity=8)
    _ch147_seed_text(parameters, small)
    before_small = _ch147_snapshot(small)
    capacity_error = _ch45_captured_error() do
        generate_hf_qwen3_vl(vision, parameters, tokenizer, valid_messages;
            cache=small, max_new_tokens=2, stop_token_ids=Int[], processor_spec=processor)
    end
    @test capacity_error isa ArgumentError
    @test occursin("existing static cache", sprint(showerror, capacity_error))
    @test !vision.touched[]
    _ch147_unchanged(small, before_small)

    vision_error = _ch45_captured_error() do
        generate_hf_qwen3_vl(vision, parameters, tokenizer, valid_messages;
            cache=large, max_new_tokens=2, stop_token_ids=Int[], processor_spec=processor)
    end
    @test vision_error isa ErrorException
    @test occursin("vision preparation sentinel", sprint(showerror, vision_error))
    @test vision.touched[]
    _ch147_unchanged(large, before)
end
