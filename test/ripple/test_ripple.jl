using Test, LifeAI, JSON3, Random, LinearAlgebra
using BFloat16s: BFloat16

const RIPPLE_FIXTURE = joinpath(@__DIR__, "fixtures")
const RIPPLE_REFERENCE = JSON3.read(read(joinpath(RIPPLE_FIXTURE, "reference.json"), String))
ripple_reference_matrix(rows) = hcat((Float32.(row) for row in rows)...)
function ripple_variant(config; kwargs...)
    options = Dict{Symbol,Any}(key => getfield(config, key) for key in fieldnames(RippleConfig))
    merge!(options, Dict(kwargs))
    return RippleConfig(; options...)
end

function ripple_numerical_checks(device=identity)
    bundle = load_hf_ripple_model(RIPPLE_FIXTURE; device)
    c, p = bundle.model, bundle.parameters
    ids = Int.(RIPPLE_REFERENCE["input_ids"]) .+ 1
    expected = ripple_reference_matrix(RIPPLE_REFERENCE["logits"])
    actual = Array(hf_ripple_forward(c, p, ids))
    @info "Ripple vs pinned official PyTorch" device max_absolute_error=maximum(abs.(actual .- expected))
    @test actual ≈ expected atol=4f-5 rtol=4f-5
    @test Array(hf_ripple_forward(c, p, ids; last_only=true)) ≈ expected[:, end:end] atol=4f-5 rtol=4f-5
    for chunks in (ones(Int, length(ids)), [2, 4, length(ids)-6], [7, length(ids)-7])
        cache = RippleKVCache(c)
        offset = 0
        output = Matrix{Float32}[]
        for n in chunks
            push!(output, Array(hf_ripple_forward(c, p, ids[offset+1:offset+n]; cache, cache_capacity=32)))
            offset += n
        end
        @test hcat(output...) ≈ expected atol=4f-5 rtol=4f-5
        @test cache.tokens == length(ids)
        @test size(cache.layers[1].key, 3) == c.sliding_window-1
        @test size(cache.layers[2].key, 3) == length(ids)
        @test cache.layers[1].first_position == length(ids)-c.sliding_window+1
        @test reset_ripple_cache!(cache) === cache
        @test cache.tokens == 0 && all(isnothing, cache.layers)
    end
    for variant in RIPPLE_REFERENCE["variants"]
        options = Dict(Symbol(k) => v for (k, v) in pairs(variant["options"]))
        config = ripple_variant(c; options...)
        baseline = ripple_reference_matrix(variant["logits"])
        @test Array(hf_ripple_forward(config, p, ids)) ≈ baseline atol=4f-5 rtol=4f-5
        cache = RippleKVCache(config)
        output = hcat((Array(hf_ripple_forward(config, p, [id]; cache)) for id in ids)...)
        @test output ≈ baseline atol=4f-5 rtol=4f-5
    end
    greedy = Int.(RIPPLE_REFERENCE["greedy_tokens"]) .+ 1
    @test generate_ripple_tokens(bundle, ids; max_new_tokens=5, temperature=0, eos_ids=Int[]) == greedy
    @test generate_ripple_tokens(bundle, ids; max_new_tokens=5, temperature=0, eos_ids=Int[], prefill_chunk_size=2) == greedy
    @test generate_ripple_tokens(bundle, ids; max_new_tokens=5, temperature=0, eos_ids=[first(greedy)]) == greedy[1:1]
    received = Int[]
    generated = generate_ripple_tokens(bundle, ids; max_new_tokens=5, temperature=0, eos_ids=Int[],
        on_token=id -> (push!(received, id); length(received) < 2))
    @test generated == received == greedy[1:2]
    progress = Tuple{Int,Int}[]
    @test isempty(generate_ripple_tokens(bundle, ids; max_new_tokens=5, prefill_chunk_size=2,
        on_prefill=(done, total) -> (push!(progress, (done, total)); false)))
    @test progress == [(2, length(ids))]
    cache = RippleKVCache(c)
    @test hf_ripple_forward(c, p, ids[1:5]; cache, output_logits=false) === nothing
    @test Array(hf_ripple_forward(c, p, ids[6:end]; cache)) ≈ expected[:,6:end] atol=4f-5 rtol=4f-5
    bf16 = load_hf_ripple_model(RIPPLE_FIXTURE; dtype=BFloat16, device)
    @test eltype(bf16.parameters["model.embed_tokens.weight"]) == BFloat16
    @test eltype(bf16.parameters["model.layers.0.self_attn.value_res_lambda"]) == Float32
    @test all(isfinite, Array(hf_ripple_forward(c, bf16.parameters, ids)))
    for scheme in (:int8, :int4, :nvfp4)
        lowbit = load_hf_ripple_model(RIPPLE_FIXTURE; scheme, group=scheme === :nvfp4 ? 16 : 4, device)
        output = Array(hf_ripple_forward(lowbit.model, lowbit.parameters, ids))
        @test all(isfinite, output)
        @test norm(output-actual)/norm(actual) < (scheme === :int8 ? 0.12 : 0.85)
        @test lowbit.parameters["model.layers.0.self_attn.g_proj.weight"] isa
            (scheme === :int8 ? Int8ChannelWeight : scheme === :int4 ? Int4GroupWeight : NVFP4Weight)
        lowcache = RippleKVCache(lowbit.model)
        decoded = hcat((Array(hf_ripple_forward(lowbit.model, lowbit.parameters, [id]; cache=lowcache)) for id in ids)...)
        @test decoded ≈ output atol=6f-5 rtol=6f-5
    end
end

@testset "Ripple official numerical reference and cached generation" begin
    ripple_numerical_checks()
end

@testset "Ripple configuration, checkpoint and input boundaries" begin
    official = RippleConfig()
    @test official.hidden_size == 5120 && official.num_hidden_layers == 80
    @test official.head_dim*official.num_attention_heads == 10240
    @test count(==("full_attention"), official.layer_types) == 16
    @test official.max_position_embeddings == 2097152
    bundle = load_hf_ripple_model(RIPPLE_FIXTURE)
    c, p = bundle.model, bundle.parameters
    @test Set(keys(p)) == Set(keys(ripple_expected_tensor_shapes(c)))
    @test c.num_attention_heads*c.head_dim != c.hidden_size
    for kw in ((num_attention_heads=3,), (head_dim=3,), (layer_types=["full_attention"],),
            (rms_norm_eps=0,), (rms_norm_eps=1e-300,), (embed_scale=Inf,), (use_valres=1,),
            (eos_token_id=Int[],), (bos_token_id=c.vocab_size,), (sliding_window=true,))
        @test_throws ArgumentError ripple_variant(c; kw...)
    end
    for ids in (Int[], [0], [c.vocab_size+1], [true], fill(1,c.max_position_embeddings+1))
        @test_throws ArgumentError hf_ripple_forward(c, p, ids)
    end
    @test_throws ArgumentError hf_ripple_forward(c, p, [1]; cache=RippleKVCache(ripple_variant(c)))
    @test_throws ArgumentError hf_ripple_forward(c, p, [1]; cache_capacity=true)
    @test_throws ArgumentError generate_ripple_tokens(bundle, [1]; max_new_tokens=-1)
    @test_throws ArgumentError generate_ripple_tokens(bundle, [1]; max_new_tokens=true)
    @test_throws ArgumentError generate_ripple_tokens(bundle, [1]; prefill_chunk_size=0)
    @test_throws ArgumentError generate_ripple_tokens(bundle, [1]; max_new_tokens=64)
    @test generate_ripple_tokens(bundle, [1]; max_new_tokens=0) == Int[]
    @test_throws ArgumentError load_hf_ripple_model(RIPPLE_FIXTURE; scheme=:int2)
    @test_throws ArgumentError load_hf_ripple_model(RIPPLE_FIXTURE; scheme=:int4, group=6)
    @test_throws ArgumentError load_hf_ripple_model(RIPPLE_FIXTURE; dtype=Float64)
    mktempdir() do dir
        original = LifeAI.ripple_config_dict(c)
        write(joinpath(dir, "config.json"), JSON3.write(original))
        @test LifeAI.ripple_config_dict(load_hf_ripple_config(dir)) == original
        # Exercise an indexed safetensors checkpoint without duplicating fixture data.
        cp(joinpath(RIPPLE_FIXTURE, "model.safetensors"), joinpath(dir,"part.safetensors"))
        write(joinpath(dir, "model.safetensors.index.json"), JSON3.write(Dict("weight_map" => Dict(name => "part.safetensors" for name in keys(p)))))
        sharded = load_hf_ripple_model(dir)
        @test sharded.parameters == p
        for (key, value) in (("model_type", "flint"), ("hidden_act", "gelu"),
                ("quantization_config", Dict("quant_method"=>"modelopt")),
                ("rope_scaling", Dict("type"=>"linear")), ("hidden_size", 7))
            write(joinpath(dir, "config.json"), JSON3.write(merge(original, Dict(key=>value))))
            @test_throws Exception load_hf_ripple_model(dir)
        end
        broken = copy(original)
        delete!(broken, "embed_scale")
        delete!(broken, "ffn_clamp_threshold")
        delete!(broken, "use_valres")
        delete!(broken, "max_position_embeddings")
        write(joinpath(dir, "config.json"), JSON3.write(broken))
        folded = load_hf_ripple_config(dir)
        @test folded.embed_scale === nothing && folded.ffn_clamp_threshold === nothing && !folded.use_valres
        @test folded.max_position_embeddings == 2048
    end
end

@testset "Ripple native BPE and chat template vs Hugging Face" begin
    tokenizer = load_hf_ripple_tokenizer(RIPPLE_FIXTURE)
    cases = JSON3.read(read(joinpath(RIPPLE_FIXTURE,"tokenizer_cases.json"), String))
    for case in cases
        case["kind"] == "tiny" || continue
        ids = Int.(case["ids"]) .+ 1
        @test encode(tokenizer, case["text"]) == ids
        @test decode(tokenizer, ids) == case["decoded_all"]
        @test decode(tokenizer, ids; skip_special_tokens=true) == case["decoded"]
    end
    @test encode(tokenizer, "hi") == [279]
    @test special_token_id(tokenizer, :bos) == 258
    @test special_token_id(tokenizer, :eos) == 266
    @test special_token_id(tokenizer, :pad) == 278
    @test_throws ArgumentError decode(tokenizer, [0])
    @test_throws ArgumentError apply_ripple_chat_template(tokenizer, []; tools=(name="tool",))
    @test_throws ArgumentError apply_ripple_chat_template(tokenizer, [(role="developer", content="x")])
    for case in parse_qwen3_json(read(joinpath(RIPPLE_FIXTURE,"chat_cases.json"), String))
        options = Dict(Symbol(k) => v for (k,v) in pairs(case["options"]))
        text = apply_ripple_chat_template(tokenizer, case["messages"]; options...)
        @test text == case["text"]
        @test apply_ripple_chat_template(tokenizer, case["messages"]; options..., tokenize=true) == encode(tokenizer, text)
    end
    mktempdir() do directory
        for name in ("tokenizer.json", "tokenizer_config.json", "config.json")
            cp(joinpath(RIPPLE_FIXTURE, name), joinpath(directory, name))
        end
        simple = load_hf_ripple_tokenizer(directory)
        @test apply_ripple_chat_template(simple, [(role="user", content="hi")]) ==
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        write(joinpath(directory, "chat_template.jinja"), "unknown template")
        unknown = load_hf_ripple_tokenizer(directory)
        @test_throws ArgumentError apply_ripple_chat_template(unknown, [(role="user", content="hi")])
    end
    if haskey(ENV, "LIFEAI_RIPPLE_TOKENIZER_DIR")
        official = load_hf_ripple_tokenizer(ENV["LIFEAI_RIPPLE_TOKENIZER_DIR"])
        for case in cases
            case["kind"] == "official" || continue
            ids = Int.(case["ids"]) .+ 1
            @test encode(official, case["text"]) == ids
            @test decode(official, ids) == case["decoded_all"]
            @test decode(official, ids; skip_special_tokens=true) == case["decoded"]
        end
    end
end
