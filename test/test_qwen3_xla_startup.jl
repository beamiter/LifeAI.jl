using Test
using TOML
using LifeAI

const _QWEN3_STARTUP_ROOT = dirname(@__DIR__)
const _QWEN3_STARTUP_LAUNCHER = joinpath(
    _QWEN3_STARTUP_ROOT, "scripts", "launch_qwen3_xla_server.sh",
)

@testset "Qwen3 launcher CPU compatibility and argument forwarding" begin
    if Sys.isunix()
        mktempdir() do directory
            recorder = joinpath(directory, "Julia recorder")
            arguments_path = joinpath(directory, "arguments")
            write(recorder, "#!/bin/sh\nprintf '%s\\0' \"\$@\" > \"\$LIFEAI_TEST_ARGV_OUTPUT\"\n")
            chmod(recorder, 0o755)
            model = joinpath(directory, "model weights")
            profile = joinpath(directory, "deployment profile.json")
            read_arguments() = split(read(arguments_path, String), '\0'; keepempty=false)
            environment = (
                "LIFEAI_JULIA_BIN" => recorder,
                "LIFEAI_TEST_ARGV_OUTPUT" => arguments_path,
            )

            run(addenv(`bash $_QWEN3_STARTUP_LAUNCHER $model --profile $profile`, environment...))
            default = read_arguments()
            @test default[1] == "--startup-file=no"
            @test !any(startswith("--cpu-target="), default)
            @test default[2] == "--project=$_QWEN3_STARTUP_ROOT"
            @test default[3] == joinpath(_QWEN3_STARTUP_ROOT, "scripts", "run_qwen3_xla_server.jl")
            @test default[4:end] == [model, "--profile", profile]

            run(addenv(`bash $_QWEN3_STARTUP_LAUNCHER $model --cpu-bf16-compat --quiet-startup --profile $profile`, environment...))
            compatible = read_arguments()
            @test compatible[1] == "--cpu-target=native,-avx512bf16"
            @test compatible[2:4] == default[1:3]
            @test compatible[5:end] == [model, "--quiet-startup", "--profile", profile]
            @test !in("--cpu-bf16-compat", compatible)
            @test occursin("--cpu-bf16-compat", read(`bash $_QWEN3_STARTUP_LAUNCHER --help`, String))
        end
    else
        @test_skip Sys.isunix()
    end
end

@testset "CPU BF16 compatibility executes vector conversion correctly" begin
    if Sys.ARCH === :x86_64
        # Exercise the actual JIT target in a fresh Julia process. Compare BF16
        # bits against independent round-to-nearest-even integer arithmetic.
        julia = joinpath(Sys.BINDIR, Base.julia_exename())
        probe = """
        using BFloat16s: BFloat16
        source = reshape(collect(range(-1.0f0, 1.0f0; length=8192)), 128, 64)
        result = BFloat16.(source)
        reference = UInt16[]
        for value in source
            bits = reinterpret(UInt32, value)
            push!(reference, UInt16((bits + UInt32(0x7fff) + ((bits >> 16) & UInt32(1))) >> 16))
        end
        @assert collect(reinterpret(UInt16, vec(result))) == reference
        println("BF16_VECTOR_COMPAT_OK")
        """
        output = read(`$julia --startup-file=no --cpu-target=native,-avx512bf16 --project=$_QWEN3_STARTUP_ROOT -e $probe`, String)
        @test occursin("BF16_VECTOR_COMPAT_OK", output)
    else
        @test_skip Sys.ARCH === :x86_64
    end
end

@testset "Optional CUDA toolkit preference example" begin
    project = TOML.parsefile(joinpath(_QWEN3_STARTUP_ROOT, "Project.toml"))
    preferences = TOML.parsefile(joinpath(
        _QWEN3_STARTUP_ROOT, "configs", "deployment", "LocalPreferences.cuda12_9.example.toml",
    ))
    for package in ("CUDA_Runtime_jll", "CUDA_Compiler_jll")
        @test haskey(project["extras"], package)
        @test preferences[package]["version"] == "12.9"
    end
end

@testset "Session startup callback preserves loading failures" begin
    mktempdir() do directory
        events = []
        failure = try
            load_qwen3_xla_http_service(directory;
                context_tokens=16, prefill_chunk_tokens=4, max_new_tokens=4,
                on_progress=event -> push!(events, event))
            nothing
        catch error
            error
        end
        @test failure isa ArgumentError
        @test occursin("required Qwen3 tokenizer file", sprint(showerror, failure))
        @test events == [(stage=:host_load, status=:start, seconds=nothing)]

        callback_error = ErrorException("progress callback failed")
        observed = try
            load_hf_qwen3_bf16_xla_session(directory;
                context_tokens=16, prefill_chunk_tokens=4,
                on_progress=_ -> throw(callback_error))
            nothing
        catch error
            error
        end
        @test observed === callback_error
    end
end

@testset "GPU startup progress and silent default (opt-in)" begin
    if get(ENV, "LIFEAI_TEST_QWEN3_XLA_STARTUP_GPU", "0") == "1"
        import Reactant
        isdefined(@__MODULE__, :_qwen3_tiny_model_fixture_dir) ||
            include(joinpath(@__DIR__, "support", "qwen3_tiny_model_fixture.jl"))
        isdefined(@__MODULE__, :write_qwen3_tokenizer_fixture) ||
            include(joinpath(@__DIR__, "support", "qwen3_tokenizer_fixture.jl"))
        Reactant.set_default_backend("gpu")
        mktempdir() do directory
            _qwen3_tiny_model_fixture_dir(directory; vocab_size=263)
            write_qwen3_tokenizer_fixture(directory)
            events = []
            service = load_qwen3_xla_http_service(directory;
                context_tokens=16, prefill_chunk_tokens=4, max_new_tokens=4,
                on_progress=event -> push!(events, event))
            stages = (:host_load, :parameter_transfer, :runtime_allocation,
                      :prefill_compile, :decode_compile)
            @test [(event.stage, event.status) for event in events] ==
                [(stage, status) for stage in stages for status in (:start, :done)]
            @test all(event.seconds === nothing for event in events[1:2:end])
            @test all(event.seconds >= 0 for event in events[2:2:end])
            @test qwen3_xla_service_status(service).load_count == 1

            with_progress = generate_hf_qwen3_bf16_xla!(service.session, [2, 3]; max_new_tokens=2)
            log = joinpath(directory, "default-startup.log")
            plain = open(log, "w") do io
                redirect_stderr(io) do
                    load_hf_qwen3_bf16_xla_session(directory;
                        context_tokens=16, prefill_chunk_tokens=4)
                end
            end
            @test isempty(read(log, String))
            without_progress = generate_hf_qwen3_bf16_xla!(plain, [2, 3]; max_new_tokens=2)
            @test with_progress.generated_ids == without_progress.generated_ids
        end
    else
        @test_skip get(ENV, "LIFEAI_TEST_QWEN3_XLA_STARTUP_GPU", "0") == "1"
    end
end

@testset "Server CLI accepts a separate output cap" begin
    server_path = joinpath(_QWEN3_STARTUP_ROOT, "scripts", "run_qwen3_xla_server.jl")
    source = read(server_path, String)
    first_definition = findfirst("const DEFAULT_PROFILE =", source)
    runtime_start = findfirst("\nmodel_dir, options = parse_args(ARGS)", source)
    cli = Module(gensym(:Qwen3ServerCLI))
    # Evaluate only CLI definitions, before any model or backend startup.
    include_string(cli, source[first(first_definition):(first(runtime_start) - 1)], server_path)
    _, defaults = cli.parse_args(["model"])
    @test defaults[:max_new_tokens] === nothing
    _, capped = cli.parse_args(["model", "--max-new-tokens", "40896", "--port", "11435"])
    @test capped[:max_new_tokens] == 40896
    @test capped[:port] == 11435
    @test_throws ErrorException cli.parse_args(["model", "--max-new-tokens"])
    @test_throws ArgumentError cli.parse_args(["model", "--max-new-tokens", "invalid"])
end
