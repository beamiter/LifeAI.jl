using Dates
using Lux
using LifeAI
using Printf
using Random
using Statistics

const WEEK04_PROFILES = (
    baseline=(;),
    rmsnorm_only=(; norm_type=:rmsnorm),
    swiglu_only=(; mlp_type=:swiglu),
    tied_only=(; tie_embeddings=true),
    modern=(;
        norm_type=:rmsnorm,
        mlp_type=:swiglu,
        tie_embeddings=true,
    ),
)
const WEEK04_BACKENDS = ("cpu", "gpu", "xla_cpu", "xla_gpu")

function parse_options(args)
    options = Dict{String,String}()
    for argument in args
        startswith(argument, "--") || error("unknown argument: $argument")
        parts = split(argument[3:end], "="; limit=2)
        length(parts) == 2 || error("expected --name=value, got: $argument")
        options[parts[1]] = parts[2]
    end
    return options
end

env_int(name, default) = parse(Int, get(ENV, name, string(default)))
env_float(name, default) = parse(Float64, get(ENV, name, string(default)))

function benchmark_seeds()
    raw = replace(get(ENV, "LIFEAI_WEEK04_SEEDS", "20260718,20260719,20260720"), ',' => ' ')
    seeds = parse.(Int, split(raw))
    length(seeds) >= 3 ||
        error("LIFEAI_WEEK04_SEEDS must contain at least three integer seeds")
    return seeds
end

function controlled_config()
    d_model = env_int("LIFEAI_WEEK04_EMBED_DIM", 32)
    num_heads = env_int("LIFEAI_WEEK04_NUM_HEADS", 4)
    num_layers = env_int("LIFEAI_WEEK04_NUM_LAYERS", 2)
    seq_len = env_int("LIFEAI_WEEK04_SEQ_LEN", 16)
    batch_size = env_int("LIFEAI_WEEK04_BATCH_SIZE", 4)
    stride = env_int("LIFEAI_WEEK04_STRIDE", 8)
    train_steps = env_int("LIFEAI_WEEK04_TRAIN_STEPS", 20)
    learning_rate = env_float("LIFEAI_WEEK04_LEARNING_RATE", 3.0e-3)

    d_model > 0 || error("LIFEAI_WEEK04_EMBED_DIM must be positive")
    num_heads > 0 || error("LIFEAI_WEEK04_NUM_HEADS must be positive")
    d_model % num_heads == 0 ||
        error("LIFEAI_WEEK04_EMBED_DIM must be divisible by LIFEAI_WEEK04_NUM_HEADS")
    num_layers > 0 || error("LIFEAI_WEEK04_NUM_LAYERS must be positive")
    seq_len > 0 || error("LIFEAI_WEEK04_SEQ_LEN must be positive")
    batch_size > 0 || error("LIFEAI_WEEK04_BATCH_SIZE must be positive")
    stride > 0 || error("LIFEAI_WEEK04_STRIDE must be positive")
    train_steps > 0 || error("LIFEAI_WEEK04_TRAIN_STEPS must be positive")
    learning_rate > 0 || error("LIFEAI_WEEK04_LEARNING_RATE must be positive")

    return (;
        d_model,
        num_heads,
        num_layers,
        seq_len,
        batch_size,
        stride,
        train_steps,
        learning_rate,
        seeds=benchmark_seeds(),
    )
end

function controlled_data(config)
    text = repeat(
        """
        生命感来自连续的观察、记忆、反馈和行动。
        小机器人会保存经验，评估变化，再学习下一步。
        """,
        120,
    )
    return train_validation_loaders(
        text;
        validation_fraction=0.15,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        stride=config.stride,
        drop_last=true,
        add_unk=true,
    )
end

function percentile(values, probability)
    isempty(values) && error("cannot summarize empty timings")
    return quantile(values, probability)
end

function run_profile_seed(profile, options, seed, config, data)
    model = GPTModel(
        vocab_size(data.tokenizer),
        config.d_model,
        config.num_heads,
        config.num_layers;
        max_seq_len=config.seq_len,
        use_rope=true,
        options...,
    )
    trainer = TrainerGPT(
        learning_rate=config.learning_rate,
        backend=:zygote,
        device=Lux.cpu_device(),
        return_gradients=false,
        max_grad_norm=1.0f0,
    )

    # Compile each architecture before measuring, then recreate the exact seeded
    # initial state so the warm-up update cannot affect the controlled run.
    warm_state = init_train_state(Xoshiro(seed), model, trainer)
    warm_state, _, _ = train_step!(trainer, warm_state, data.train[1])
    train_state = init_train_state(Xoshiro(seed), model, trainer)

    initial_metrics, _ = evaluate_gpt(
        model,
        train_state.parameters,
        train_state.states,
        data.validation,
    )

    GC.gc()
    step_seconds = Float64[]
    last_loss = NaN32
    for step in 1:config.train_steps
        batch = data.train[mod1(step, length(data.train))]
        started = time_ns()
        train_state, last_loss, _ = train_step!(
            trainer,
            train_state,
            batch,
        )
        push!(step_seconds, Float64(time_ns() - started) / 1.0e9)
    end

    final_metrics, _ = evaluate_gpt(
        model,
        train_state.parameters,
        train_state.states,
        data.validation,
    )
    correctness = kv_cache_correctness(
        model,
        train_state.parameters,
        train_state.states,
        collect(1:min(4, config.seq_len - 2)),
        [5, 6],
    )

    checkpoint_bytes = mktempdir() do directory
        path = joinpath(directory, "week04.checkpoint")
        save_checkpoint(
            path,
            model,
            data.tokenizer,
            trainer,
            train_state;
            progress=(; epoch=1, batch=config.train_steps),
            train_config=(; profile, seed, steps=config.train_steps),
        )
        filesize(path)
    end

    tokens_per_step = config.seq_len * config.batch_size
    return (;
        profile=String(profile),
        seed,
        norm_type=String(model.norm_type),
        mlp_type=String(model.mlp_type),
        tie_embeddings=model.tie_embeddings,
        mlp_hidden_dim=model.mlp_hidden_dim,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        seq_len=config.seq_len,
        batch_size=config.batch_size,
        train_steps=config.train_steps,
        parameter_count=Lux.parameterlength(train_state.parameters),
        checkpoint_bytes,
        initial_validation_loss=Float64(initial_metrics.loss),
        initial_perplexity=Float64(initial_metrics.perplexity),
        final_train_loss=Float64(last_loss),
        final_validation_loss=Float64(final_metrics.loss),
        final_perplexity=Float64(final_metrics.perplexity),
        training_p50_ms=median(step_seconds) * 1.0e3,
        training_p90_ms=percentile(step_seconds, 0.90) * 1.0e3,
        training_tokens_per_second=tokens_per_step / median(step_seconds),
        cache_correctness=correctness.passed,
        cache_prefill_max_abs_error=max(
            correctness.prefill.dynamic_max_error,
            correctness.prefill.static_max_error,
        ),
        cache_decode_max_abs_error=max(
            correctness.decode.dynamic_max_error,
            correctness.decode.static_max_error,
        ),
    )
end

function write_cpu_matrix(path)
    config = controlled_config()
    data = controlled_data(config)
    rows = NamedTuple[]

    for (profile, options) in pairs(WEEK04_PROFILES)
        for seed in config.seeds
            println("[cpu-matrix] profile=$profile seed=$seed")
            row = run_profile_seed(profile, options, seed, config, data)
            push!(rows, row)
            println(
                "[cpu-matrix] profile=$profile seed=$seed " *
                "ppl=$(round(row.final_perplexity; digits=3)) " *
                "tokens/s=$(round(row.training_tokens_per_second; digits=1))",
            )
        end
    end

    absolute_path = abspath(path)
    mkpath(dirname(absolute_path))
    open(absolute_path, "w") do io
        header = keys(first(rows))
        println(io, join(string.(header), '\t'))
        for row in rows
            println(io, join(string.(values(row)), '\t'))
        end
    end

    println("[cpu-matrix] result: $absolute_path")
    return absolute_path
end

function read_table(path)
    lines = readlines(path)
    isempty(lines) && error("empty table: $path")
    header = split(first(lines), '\t')
    return [
        Dict(zip(header, split(line, '\t'; keepempty=true)))
        for line in Iterators.drop(lines, 1) if !isempty(line)
    ]
end

function read_metrics(path)
    metrics = Dict{String,String}()
    for line in eachline(path)
        columns = split(line, '\t'; keepempty=true)
        length(columns) >= 3 || continue
        metrics[columns[2]] = columns[3]
    end
    return metrics
end

metric_float(row, key) = parse(Float64, row[key])

function format_number(value; digits=2)
    isfinite(value) || return string(value)
    return @sprintf("%.*f", digits, value)
end

function aggregate_profile(rows, profile)
    selected = filter(row -> row["profile"] == profile, rows)
    isempty(selected) && error("missing CPU rows for profile $profile")

    values(key) = metric_float.(selected, Ref(key))
    return (;
        row=first(selected),
        seeds=length(selected),
        checkpoint_kib=mean(values("checkpoint_bytes")) / 1024,
        initial_ppl=mean(values("initial_perplexity")),
        final_loss=mean(values("final_validation_loss")),
        final_ppl=mean(values("final_perplexity")),
        final_ppl_min=minimum(values("final_perplexity")),
        final_ppl_max=maximum(values("final_perplexity")),
        tokens_per_second=mean(values("training_tokens_per_second")),
        p50_ms=mean(values("training_p50_ms")),
    )
end

function backend_statuses(directory)
    path = joinpath(directory, "status.tsv")
    isfile(path) || return Dict{Tuple{String,String},String}()
    statuses = Dict{Tuple{String,String},String}()
    for row in read_table(path)
        statuses[(row["profile"], row["backend"])] = row["status"]
    end
    return statuses
end

function summarize(directory)
    directory = abspath(directory)
    cpu_rows = read_table(joinpath(directory, "cpu_matrix.tsv"))
    statuses = backend_statuses(directory)
    output = IOBuffer()

    println(output, "# Week 04 model modernization benchmark")
    println(output)
    println(output, "生成时间：", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    println(output)
    println(output, "## 五配置 CPU 受控训练")
    println(output)
    println(
        output,
        "| 配置 | Norm | MLP | Tied | MLP width | 参数量 | Checkpoint KiB | ",
        "初始 PPL | 最终 validation loss | 最终 PPL mean [range] | 训练 tokens/s |",
    )
    println(
        output,
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
    )

    for profile_symbol in keys(WEEK04_PROFILES)
        profile = String(profile_symbol)
        aggregate = aggregate_profile(cpu_rows, profile)
        row = aggregate.row
        println(
            output,
            "| ", profile,
            " | ", row["norm_type"],
            " | ", row["mlp_type"],
            " | ", row["tie_embeddings"],
            " | ", row["mlp_hidden_dim"],
            " | ", row["parameter_count"],
            " | ", format_number(aggregate.checkpoint_kib; digits=1),
            " | ", format_number(aggregate.initial_ppl; digits=3),
            " | ", format_number(aggregate.final_loss; digits=4),
            " | ", format_number(aggregate.final_ppl; digits=3),
            " [", format_number(aggregate.final_ppl_min; digits=3),
            ", ", format_number(aggregate.final_ppl_max; digits=3), "]",
            " | ", format_number(aggregate.tokens_per_second; digits=1),
            " |",
        )
    end

    println(output)
    first_row = first(cpu_rows)
    seed_count = length(filter(
        row -> row["profile"] == "baseline",
        cpu_rows,
    ))
    println(
        output,
        "每个配置使用 ", seed_count,
        " 个固定 seed、", first_row["train_steps"],
        " 个训练 step；模型 d_model=", first_row["d_model"],
        "、heads=", first_row["num_heads"],
        "、layers=", first_row["num_layers"],
        "、seq_len=", first_row["seq_len"],
        "、batch=", first_row["batch_size"], "。",
    )
    all(row["cache_correctness"] == "true" for row in cpu_rows) &&
        println(output, "五组配置、全部 seed 的 full / dynamic / static cache correctness 均通过。")

    println(output)
    println(output, "## Baseline vs Modern 四后端")
    println(output)
    println(
        output,
        "| 配置 | 后端 | 状态 | correctness | 参数量 | 训练 cold ms | ",
        "训练 p50/p90 ms | 训练 tokens/s | Prefill p50 ms | Decode p50 ms/token |",
    )
    println(
        output,
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    )

    successful = Dict{Tuple{String,String},Dict{String,String}}()
    for profile in ("baseline", "modern")
        for backend in WEEK04_BACKENDS
            path = joinpath(directory, "$(profile)_$(backend).tsv")
            status = get(statuses, (profile, backend), isfile(path) ? "ok" : "missing")
            if isfile(path)
                metrics = read_metrics(path)
                successful[(profile, backend)] = metrics
                println(
                    output,
                    "| ", profile,
                    " | ", backend,
                    " | ", status,
                    " | ", get(metrics, "inference_correctness_passed", "—"),
                    " | ", get(metrics, "parameter_count", "—"),
                    " | ", format_number(parse(Float64, get(metrics, "training_first_step_ms", "NaN"))),
                    " | ", format_number(parse(Float64, get(metrics, "training_steady_p50_ms", "NaN"))),
                    " / ", format_number(parse(Float64, get(metrics, "training_steady_p90_ms", "NaN"))),
                    " | ", format_number(parse(Float64, get(metrics, "training_steady_tokens_per_second", "NaN")); digits=1),
                    " | ", format_number(parse(Float64, get(metrics, "inference_prefill_steady_p50_ms", "NaN"))),
                    " | ", format_number(parse(Float64, get(metrics, "inference_decode_steady_p50_ms_per_token", "NaN"))),
                    " |",
                )
            else
                println(
                    output,
                    "| ", profile, " | ", backend, " | ", status,
                    " | — | — | — | — | — | — | — |",
                )
            end
        end
    end

    println(output)
    println(output, "## 口径与边界")
    println(output)
    println(output, "- CPU 单变量表的训练结果按 seed 汇总；性能编译 warm-up 不计入固定训练 step。")
    println(output, "- 四后端表的每个配置/后端运行在独立 Julia 进程，cold 与 steady-state 分开记录。")
    println(output, "- 单变量 CPU 实验与四后端性能实验使用不同模型规模，各自只在表内横向比较。")
    println(output, "- tiny model / tiny corpus 结果用于验证可归因性和工程兼容性，不代表真实模型质量排名。")
    println(output, "- 原始 TSV、逐样本时延和完整日志保留在同一目录。")

    summary = String(take!(output))
    path = joinpath(directory, "summary.md")
    open(path, "w") do io
        write(io, summary)
    end
    print(summary)
    return path
end

function main(args)
    options = parse_options(args)
    if haskey(options, "output")
        write_cpu_matrix(options["output"])
        return
    elseif haskey(options, "summarize")
        summarize(options["summarize"])
        return
    end
    error("expected --output=PATH or --summarize=DIRECTORY")
end

main(ARGS)
