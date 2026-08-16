using Test
using JSON3
using BFloat16s: BFloat16
using LifeAI
using LifeAI:
    EvalItemResult,
    GSM8KItem,
    MMLUItem,
    accuracy_report,
    encode,
    extract_gsm8k_answer,
    extract_mmlu_letter,
    gsm8k_answer_matches,
    gsm8k_chat_messages,
    load_eval_tasks,
    load_hf_qwen3_bundle,
    load_hf_qwen3_tokenizer,
    mmlu_chat_messages,
    mmlu_loglikelihood_prompt,
    mmlu_loglikelihood_scores,
    subject_report

if !isdefined(@__MODULE__, :LIFEAI_REPO_ROOT)
    @eval const LIFEAI_REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
end

const CHAPTER37_FIXTURES = joinpath(@__DIR__, "fixtures")
const CHAPTER37_TASKS = joinpath(CHAPTER37_FIXTURES, "eval_tasks.json")

const CHAPTER37_SAMPLE_ITEM = MMLUItem(
    "mmlu/high_school_mathematics/0",
    "high_school_mathematics",
    "What is 2 + 2?",
    ["3", "4", "5", "6"],
    1,
)

@testset "Chapter 37 — frozen task set loads and validates" begin
    tasks = load_eval_tasks(CHAPTER37_TASKS)
    @test length(tasks.mmlu) == 200
    @test length(tasks.gsm8k) == 150
    @test length(unique(item.subject for item in tasks.mmlu)) == 8
    # 8 subjects x 25 items, so every subject must be equally represented; an
    # unbalanced fixture would quietly reweight the reported accuracy.
    for subject in unique(item.subject for item in tasks.mmlu)
        @test count(item -> item.subject == subject, tasks.mmlu) == 25
    end
    @test all(item -> length(item.choices) == 4, tasks.mmlu)
    @test all(item -> 0 <= item.answer_index <= 3, tasks.mmlu)
    @test all(item -> !isempty(strip(item.question)), tasks.mmlu)
    @test all(item -> tryparse(Float64, replace(item.answer, "," => "")) !== nothing, tasks.gsm8k)
    @test length(tasks.sha256) == 64

    payload = JSON3.read(read(CHAPTER37_TASKS, String))
    mktempdir() do directory
        broken(mutate) = begin
            data = JSON3.read(JSON3.write(payload), Dict{String,Any})
            mutate(data)
            path = joinpath(directory, "broken.json")
            open(path, "w") do io
                JSON3.write(io, data)
            end
            path
        end
        # Three shapes of corruption a hand-edited fixture could take.
        @test_throws ArgumentError load_eval_tasks(broken(
            data -> data["mmlu"][1]["choices"] = ["only", "three", "choices"]))
        @test_throws ArgumentError load_eval_tasks(broken(
            data -> data["mmlu"][1]["answer_index"] = 4))
        @test_throws ArgumentError load_eval_tasks(broken(
            data -> data["gsm8k"][1]["answer"] = "not a number"))
        # `tryparse(Float64, ...)` accepts these; an item whose target is NaN would
        # load cleanly and then be impossible to score correct.
        for poison in ("nan", "inf", "-inf", "0x10")
            @test_throws ArgumentError load_eval_tasks(broken(
                data -> data["gsm8k"][1]["answer"] = poison))
        end
        @test_throws ArgumentError load_eval_tasks(broken(
            data -> data["mmlu"][2]["id"] = data["mmlu"][1]["id"]))
    end
end

@testset "Chapter 37 — protocols are frozen prompts" begin
    @test mmlu_loglikelihood_prompt(CHAPTER37_SAMPLE_ITEM) ==
          "The following are multiple choice questions (with answers) about " *
          "high school mathematics.\n\nWhat is 2 + 2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
    # No chat template is applied on the loglikelihood path: it is a base-model
    # protocol, and mixing in the chat template would change what is measured.
    @test !occursin("<|im_start|>", mmlu_loglikelihood_prompt(CHAPTER37_SAMPLE_ITEM))

    message = only(mmlu_chat_messages(CHAPTER37_SAMPLE_ITEM))
    @test message[:role] == "user"
    @test occursin("Answer with a single letter (A, B, C or D) and nothing else.",
        message[:content])
    @test occursin("A. 3\nB. 4\nC. 5\nD. 6", message[:content])

    gsm = only(gsm8k_chat_messages(GSM8KItem("gsm8k/0", "How many?", "7")))
    @test gsm[:role] == "user"
    @test occursin("#### <number>", gsm[:content])
    @test endswith(gsm[:content], "How many?")
end

@testset "Chapter 37 — MMLU answer extraction" begin
    for (completion, expected) in [
        ("B", "B"),
        ("The answer is C.", "C"),
        ("**D**", "D"),
        ("Answer: A", "A"),
        ("Answer:\nB", "B"),
        # Anchored on the last "answer", so options discussed on the way do not win.
        ("I think A but the answer is D", "D"),
        ("Let me think. A is wrong, B is wrong, so the answer is C", "C"),
        # No "answer" anywhere falls back to the first standalone letter.
        ("C. 5 is correct", "C"),
        ("no letters here", nothing),
        ("", nothing),
        # A bare word starting with a letter must not be mistaken for an option.
        ("Absolutely nothing here", nothing),
        # Without an "answer" anchor only a marked letter counts. The English
        # article, a set named A and an "options A-D" phrase all used to score as
        # option A; on the frozen 1.7B run that returned A for 18 of the 24 items
        # reaching this branch, against a 27% gold base rate.
        ("A factor group of a non-Abelian group is non-Abelian.", nothing),
        ("A student measured the mass of a sample.", nothing),
        ("Set \$ A = \\{1, 2, 3\\} \$ and proceed.", nothing),
        ("Consider options A-D carefully", nothing),
        ("Which of the options A-D is true? Let us check B) first", "B"),
        ("A. 2x + 5", "A"),
        ("D)", "D"),
        ("C:", "C"),
        # The anchor overrides the marker requirement.
        ("A factor group is odd, so the answer is B", "B"),
    ]
        @test extract_mmlu_letter(completion) == expected
    end
end

@testset "Chapter 37 — GSM8K answer extraction" begin
    for (completion, expected) in [
        ("#### 18", 18.0),
        ("So the total is 42.\n#### 42", 42.0),
        ("The answer is 1,234", 1234.0),
        ("#### 18\nextra 99", 18.0),
        ("he has 7 apples", 7.0),
        ("first 3 then 5 then finally 11", 11.0),
        ("#### -4", -4.0),
        ("#### 2.5", 2.5),
        ("no numbers", nothing),
    ]
        @test extract_gsm8k_answer(completion) == expected
    end
    @test gsm8k_answer_matches("18", 18.0)
    @test gsm8k_answer_matches("18", 18.0000001)
    @test !gsm8k_answer_matches("18", 19.0)
    @test !gsm8k_answer_matches("18", nothing)
    @test !gsm8k_answer_matches("not a number", 18.0)
end

@testset "Chapter 37 — accuracy reporting" begin
    result(id, subject, correct, parsed=true) = EvalItemResult(
        id, subject, :mmlu_generative, "sha", 10, "", "", "", correct, parsed, nothing)
    results = [
        result("a", "algebra", true),
        result("b", "algebra", false),
        result("c", "algebra", false, false),
        result("d", "logic", true),
    ]
    report = accuracy_report(results)
    @test report.total == 4
    @test report.correct == 2
    # An unparseable completion counts as wrong, but is also reported on its own so
    # a harness bug never hides inside the error rate.
    @test report.unparsed == 1
    @test report.accuracy == 0.5
    @test report.wilson_lower < 0.5 < report.wilson_upper
    @test report.wilson_upper - report.wilson_lower > 0.5

    subjects = subject_report(results)
    @test length(subjects) == 2
    @test subjects[1].subject == "algebra"
    @test subjects[1].total == 3
    @test subjects[1].correct == 1
    @test subjects[2].subject == "logic"
    @test subjects[2].accuracy == 1.0

    @test_throws ArgumentError accuracy_report(EvalItemResult[])
end

# Opt-in: the offline tests above cover the harness. Only the real 0.6B checkpoint
# can show that our loglikelihood scoring agrees with HuggingFace item by item.
const CHAPTER37_REFERENCE =
    joinpath(CHAPTER37_FIXTURES, "mmlu_loglikelihood_reference_qwen3_0_6b.json")

# How many items the protocol-matched parity check scores. Each item costs four CPU
# forwards on the :general path, so the full 200 would take roughly 40 minutes; the
# frozen 200-item comparison lives in the chapter notes instead.
#
# Sampled with a stride rather than taken as a prefix: the fixture is ordered by
# subject, so the first 60 items are two subjects only. On CPU that prefix agreed on
# 55/60 against a 0.9 bar — three items from failing for a reason that says nothing
# about the code. The stride covers all eight subjects at the same cost and agreed on
# 49/50.
const CHAPTER37_PARITY_ITEMS = 60

if haskey(ENV, "LIFEAI_QWEN3_MODEL_DIR") && isfile(CHAPTER37_REFERENCE)
    @testset "Chapter 37 — MMLU loglikelihood parity with Transformers" begin
        reference = JSON3.read(read(CHAPTER37_REFERENCE, String))
        tasks = load_eval_tasks(CHAPTER37_TASKS)
        @test String(reference.tasks_sha256) == tasks.sha256
        @test String(reference.protocol) == "loglikelihood-unnormalized"

        model_dir = ENV["LIFEAI_QWEN3_MODEL_DIR"]
        bundle = load_hf_qwen3_bundle(model_dir; weight_dtype=BFloat16, max_seq_len=1024)
        tokenizer = load_hf_qwen3_tokenizer(model_dir)
        by_id = Dict(item.id => item for item in tasks.mmlu)

        agreements = 0
        largest_gap = 0.0
        all_items = collect(reference.items)
        stride = max(1, cld(length(all_items), CHAPTER37_PARITY_ITEMS))
        scored = all_items[1:stride:end]
        for entry in scored
            item = by_id[String(entry.id)]
            # `:general` on purpose: the reference forwards prompt+continuation, and
            # the cheaper `:fast` shortcut is not BF16-equivalent to it, so comparing
            # `:fast` here would measure our shortcut as well as the dtype.
            scores, prompt_tokens = mmlu_loglikelihood_scores(
                bundle.model, bundle.parameters, tokenizer, item; path=:general)
            # Catches prompt drift outright: a changed template or a changed
            # continuation would move this before it moves any log probability.
            @test prompt_tokens == Int(entry.prompt_token_count)
            expected = Float64[Float64(value) for value in entry.logprobs]
            largest_gap = max(largest_gap, maximum(abs.(scores .- expected)))
            argmax(scores) == argmax(expected) && (agreements += 1)
        end
        total = length(scored)
        @info "Chapter 37 loglikelihood parity" total agreements largest_gap
        # BF16 against a Float32 reference shifts the log probabilities by more than
        # the smallest decision margins, so "every decision agrees" is simply false —
        # the frozen 200-item protocol-matched run agrees on 193. An earlier version
        # of this test asserted `margin <= 2 * deviation` for each flip; that is a
        # tautology (s_c >= s_a, s_c <= e_c + d and s_a >= e_a - d force it), so it
        # could never fail and caught nothing. The agreement rate is what has power:
        # swapping two continuations drops it to 0.77.
        @test agreements / total >= 0.9
    end
end
