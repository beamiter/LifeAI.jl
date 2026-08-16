"""
Task-level quality evaluation for the reproduced Qwen3 dense models.

Every earlier chapter asked "does LifeAI compute the same numbers as HuggingFace".
This one asks the question those chapters never did: does the model get the answer
right. Two protocols run over the same frozen items so the gap between them is
measurable rather than assumed:

  * `:loglikelihood` — the classic MMLU protocol. A base-model prompt, no chat
    template, scored by comparing the log probability of ` A` / ` B` / ` C` / ` D`.
    Deterministic, one forward pass, and directly checkable against a HuggingFace
    reference.
  * `:generative` — the deployment path. The official chat template from Chapter 36,
    greedy decoding, and an answer parsed out of the text the model actually wrote.

A protocol that cannot parse an answer scores the item as wrong and records why, so
"the model was wrong" and "we could not read the model" never collapse into one number.
"""

"""One MMLU multiple-choice item. `answer_index` is 0-based, as upstream."""
struct MMLUItem
    id::String
    subject::String
    question::String
    choices::Vector{String}
    answer_index::Int
end

"""One GSM8K item; `answer` is the final numeric string after `####`."""
struct GSM8KItem
    id::String
    question::String
    answer::String
end

"""The frozen evaluation task set."""
struct EvalTaskSet
    mmlu::Vector{MMLUItem}
    gsm8k::Vector{GSM8KItem}
    source::Any
    sha256::String
end

const MMLU_LETTERS = ("A", "B", "C", "D")
const MMLU_CONTINUATIONS = (" A", " B", " C", " D")

"""
    load_eval_tasks(path)

Read the frozen task fixture written by `scripts/export_qwen3_eval_tasks.py` and
validate every item, so a truncated or edited fixture fails here rather than
silently changing a reported accuracy.
"""
function load_eval_tasks(path::AbstractString)
    raw = read(path, String)
    payload = JSON3.read(raw)
    mmlu = MMLUItem[]
    for entry in get(payload, :mmlu, ())
        choices = String[String(choice) for choice in entry.choices]
        length(choices) == 4 || throw(ArgumentError(
            "MMLU item $(entry.id) has $(length(choices)) choices, expected 4",
        ))
        index = Int(entry.answer_index)
        0 <= index <= 3 || throw(ArgumentError(
            "MMLU item $(entry.id) has answer_index $index outside 0:3",
        ))
        push!(mmlu, MMLUItem(
            String(entry.id),
            String(entry.subject),
            String(entry.question),
            choices,
            index,
        ))
    end
    gsm8k = GSM8KItem[]
    for entry in get(payload, :gsm8k, ())
        answer = strip(String(entry.answer))
        _parse_numeric_answer(answer) === nothing && throw(ArgumentError(
            "GSM8K item $(entry.id) has a non-numeric answer $(repr(answer))",
        ))
        push!(gsm8k, GSM8KItem(String(entry.id), String(entry.question), String(answer)))
    end
    isempty(mmlu) && isempty(gsm8k) && throw(ArgumentError("task set is empty: $path"))
    ids = vcat([item.id for item in mmlu], [item.id for item in gsm8k])
    length(unique(ids)) == length(ids) || throw(ArgumentError("task ids are not unique"))
    return EvalTaskSet(mmlu, gsm8k, get(payload, :source, nothing), _sha256_hex(raw))
end

# ---------------------------------------------------------------------------
# Prompts. Both protocols are frozen here: a reported accuracy is only meaningful
# next to the exact prompt that produced it.
# ---------------------------------------------------------------------------

"""Base-model MMLU prompt scored by log likelihood; no chat template is applied."""
function mmlu_loglikelihood_prompt(item::MMLUItem)
    output = IOBuffer()
    print(output, "The following are multiple choice questions (with answers) about ",
        replace(item.subject, "_" => " "), ".\n\n")
    print(output, item.question, "\n")
    for (letter, choice) in zip(MMLU_LETTERS, item.choices)
        print(output, letter, ". ", choice, "\n")
    end
    print(output, "Answer:")
    return String(take!(output))
end

"""Chat messages for the generative MMLU protocol."""
function mmlu_chat_messages(item::MMLUItem)
    output = IOBuffer()
    print(output, "The following is a multiple choice question about ",
        replace(item.subject, "_" => " "), ".\n",
        "Answer with a single letter (A, B, C or D) and nothing else.\n\n")
    print(output, item.question, "\n")
    for (letter, choice) in zip(MMLU_LETTERS, item.choices)
        print(output, letter, ". ", choice, "\n")
    end
    return Any[Dict{Symbol,Any}(:role => "user", :content => String(take!(output)))]
end

"""Chat messages for the generative GSM8K protocol."""
function gsm8k_chat_messages(item::GSM8KItem)
    content = "Solve the problem. Reason step by step, then give the final answer " *
        "on the last line in the form \"#### <number>\".\n\n" * item.question
    return Any[Dict{Symbol,Any}(:role => "user", :content => content)]
end

# ---------------------------------------------------------------------------
# Answer extraction. Both rules are deliberately mechanical and documented; an
# unparseable completion returns `nothing` and is reported separately.
# ---------------------------------------------------------------------------

"""Every standalone `A`/`B`/`C`/`D`, as `(offset of the letter, letter, is_marked)`.

`is_marked` records whether the letter is immediately followed by an option marker
(`.`, `)`, `:`) or ends the text — the shapes a model uses when it is naming an
option rather than writing the English article "A" or a variable called `B`.
"""
function _mmlu_letter_positions(text::AbstractString)
    found = Tuple{Int,String,Bool}[]
    for match in eachmatch(r"(?:^|[^A-Za-z])([ABCD])(?![A-Za-z])", text)
        capture = only(match.captures)
        offset = match.offsets[1]
        stop = offset + ncodeunits(capture) - 1
        following = stop < lastindex(text) ?
            text[nextind(text, stop)] : nothing
        marked = following === nothing || following in ('.', ')', ':', ',')
        push!(found, (offset, String(capture), marked))
    end
    return found
end

"""
    extract_mmlu_letter(completion)

The answer letter a completion commits to, or `nothing`.

Two rules, because a bare capital letter is ambiguous in running English text:

  * after the last case-insensitive `answer`, any standalone `A`/`B`/`C`/`D` counts —
    a model that has just written "answer" is naming an option;
  * with no `answer` anywhere, only a *marked* letter counts (`A.`, `B)`, `C:`, or a
    letter ending the text). Without this, "A factor group of a non-Abelian group…"
    scores as option A. That is not hypothetical: on the frozen Qwen3-1.7B run, 24 of
    200 completions reach this branch and the permissive rule returned `A` for 18 of
    them against a gold base rate of 27%.
"""
function extract_mmlu_letter(completion::AbstractString)
    text = replace(String(completion), r"[*`_]" => "")
    positions = _mmlu_letter_positions(text)
    isempty(positions) && return nothing
    anchor = 0
    for match in eachmatch(r"(?i)answer", text)
        anchor = match.offset
    end
    if anchor > 0
        for (offset, letter, _) in positions
            offset >= anchor && return letter
        end
    end
    for (_, letter, marked) in positions
        marked && return letter
    end
    return nothing
end

"""
Parse a numeric answer string, tolerating separators a model may emit.

Restricted to plain decimal notation on purpose. `tryparse(Float64, ...)` also accepts
`"nan"`, `"inf"` and `"0x10"`; a fixture carrying `"nan"` would then load cleanly and
sit in the accuracy denominator while being impossible to score correct, because
`isapprox(x, NaN)` is false for every `x`.
"""
function _parse_numeric_answer(text::AbstractString)
    cleaned = replace(String(text), "," => "", "\$" => "", " " => "")
    cleaned = strip(cleaned, ['.', ':', '*', '\n', '\r', '"', '\''])
    occursin(r"^[+-]?\d+(?:\.\d+)?$", cleaned) || return nothing
    value = tryparse(Float64, cleaned)
    value === nothing && return nothing
    return isfinite(value) ? value : nothing
end

"""
    extract_gsm8k_answer(completion)

The numeric answer a completion commits to, or `nothing`.

Rule: if the text contains `####`, parse the first number after the last `####`;
otherwise parse the last number anywhere in the text.
"""
function extract_gsm8k_answer(completion::AbstractString)
    text = String(completion)
    number = r"-?\d[\d,]*(?:\.\d+)?"
    marker = findlast("####", text)
    if marker !== nothing
        tail = SubString(text, last(marker) + 1)
        found = match(number, tail)
        found === nothing || return _parse_numeric_answer(found.match)
    end
    matches = collect(eachmatch(number, text))
    isempty(matches) && return nothing
    return _parse_numeric_answer(last(matches).match)
end

"""Numeric comparison used for GSM8K, tolerant of `18` versus `18.0`."""
function gsm8k_answer_matches(expected::AbstractString, predicted)
    predicted === nothing && return false
    target = _parse_numeric_answer(expected)
    target === nothing && return false
    return isapprox(predicted, target; atol=1e-6, rtol=0)
end

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

"""Log probability of `token` under one column of logits, computed in Float64."""
function _column_logprob(column::AbstractVector, token::Integer)
    values = Float64.(column)
    ceiling = maximum(values)
    return values[token] - (ceiling + log(sum(exp.(value - ceiling) for value in values)))
end

"""
    mmlu_loglikelihood_scores(model, parameters, tokenizer, item; path=:auto, forward)

Unnormalized log likelihood of each ` A` / ` B` / ` C` / ` D` continuation.

`path` selects how the scores are obtained:

  * `:fast` — one forward over the prompt, reading all four single-token log
    probabilities off the last position. Four times cheaper.
  * `:general` — one forward per continuation over prompt+continuation, scored at the
    predicting position. This is what a reference implementation does.
  * `:auto` (default) — `:fast` when every continuation is a single token, else
    `:general`.

The two are identical in exact arithmetic and **not** identical in BF16: changing the
sequence length changes reduction order. On the frozen Qwen3-0.6B run they disagreed
on 6 of 200 decisions. Comparisons against an external reference must therefore use
`:general`, or they measure the shortcut as well as the dtype.
"""
function mmlu_loglikelihood_scores(
    model,
    parameters,
    tokenizer::HFQwen3Tokenizer,
    item::MMLUItem;
    path::Symbol=:auto,
    forward=hf_qwen3_bf16_accel_forward,
)
    path in (:auto, :fast, :general) || throw(ArgumentError(
        "path must be :auto, :fast or :general",
    ))
    prompt = mmlu_loglikelihood_prompt(item)
    prompt_ids = encode(tokenizer, prompt; add_special_tokens=false)
    continuations = [
        encode(tokenizer, continuation; add_special_tokens=false)
        for continuation in MMLU_CONTINUATIONS
    ]
    path === :fast && !all(ids -> length(ids) == 1, continuations) && throw(ArgumentError(
        "path=:fast requires single-token continuations",
    ))
    if path !== :general && all(ids -> length(ids) == 1, continuations)
        result = forward(model, parameters, reshape(prompt_ids, :, 1))
        column = Array(result.logits[:, end, 1])
        return Float64[_column_logprob(column, only(ids)) for ids in continuations],
            length(prompt_ids)
    end
    scores = Float64[]
    for ids in continuations
        tokens = vcat(prompt_ids, ids)
        result = forward(model, parameters, reshape(tokens, :, 1))
        logits = Array(result.logits)
        total = 0.0
        for (offset, token) in enumerate(ids)
            total += _column_logprob(
                view(logits, :, length(prompt_ids) + offset - 1, 1),
                token,
            )
        end
        push!(scores, total)
    end
    return scores, length(prompt_ids)
end

"""One scored item."""
struct EvalItemResult
    id::String
    subject::String
    protocol::Symbol
    prompt_sha256::String
    prompt_token_count::Int
    completion::String
    extracted::String
    expected::String
    correct::Bool
    parsed::Bool
    detail::Any
end

"""
    accuracy_report(results)

Accuracy with its Wilson 95% interval, plus how many items were unparseable. The
interval is always reported: at a few hundred items the point estimate alone
overstates how much has been measured.
"""
function accuracy_report(results::AbstractVector{EvalItemResult})
    total = length(results)
    total > 0 || throw(ArgumentError("no results to report"))
    correct = count(result -> result.correct, results)
    unparsed = count(result -> !result.parsed, results)
    interval = wilson_interval(correct, total)
    return (;
        total,
        correct,
        unparsed,
        accuracy=interval.point,
        wilson_lower=interval.lower,
        wilson_upper=interval.upper,
    )
end

"""Per-subject breakdown, ordered by subject name."""
function subject_report(results::AbstractVector{EvalItemResult})
    subjects = sort(unique(result.subject for result in results))
    return [
        begin
            selected = filter(result -> result.subject == subject, results)
            report = accuracy_report(selected)
            (; subject, report.total, report.correct, report.accuracy)
        end
        for subject in subjects if !isempty(subject)
    ]
end
