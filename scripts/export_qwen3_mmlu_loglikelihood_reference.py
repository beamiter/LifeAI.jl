#!/usr/bin/env python3
"""Export a per-item HuggingFace loglikelihood reference for Qwen3 on MMLU.

This is the Python/HuggingFace side of the Chapter 37 task-quality baseline.
It reads the MMLU half of the frozen task fixture
(``fixtures/eval_tasks.json``), scores every item with the standard
lm-eval-harness *unnormalized* loglikelihood protocol, and writes one record
per item so the Julia evaluation pipeline can be diffed against it question by
question -- not just on the aggregate accuracy number.

Protocol (the Julia side replicates this exactly; do not "improve" it here
without changing it there too):

1.  Base prompt, built verbatim, with the subject's underscores turned into
    spaces::

        The following are multiple choice questions (with answers) about {subject}.

        {question}
        A. {choice0}
        B. {choice1}
        C. {choice2}
        D. {choice3}
        Answer:

    No chat template. This is a base-model protocol, so the Qwen3 chat
    template is deliberately NOT applied, and every ``tokenizer(...)`` call
    passes ``add_special_tokens=False``.

2.  For each of the four continuations " A", " B", " C", " D": encode
    ``prompt + continuation`` in one go, encode ``prompt`` on its own, and
    take the continuation's token ids as the tail of the joint encoding past
    the prompt's length. This is lm-eval-harness ``_encode_pair`` -- encoding
    the pair jointly is what keeps a BPE merge across the prompt/continuation
    seam from silently changing the scored tokens.

3.  One forward pass over the concatenation, then sum ``log_softmax`` at the
    continuation's token positions. **No length normalization** (all four
    continuations are the same shape here anyway, but the harness convention
    is the unnormalized sum and that is what is recorded).

4.  Prediction = argmax over the four sums.

Numerics: float32 on CPU under ``torch.no_grad()``, eager attention, batch
size 1, no padding. The checkpoint ships bfloat16; it is upcast on load. A
float32 batch-1 unpadded forward is the most reproducible thing to hand the
Julia side -- no padding masks and no batching-induced GEMM reassociation to
explain away when the two sides disagree in the last digits. CPU reduction
order does depend on the thread count, so the effective thread count is
recorded in the output.

Environment: requirements/chapter37-eval-reference.txt (pinned to the same
torch/transformers as requirements/week22-reference.txt).

Example::

    "$HOME/.venvs/lifeai-chapter37-eval/bin/python" \
        scripts/export_qwen3_mmlu_loglikelihood_reference.py \
        --out test/episodes/episode07_agent_closed_loop/\
chapter37_qwen3_task_quality/fixtures/\
mmlu_loglikelihood_reference_qwen3_0_6b.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

DEFAULT_MODEL_DIR = (
    "/home/yj/models/huggingface/Qwen/Qwen3-0.6B/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
DEFAULT_TASKS = (
    "test/episodes/episode07_agent_closed_loop/chapter37_qwen3_task_quality/"
    "fixtures/eval_tasks.json"
)

CHOICE_LETTERS = ("A", "B", "C", "D")
# The four scored continuations. The leading space is part of the
# continuation, not of the prompt: the prompt ends at "Answer:" with no
# trailing whitespace, which is what lets the joint-encoding split below be a
# clean token boundary.
CONTINUATIONS = tuple(" " + letter for letter in CHOICE_LETTERS)

PROMPT_HEADER = (
    "The following are multiple choice questions (with answers) about {subject}.\n\n"
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_prompt(subject: str, question: str, choices: list) -> str:
    """Build the base-model MMLU prompt for one item."""
    if len(choices) != len(CHOICE_LETTERS):
        raise ValueError(
            f"expected {len(CHOICE_LETTERS)} choices, got {len(choices)}"
        )
    parts = [PROMPT_HEADER.format(subject=subject.replace("_", " ")), question]
    for letter, choice in zip(CHOICE_LETTERS, choices):
        parts.append(f"\n{letter}. {choice}")
    parts.append("\nAnswer:")
    return "".join(parts)


def encode_pair(tokenizer, context: str, continuation: str):
    """lm-eval-harness style joint encoding of (context, continuation).

    Returns ``(whole_ids, context_len)``. The continuation's ids are
    ``whole_ids[context_len:]``.
    """
    # The harness migrates trailing context whitespace into the continuation.
    # Our prompts end in "Answer:", so this is a no-op -- assert it rather
    # than carrying dead code that could quietly start firing.
    if context != context.rstrip():
        raise AssertionError(
            "prompt ends in whitespace; the harness whitespace migration "
            "would apply and this script does not implement it"
        )
    whole_ids = tokenizer(context + continuation, add_special_tokens=False)[
        "input_ids"
    ]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    context_len = len(context_ids)
    if whole_ids[:context_len] != context_ids:
        raise AssertionError(
            "tokenizer merged across the prompt/continuation boundary; the "
            "harness split is not valid for this item"
        )
    if len(whole_ids) <= context_len:
        raise AssertionError("continuation encoded to zero tokens")
    return whole_ids, context_len


def loglikelihood(model, torch, whole_ids, context_len: int) -> float:
    """Sum of log-softmax at the continuation's token positions."""
    input_ids = torch.tensor([whole_ids], dtype=torch.long)
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits[0]
    # Token at position i is predicted by the logits at position i-1.
    scored = logits[context_len - 1 : len(whole_ids) - 1, :]
    targets = torch.tensor(whole_ids[context_len:], dtype=torch.long)
    logprobs = torch.log_softmax(scored.float(), dim=-1)
    return logprobs.gather(-1, targets.unsqueeze(-1)).sum().item()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Export per-item Qwen3 MMLU loglikelihood scores as a HuggingFace "
            "reference for the Julia evaluation pipeline."
        )
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="local HuggingFace snapshot directory (default: %(default)s)",
    )
    parser.add_argument(
        "--tasks",
        default=DEFAULT_TASKS,
        help="task fixture JSON (default: %(default)s)",
    )
    parser.add_argument("--out", required=True, help="output JSON path")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N MMLU items (smoke tests / partial runs)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help=(
            "torch CPU thread count; 0 (default) leaves the torch default. "
            "The effective value is recorded in the output because CPU "
            "float32 reduction order depends on it."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    model_dir = os.path.abspath(os.path.expanduser(args.model_dir))
    if not os.path.isdir(model_dir):
        raise SystemExit(f"model directory does not exist: {model_dir}")
    for required in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not os.path.isfile(os.path.join(model_dir, required)):
            raise SystemExit(f"model directory is missing {required}: {model_dir}")

    tasks_path = os.path.abspath(os.path.expanduser(args.tasks))
    if not os.path.isfile(tasks_path):
        raise SystemExit(f"task fixture does not exist: {tasks_path}")

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    num_threads = torch.get_num_threads()

    tasks_sha256 = sha256_file(tasks_path)
    with open(tasks_path, "r", encoding="utf-8") as handle:
        tasks = json.load(handle)
    mmlu_items = tasks["mmlu"]
    if args.limit is not None:
        if args.limit <= 0:
            raise SystemExit("--limit must be positive")
        mmlu_items = mmlu_items[: args.limit]
    if not mmlu_items:
        raise SystemExit("no MMLU items to score")

    # Assertion 1: the config loads and really is the model we think it is.
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    if config.model_type != "qwen3":
        raise SystemExit(f"unexpected model_type: {config.model_type!r}")

    # Assertion 2: the tokenizer carries a chat template. We do NOT use it
    # (this is a base-model protocol) but its digest pins which tokenizer
    # snapshot produced this fixture.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise SystemExit("tokenizer has no chat_template")
    chat_template_sha256 = sha256_text(chat_template)

    print(f"model_dir            {model_dir}", file=sys.stderr)
    print(f"torch                {torch.__version__}", file=sys.stderr)
    print(f"transformers         {transformers.__version__}", file=sys.stderr)
    print(f"threads              {num_threads}", file=sys.stderr)
    print(f"tasks_sha256         {tasks_sha256}", file=sys.stderr)
    print(f"chat_template_sha256 {chat_template_sha256}", file=sys.stderr)
    print(f"items                {len(mmlu_items)}", file=sys.stderr)

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()
    print(
        f"model loaded in {time.perf_counter() - load_start:.1f}s "
        f"(dtype={next(model.parameters()).dtype})",
        file=sys.stderr,
    )

    revision = os.path.basename(model_dir)
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        revision = None

    items = []
    correct_count = 0
    run_start = time.perf_counter()
    with torch.no_grad():
        for index, task in enumerate(mmlu_items):
            item_start = time.perf_counter()
            prompt = build_prompt(task["subject"], task["question"], task["choices"])

            logprobs = []
            prompt_token_count = None
            for continuation in CONTINUATIONS:
                whole_ids, context_len = encode_pair(tokenizer, prompt, continuation)
                if prompt_token_count is None:
                    prompt_token_count = context_len
                elif prompt_token_count != context_len:
                    raise AssertionError(
                        "prompt tokenized to different lengths across "
                        "continuations"
                    )
                logprobs.append(loglikelihood(model, torch, whole_ids, context_len))

            prediction_index = max(range(len(logprobs)), key=logprobs.__getitem__)
            answer_index = task["answer_index"]
            is_correct = prediction_index == answer_index
            correct_count += int(is_correct)

            items.append(
                {
                    "id": task["id"],
                    "subject": task["subject"],
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_token_count": prompt_token_count,
                    "logprobs": logprobs,
                    "prediction_index": prediction_index,
                    "answer_index": answer_index,
                    "correct": is_correct,
                }
            )

            elapsed = time.perf_counter() - item_start
            done = index + 1
            mean = (time.perf_counter() - run_start) / done
            print(
                f"[{done}/{len(mmlu_items)}] {task['id']} "
                f"tokens={prompt_token_count} {elapsed:.2f}s "
                f"(mean {mean:.2f}s, running acc {correct_count / done:.4f}, "
                f"eta {mean * (len(mmlu_items) - done) / 60:.1f}min)",
                file=sys.stderr,
            )

    total = len(items)
    payload = {
        "model_dir": model_dir,
        "revision": revision,
        "tasks_sha256": tasks_sha256,
        "protocol": "loglikelihood-unnormalized",
        "dtype": "float32",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "chat_template_sha256": chat_template_sha256,
        "num_threads": num_threads,
        "limit": args.limit,
        "items": items,
        "accuracy": correct_count / total,
        "correct_count": correct_count,
        "total": total,
    }

    out_path = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # json.dump uses float.__repr__, i.e. the shortest round-tripping form:
    # the logprobs go out at full float precision, unrounded.
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"wrote {out_path}\n"
        f"accuracy {correct_count}/{total} = {correct_count / total:.4f}\n"
        f"total {(time.perf_counter() - run_start) / 60:.1f}min",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
