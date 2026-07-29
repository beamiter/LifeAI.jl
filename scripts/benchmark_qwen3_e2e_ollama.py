#!/usr/bin/env python3
"""Benchmark an exact, already-rendered Qwen3 prompt through Ollama.

The client deliberately uses ``raw=true`` so Ollama does not apply a second
chat template.  Client timings use a monotonic clock; Ollama's final streaming
message supplies the server-side timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional


DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "qwen3:14b-q8_0"


def first_environment_value(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        if name in os.environ:
            return os.environ[name]
    return default


def normalize_endpoint(value: str) -> str:
    endpoint = value.strip()
    if not endpoint:
        raise ValueError("Ollama endpoint must not be empty")
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/api/generate"):
        endpoint += "/api/generate"
    return endpoint


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Qwen3 end-to-end latency through Ollama's streaming "
            "/api/generate endpoint. The prompt must already contain the exact "
            "chat template to benchmark."
        )
    )
    parser.add_argument(
        "--url",
        default=first_environment_value(
            "LIFEAI_OLLAMA_URL", "OLLAMA_URL", "OLLAMA_HOST", default=DEFAULT_ENDPOINT
        ),
        help=(
            "Ollama host or /api/generate URL "
            "(env: LIFEAI_OLLAMA_URL, OLLAMA_URL, or OLLAMA_HOST)"
        ),
    )
    parser.add_argument(
        "--model",
        default=first_environment_value(
            "LIFEAI_OLLAMA_MODEL", "OLLAMA_MODEL", default=DEFAULT_MODEL
        ),
        help="Ollama model tag (env: LIFEAI_OLLAMA_MODEL or OLLAMA_MODEL)",
    )
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument(
        "--prompt",
        help="exact rendered prompt string (or env: LIFEAI_QWEN3_PROMPT)",
    )
    prompt_group.add_argument(
        "--prompt-file",
        help=(
            "UTF-8 file containing the exact rendered prompt, read without "
            "newline normalization (or env: LIFEAI_QWEN3_PROMPT_FILE)"
        ),
    )
    prompt_group.add_argument(
        "--prompts-json",
        help=(
            "UTF-8 JSON file containing multiple cases with name, "
            "rendered_prompt, and LifeAI token_count; phase-aware inputs are "
            "explicit schedules and require --warmup=0 --samples=1 "
            "(or env: LIFEAI_QWEN3_PROMPTS_JSON/PROMPTS_JSON)"
        ),
    )
    parser.add_argument(
        "--expected-prompt-tokens",
        type=int,
        default=first_environment_value(
            "LIFEAI_QWEN3_EXPECTED_PROMPT_TOKENS", default=None
        ),
        help=(
            "expected LifeAI prompt token count for a single --prompt or "
            "--prompt-file; multi-case JSON carries this per case"
        ),
    )
    parser.add_argument(
        "--output",
        default=first_environment_value(
            "LIFEAI_OLLAMA_BENCHMARK_OUTPUT",
            "OLLAMA_BENCHMARK_OUTPUT",
            default="-",
        ),
        help="JSON output path, or - for stdout",
    )
    parser.add_argument(
        "--warmup",
        "--warmups",
        dest="warmups",
        type=int,
        default=first_environment_value(
            "LIFEAI_BENCHMARK_WARMUPS",
            "OLLAMA_BENCHMARK_WARMUPS",
            "LIFEAI_BENCHMARK_WARMUP",
            "WARMUPS",
            default="1",
        ),
        help=(
            "unreported warmup requests "
            "(env: LIFEAI_BENCHMARK_WARMUPS or WARMUPS)"
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=first_environment_value(
            "LIFEAI_BENCHMARK_SAMPLES",
            "OLLAMA_BENCHMARK_SAMPLES",
            "SAMPLES",
            default="5",
        ),
        help=(
            "measured requests "
            "(env: LIFEAI_BENCHMARK_SAMPLES or SAMPLES)"
        ),
    )
    parser.add_argument(
        "--num-predict",
        "--max-new-tokens",
        dest="num_predict",
        type=int,
        default=first_environment_value(
            "LIFEAI_QWEN3_NUM_PREDICT",
            "OLLAMA_NUM_PREDICT",
            "MAX_NEW_TOKENS",
            default="128",
        ),
        help=(
            "maximum generated tokens for every request "
            "(env: LIFEAI_QWEN3_NUM_PREDICT or MAX_NEW_TOKENS)"
        ),
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=first_environment_value(
            "LIFEAI_QWEN3_NUM_CTX", "OLLAMA_NUM_CTX", default="4096"
        ),
        help="context window for every request",
    )
    parser.add_argument(
        "--keep-alive",
        default=first_environment_value(
            "LIFEAI_OLLAMA_KEEP_ALIVE", "OLLAMA_KEEP_ALIVE", default="10m"
        ),
        help="Ollama keep_alive value used for every request",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=first_environment_value(
            "LIFEAI_OLLAMA_TIMEOUT", "OLLAMA_BENCHMARK_TIMEOUT", default="600"
        ),
        help="per-request HTTP timeout in seconds",
    )
    args = parser.parse_args(argv)

    if args.warmups < 0:
        parser.error("--warmup must be non-negative")
    if args.samples <= 0:
        parser.error("--samples must be positive")
    if args.num_predict <= 0:
        parser.error("--num-predict must be positive")
    if args.num_ctx <= 0:
        parser.error("--num-ctx must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if (
        args.expected_prompt_tokens is not None
        and args.expected_prompt_tokens < 0
    ):
        parser.error("--expected-prompt-tokens must be non-negative")
    if not args.model:
        parser.error("--model must not be empty")
    if not args.keep_alive:
        parser.error("--keep-alive must not be empty")

    if (
        args.prompt is None
        and args.prompt_file is None
        and args.prompts_json is None
    ):
        environment_prompt = first_environment_value(
            "LIFEAI_QWEN3_PROMPT", "OLLAMA_PROMPT"
        )
        environment_prompt_file = first_environment_value(
            "LIFEAI_QWEN3_PROMPT_FILE", "OLLAMA_PROMPT_FILE"
        )
        environment_prompts_json = first_environment_value(
            "LIFEAI_QWEN3_PROMPTS_JSON", "OLLAMA_PROMPTS_JSON", "PROMPTS_JSON"
        )
        environment_inputs = [
            value
            for value in (
                environment_prompt,
                environment_prompt_file,
                environment_prompts_json,
            )
            if value is not None
        ]
        if len(environment_inputs) > 1:
            parser.error(
                "set only one of LIFEAI_QWEN3_PROMPT, "
                "LIFEAI_QWEN3_PROMPT_FILE, and LIFEAI_QWEN3_PROMPTS_JSON"
            )
        if environment_prompts_json is not None:
            args.prompts_json = environment_prompts_json
        elif environment_prompt_file is not None:
            args.prompt_file = environment_prompt_file
        elif environment_prompt is not None:
            args.prompt = environment_prompt
        else:
            parser.error(
                "provide --prompt, --prompt-file, or --prompts-json (or set "
                "the corresponding LIFEAI_QWEN3_* environment variable)"
            )
    if args.prompts_json is not None and args.expected_prompt_tokens is not None:
        parser.error(
            "--expected-prompt-tokens is only for a single prompt; put "
            "token_count in every --prompts-json case"
        )

    try:
        args.url = normalize_endpoint(args.url)
    except ValueError as error:
        parser.error(str(error))
    return args


def describe_prompt(prompt: str, source: dict[str, Any]) -> dict[str, Any]:
    prompt_bytes = prompt.encode("utf-8")
    source.update(
        {
            "utf8_bytes": len(prompt_bytes),
            "characters": len(prompt),
            "sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        }
    )
    return source


def aliased_value(
    entry: dict[str, Any],
    aliases: tuple[str, ...],
    *,
    required: bool,
    label: str,
) -> Any:
    present = [(name, entry[name]) for name in aliases if name in entry]
    if not present:
        if required:
            raise ValueError(
                f"prompt case is missing {label}; accepted keys: "
                + ", ".join(aliases)
            )
        return None
    first_name, first_value = present[0]
    for name, value in present[1:]:
        if value != first_value:
            raise ValueError(
                f"prompt case has conflicting {label} values in "
                f"{first_name!r} and {name!r}"
            )
    return first_value


def read_prompt_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.prompts_json is not None:
        prompts_path = Path(args.prompts_json).expanduser().resolve()
        prompts_bytes = prompts_path.read_bytes()
        try:
            decoded = prompts_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                f"prompts JSON is not valid UTF-8: {prompts_path}"
            ) from error
        parsed = json.loads(decoded)
        if isinstance(parsed, list):
            entries = parsed
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("cases"), list):
                entries = parsed["cases"]
            elif isinstance(parsed.get("prompts"), list):
                entries = parsed["prompts"]
            else:
                raise ValueError(
                    "prompts JSON object must contain a 'cases' array"
                )
        else:
            raise ValueError("prompts JSON must be an array or an object with 'cases'")
        if not entries:
            raise ValueError("prompts JSON contains no cases")

        cases: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for case_index, entry in enumerate(entries, start=1):
            if not isinstance(entry, dict):
                raise ValueError(f"prompt case {case_index} is not a JSON object")
            name = aliased_value(
                entry,
                ("name", "case"),
                required=True,
                label="case name",
            )
            prompt = aliased_value(
                entry,
                ("rendered_prompt", "prompt"),
                required=True,
                label="rendered prompt",
            )
            expected_tokens = aliased_value(
                entry,
                (
                    "expected_lifeai_token_count",
                    "lifeai_token_count",
                    "lifeai_prompt_tokens",
                    "prompt_tokens",
                    "token_count",
                    "prompt_token_count",
                    "expected_prompt_eval_count",
                ),
                required=True,
                label="expected LifeAI token count",
            )
            expected_token_ids = aliased_value(
                entry,
                (
                    "prompt_token_ids_0_based",
                    "lifeai_prompt_token_ids_0_based",
                ),
                required=False,
                label="LifeAI prompt token IDs",
            )
            phase = entry.get("phase")
            if not isinstance(name, str) or not name:
                raise ValueError(f"prompt case {case_index} name must be a string")
            if name in seen_names:
                raise ValueError(f"duplicate prompt case name: {name!r}")
            seen_names.add(name)
            if not isinstance(prompt, str):
                raise ValueError(
                    f"rendered_prompt for case {name!r} must be a string"
                )
            if phase is not None and (
                not isinstance(phase, str) or not phase
            ):
                raise ValueError(
                    f"phase for case {name!r} must be a non-empty string"
                )
            if (
                isinstance(expected_tokens, bool)
                or not isinstance(expected_tokens, int)
                or expected_tokens < 0
            ):
                raise ValueError(
                    f"token_count for case {name!r} must be a non-negative integer"
                )
            if expected_token_ids is not None:
                if (
                    not isinstance(expected_token_ids, list)
                    or any(
                        isinstance(token_id, bool)
                        or not isinstance(token_id, int)
                        or token_id < 0
                        for token_id in expected_token_ids
                    )
                ):
                    raise ValueError(
                        f"prompt_token_ids_0_based for case {name!r} must be "
                        "an array of non-negative integers"
                    )
                if len(expected_token_ids) != expected_tokens:
                    raise ValueError(
                        f"prompt_token_ids_0_based length for case {name!r} "
                        f"is {len(expected_token_ids)}, expected {expected_tokens}"
                    )
            source = describe_prompt(
                prompt,
                {
                    "kind": "prompts_json",
                    "path": str(prompts_path),
                    "case_index": case_index,
                },
            )
            cases.append(
                {
                    "name": name,
                    "prompt": prompt,
                    "prompt_source": source,
                    "expected_lifeai_prompt_tokens": expected_tokens,
                    "expected_lifeai_prompt_token_ids_0_based": expected_token_ids,
                    "phase": phase,
                }
            )
        return cases

    if args.prompt_file is not None:
        prompt_path = Path(args.prompt_file).expanduser().resolve()
        prompt_bytes = prompt_path.read_bytes()
        try:
            prompt = prompt_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"prompt file is not valid UTF-8: {prompt_path}") from error
        source = {
            "kind": "file",
            "path": str(prompt_path),
        }
    else:
        prompt = args.prompt
        assert prompt is not None
        source = {
            "kind": "argument_or_environment",
        }
    return [
        {
            "name": "default",
            "prompt": prompt,
            "prompt_source": describe_prompt(prompt, source),
            "expected_lifeai_prompt_tokens": args.expected_prompt_tokens,
            "expected_lifeai_prompt_token_ids_0_based": None,
            "phase": None,
        }
    ]


def phase_plan(
    cases: list[dict[str, Any]],
    warmups: int,
    samples: int,
) -> Optional[dict[str, Any]]:
    phases = [case["phase"] for case in cases]
    if not any(phase is not None for phase in phases):
        return None
    if not all(phase is not None for phase in phases):
        raise ValueError(
            "prompts JSON must set phase on every case when any case uses it"
        )

    allowed = {"cold", "warmup", "measured", "semantic_smoke"}
    if not all(phase in allowed for phase in phases):
        raise ValueError(
            "phase must be cold, warmup, measured, or semantic_smoke"
        )
    if warmups != 0:
        raise ValueError(
            "phase-aware prompts JSON is an explicit request schedule; "
            "set --warmup=0 or WARMUPS=0"
        )
    if samples != 1:
        raise ValueError(
            "phase-aware prompts JSON is an explicit request schedule; "
            "set --samples=1 or SAMPLES=1"
        )
    if phases.count("cold") != 1:
        raise ValueError(
            "phase-aware prompts JSON requires exactly one cold case"
        )
    if phases[0] != "cold":
        raise ValueError(
            "the cold case must be first in phase-aware prompts JSON"
        )
    if "measured" not in phases:
        raise ValueError(
            "phase-aware prompts JSON requires at least one measured case"
        )

    first_measured = phases.index("measured")
    if any(
        phase == "warmup" and index > first_measured
        for index, phase in enumerate(phases)
    ):
        raise ValueError(
            "warmup cases must precede measured cases in phase-aware prompts JSON"
        )
    if any(
        phase == "semantic_smoke" and index < first_measured
        for index, phase in enumerate(phases)
    ):
        raise ValueError("semantic_smoke cases must follow measured cases")

    measured_case_names = [
        case["name"] for case in cases if case["phase"] == "measured"
    ]
    excluded_case_names = [
        case["name"] for case in cases if case["phase"] != "measured"
    ]
    return {
        "enabled": True,
        "scheduling": "one request per case in JSON order",
        "case_order": [case["name"] for case in cases],
        "phases": phases,
        "measured_case_names": measured_case_names,
        "excluded_from_measured_summary": excluded_case_names,
    }


def request_payload(args: argparse.Namespace, prompt: str) -> dict[str, Any]:
    return {
        "model": args.model,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "think": False,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": 0.0,
            "repeat_penalty": 1.0,
            "repeat_last_n": 0,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
            "seed": 0,
        },
    }


def optional_integer(message: dict[str, Any], key: str) -> Optional[int]:
    value = message.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Ollama final field {key!r} is not an integer: {value!r}")
    return value


def nanoseconds_to_seconds(value: Optional[int]) -> Optional[float]:
    return None if value is None else value / 1_000_000_000.0


def rate(count: Optional[int], duration_ns: Optional[int]) -> Optional[float]:
    if count is None or duration_ns is None or duration_ns <= 0:
        return None
    return count * 1_000_000_000.0 / duration_ns


def run_sample(
    endpoint: str,
    payload: dict[str, Any],
    timeout: float,
    sample_index: int,
) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Accept": "application/x-ndjson",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "LifeAI-Qwen3-E2E-Benchmark/1",
        },
        method="POST",
    )

    start_ns = time.perf_counter_ns()
    chunk_receipt_ns: list[int] = []
    completion_parts: list[str] = []
    final_message: Optional[dict[str, Any]] = None
    final_receipt_ns: Optional[int] = None
    http_status: Optional[int] = None

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.status
            for line_number, line in enumerate(response, start=1):
                receipt_ns = time.perf_counter_ns()
                if not line.strip():
                    continue
                try:
                    message = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        f"invalid Ollama NDJSON at response line {line_number}"
                    ) from error
                if not isinstance(message, dict):
                    raise RuntimeError(
                        f"Ollama response line {line_number} is not a JSON object"
                    )
                if "error" in message:
                    raise RuntimeError(f"Ollama error: {message['error']}")

                response_piece = message.get("response", "")
                if not isinstance(response_piece, str):
                    raise RuntimeError(
                        f"Ollama response line {line_number} has a non-string "
                        "'response' field"
                    )
                if response_piece:
                    chunk_receipt_ns.append(receipt_ns)
                    completion_parts.append(response_piece)

                if message.get("done") is True:
                    final_message = message
                    final_receipt_ns = receipt_ns
                    break
    except urllib.error.HTTPError as error:
        try:
            detail = error.read().decode("utf-8", errors="replace").strip()
        except Exception:
            detail = ""
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"Ollama HTTP {error.code}{suffix}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot reach Ollama at {endpoint}: {error.reason}") from error

    if final_message is None or final_receipt_ns is None:
        raise RuntimeError("Ollama stream ended without a final done=true message")

    total_duration_ns = optional_integer(final_message, "total_duration")
    load_duration_ns = optional_integer(final_message, "load_duration")
    prompt_eval_duration_ns = optional_integer(final_message, "prompt_eval_duration")
    eval_duration_ns = optional_integer(final_message, "eval_duration")
    prompt_eval_count = optional_integer(final_message, "prompt_eval_count")
    eval_count = optional_integer(final_message, "eval_count")
    context_present = "context" in final_message
    final_context = final_message.get("context")

    received_offsets_seconds = [
        (receipt_ns - start_ns) / 1_000_000_000.0
        for receipt_ns in chunk_receipt_ns
    ]
    chunk_itl_seconds = [
        (current_ns - previous_ns) / 1_000_000_000.0
        for previous_ns, current_ns in zip(
            chunk_receipt_ns, chunk_receipt_ns[1:]
        )
    ]
    ttft_seconds = (
        None
        if not chunk_receipt_ns
        else (chunk_receipt_ns[0] - start_ns) / 1_000_000_000.0
    )
    total_wall_seconds = (final_receipt_ns - start_ns) / 1_000_000_000.0
    api_total_seconds = nanoseconds_to_seconds(total_duration_ns)
    completion = "".join(completion_parts)

    return {
        "sample": sample_index,
        "http_status": http_status,
        "client": {
            "ttft_seconds": ttft_seconds,
            "chunk_itl_seconds": chunk_itl_seconds,
            "chunk_received_offsets_seconds": received_offsets_seconds,
            "response_chunk_count": len(chunk_receipt_ns),
            "total_wall_seconds": total_wall_seconds,
            "wall_minus_api_total_seconds": (
                None
                if api_total_seconds is None
                else total_wall_seconds - api_total_seconds
            ),
        },
        "ollama": {
            "model": final_message.get("model"),
            "created_at": final_message.get("created_at"),
            "done_reason": final_message.get("done_reason"),
            "total_duration_ns": total_duration_ns,
            "load_duration_ns": load_duration_ns,
            "prompt_eval_duration_ns": prompt_eval_duration_ns,
            "eval_duration_ns": eval_duration_ns,
            "total_duration_seconds": api_total_seconds,
            "load_duration_seconds": nanoseconds_to_seconds(load_duration_ns),
            "prompt_eval_duration_seconds": nanoseconds_to_seconds(
                prompt_eval_duration_ns
            ),
            "eval_duration_seconds": nanoseconds_to_seconds(eval_duration_ns),
            "prompt_eval_count": prompt_eval_count,
            "eval_count": eval_count,
            "context_present": context_present,
            "context": final_context if context_present else None,
            "context_token_count": (
                len(final_context) if isinstance(final_context, list) else None
            ),
            "prompt_eval_tokens_per_second": rate(
                prompt_eval_count, prompt_eval_duration_ns
            ),
            "eval_tokens_per_second": rate(eval_count, eval_duration_ns),
        },
        "completion": completion,
        "completion_utf8_bytes": len(completion.encode("utf-8")),
        "completion_sha256": hashlib.sha256(completion.encode("utf-8")).hexdigest(),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def distribution(values: Iterable[Optional[float]]) -> dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    return {
        "count": len(present),
        "median": statistics.median(present) if present else None,
        "p90": percentile(present, 0.90) if present else None,
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "percentile_method": "linear interpolation at (n - 1) * q",
        "client_ttft_seconds": distribution(
            sample["client"]["ttft_seconds"] for sample in samples
        ),
        "client_chunk_itl_seconds": distribution(
            interval
            for sample in samples
            for interval in sample["client"]["chunk_itl_seconds"]
        ),
        "client_total_wall_seconds": distribution(
            sample["client"]["total_wall_seconds"] for sample in samples
        ),
        "client_wall_minus_api_total_seconds": distribution(
            sample["client"]["wall_minus_api_total_seconds"] for sample in samples
        ),
        "api_total_duration_seconds": distribution(
            sample["ollama"]["total_duration_seconds"] for sample in samples
        ),
        "api_load_duration_seconds": distribution(
            sample["ollama"]["load_duration_seconds"] for sample in samples
        ),
        "api_prompt_eval_duration_seconds": distribution(
            sample["ollama"]["prompt_eval_duration_seconds"] for sample in samples
        ),
        "api_eval_duration_seconds": distribution(
            sample["ollama"]["eval_duration_seconds"] for sample in samples
        ),
        "api_prompt_eval_count": distribution(
            sample["ollama"]["prompt_eval_count"] for sample in samples
        ),
        "api_eval_count": distribution(
            sample["ollama"]["eval_count"] for sample in samples
        ),
        "api_prompt_eval_tokens_per_second": distribution(
            sample["ollama"]["prompt_eval_tokens_per_second"] for sample in samples
        ),
        "api_eval_tokens_per_second": distribution(
            sample["ollama"]["eval_tokens_per_second"] for sample in samples
        ),
    }


def write_report(report: dict[str, Any], output: str) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if output == "-":
        sys.stdout.write(rendered)
        return
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {output_path}", file=sys.stderr)


def context_prefix_validation(
    ollama_metrics: dict[str, Any],
    expected_prompt_token_ids: Optional[list[int]],
) -> dict[str, Any]:
    """Describe why raw-mode context cannot be used as a token-prefix gate.

    Ollama's generate API documents ``context`` as deprecated and explicitly
    states that raw mode does not return it.  An unexpected context value is
    retained for diagnostics, but its prompt-prefix semantics are therefore
    not a stable API contract that this fairness harness can hard-gate.
    """

    if not ollama_metrics["context_present"]:
        reason = "raw_mode_api_contract_does_not_return_context"
    elif expected_prompt_token_ids is None:
        reason = "lifeai_prompt_token_ids_unavailable"
    else:
        reason = "raw_mode_context_prefix_semantics_not_guaranteed"
    return {
        "checked": False,
        "matches": None,
        "reason": reason,
        "context_present": ollama_metrics["context_present"],
        "lifeai_prompt_token_ids_available": expected_prompt_token_ids is not None,
    }


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        prompt_cases = read_prompt_cases(args)
        resolved_phase_plan = phase_plan(
            prompt_cases, args.warmups, args.samples
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    try:
        case_results: list[dict[str, Any]] = []
        for case_index, prompt_case in enumerate(prompt_cases, start=1):
            payload = request_payload(args, prompt_case["prompt"])
            case_label = (
                f"{prompt_case['name']} ({case_index}/{len(prompt_cases)})"
            )
            for warmup_index in range(1, args.warmups + 1):
                print(
                    f"Ollama case {case_label}, warmup "
                    f"{warmup_index}/{args.warmups}",
                    file=sys.stderr,
                    flush=True,
                )
                run_sample(args.url, payload, args.timeout, warmup_index)

            samples: list[dict[str, Any]] = []
            for sample_index in range(1, args.samples + 1):
                print(
                    f"Ollama case {case_label}, sample "
                    f"{sample_index}/{args.samples}",
                    file=sys.stderr,
                    flush=True,
                )
                sample = run_sample(
                    args.url, payload, args.timeout, sample_index
                )
                expected_tokens = prompt_case[
                    "expected_lifeai_prompt_tokens"
                ]
                expected_token_ids = prompt_case[
                    "expected_lifeai_prompt_token_ids_0_based"
                ]
                prompt_count_matches = (
                    None
                    if expected_tokens is None
                    else sample["ollama"]["prompt_eval_count"]
                    == expected_tokens
                )
                eval_count_matches = (
                    sample["ollama"]["eval_count"] == args.num_predict
                )
                done_reason_matches = (
                    sample["ollama"]["done_reason"] == "length"
                )
                sample["validation"] = {
                    "expected_lifeai_prompt_tokens": expected_tokens,
                    "prompt_eval_count_matches_lifeai": prompt_count_matches,
                    "expected_eval_count": args.num_predict,
                    "eval_count_matches_num_predict": eval_count_matches,
                    "expected_done_reason": "length",
                    "done_reason_is_length": done_reason_matches,
                    "context_prompt_prefix": context_prefix_validation(
                        sample["ollama"], expected_token_ids
                    ),
                    "fairness_hard_gates_passed": (
                        prompt_count_matches is not False
                        and eval_count_matches
                        and done_reason_matches
                    ),
                }
                samples.append(sample)

            expected_tokens = prompt_case["expected_lifeai_prompt_tokens"]
            observed_counts = [
                sample["ollama"]["prompt_eval_count"] for sample in samples
            ]
            validation_checked = expected_tokens is not None
            validation_passed = (
                None
                if not validation_checked
                else all(count == expected_tokens for count in observed_counts)
            )
            eval_counts = [
                sample["ollama"]["eval_count"] for sample in samples
            ]
            done_reasons = [
                sample["ollama"]["done_reason"] for sample in samples
            ]
            eval_counts_passed = all(
                count == args.num_predict for count in eval_counts
            )
            done_reasons_passed = all(
                reason == "length" for reason in done_reasons
            )
            fairness_hard_gates_passed = (
                validation_passed is not False
                and eval_counts_passed
                and done_reasons_passed
            )
            case_results.append(
                {
                    "name": prompt_case["name"],
                    "phase": prompt_case["phase"],
                    "included_in_measured_summary": (
                        prompt_case["phase"] is None
                        or prompt_case["phase"] == "measured"
                    ),
                    "prompt_source": prompt_case["prompt_source"],
                    "prompt": prompt_case["prompt"],
                    "expected_lifeai_prompt_tokens": expected_tokens,
                    "expected_lifeai_prompt_token_ids_0_based": prompt_case[
                        "expected_lifeai_prompt_token_ids_0_based"
                    ],
                    "raw_samples": samples,
                    "summary": summarize(samples),
                    "validation": {
                        "checked": validation_checked,
                        "expected_lifeai_prompt_tokens": expected_tokens,
                        "observed_ollama_prompt_eval_counts": observed_counts,
                        "all_samples_match": validation_passed,
                        "expected_eval_count": args.num_predict,
                        "observed_ollama_eval_counts": eval_counts,
                        "all_eval_counts_match_num_predict": eval_counts_passed,
                        "expected_done_reason": "length",
                        "observed_ollama_done_reasons": done_reasons,
                        "all_done_reasons_are_length": done_reasons_passed,
                        "context_prompt_prefix_checked": False,
                        "context_prompt_prefix_reason": (
                            "raw_mode_context_prefix_semantics_not_guaranteed"
                        ),
                        "fairness_hard_gates_passed": (
                            fairness_hard_gates_passed
                        ),
                    },
                }
            )
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    checked_case_results = [
        case for case in case_results if case["validation"]["checked"]
    ]
    prompt_count_failed_case_names = [
        case["name"]
        for case in checked_case_results
        if not case["validation"]["all_samples_match"]
    ]
    eval_count_failed_case_names = [
        case["name"]
        for case in case_results
        if not case["validation"]["all_eval_counts_match_num_predict"]
    ]
    done_reason_failed_case_names = [
        case["name"]
        for case in case_results
        if not case["validation"]["all_done_reasons_are_length"]
    ]
    fairness_failed_case_names = [
        case["name"]
        for case in case_results
        if not case["validation"]["fairness_hard_gates_passed"]
    ]
    phased_execution: Optional[dict[str, Any]]
    if resolved_phase_plan is None:
        phased_execution = None
    else:
        measured_phase_samples = [
            sample
            for case in case_results
            if case["phase"] == "measured"
            for sample in case["raw_samples"]
        ]
        phased_execution = {
            **resolved_phase_plan,
            "measured_sample_count": len(measured_phase_samples),
            "measured_summary": summarize(measured_phase_samples),
        }
    report = {
        "schema_version": 1,
        "benchmark": "qwen3_e2e_ollama",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "clock": "time.perf_counter_ns",
        },
        "endpoint": args.url,
        "request": {
            "model": args.model,
            "raw": True,
            "stream": True,
            "think": False,
            "keep_alive": args.keep_alive,
            "options": payload["options"],
            "timeout_seconds": args.timeout,
            "phase_aware": resolved_phase_plan is not None,
        },
        "warmups": args.warmups,
        "samples": args.samples,
        "case_count": len(case_results),
        "cases": case_results,
        "phased_execution": phased_execution,
        "validation": {
            "prompt_eval_count_checked_cases": len(checked_case_results),
            "prompt_eval_count_failed_cases": prompt_count_failed_case_names,
            "all_checked_cases_match": (
                None
                if not checked_case_results
                else not prompt_count_failed_case_names
            ),
            "eval_count_failed_cases": eval_count_failed_case_names,
            "done_reason_failed_cases": done_reason_failed_case_names,
            "context_prompt_prefix_checked": False,
            "context_prompt_prefix_reason": (
                "Ollama raw mode does not provide stable context-prefix semantics"
            ),
            "fairness_hard_gate_failed_cases": fairness_failed_case_names,
            "all_fairness_hard_gates_pass": not fairness_failed_case_names,
        },
    }
    try:
        write_report(report, args.output)
    except OSError as error:
        print(f"error writing report: {error}", file=sys.stderr)
        return 2
    if fairness_failed_case_names:
        print(
            "error: Ollama fairness hard gate failed for case(s): "
            + ", ".join(fairness_failed_case_names),
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
