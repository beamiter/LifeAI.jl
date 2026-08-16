#!/usr/bin/env python3
"""Render the official Qwen3 Jinja chat template over the Chapter 36 case set.

This is the Python oracle that ``apply_qwen3_chat_template`` (Julia) is checked
against byte for byte.  It deliberately does *not* import ``transformers`` or
``torch``: the only thing needed to reproduce transformers' rendering is Jinja2
configured exactly the way ``transformers`` configures it in
``PreTrainedTokenizerBase._compile_jinja_template``::

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = lambda v, indent=None: json.dumps(
        v, ensure_ascii=False, indent=indent
    )

The template itself is read from ``tokenizer_config.json`` of a local Qwen3
checkpoint and its sha256 is asserted, so a checkpoint carrying a different
template fails loudly instead of silently producing a different oracle.

Dependencies are pinned in ``requirements/chapter36-template-oracle.txt``
(jinja2==3.1.6, markupsafe==3.0.3).

Example::

    uv venv --python 3.12 "$HOME/.venvs/lifeai-chapter36-oracle"
    uv pip install --python "$HOME/.venvs/lifeai-chapter36-oracle/bin/python" \\
        -r requirements/chapter36-template-oracle.txt
    "$HOME/.venvs/lifeai-chapter36-oracle/bin/python" \\
        scripts/export_qwen3_chat_template_reference.py \\
        --model-dir /path/to/Qwen3-4B/<revision> \\
        --out test/episodes/episode07_agent_closed_loop/\\
chapter36_qwen3_tools_chat_template/fixtures/chat_template_reference.json \\
        --template-out test/episodes/episode07_agent_closed_loop/\\
chapter36_qwen3_tools_chat_template/fixtures/official_chat_template.jinja
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# sha256 of the "chat_template" string of Qwen/Qwen3-4B revision
# 1cfa9a7208912126459214e8b04321603b3df60c (the same template ships with the
# other Qwen3 dense checkpoints).  Frozen on purpose: the Chapter 36 fixtures
# are only meaningful for this exact template.
EXPECTED_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASES = (
    REPO_ROOT
    / "test"
    / "episodes"
    / "episode07_agent_closed_loop"
    / "chapter36_qwen3_tools_chat_template"
    / "fixtures"
    / "chat_template_cases.json"
)

REQUIRED_CASE_KEYS = (
    "name",
    "messages",
    "add_generation_prompt",
    "enable_thinking",
)


class OracleError(RuntimeError):
    """A failure that should end the process with a readable message."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def import_jinja2():
    """Import Jinja2, turning the usual ModuleNotFoundError into advice."""
    try:
        import jinja2  # noqa: PLC0415
        import jinja2.ext  # noqa: PLC0415
        from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment dependent
        raise OracleError(
            f"cannot import jinja2 ({error}).\n"
            "Install the pinned oracle dependencies, e.g.\n"
            '  uv venv --python 3.12 "$HOME/.venvs/lifeai-chapter36-oracle"\n'
            '  uv pip install --python "$HOME/.venvs/lifeai-chapter36-oracle/bin/python" '
            "-r requirements/chapter36-template-oracle.txt\n"
            "See requirements/chapter36-template-oracle.txt for an offline "
            "fallback that only needs the wheels plus PYTHONPATH."
        ) from error
    return jinja2, ImmutableSandboxedEnvironment


def build_environment():
    """Reproduce transformers' ``_compile_jinja_template`` environment exactly."""
    jinja2, ImmutableSandboxedEnvironment = import_jinja2()

    def tojson(value: Any, indent: int | None = None) -> str:
        return json.dumps(value, ensure_ascii=False, indent=indent)

    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[jinja2.ext.loopcontrols],
    )
    environment.filters["tojson"] = tojson
    return jinja2, environment


def load_template_text(model_dir: Path) -> str:
    config_path = model_dir / "tokenizer_config.json"
    if not config_path.is_file():
        raise OracleError(f"no tokenizer_config.json under --model-dir: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OracleError(f"{config_path} is not valid JSON: {error}") from error
    template = config.get("chat_template")
    if template is None:
        raise OracleError(f'{config_path} has no "chat_template" field')
    if not isinstance(template, str):
        raise OracleError(
            f'{config_path}: "chat_template" is {type(template).__name__}, '
            "expected a string (chat-template lists are not supported)"
        )
    return template


def load_cases(cases_path: Path) -> list[dict[str, Any]]:
    if not cases_path.is_file():
        raise OracleError(f"cases file not found: {cases_path}")
    try:
        # Python 3.7+ dicts keep insertion order, and json.load builds them in
        # document order, so tool schemas reach `tojson` with the key order the
        # fixture file spells out.  That order is load bearing for the bytes.
        document = json.loads(cases_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OracleError(f"{cases_path} is not valid JSON: {error}") from error
    if not isinstance(document, dict) or "cases" not in document:
        raise OracleError(f'{cases_path} has no top-level "cases" array')
    cases = document["cases"]
    if not isinstance(cases, list) or not cases:
        raise OracleError(f'{cases_path}: "cases" must be a non-empty array')

    declared = document.get("template_sha256")
    if declared is not None and declared != EXPECTED_TEMPLATE_SHA256:
        raise OracleError(
            f'{cases_path}: "template_sha256" is {declared}, but this script is '
            f"frozen against {EXPECTED_TEMPLATE_SHA256}"
        )

    seen: dict[str, int] = {}
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise OracleError(f"{cases_path}: case #{index} is not an object")
        missing = [key for key in REQUIRED_CASE_KEYS if key not in case]
        if missing:
            raise OracleError(
                f"{cases_path}: case #{index} is missing {', '.join(missing)}"
            )
        name = case["name"]
        if not isinstance(name, str) or not name:
            raise OracleError(f'{cases_path}: case #{index} has a bad "name"')
        if name in seen:
            raise OracleError(
                f'{cases_path}: duplicate case name "{name}" '
                f"(cases #{seen[name]} and #{index})"
            )
        seen[name] = index
    return cases


def render_cases(template, cases: list[dict[str, Any]]) -> dict[str, str]:
    renders: dict[str, str] = {}
    for case in cases:
        name = case["name"]
        try:
            renders[name] = template.render(
                messages=case["messages"],
                tools=case.get("tools"),  # JSON null -> None, as transformers does
                add_generation_prompt=case["add_generation_prompt"],
                enable_thinking=case["enable_thinking"],
            )
        except Exception as error:  # noqa: BLE001 - report which case died
            raise OracleError(
                f'case "{name}" failed to render: '
                f"{type(error).__name__}: {error}"
            ) from error
    return renders


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export_qwen3_chat_template_reference.py",
        description=(
            "Render the official Qwen3 Jinja chat template over the Chapter 36 "
            "cases and write a byte-exact reference JSON. Needs jinja2 + "
            "markupsafe only (no torch, no transformers)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The chat template is read from <model-dir>/tokenizer_config.json and "
            "its sha256 must equal\n  " + EXPECTED_TEMPLATE_SHA256 + "\n"
            "otherwise the script exits non-zero and prints the actual digest."
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Local Qwen3 checkpoint directory containing tokenizer_config.json.",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
        help="Frozen case specification JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the reference JSON (UTF-8, indent=2).",
    )
    parser.add_argument(
        "--template-out",
        type=Path,
        default=None,
        help="Optional path to also dump the raw chat template text.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model_dir = args.model_dir.resolve()
        if not model_dir.is_dir():
            raise OracleError(f"--model-dir is not a directory: {model_dir}")

        template_text = load_template_text(model_dir)
        template_sha256 = sha256_text(template_text)
        if template_sha256 != EXPECTED_TEMPLATE_SHA256:
            raise OracleError(
                "chat_template sha256 mismatch\n"
                f"  expected: {EXPECTED_TEMPLATE_SHA256}\n"
                f"  actual:   {template_sha256}\n"
                f"  source:   {model_dir / 'tokenizer_config.json'}\n"
                "This checkpoint ships a different chat template; the Chapter 36 "
                "fixtures are frozen against the one above."
            )

        cases_path = args.cases.resolve()
        cases = load_cases(cases_path)

        jinja2, environment = build_environment()
        template = environment.from_string(template_text)
        renders = render_cases(template, cases)

        payload = {
            "model_dir": str(model_dir),
            "template_sha256": template_sha256,
            "cases_sha256": sha256_file(cases_path),
            "jinja2_version": jinja2.__version__,
            "renders": renders,
        }
        write_json(args.out, payload)

        if args.template_out is not None:
            args.template_out.parent.mkdir(parents=True, exist_ok=True)
            # newline="" so the template's own "\n" bytes survive verbatim and the
            # file's sha256 equals the sha256 of the chat_template string.
            args.template_out.write_text(
                template_text, encoding="utf-8", newline=""
            )

        print(
            f"rendered {len(renders)} case(s) from {cases_path}\n"
            f"  template sha256: {template_sha256}\n"
            f"  jinja2:          {jinja2.__version__}\n"
            f"  wrote:           {args.out.resolve()}"
            + (
                f"\n  wrote:           {args.template_out.resolve()}"
                if args.template_out is not None
                else ""
            )
        )
        return 0
    except OracleError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
