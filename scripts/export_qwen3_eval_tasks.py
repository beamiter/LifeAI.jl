#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 Chapter 37「Qwen3 dense 真实任务质量基线」所用的公开任务集 fixture。

本脚本只用 Python 标准库（urllib + json + hashlib），不依赖 datasets / pyarrow /
pandas，通过 HuggingFace datasets-server 的 JSON API 抓取上游原始数据，并把选定
的子集冻结成 `eval_tasks.json` + `eval_tasks_provenance.json`。

数据来源与许可
--------------
MMLU (Measuring Massive Multitask Language Understanding)
  - HuggingFace dataset: `cais/mmlu`，config `all`，split `test`
  - 论文: Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt,
    "Measuring Massive Multitask Language Understanding", ICLR 2021.
    (另见 Hendrycks et al., "Aligning AI With Shared Human Values", ICLR 2021)
  - 上游仓库: https://github.com/hendrycks/test
  - 许可: MIT License

GSM8K (Grade School Math 8K)
  - HuggingFace dataset: `openai/gsm8k`，config `main`，split `test`
  - 论文: Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek,
    Hilton, Nakano, Hesse, Schulman,
    "Training Verifiers to Solve Math Word Problems", arXiv:2110.14168, 2021.
  - 上游仓库: https://github.com/openai/grade-school-math
  - 许可: MIT License

两个数据集均为 MIT 许可，允许再分发；本仓库冻结的是其 test split 的一个确定性
子集（见下方 SUBJECTS / PER_SUBJECT / GSM8K_FIRST_N 常量），不做任何改写，
仅做字段裁剪：GSM8K 的 `answer` 只保留 "#### " 之后的最终数值字符串（去掉逗号），
完整推理过程不进入 fixture。

选题规则（写死为常量，无随机性）
--------------------------------
- MMLU: 8 个 subject，每个取该 subject 在 test split 中的**前 25 题**（数据集原始
  顺序，按上游 row_idx 升序）。合计 200 题。
- GSM8K: test split 的**前 150 题**（数据集原始顺序）。

用法
----
    python3 scripts/export_qwen3_eval_tasks.py \
        --out test/episodes/episode07_agent_closed_loop/chapter37_qwen3_task_quality/fixtures/eval_tasks.json

provenance 文件默认写到 `--out` 同目录的 `eval_tasks_provenance.json`。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# 冻结常量：选题规则。修改这里等于换一个 fixture 版本，必须同步更新下游 sha256。
# ---------------------------------------------------------------------------

DATASETS_SERVER = "https://datasets-server.huggingface.co"

MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"
MMLU_SUBJECTS = (
    "abstract_algebra",
    "college_computer_science",
    "high_school_mathematics",
    "professional_medicine",
    "world_religions",
    "moral_scenarios",
    "econometrics",
    "machine_learning",
)
PER_SUBJECT = 25
MMLU_SELECTION = "first N rows of each subject, in dataset order"

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"
GSM8K_FIRST_N = 150
GSM8K_SELECTION = "first N rows in dataset order"

PAGE_LENGTH = 100  # datasets-server 单次请求上限

LICENSES = {
    "mmlu": {
        "name": "MMLU (Measuring Massive Multitask Language Understanding)",
        "hf_dataset": MMLU_DATASET,
        "license": "MIT",
        "citation": (
            "Hendrycks, Burns, Basart, Zou, Mazeika, Song, Steinhardt. "
            "Measuring Massive Multitask Language Understanding. ICLR 2021."
        ),
        "upstream": "https://github.com/hendrycks/test",
    },
    "gsm8k": {
        "name": "GSM8K (Grade School Math 8K)",
        "hf_dataset": GSM8K_DATASET,
        "license": "MIT",
        "citation": (
            "Cobbe, Kosaraju, Bavarian, Chen, Jun, Kaiser, Plappert, Tworek, "
            "Hilton, Nakano, Hesse, Schulman. Training Verifiers to Solve Math "
            "Word Problems. arXiv:2110.14168, 2021."
        ),
        "upstream": "https://github.com/openai/grade-school-math",
    },
}


class ExportError(RuntimeError):
    """校验失败或抓取失败，脚本必须以非零码退出。"""


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def build_url(path: str, params: dict) -> str:
    return f"{DATASETS_SERVER}/{path}?" + urllib.parse.urlencode(params)


def fetch_json(url: str, timeout: float, retries: int) -> dict:
    """GET 一个 JSON 接口，失败时指数退避重试。"""
    last_err = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "LifeAI.jl/export_qwen3_eval_tasks (stdlib urllib)",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            return json.loads(payload.decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as err:
            last_err = err
            if attempt >= retries:
                break
            backoff = 2.0 ** attempt
            print(
                f"  [retry {attempt + 1}/{retries}] {type(err).__name__}: {err} "
                f"-> sleep {backoff:.1f}s",
                file=sys.stderr,
            )
            time.sleep(backoff)
    raise ExportError(f"抓取失败（已重试 {retries} 次）: {url}\n  最后错误: {last_err!r}")


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------


def mmlu_url(subject: str, offset: int, length: int) -> str:
    return build_url(
        "filter",
        {
            "dataset": MMLU_DATASET,
            "config": MMLU_CONFIG,
            "split": MMLU_SPLIT,
            "where": f"\"subject\"='{subject}'",
            "offset": offset,
            "length": length,
        },
    )


def fetch_mmlu_subject(subject: str, timeout: float, retries: int) -> tuple[list, dict]:
    """抓取单个 subject 的前 PER_SUBJECT 题，返回 (items, meta)。"""
    raw_rows: list[dict] = []
    urls: list[str] = []
    num_rows_total = None
    offset = 0
    while len(raw_rows) < PER_SUBJECT:
        url = mmlu_url(subject, offset, PAGE_LENGTH)
        urls.append(url)
        data = fetch_json(url, timeout, retries)
        num_rows_total = data.get("num_rows_total")
        page = data.get("rows", [])
        if not page:
            break
        raw_rows.extend(page)
        offset += len(page)
        if num_rows_total is not None and offset >= num_rows_total:
            break

    if num_rows_total is None:
        raise ExportError(f"MMLU subject={subject}: 接口未返回 num_rows_total")
    if len(raw_rows) < PER_SUBJECT:
        raise ExportError(
            f"MMLU subject={subject}: 只抓到 {len(raw_rows)} 行，不足 {PER_SUBJECT} 行"
        )

    # 严格按上游 row_idx 升序 = 数据集原始顺序，然后取前 N。
    raw_rows.sort(key=lambda r: r["row_idx"])
    selected = raw_rows[:PER_SUBJECT]

    items = []
    row_indices = []
    for i, entry in enumerate(selected):
        row_idx = entry["row_idx"]
        row = entry["row"]
        truncated = entry.get("truncated_cells") or []
        if truncated:
            raise ExportError(
                f"MMLU subject={subject} row_idx={row_idx}: datasets-server 截断了单元格 "
                f"{truncated}，冻结这样的题目会污染评测，拒绝写出"
            )
        if row.get("subject") != subject:
            raise ExportError(
                f"MMLU row_idx={row_idx}: subject 不匹配，期望 {subject!r} 实得 {row.get('subject')!r}"
            )

        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ExportError(f"MMLU subject={subject} row_idx={row_idx}: question 为空或非字符串")

        choices = row.get("choices")
        if not isinstance(choices, list) or len(choices) != 4:
            raise ExportError(
                f"MMLU subject={subject} row_idx={row_idx}: choices 必须恰好 4 个，"
                f"实得 {len(choices) if isinstance(choices, list) else type(choices).__name__}"
            )
        for c_i, choice in enumerate(choices):
            if not isinstance(choice, str) or not choice.strip():
                raise ExportError(
                    f"MMLU subject={subject} row_idx={row_idx}: choices[{c_i}] 为空或非字符串"
                )

        answer_index = row.get("answer")
        if isinstance(answer_index, bool) or not isinstance(answer_index, int):
            raise ExportError(
                f"MMLU subject={subject} row_idx={row_idx}: answer 非整数（{answer_index!r}）"
            )
        if not 0 <= answer_index <= 3:
            raise ExportError(
                f"MMLU subject={subject} row_idx={row_idx}: answer_index={answer_index} 越界，必须在 0..3"
            )

        items.append(
            {
                "id": f"mmlu/{subject}/{i}",
                "subject": subject,
                "question": question,
                "choices": choices,
                "answer_index": answer_index,
            }
        )
        row_indices.append(row_idx)

    meta = {
        "subject": subject,
        "num_rows_total": num_rows_total,
        "taken": len(items),
        "urls": urls,
        "upstream_row_idx_first": row_indices[0],
        "upstream_row_idx_last": row_indices[-1],
        "upstream_row_idx": row_indices,
    }
    return items, meta


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------


def gsm8k_url(offset: int, length: int) -> str:
    return build_url(
        "rows",
        {
            "dataset": GSM8K_DATASET,
            "config": GSM8K_CONFIG,
            "split": GSM8K_SPLIT,
            "offset": offset,
            "length": length,
        },
    )


def parse_final_answer(answer_text: str, row_idx: int) -> str:
    """从 GSM8K 的 answer 字段里取出 '#### ' 之后的最终数值字符串（去逗号）。"""
    if not isinstance(answer_text, str):
        raise ExportError(f"GSM8K row_idx={row_idx}: answer 非字符串")
    if "####" not in answer_text:
        raise ExportError(f"GSM8K row_idx={row_idx}: answer 缺少 '####' 终答标记")
    final = answer_text.rsplit("####", 1)[1].strip().replace(",", "")
    if not final:
        raise ExportError(f"GSM8K row_idx={row_idx}: '####' 之后为空")
    # 必须能 parse 成整数或小数
    try:
        int(final)
    except ValueError:
        try:
            value = float(final)
        except ValueError:
            raise ExportError(
                f"GSM8K row_idx={row_idx}: 最终答案 {final!r} 无法解析为整数或小数"
            ) from None
        if value != value or value in (float("inf"), float("-inf")):
            raise ExportError(f"GSM8K row_idx={row_idx}: 最终答案 {final!r} 不是有限数值")
    return final


def fetch_gsm8k(timeout: float, retries: int) -> tuple[list, dict]:
    raw_rows: list[dict] = []
    urls: list[str] = []
    num_rows_total = None
    offset = 0
    while len(raw_rows) < GSM8K_FIRST_N:
        url = gsm8k_url(offset, PAGE_LENGTH)
        urls.append(url)
        data = fetch_json(url, timeout, retries)
        num_rows_total = data.get("num_rows_total")
        page = data.get("rows", [])
        if not page:
            break
        raw_rows.extend(page)
        offset += len(page)
        if num_rows_total is not None and offset >= num_rows_total:
            break

    if num_rows_total is None:
        raise ExportError("GSM8K: 接口未返回 num_rows_total")
    if len(raw_rows) < GSM8K_FIRST_N:
        raise ExportError(f"GSM8K: 只抓到 {len(raw_rows)} 行，不足 {GSM8K_FIRST_N} 行")

    raw_rows.sort(key=lambda r: r["row_idx"])
    selected = raw_rows[:GSM8K_FIRST_N]

    items = []
    for i, entry in enumerate(selected):
        row_idx = entry["row_idx"]
        if row_idx != i:
            raise ExportError(
                f"GSM8K: 期望连续的前 {GSM8K_FIRST_N} 行，第 {i} 项的 row_idx={row_idx}"
            )
        truncated = entry.get("truncated_cells") or []
        if truncated:
            raise ExportError(
                f"GSM8K row_idx={row_idx}: datasets-server 截断了单元格 {truncated}，拒绝写出"
            )
        row = entry["row"]
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ExportError(f"GSM8K row_idx={row_idx}: question 为空或非字符串")
        final = parse_final_answer(row.get("answer"), row_idx)
        items.append({"id": f"gsm8k/{i}", "question": question, "answer": final})

    meta = {
        "num_rows_total": num_rows_total,
        "taken": len(items),
        "urls": urls,
        "upstream_row_idx_first": 0,
        "upstream_row_idx_last": GSM8K_FIRST_N - 1,
    }
    return items, meta


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_relpath(path: str) -> str:
    """相对仓库根目录的路径（与 cwd 无关，保证 provenance 可复现）。"""
    abspath = os.path.abspath(path)
    if abspath.startswith(REPO_ROOT + os.sep):
        return os.path.relpath(abspath, start=REPO_ROOT).replace(os.sep, "/")
    return abspath


def dump_json(path: str, obj: dict) -> str:
    """写 UTF-8 JSON（ensure_ascii=False, indent=2, 末尾换行），返回 sha256。"""
    text = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    data = text.encode("utf-8")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="导出 Chapter 37 的 Qwen3 真实任务质量评测 fixture（MMLU + GSM8K）。"
    )
    parser.add_argument("--out", required=True, help="eval_tasks.json 输出路径")
    parser.add_argument(
        "--provenance",
        default=None,
        help="provenance 输出路径（默认 --out 同目录下的 eval_tasks_provenance.json）",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="单次 HTTP 超时秒数（默认 30）")
    parser.add_argument("--retries", type=int, default=3, help="失败重试次数（默认 3，指数退避）")
    args = parser.parse_args(argv)

    if args.retries < 0:
        parser.error("--retries 不能为负")
    if args.timeout <= 0:
        parser.error("--timeout 必须为正")

    out_path = os.path.abspath(args.out)
    prov_path = os.path.abspath(
        args.provenance
        or os.path.join(os.path.dirname(out_path), "eval_tasks_provenance.json")
    )

    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"fetched_at (UTC) = {fetched_at}")

    mmlu_items: list = []
    mmlu_meta: list = []
    for subject in MMLU_SUBJECTS:
        print(f"[MMLU] {subject} ...", flush=True)
        items, meta = fetch_mmlu_subject(subject, args.timeout, args.retries)
        mmlu_items.extend(items)
        mmlu_meta.append(meta)
        print(
            f"  taken={meta['taken']} / subject_total={meta['num_rows_total']} "
            f"upstream row_idx {meta['upstream_row_idx_first']}..{meta['upstream_row_idx_last']}"
        )

    print("[GSM8K] test split ...", flush=True)
    gsm8k_items, gsm8k_meta = fetch_gsm8k(args.timeout, args.retries)
    print(f"  taken={gsm8k_meta['taken']} / split_total={gsm8k_meta['num_rows_total']}")

    expected_mmlu = len(MMLU_SUBJECTS) * PER_SUBJECT
    if len(mmlu_items) != expected_mmlu:
        raise ExportError(f"MMLU 总题数 {len(mmlu_items)} != 期望 {expected_mmlu}")
    if len(gsm8k_items) != GSM8K_FIRST_N:
        raise ExportError(f"GSM8K 总题数 {len(gsm8k_items)} != 期望 {GSM8K_FIRST_N}")
    all_ids = [it["id"] for it in mmlu_items] + [it["id"] for it in gsm8k_items]
    if len(set(all_ids)) != len(all_ids):
        raise ExportError("存在重复的 task id")

    fixture = {
        "description": (
            "Chapter 37「Qwen3 dense 真实任务质量基线」冻结的公开任务集子集："
            f"MMLU test split {len(MMLU_SUBJECTS)} 个 subject 各前 {PER_SUBJECT} 题（共 {expected_mmlu} 题）"
            f"+ GSM8K test split 前 {GSM8K_FIRST_N} 题。答案为上游标注，未做任何改写；"
            "GSM8K 仅保留 '#### ' 之后的最终数值。MMLU 与 GSM8K 均为 MIT 许可。"
        ),
        "generated_by": "scripts/export_qwen3_eval_tasks.py",
        "source": {
            "mmlu": {
                "dataset": MMLU_DATASET,
                "config": MMLU_CONFIG,
                "split": MMLU_SPLIT,
                "subjects": list(MMLU_SUBJECTS),
                "per_subject": PER_SUBJECT,
                "selection": MMLU_SELECTION,
            },
            "gsm8k": {
                "dataset": GSM8K_DATASET,
                "config": GSM8K_CONFIG,
                "split": GSM8K_SPLIT,
                "first_n": GSM8K_FIRST_N,
                "selection": GSM8K_SELECTION,
            },
        },
        "mmlu": mmlu_items,
        "gsm8k": gsm8k_items,
    }

    out_sha = dump_json(out_path, fixture)

    provenance = {
        "description": (
            "eval_tasks.json 的抓取来源与冻结校验信息，供第三方核对冻结的是哪一版上游数据。"
        ),
        "generated_by": "scripts/export_qwen3_eval_tasks.py",
        "fetched_at_utc": fetched_at,
        "datasets_server": DATASETS_SERVER,
        "licenses": LICENSES,
        "mmlu": {
            "dataset": MMLU_DATASET,
            "config": MMLU_CONFIG,
            "split": MMLU_SPLIT,
            "per_subject": PER_SUBJECT,
            "selection": MMLU_SELECTION,
            "total_taken": len(mmlu_items),
            "subjects": mmlu_meta,
        },
        "gsm8k": {
            "dataset": GSM8K_DATASET,
            "config": GSM8K_CONFIG,
            "split": GSM8K_SPLIT,
            "first_n": GSM8K_FIRST_N,
            "selection": GSM8K_SELECTION,
            "num_rows_total": gsm8k_meta["num_rows_total"],
            "total_taken": gsm8k_meta["taken"],
            "urls": gsm8k_meta["urls"],
            "upstream_row_idx_first": gsm8k_meta["upstream_row_idx_first"],
            "upstream_row_idx_last": gsm8k_meta["upstream_row_idx_last"],
        },
        "outputs": {
            "eval_tasks.json": {
                "path": repo_relpath(out_path),
                "sha256": out_sha,
                "json_dump": "ensure_ascii=False, indent=2, trailing newline, UTF-8",
            }
        },
    }
    prov_sha = dump_json(prov_path, provenance)

    print()
    print("=== 报告 ===")
    print(f"MMLU subjects ({len(MMLU_SUBJECTS)}):")
    for meta in mmlu_meta:
        print(
            f"  {meta['subject']:<28} taken={meta['taken']:>3}  "
            f"subject_num_rows_total={meta['num_rows_total']}"
        )
    print(f"MMLU total          = {len(mmlu_items)}")
    print(f"GSM8K total         = {len(gsm8k_items)} (split num_rows_total={gsm8k_meta['num_rows_total']})")
    print(f"TASKS total         = {len(mmlu_items) + len(gsm8k_items)}")
    print(f"{out_path}")
    print(f"  sha256 = {out_sha}")
    print(f"{prov_path}")
    print(f"  sha256 = {prov_sha}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
