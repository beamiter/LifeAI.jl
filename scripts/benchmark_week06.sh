#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

julia --project=. examples/benchmark_week06_gqa.jl

printf '\nWeek 06 GQA benchmark written to %s/benchmark_results/week06\n' "${ROOT_DIR}"
