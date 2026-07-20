#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${ROOT_DIR}/benchmark_results/week05-$(date +%Y%m%d-%H%M%S)}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

julia --project=. examples/benchmark_week05_tokenizers.jl "${OUTPUT_DIR}"
julia --project=. examples/benchmark_week05_tokenizer_artifacts.jl "${OUTPUT_DIR}"

printf '\nWeek 05 benchmark written to %s\n' "${OUTPUT_DIR}"
