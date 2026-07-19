#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
OUTPUT_DIR="${1:-${ROOT_DIR}/benchmark_results/week04-${TIMESTAMP}}"
BACKENDS="${LIFEAI_WEEK04_BACKENDS:-cpu gpu xla_cpu xla_gpu}"
JULIA_BIN="${JULIA_BIN:-julia}"
BACKEND_WORKER="${ROOT_DIR}/examples/benchmark_week03_backends.jl"
WEEK04_WORKER="${ROOT_DIR}/examples/benchmark_week04_components.jl"

export LIFEAI_BENCH_EMBED_DIM="${LIFEAI_BENCH_EMBED_DIM:-64}"
export LIFEAI_BENCH_NUM_HEADS="${LIFEAI_BENCH_NUM_HEADS:-4}"
export LIFEAI_BENCH_NUM_LAYERS="${LIFEAI_BENCH_NUM_LAYERS:-2}"
export LIFEAI_BENCH_SEQ_LEN="${LIFEAI_BENCH_SEQ_LEN:-64}"
export LIFEAI_BENCH_BATCH_SIZE="${LIFEAI_BENCH_BATCH_SIZE:-4}"
export LIFEAI_BENCH_PROMPT_TOKENS="${LIFEAI_BENCH_PROMPT_TOKENS:-64}"
export LIFEAI_BENCH_DECODE_TOKENS="${LIFEAI_BENCH_DECODE_TOKENS:-16}"
export LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS="${LIFEAI_BENCH_XLA_MODE_DECODE_TOKENS:-2}"
export LIFEAI_BENCH_WARMUP_STEPS="${LIFEAI_BENCH_WARMUP_STEPS:-3}"
export LIFEAI_BENCH_SAMPLES="${LIFEAI_BENCH_SAMPLES:-30}"

mkdir -p "${OUTPUT_DIR}"
printf 'profile\tbackend\tstatus\tmessage\n' > "${OUTPUT_DIR}/status.tsv"

sample_gpu_memory() {
    local pid="$1"
    local output="$2"
    local metric_prefix="$3"
    local peak=0
    local current

    command -v nvidia-smi >/dev/null 2>&1 || return 0
    while kill -0 "${pid}" >/dev/null 2>&1; do
        current="$(
            nvidia-smi \
                --query-compute-apps=pid,used_memory \
                --format=csv,noheader,nounits 2>/dev/null |
            awk -F',' -v target="${pid}" '
                {
                    gsub(/ /, "", $1)
                    gsub(/ /, "", $2)
                    if ($1 == target && $2 + 0 > peak) peak = $2 + 0
                }
                END { if (peak > 0) print peak }
            '
        )"
        if [[ -n "${current}" ]] && (( current > peak )); then
            peak="${current}"
        fi
        sleep 0.2
    done

    if (( peak > 0 )) && [[ -f "${output}" ]]; then
        printf '%s\tgpu_peak_memory_mb\t%s\tMiB\n' \
            "${metric_prefix}" "${peak}" >> "${output}"
    fi
}

run_cpu_matrix() {
    local result="${OUTPUT_DIR}/cpu_matrix.tsv"
    local log="${OUTPUT_DIR}/cpu_matrix.log"

    printf '\n==> five-profile CPU controlled matrix\n'
    if "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" \
        "${WEEK04_WORKER}" "--output=${result}" >"${log}" 2>&1; then
        printf '%s\t%s\tok\t\n' "all" "cpu_matrix" >> "${OUTPUT_DIR}/status.tsv"
        tail -n 4 "${log}"
    else
        local exit_code=$?
        local message
        message="$(tail -n 1 "${log}" | tr '\t' ' ')"
        printf '%s\t%s\tfailed (exit %s)\t%s\n' \
            "all" "cpu_matrix" "${exit_code}" "${message}" >> "${OUTPUT_DIR}/status.tsv"
        printf 'FAILED: CPU matrix (see %s)\n' "${log}"
    fi
}

run_backend() {
    local profile="$1"
    local backend="$2"
    local norm_type="$3"
    local mlp_type="$4"
    local tie_embeddings="$5"
    local result="${OUTPUT_DIR}/${profile}_${backend}.tsv"
    local log="${OUTPUT_DIR}/${profile}_${backend}.log"
    local monitor_pid=""
    local exit_code

    printf '\n==> %s / %s\n' "${profile}" "${backend}"
    LIFEAI_BENCH_PROFILE="${profile}" \
    LIFEAI_BENCH_NORM_TYPE="${norm_type}" \
    LIFEAI_BENCH_MLP_TYPE="${mlp_type}" \
    LIFEAI_BENCH_TIE_EMBEDDINGS="${tie_embeddings}" \
    "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" \
        "${BACKEND_WORKER}" "--backend=${backend}" "--output=${result}" \
        >"${log}" 2>&1 &
    local worker_pid=$!

    if [[ "${backend}" == "gpu" || "${backend}" == "xla_gpu" ]]; then
        sample_gpu_memory "${worker_pid}" "${result}" "${backend}" &
        monitor_pid=$!
    fi

    wait "${worker_pid}"
    exit_code=$?
    if [[ -n "${monitor_pid}" ]]; then
        wait "${monitor_pid}" || true
    fi

    if (( exit_code == 0 )); then
        printf '%s\t%s\tok\t\n' \
            "${profile}" "${backend}" >> "${OUTPUT_DIR}/status.tsv"
        tail -n 3 "${log}"
    else
        local message
        message="$(tail -n 1 "${log}" | tr '\t' ' ')"
        printf '%s\t%s\tfailed (exit %s)\t%s\n' \
            "${profile}" "${backend}" "${exit_code}" "${message}" \
            >> "${OUTPUT_DIR}/status.tsv"
        printf 'FAILED: %s / %s (see %s)\n' \
            "${profile}" "${backend}" "${log}"
    fi
}

run_cpu_matrix

for profile in baseline modern; do
    if [[ "${profile}" == "baseline" ]]; then
        norm_type="layernorm"
        mlp_type="gelu"
        tie_embeddings="false"
    else
        norm_type="rmsnorm"
        mlp_type="swiglu"
        tie_embeddings="true"
    fi

    for backend in ${BACKENDS}; do
        case "${backend}" in
            cpu|gpu|xla_cpu|xla_gpu)
                run_backend \
                    "${profile}" \
                    "${backend}" \
                    "${norm_type}" \
                    "${mlp_type}" \
                    "${tie_embeddings}"
                ;;
            *)
                printf 'Unknown backend: %s\n' "${backend}" >&2
                exit 2
                ;;
        esac
    done
done

printf '\n==> summary\n'
if [[ -f "${OUTPUT_DIR}/cpu_matrix.tsv" ]]; then
    "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" \
        "${WEEK04_WORKER}" "--summarize=${OUTPUT_DIR}"
else
    printf 'CPU matrix missing; summary not generated.\n'
fi
printf '\nResults: %s\n' "${OUTPUT_DIR}"
