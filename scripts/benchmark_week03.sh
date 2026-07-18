#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date '+%Y%m%d-%H%M%S')"
OUTPUT_DIR="${1:-${ROOT_DIR}/benchmark_results/week03-${TIMESTAMP}}"
BACKENDS="${LIFEAI_BENCH_BACKENDS:-cpu gpu xla_cpu xla_gpu}"
JULIA_BIN="${JULIA_BIN:-julia}"
WORKER="${ROOT_DIR}/examples/benchmark_week03_backends.jl"

mkdir -p "${OUTPUT_DIR}"
printf 'backend\tstatus\tmessage\n' > "${OUTPUT_DIR}/status.tsv"

sample_gpu_memory() {
    local pid="$1"
    local output="$2"
    local peak=0
    local current
    local gpu_name=""
    local driver_version=""

    command -v nvidia-smi >/dev/null 2>&1 || return 0
    gpu_name="$(
        nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null |
        head -n 1
    )"
    driver_version="$(
        nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null |
        head -n 1
    )"
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
            "$(basename "${output}" .tsv)" "${peak}" >> "${output}"
    fi
    if [[ -n "${gpu_name}" && -f "${output}" ]]; then
        printf '%s\tgpu_name\t%s\t\n' \
            "$(basename "${output}" .tsv)" "${gpu_name}" >> "${output}"
    fi
    if [[ -n "${driver_version}" && -f "${output}" ]]; then
        printf '%s\tgpu_driver_version\t%s\t\n' \
            "$(basename "${output}" .tsv)" "${driver_version}" >> "${output}"
    fi
}

run_backend() {
    local backend="$1"
    local result="${OUTPUT_DIR}/${backend}.tsv"
    local log="${OUTPUT_DIR}/${backend}.log"
    local monitor_pid=""
    local exit_code

    printf '\n==> %s\n' "${backend}"
    "${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" \
        "${WORKER}" "--backend=${backend}" "--output=${result}" \
        >"${log}" 2>&1 &
    local worker_pid=$!

    if [[ "${backend}" == "gpu" || "${backend}" == "xla_gpu" ]]; then
        sample_gpu_memory "${worker_pid}" "${result}" &
        monitor_pid=$!
    fi

    wait "${worker_pid}"
    exit_code=$?
    if [[ -n "${monitor_pid}" ]]; then
        wait "${monitor_pid}" || true
    fi

    if (( exit_code == 0 )); then
        printf '%s\tok\t\n' "${backend}" >> "${OUTPUT_DIR}/status.tsv"
        tail -n 3 "${log}"
    else
        local message
        message="$(tail -n 1 "${log}" | tr '\t' ' ')"
        printf '%s\tfailed (exit %s)\t%s\n' \
            "${backend}" "${exit_code}" "${message}" >> "${OUTPUT_DIR}/status.tsv"
        printf 'FAILED: %s (see %s)\n' "${backend}" "${log}"
    fi
}

for backend in ${BACKENDS}; do
    case "${backend}" in
        cpu|gpu|xla_cpu|xla_gpu)
            run_backend "${backend}"
            ;;
        *)
            printf 'Unknown backend: %s\n' "${backend}" >&2
            exit 2
            ;;
    esac
done

printf '\n==> summary\n'
"${JULIA_BIN}" --startup-file=no --project="${ROOT_DIR}" \
    "${WORKER}" "--summarize=${OUTPUT_DIR}"
printf '\nResults: %s\n' "${OUTPUT_DIR}"
