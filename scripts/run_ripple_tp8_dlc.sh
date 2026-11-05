#!/usr/bin/env bash
set -euo pipefail
cd "${LIFEAI_CODE_ROOT:?missing immutable code snapshot}"
test "$(cat CODE_REVISION)" = "${LIFEAI_CODE_REVISION:?missing expected code revision}"
sha256sum --check --status CODE_SHA256SUMS
test -r "${LIFEAI_MODEL_DIR:?}/verification.json"
test -r "${LIFEAI_REFERENCE_DIR:?}/reference.safetensors"
mkdir -p "${LIFEAI_RESULT_DIR:?}"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv > "$LIFEAI_RESULT_DIR/gpus.csv"
nvidia-smi topo -m > "$LIFEAI_RESULT_DIR/topology.txt"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-16}"
export OPENBLAS_NUM_THREADS=1
python3 scripts/monitor_ripple_tp8.py "$$" "$LIFEAI_RESULT_DIR" > "$LIFEAI_RESULT_DIR/monitor.log" 2>&1 &
exec > >(tee -a "$LIFEAI_RESULT_DIR/service.log") 2>&1
exec "${LIFEAI_JULIA:?}" --startup-file=no -O1 --project=configs/deployment/ripple_tp8 \
  scripts/run_ripple_tp8_server.jl --gpus 8 \
  --model "$LIFEAI_MODEL_DIR" --reference "$LIFEAI_REFERENCE_DIR" \
  --result-dir "$LIFEAI_RESULT_DIR" --api-key-file "$LIFEAI_API_KEY_FILE" \
  --host 0.0.0.0 --port "${LIFEAI_SERVICE_PORT:-18085}"
