#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/working/eduseg/20260808-prod2/234-3-514561638"
SOURCE_405="/working/eduseg/20260808-prod2/234-3-405"
INPUT_SUMMARY="${RUN_ROOT}/xreg-405-m6/fused_fixed_method8_summary.json"
RECOVERY_DIR="${RUN_ROOT}/xreg-405-m6-recovery-v2"
FINAL_DIR="${RUN_ROOT}/xreg-405-m6-final-v2"
RECOVERY_SCRIPT="/home/chaichontat/squisher/lightsheet/scripts/recover_fused_fixed_method10_outliers.py"
LOG_PATH="${RUN_ROOT}/run-logs/recovery-405-v2.log"
EXPECTED_WINDOWS=2480

GPU_5090="GPU-5d7f9022-ea7f-32f4-71b5-e424ae96aafd"
GPU_4090="GPU-94714ab6-ba56-a7d9-6dc3-adc866b00bd6"
export CUDA_VISIBLE_DEVICES="${GPU_5090},${GPU_4090}"

source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate multi
EXPECTED_PYTHON="/home/chaichontat/miniforge3/envs/multi/bin/python"
if [[ "$(command -v python)" != "${EXPECTED_PYTHON}" ]]; then
  echo "Expected ${EXPECTED_PYTHON}, got $(command -v python)" >&2
  exit 1
fi

export PYTHONPATH="/home/chaichontat/squisher/deconv/src:/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/squisher/src"
export CUDA_PATH="${CONDA_PREFIX}/targets/x86_64-linux"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}/run-logs"
exec > >(tee -a "${LOG_PATH}") 2>&1

jq -e --argjson expected "${EXPECTED_WINDOWS}" \
  '.aggregate.window_count == $expected and .aggregate.error_window_count == 0' \
  "${INPUT_SUMMARY}" >/dev/null

echo "405 failed-window recovery v2"
echo "python=${EXPECTED_PYTHON}"
echo "gpu_0=${GPU_5090}"
echo "gpu_1=${GPU_4090}"

"${EXPECTED_PYTHON}" -u "${RECOVERY_SCRIPT}" \
  --summary "${INPUT_SUMMARY}" \
  --output-dir "${RECOVERY_DIR}" \
  --maximum-outlier-px 5 \
  --maximum-refit-displacement-px 10 \
  --devices 0,1 \
  --adjacency-json "${SOURCE_405}/registration-basic/registration.measurements.json" \
  --resume

RECOVERY_SUMMARY="${RECOVERY_DIR}/fused_fixed_method8_summary.json"
jq -e --argjson expected "${EXPECTED_WINDOWS}" \
  '(.windows | length) == $expected and .aggregate.error == 0' \
  "${RECOVERY_SUMMARY}" >/dev/null

if [[ -e "${FINAL_DIR}" ]]; then
  echo "Final output already exists: ${FINAL_DIR}" >&2
  exit 1
fi
"${EXPECTED_PYTHON}" -u "${RECOVERY_SCRIPT}" \
  --summary "${RECOVERY_SUMMARY}" \
  --output-dir "${FINAL_DIR}" \
  --exclude-nonlinear-only \
  --maximum-outlier-px 5 \
  --maximum-refit-displacement-px 10

FINAL_SUMMARY="${FINAL_DIR}/fused_fixed_method8_summary.json"
jq -e --argjson expected "${EXPECTED_WINDOWS}" \
  '(.windows | length) == $expected and .aggregate.error == 0' \
  "${FINAL_SUMMARY}" >/dev/null
echo "405 recovery v2 complete: ${FINAL_SUMMARY}"
