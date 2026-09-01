#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/working/eduseg/20260808-prod2/234-3-514561638"
SOURCE_405="/working/eduseg/20260808-prod2/234-3-405"
FIT_514="${RUN_ROOT}/xreg-514-m6-v10"
RECOVERY_514="${RUN_ROOT}/xreg-514-m6-v10-recovery"
FINAL_514="${RUN_ROOT}/xreg-514-m6-v10-final"
RECOVERY_405="${RUN_ROOT}/xreg-405-m6-recovery"
FINAL_405="${RUN_ROOT}/xreg-405-m6-final"
MATERIALIZED_405="${RUN_ROOT}/materialized-405-l0-jxr"
MATERIALIZED_514="${RUN_ROOT}/materialized-514-l0-jxr"
FUSION_405="${RUN_ROOT}/fusion-405-l0/fused.ch0.ome.zarr"
FUSION_514="${RUN_ROOT}/fusion-514-l0/fused.ch0.ome.zarr"
FIXED_FUSED="${RUN_ROOT}/fusion-561/fused.ch0.ome.zarr"
RECOVERY_SCRIPT="/home/chaichontat/squisher/lightsheet/scripts/recover_fused_fixed_method10_outliers.py"
LOG_PATH="${RUN_ROOT}/run-logs/post-514-l0.log"
TOTAL_514_WINDOWS=2460
TOTAL_405_WINDOWS=2480

GPU_5090="GPU-5d7f9022-ea7f-32f4-71b5-e424ae96aafd"
GPU_4090="GPU-94714ab6-ba56-a7d9-6dc3-adc866b00bd6"
export CUDA_VISIBLE_DEVICES="${GPU_5090},${GPU_4090}"

source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate multi
EXPECTED_PYTHON="/home/chaichontat/miniforge3/envs/multi/bin/python"
LIGHTSHEET="/home/chaichontat/miniforge3/envs/multi/bin/lightsheet"
if [[ "$(command -v python)" != "${EXPECTED_PYTHON}" ]]; then
  echo "Expected ${EXPECTED_PYTHON}, got $(command -v python)" >&2
  exit 1
fi

export PYTHONPATH="/home/chaichontat/squisher/deconv/src:/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/squisher/src"
export CUDA_PATH="${CONDA_PREFIX}/targets/x86_64-linux"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}/run-logs"
exec > >(tee -a "${LOG_PATH}") 2>&1

base_summary_complete() {
  local summary="$1"
  local expected="$2"
  [[ -f "${summary}" ]] && jq -e \
    --argjson expected "${expected}" \
    '.aggregate.window_count == $expected and .aggregate.error_window_count == 0' \
    "${summary}" >/dev/null
}

final_summary_complete() {
  local summary="$1"
  local expected="$2"
  [[ -f "${summary}" ]] && jq -e \
    --argjson expected "${expected}" \
    '(.windows | length) == $expected and .aggregate.error == 0' \
    "${summary}" >/dev/null
}

fusion_complete() {
  local output="$1"
  local manifest="${output}/provenance/manifest.json"
  [[ -f "${output}/0/zarr.json" && -f "${manifest}" ]] && \
    jq -e '.status == "complete"' "${manifest}" >/dev/null
}

echo "post-514 level-0 pipeline"
echo "python=${EXPECTED_PYTHON}"
echo "gpu_0=${GPU_5090}"
echo "gpu_1=${GPU_4090}"

FIT_514_SUMMARY="${FIT_514}/fused_fixed_method8_summary.json"
until base_summary_complete "${FIT_514_SUMMARY}" "${TOTAL_514_WINDOWS}"; do
  completed=$(find "${FIT_514}/window_json" -maxdepth 1 -name '*.json' -type f 2>/dev/null | wc -l)
  echo "waiting for 514 fit: ${completed}/${TOTAL_514_WINDOWS} windows"
  sleep 60
done
echo "514 fit complete"

RECOVERY_514_SUMMARY="${RECOVERY_514}/fused_fixed_method8_summary.json"
if ! final_summary_complete "${RECOVERY_514_SUMMARY}" "${TOTAL_514_WINDOWS}"; then
  "${EXPECTED_PYTHON}" -u "${RECOVERY_SCRIPT}" \
    --summary "${FIT_514_SUMMARY}" \
    --output-dir "${RECOVERY_514}" \
    --maximum-outlier-px 5 \
    --maximum-refit-displacement-px 10 \
    --devices 0,1 \
    --adjacency-json "${RUN_ROOT}/514-reg/reg/registration.measurements.json" \
    --resume
fi
final_summary_complete "${RECOVERY_514_SUMMARY}" "${TOTAL_514_WINDOWS}"
echo "514 recovery complete"

FINAL_514_SUMMARY="${FINAL_514}/fused_fixed_method8_summary.json"
if ! final_summary_complete "${FINAL_514_SUMMARY}" "${TOTAL_514_WINDOWS}"; then
  if [[ -e "${FINAL_514}" ]]; then
    echo "Incomplete 514 final-filter output already exists: ${FINAL_514}" >&2
    exit 1
  fi
  "${EXPECTED_PYTHON}" -u "${RECOVERY_SCRIPT}" \
    --summary "${RECOVERY_514_SUMMARY}" \
    --output-dir "${FINAL_514}" \
    --exclude-nonlinear-only \
    --adjacency-json "${RUN_ROOT}/514-reg/reg/registration.measurements.json" \
    --maximum-outlier-px 5 \
    --maximum-refit-displacement-px 10
fi
final_summary_complete "${FINAL_514_SUMMARY}" "${TOTAL_514_WINDOWS}"
echo "514 final linear-consistency filter complete"

FINAL_405_SUMMARY="${FINAL_405}/fused_fixed_method8_summary.json"
if ! final_summary_complete "${FINAL_405_SUMMARY}" "${TOTAL_405_WINDOWS}"; then
  if [[ -e "${FINAL_405}" ]]; then
    echo "Incomplete 405 final-filter output already exists: ${FINAL_405}" >&2
    exit 1
  fi
  "${EXPECTED_PYTHON}" -u "${RECOVERY_SCRIPT}" \
    --summary "${RECOVERY_405}/fused_fixed_method8_summary.json" \
    --output-dir "${FINAL_405}" \
    --exclude-nonlinear-only \
    --adjacency-json "${SOURCE_405}/registration-basic/registration.measurements.json" \
    --maximum-outlier-px 5 \
    --maximum-refit-displacement-px 10
fi
final_summary_complete "${FINAL_405_SUMMARY}" "${TOTAL_405_WINDOWS}"
echo "405 final linear-consistency filter complete"

"${LIGHTSHEET}" fused-fixed-materialize-overlap \
  --moving-position "${SOURCE_405}/registration-basic/registration.positions.json" \
  --source-summary "${FINAL_405_SUMMARY}" \
  --output-dir "${MATERIALIZED_405}" \
  --output-codec jpegxr \
  --jpegxr-level 0.7 \
  --source-channel 0 \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --level-factor-zyx 1,1,1 \
  --workers 4 \
  --resume

if ! fusion_complete "${FUSION_405}"; then
  "${LIGHTSHEET}" fuse "${MATERIALIZED_405}/materialized_tiles" \
    --position-input "${MATERIALIZED_405}/fused_fixed_materialized_chunks.positions.json" \
    --registration-input "${MATERIALIZED_405}/fused_fixed_materialized_chunks.registration.json" \
    --output "${FUSION_405}" \
    --channel 0 \
    --fusion-level 0 \
    --fusion-weight-mode content-preibisch-coarse \
    --batch-size 1 \
    --output-chunksize-zyx 12,960,960 \
    --output-grid-template "${FIXED_FUSED}" \
    --output-grid-template-level 0 \
    --output-codec jpegxr \
    --jpegxr-level 0.7 \
    --resume-fusion
fi
fusion_complete "${FUSION_405}"
echo "405 level-0 fusion complete"

"${LIGHTSHEET}" fused-fixed-materialize-overlap \
  --moving-position "${RUN_ROOT}/514-reg/reg/registration.positions.json" \
  --moving-source-registration "${RUN_ROOT}/514-reg/reg/registration.json" \
  --source-summary "${FINAL_514_SUMMARY}" \
  --output-dir "${MATERIALIZED_514}" \
  --output-codec jpegxr \
  --jpegxr-level 0.7 \
  --source-channel 2 \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --level-factor-zyx 1,1,1 \
  --workers 4 \
  --resume

if ! fusion_complete "${FUSION_514}"; then
  "${LIGHTSHEET}" fuse "${MATERIALIZED_514}/materialized_tiles" \
    --position-input "${MATERIALIZED_514}/fused_fixed_materialized_chunks.positions.json" \
    --registration-input "${MATERIALIZED_514}/fused_fixed_materialized_chunks.registration.json" \
    --output "${FUSION_514}" \
    --channel 0 \
    --fusion-level 0 \
    --fusion-weight-mode content-preibisch-coarse \
    --batch-size 1 \
    --output-chunksize-zyx 12,960,960 \
    --output-grid-template "${FIXED_FUSED}" \
    --output-grid-template-level 0 \
    --output-codec jpegxr \
    --jpegxr-level 0.7 \
    --resume-fusion
fi
fusion_complete "${FUSION_514}"
echo "514 level-0 fusion complete"
