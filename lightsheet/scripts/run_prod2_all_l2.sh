#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/working/eduseg/20260808-prod2/234-3-514561638"
INPUT_DIR="${RUN_ROOT}/squisher-deconv-run-u16"
POSITION="${RUN_ROOT}/561-ref-405-xreg/reg-638/registration.positions.json"
REGISTRATION_561="${RUN_ROOT}/561-ref-405-xreg/reg-638/registration.json"
REGISTRATION_638="${RUN_ROOT}/reg-638-to-561/registration.json"
MATERIALIZED_514="${RUN_ROOT}/materialized-514-l2-zstd"
OUTPUT_ROOT="${RUN_ROOT}/fusion-l2-zstd"
PREVIEW_GRID="${OUTPUT_ROOT}/561/fused.ch0.ome.zarr"
LOG_PATH="${RUN_ROOT}/run-logs/fusion-all-l2.log"

GPU_5090="GPU-5d7f9022-ea7f-32f4-71b5-e424ae96aafd"
export CUDA_VISIBLE_DEVICES="${GPU_5090}"

source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate multi
EXPECTED_PYTHON="/home/chaichontat/miniforge3/envs/multi/bin/python"
LIGHTSHEET="/home/chaichontat/miniforge3/envs/multi/bin/lightsheet"
if [[ "$(command -v python)" != "${EXPECTED_PYTHON}" ]]; then
  echo "Expected ${EXPECTED_PYTHON}, got $(command -v python)" >&2
  exit 1
fi

export PYTHONPATH="/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/squisher/src"
export CUDA_PATH="${CONDA_PREFIX}/targets/x86_64-linux"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/targets/x86_64-linux/lib:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${RUN_ROOT}/run-logs"
exec > >(tee -a "${LOG_PATH}") 2>&1

fusion_complete() {
  local output="$1"
  [[ -f "${output}/provenance/manifest.json" ]] &&
    jq -e '.status == "complete"' "${output}/provenance/manifest.json" >/dev/null
}

echo "Prod2 sequential level-2 Zstd fusion"
echo "channels=0:561,1:638,2:514"
echo "python=${EXPECTED_PYTHON}"
echo "gpu_0=${GPU_5090}"

if ! fusion_complete "${OUTPUT_ROOT}/561/fused.ch0.ome.zarr"; then
  "${LIGHTSHEET}" fuse "${INPUT_DIR}" \
    --position-input "${POSITION}" \
    --registration-input "${REGISTRATION_561}" \
    --output "${OUTPUT_ROOT}/561/fused.ome.zarr" \
    --channel 0 \
    --fusion-level 2 \
    --fusion-weight-mode content-preibisch-coarse \
    --batch-size 1 \
    --output-chunksize-zyx 12,960,960 \
    --output-codec zstd \
    --zstd-level 3
fi
echo "561 level-2 fusion complete"

if ! fusion_complete "${OUTPUT_ROOT}/638/fused.ch1.ome.zarr"; then
  "${LIGHTSHEET}" fuse "${INPUT_DIR}" \
    --position-input "${POSITION}" \
    --registration-input "${REGISTRATION_638}" \
    --output "${OUTPUT_ROOT}/638/fused.ome.zarr" \
    --channel 1 \
    --fusion-level 2 \
    --fusion-weight-mode content-preibisch-coarse \
    --batch-size 1 \
    --output-chunksize-zyx 12,960,960 \
    --output-grid-template "${PREVIEW_GRID}" \
    --output-grid-template-level 0 \
    --output-codec zstd \
    --zstd-level 3
fi
echo "638 level-2 fusion complete"

"${LIGHTSHEET}" fused-fixed-materialize-overlap \
  --moving-position "${RUN_ROOT}/xreg-514-rough/global-phase.positions.json" \
  --moving-source-registration "${RUN_ROOT}/514-reg/reg/registration.json" \
  --source-summary "${RUN_ROOT}/xreg-514-m6-v10-final/fused_fixed_method8_summary.json" \
  --output-dir "${MATERIALIZED_514}" \
  --output-codec zstd \
  --zstd-level 3 \
  --source-channel 2 \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --level-factor-zyx 1,4,4 \
  --workers 4 \
  --resume

if ! fusion_complete "${OUTPUT_ROOT}/514/fused.ch0.ome.zarr"; then
  "${LIGHTSHEET}" fuse "${MATERIALIZED_514}/materialized_tiles" \
    --position-input "${MATERIALIZED_514}/fused_fixed_materialized_chunks.positions.json" \
    --registration-input "${MATERIALIZED_514}/fused_fixed_materialized_chunks.registration.json" \
    --output "${OUTPUT_ROOT}/514/fused.ome.zarr" \
    --channel 0 \
    --fusion-level 0 \
    --fusion-weight-mode content-preibisch-coarse \
    --batch-size 1 \
    --output-chunksize-zyx 12,960,960 \
    --output-grid-template "${PREVIEW_GRID}" \
    --output-grid-template-level 0 \
    --output-codec zstd \
    --zstd-level 3 \
    --resume-fusion
fi
echo "514 level-2 fusion complete"
