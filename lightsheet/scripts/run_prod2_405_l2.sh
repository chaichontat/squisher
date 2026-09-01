#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/working/eduseg/20260808-prod2/234-3-514561638"
SOURCE_405="/working/eduseg/20260808-prod2/234-3-405"
PREVIEW_ROOT="${RUN_ROOT}/fusion-l2-zstd"
MATERIALIZED="${RUN_ROOT}/materialized-405-l2-zstd"
OUTPUT="${PREVIEW_ROOT}/405/fused.ome.zarr"
LOG_PATH="${RUN_ROOT}/run-logs/fusion-405-l2.log"

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

echo "Waiting for sequential 561, 638, and 514 level-2 fusion"
while ! jq -e '.status == "complete"' \
  "${PREVIEW_ROOT}/514/fused.ch0.ome.zarr/provenance/manifest.json" >/dev/null 2>&1; do
  sleep 60
done

echo "405 level-2 Zstd materialization"
echo "python=${EXPECTED_PYTHON}"
echo "gpu_0=${GPU_5090}"

"${LIGHTSHEET}" fused-fixed-materialize-overlap \
  --moving-position "${SOURCE_405}/registration-basic/registration.positions.json" \
  --source-summary "${RUN_ROOT}/xreg-405-m6-final/fused_fixed_method8_summary.json" \
  --output-dir "${MATERIALIZED}" \
  --output-codec zstd \
  --zstd-level 3 \
  --source-channel 0 \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --level-factor-zyx 1,4,4 \
  --workers 4 \
  --resume

"${LIGHTSHEET}" fuse "${MATERIALIZED}/materialized_tiles" \
  --position-input "${MATERIALIZED}/fused_fixed_materialized_chunks.positions.json" \
  --registration-input "${MATERIALIZED}/fused_fixed_materialized_chunks.registration.json" \
  --output "${OUTPUT}" \
  --channel 0 \
  --fusion-level 0 \
  --fusion-weight-mode content-preibisch-coarse \
  --batch-size 1 \
  --output-chunksize-zyx 12,960,960 \
  --output-grid-template "${PREVIEW_ROOT}/561/fused.ch0.ome.zarr" \
  --output-grid-template-level 0 \
  --output-codec zstd \
  --zstd-level 3 \
  --resume-fusion

echo "405 level-2 fusion complete"
