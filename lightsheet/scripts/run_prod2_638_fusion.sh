#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/working/eduseg/20260808-prod2/234-3-514561638"
INPUT_DIR="${RUN_ROOT}/squisher-deconv-run-u16"
POSITION="${RUN_ROOT}/561-ref-405-xreg/reg-638/registration.positions.json"
REGISTRATION="${RUN_ROOT}/reg-638-to-561/registration.json"
FIXED_FUSED="${RUN_ROOT}/fusion-561/fused.ch0.ome.zarr"
OUTPUT="${RUN_ROOT}/fusion-638-l0/fused.ome.zarr"
LOG_PATH="${RUN_ROOT}/run-logs/fusion-638-l0.log"

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

jq -e \
  --arg fixed "${FIXED_FUSED}" \
  '.artifact_type == "squisher_lightsheet.global_channel_affine_registration.v1"
   and (.tiles | length) == 82
   and .diagnostics.global_channel_affine.expected_moving_channel == 1
   and .diagnostics.global_channel_affine.expected_fixed_fused == $fixed
   and .transform_contract.source_space == "638_stage_um"
   and .transform_contract.target_space == "561_registered_um"' \
  "${REGISTRATION}" >/dev/null

echo "638 level-0 fusion"
echo "python=${EXPECTED_PYTHON}"
echo "gpu_0=${GPU_5090}"
echo "registration=${REGISTRATION}"

exec "${LIGHTSHEET}" fuse "${INPUT_DIR}" \
  --position-input "${POSITION}" \
  --registration-input "${REGISTRATION}" \
  --output "${OUTPUT}" \
  --channel 1 \
  --fusion-level 0 \
  --fusion-weight-mode content-preibisch-coarse \
  --batch-size 1 \
  --output-chunksize-zyx 12,960,960 \
  --output-grid-template "${FIXED_FUSED}" \
  --output-grid-template-level 0 \
  --output-codec jpegxr \
  --jpegxr-level 0.7
