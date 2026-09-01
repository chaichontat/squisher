#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="/working/eduseg/20260808-prod2/234-3-514561638"
OUTPUT_DIR="${RUN_ROOT}/xreg-638-trial"
LOG_PATH="${RUN_ROOT}/run-logs/xreg-638-trial.log"
SWEEP="/home/chaichontat/nvme/lightsheet/scripts/run_fused_fixed_method8_sweep.py"
WINDOWS="/home/chaichontat/squisher/lightsheet/scripts/prod2_561_638_windows.json"

# Acquisition metadata records source channel 0 as 561 and source channel 1 as
# 638. This trial fits moving 638 to the completed source-channel-0 561 mosaic.
MOVING_CHANNEL=1
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

echo "638->561 phase-correlation and Method 6 trial"
echo "python=${EXPECTED_PYTHON}"
echo "moving_channel=${MOVING_CHANNEL}"
echo "gpu_0=${GPU_5090}"
echo "gpu_1=${GPU_4090}"

exec "${EXPECTED_PYTHON}" -u "${SWEEP}" \
  --fixed-position "${RUN_ROOT}/561-ref-405-xreg/reg-638/registration.positions.json" \
  --moving-position "${RUN_ROOT}/561-ref-405-xreg/reg-638/registration.positions.json" \
  --moving-source-position "${RUN_ROOT}/561-ref-405-xreg/reg-638/registration.json" \
  --fixed-fused "${RUN_ROOT}/fusion-561/fused.ch0.ome.zarr" \
  --output-dir "${OUTPUT_DIR}" \
  --window-filter-json "${WINDOWS}" \
  --core-shape-zyx 480,480,480 \
  --window-shape-zyx 528,528,528 \
  --fit-downsample-zyx 1,1,1 \
  --moving-channel "${MOVING_CHANNEL}" \
  --native-lib-dir /home/chaichontat/microImageLib/bin/linux \
  --native-method method6 \
  --starting-affine-matrix-zyx 1,0,0,0,1,0,0,0,1 \
  --fit-intensity-transform log1p \
  --ftol 0.0001 \
  --max-iterations 300 \
  --phase-upsample-factor 10 \
  --min-corr 0.15 \
  --min-grad-ncc 0.24 \
  --fixed-mask-threshold 50 \
  --fixed-mask-level 2 \
  --fixed-mask-min-voxels 256 \
  --fixed-mask-max-masked-fraction 0.95 \
  --workers 2 \
  --max-tasks-per-worker 6 \
  --devices 0,1 \
  --level0-initializer level2-method8 \
  --resume
