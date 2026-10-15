#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: run-edu-xreg-recovery.sh CONFIG

CONFIG is a shell file containing:
  OUTPUT_BASE DATASET_LABEL FIXED_POSITION FIXED_TILE_DIR FIXED_FUSED
  FIXED_THRESHOLD GPU_514 POSITION_514 REGISTRATION_514 TILE_DIR_514
  MEASUREMENTS_514 GPU_405 POSITION_405 REGISTRATION_405 TILE_DIR_405
  MEASUREMENTS_405

Optional:
  RUN_TAG PYRAMID_LEVEL QC_DIR LOG_FILE STATUS_FILE
  UPSTREAM_PID UPSTREAM_STATUS_FILE LIGHTSHEET_BIN
  MANUAL_INITIALIZATIONS (dataset master file of curated window initializations)
EOF
}

if [[ $# -ne 1 || ! -f $1 ]]; then
  usage
  exit 2
fi

config=$(realpath "$1")
# shellcheck source=/dev/null
source "$config"

required=(
  OUTPUT_BASE DATASET_LABEL FIXED_POSITION FIXED_TILE_DIR FIXED_FUSED
  FIXED_THRESHOLD GPU_514 POSITION_514 REGISTRATION_514 TILE_DIR_514
  MEASUREMENTS_514 GPU_405 POSITION_405 REGISTRATION_405 TILE_DIR_405
  MEASUREMENTS_405
)
for variable in "${required[@]}"; do
  if [[ -z ${!variable:-} ]]; then
    echo "Missing required config value: $variable" >&2
    exit 2
  fi
done

RUN_TAG=${RUN_TAG:-v1}
PYRAMID_LEVEL=${PYRAMID_LEVEL:-2}
QC_DIR=${QC_DIR:-"$OUTPUT_BASE/qc-prefusion-$RUN_TAG"}
LOG_FILE=${LOG_FILE:-"$OUTPUT_BASE/run-logs/xreg-recovery-514-405-$RUN_TAG.log"}
STATUS_FILE=${STATUS_FILE:-"$OUTPUT_BASE/run-logs/xreg-recovery-514-405-$RUN_TAG.exit"}
MATRIX_514=${MATRIX_514:-0.9999090828,-0.0083501824,-0.0029757332,0.0141554887,0.9999284579,-0.0096124636,0.0026881526,0.0099598448,0.9999495898}
MATRIX_405=${MATRIX_405:-1.0000085830688477,0.0029159076511859894,-0.0002120283606927842,0.0029159076511859894,1.0000085830688477,0.00018651569553185254,-0.0002120283606927842,0.00018651569553185254,1.0000001192092896}
SQUISHER_DIR=${SQUISHER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
CONDA_ENV=${CONDA_ENV:-multi}
LIGHTSHEET_BIN=${LIGHTSHEET_BIN:-lightsheet}

mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$STATUS_FILE")" "$QC_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

pids=()
finish() {
  status=$?
  for pid in "${pids[@]}"; do
    kill -INT "$pid" 2>/dev/null || true
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  printf '%s\n' "$status" > "$STATUS_FILE"
  printf '[edu-xreg] exit=%s %s\n' "$status" "$(date --iso-8601=seconds)"
}
trap finish EXIT

source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$SQUISHER_DIR/lightsheet/src:$SQUISHER_DIR/squisher/src:$SQUISHER_DIR/deconv/src"

if [[ -n ${UPSTREAM_PID:-} ]]; then
  echo "[edu-xreg] waiting for upstream pid=$UPSTREAM_PID"
  tail --pid="$UPSTREAM_PID" -f /dev/null
fi
if [[ -n ${UPSTREAM_STATUS_FILE:-} ]]; then
  [[ $(<"$UPSTREAM_STATUS_FILE") == 0 ]]
fi

run_channel() {
  local label=$1
  local channel=$2
  local position=$3
  local registration=$4
  local tile_dir=$5
  local measurements=$6
  local rough_transform=$7
  local fit_transform=$8
  local matrix=$9
  local min_grad=${10}
  local rough="$OUTPUT_BASE/xreg-$label-global-phase-$RUN_TAG"
  local method6="$OUTPUT_BASE/xreg-$label-method6-$RUN_TAG"
  local native="$OUTPUT_BASE/xreg-$label-native-recovery-$RUN_TAG"
  local complete="$OUTPUT_BASE/xreg-$label-complete-recovery-$RUN_TAG"

  "$LIGHTSHEET_BIN" cross-register global-phase \
    --fixed-position "$FIXED_POSITION" \
    --moving-position "$position" \
    --fixed-tile-dir "$FIXED_TILE_DIR" \
    --moving-tile-dir "$tile_dir" \
    --output-dir "$rough" \
    --output-position "$rough/global-phase.positions.json" \
    --fixed-channel 0 \
    --moving-channel "$channel" \
    --level "$PYRAMID_LEVEL" \
    --z-slab-planes 32 \
    --fixed-intensity-transform log1p \
    --moving-intensity-transform "$rough_transform" \
    --highpass-sigma-px 8

  "$LIGHTSHEET_BIN" cross-register method6 \
    --fixed-position "$FIXED_POSITION" \
    --moving-position "$rough/global-phase.positions.json" \
    --moving-source-position "$registration" \
    --fixed-fused "$FIXED_FUSED" \
    --output-dir "$method6" \
    --output-registration "$method6/registration.json" \
    --moving-channel "$channel" \
    --fit-intensity-transform "$fit_transform" \
    --starting-affine-matrix-zyx "$matrix" \
    --min-corr 0.15 \
    --min-grad-ncc "$min_grad" \
    --fixed-mask-threshold "$FIXED_THRESHOLD" \
    --source-label "$label" \
    --target-label 561 \
    --workers 1 \
    --devices 0 \
    --resume

  local manual=()
  if [[ -n ${MANUAL_INITIALIZATIONS:-} ]]; then
    manual=(
      --manual-initializations "$MANUAL_INITIALIZATIONS"
      --manual-initializations-channel "$label"
    )
  fi
  "$LIGHTSHEET_BIN" cross-register recover-native \
    --summary "$method6/fused_fixed_method8_summary.json" \
    --output-dir "$native" \
    --adjacency-json "$measurements" \
    --devices 0 \
    --resume \
    "${manual[@]}"

  "$LIGHTSHEET_BIN" cross-register complete-recovery \
    --native-summary "$native/fused_fixed_method8_summary.json" \
    --output-dir "$complete"
}

echo "[edu-xreg] starting 514 and 405 branches $(date --iso-8601=seconds)"
(
  export CUDA_VISIBLE_DEVICES="$GPU_514"
  export MPLCONFIGDIR="${MPLCONFIGDIR_514:-${TMPDIR:-/tmp}/squisher-edu-xreg-514-$$}"
  mkdir -p "$MPLCONFIGDIR"
  run_channel 514 2 "$POSITION_514" "$REGISTRATION_514" "$TILE_DIR_514" \
    "$MEASUREMENTS_514" log1p log1p "$MATRIX_514" 0.24
) &
pids+=("$!")

(
  export CUDA_VISIBLE_DEVICES="$GPU_405"
  export MPLCONFIGDIR="${MPLCONFIGDIR_405:-${TMPDIR:-/tmp}/squisher-edu-xreg-405-$$}"
  mkdir -p "$MPLCONFIGDIR"
  run_channel 405 0 "$POSITION_405" "$REGISTRATION_405" "$TILE_DIR_405" \
    "$MEASUREMENTS_405" identity linear "$MATRIX_405" 0.20
) &
pids+=("$!")

branch_status=0
for pid in "${pids[@]}"; do
  wait "$pid" || branch_status=1
done
pids=()
[[ $branch_status == 0 ]]

summary_514="$OUTPUT_BASE/xreg-514-complete-recovery-$RUN_TAG/fused_fixed_method8_summary.json"
summary_405="$OUTPUT_BASE/xreg-405-complete-recovery-$RUN_TAG/fused_fixed_method8_summary.json"
"$LIGHTSHEET_BIN" cross-register prefusion-qc \
  --summary "514=$summary_514" \
  --summary "405=$summary_405" \
  --fixed-volume "$FIXED_FUSED" \
  --fixed-threshold "$FIXED_THRESHOLD" \
  --output-dir "$QC_DIR"
"$LIGHTSHEET_BIN" cross-register interpolated-window-qc \
  --summary "$summary_514" \
  --moving-label 514 \
  --fixed-label 561 \
  --dataset-label "$DATASET_LABEL" \
  --output "$QC_DIR/interpolated-514.png"
"$LIGHTSHEET_BIN" cross-register interpolated-window-qc \
  --summary "$summary_405" \
  --moving-label 405 \
  --fixed-label 561 \
  --dataset-label "$DATASET_LABEL" \
  --output "$QC_DIR/interpolated-405.png"

echo "[edu-xreg] pre-fusion QC ready at $QC_DIR"
