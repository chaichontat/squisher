#!/usr/bin/env bash

set -eo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/one.ome.tif [lightsheet edu-workflow options]" >&2
  exit 2
fi

source /home/chaichontat/miniforge3/etc/profile.d/conda.sh
conda activate multi
set -u

expected_python=/home/chaichontat/miniforge3/envs/multi/bin/python
actual_python=$(python -c 'import sys; print(sys.executable)')
if [[ "$actual_python" != "$expected_python" ]]; then
  echo "Expected $expected_python, got $actual_python" >&2
  exit 1
fi

export PYTHONPATH=/home/chaichontat/squisher/lightsheet/src:/home/chaichontat/squisher/deconv/src:/home/chaichontat/squisher/squisher/src:/home/chaichontat/squisher/segment
export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
export LD_LIBRARY_PATH="$CUDA_PATH/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

exec lightsheet edu-workflow "$@"
