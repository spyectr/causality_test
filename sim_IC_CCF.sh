#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-sim_ic_ccf}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda command not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "Conda env '$ENV_NAME' already exists. Ensuring required packages are installed..."
  conda install -n "$ENV_NAME" -y "python=$PYTHON_VERSION" numpy matplotlib numba 
else
  echo "Creating conda env '$ENV_NAME'..."
  conda create -n "$ENV_NAME" -y "python=$PYTHON_VERSION" numpy matplotlib numba 
fi


conda activate "$ENV_NAME"
conda install -c conda-forge statsmodels pandas 
pip install tqdm-joblib



# install CCM
current_dir=$(pwd)
RNNC_REPO_URL="${RNNC_REPO_URL:-https://github.com/spyectr/RNNCausality.git}"
RNNC_DIRR="${RNNC_DIRR:-${PWD}/RNNCausality}"
echo "Cloning RNNCausality fork into $RNNC_DIRR"
mkdir -p "$(dirname "$RNNC_DIRR")"
git clone "$RNNC_REPO_URL" "$RNNC_DIRR"
cd RNNC_DIRR
if [ -f "$RNNC_DIRR/requirements.txt" ]; then
  python -m pip install --upgrade -r requirements.txt || true
fi
cd "$current_dir"




export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/${USER:-user}/mplconfig_sim_ic}"
mkdir -p "$MPLCONFIGDIR"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/${USER:-user}/xdg_cache_sim_ic}"
mkdir -p "$XDG_CACHE_HOME"
# Keep OpenMP nested runtime noise off in multi-worker IC runs.
export OMP_MAX_ACTIVE_LEVELS="${OMP_MAX_ACTIVE_LEVELS:-1}"
export KMP_WARNINGS="${KMP_WARNINGS:-0}"

python - <<'PY'
import numpy as np
import matplotlib
import numba
print("Environment check passed.")
print(f"numpy={np.__version__}")
print(f"matplotlib={matplotlib.__version__}")
print(f"numba={numba.__version__}")
PY

echo
echo "Environment is ready."
echo "Run pipeline with:"
echo "  conda activate $ENV_NAME"
echo "  MPLCONFIGDIR=$MPLCONFIGDIR XDG_CACHE_HOME=$XDG_CACHE_HOME OMP_MAX_ACTIVE_LEVELS=$OMP_MAX_ACTIVE_LEVELS KMP_WARNINGS=$KMP_WARNINGS MPLBACKEND=Agg python sim_IC.py"
