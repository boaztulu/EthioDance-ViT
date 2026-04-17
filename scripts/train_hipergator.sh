#!/bin/bash
#SBATCH --job-name=ethiodance
#SBATCH --account=rcstudents
#SBATCH --qos=rcstudents
#SBATCH --partition=hpg-turin
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@180          # deliver SIGUSR1 3 min before time limit
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=btulu@ufl.edu
#SBATCH --output=slurm_logs/ethiodance_%j.out
#SBATCH --error=slurm_logs/ethiodance_%j.err

# =============================================================================
# EthioDance-ViT — TimeSformer fine-tuning on HiPerGator
# =============================================================================
# Usage (from the repo root, i.e. the folder that contains configs/, src/, ...)
#   sbatch scripts/train_hipergator.sh                       # full run
#   sbatch scripts/train_hipergator.sh --debug               # GPU smoke test
#   sbatch scripts/train_hipergator.sh -o train.epochs=120   # override a field
#
# Optional env-var overrides (export before sbatch):
#   ETHIODANCE_EXP_DIR=/path/outside/repo   # where run dirs + checkpoints land
#   ETHIODANCE_DATA_ROOT=/path/to/Data/ALL  # if dataset is not at ../Data/ALL
#
# On SIGUSR1 (3 min before wall-time) or --requeue: python is signaled to save
# a resumable checkpoint, SLURM requeue is called, and the next allocation
# re-enters this script with the SAME base-job-id so the run dir is reused.
# =============================================================================

# ---- Step 1: resolve repo root (BEFORE `set -e`) ----------------------------
# SLURM copies the batch script to a spool dir (/var/spool/slurmd/...) before
# executing it, so $BASH_SOURCE does NOT point at your clone. The authoritative
# location when running under SLURM is $SLURM_SUBMIT_DIR (set by sbatch to the
# cwd at submission time). Fall back to BASH_SOURCE for local testing.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}" || { echo "FATAL: cannot cd to REPO_ROOT=${REPO_ROOT}" >&2; exit 2; }

# ---- Step 2: diagnostics FIRST (before any mkdir that can fail) -------------
echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID:-not-slurm}"
echo "Array id     : ${SLURM_ARRAY_JOB_ID:-n/a}"
echo "User         : $(whoami)"
echo "Host         : $(hostname)"
echo "Submit dir   : ${SLURM_SUBMIT_DIR:-<unset>}"
echo "Script src   : ${BASH_SOURCE[0]}"
echo "Repo root    : ${REPO_ROOT}"
echo "PWD          : $(pwd)"
echo "Repo perms   : $(ls -ld . | awk '{print $1, $3":"$4}')"
echo "Parent perms : $(ls -ld .. | awk '{print $1, $3":"$4}')"
echo "GPU          : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Started      : $(date -Iseconds)"
echo "============================================================"

# ---- Step 3: writability sanity check ---------------------------------------
if [[ ! -w "${REPO_ROOT}" ]]; then
    echo "FATAL: repo root '${REPO_ROOT}' is not writable by $(whoami)." >&2
    echo "       Re-clone the repo somewhere you own, e.g." >&2
    echo "         mkdir -p /blue/rcstudents/$(whoami)/ethiodance && cd \$_" >&2
    echo "         git clone <url> EthioDance-ViT" >&2
    exit 2
fi

# ---- Step 4: directory setup (all guarded — no set -e trap here) ------------
mkdir -p slurm_logs

# EXP_ROOT holds checkpoints + figures. It MUST live outside the repo so large
# files don't drift into git. Precedence:
#   1. $ETHIODANCE_EXP_DIR (explicit override)
#   2. ../experiments next to the repo (if parent is writable)
#   3. $HOME/ethiodance_experiments (always-safe fallback)
pick_experiments_root() {
    if [[ -n "${ETHIODANCE_EXP_DIR:-}" ]]; then
        if mkdir -p "${ETHIODANCE_EXP_DIR}" 2>/dev/null; then
            printf '%s' "${ETHIODANCE_EXP_DIR}"; return
        fi
    fi
    local sibling
    sibling="$(dirname "${REPO_ROOT}")/experiments"
    if mkdir -p "${sibling}" 2>/dev/null; then
        printf '%s' "${sibling}"; return
    fi
    mkdir -p "${HOME}/ethiodance_experiments"
    printf '%s' "${HOME}/ethiodance_experiments"
}
EXP_ROOT="$(pick_experiments_root)"
echo "Experiments  : ${EXP_ROOT}"

BASE_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local$(date +%s)}}"
RUN_DIR="${RUN_DIR:-${EXP_ROOT}/ethiodance_${BASE_JOB_ID}}"
mkdir -p "${RUN_DIR}/checkpoints"
echo "Run dir      : ${RUN_DIR}"

# ---- Step 5: data root (default ../Data/ALL, overridable) -------------------
DATA_ROOT="${ETHIODANCE_DATA_ROOT:-${REPO_ROOT}/../Data/ALL}"
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "FATAL: data root not found: ${DATA_ROOT}" >&2
    echo "       Set ETHIODANCE_DATA_ROOT or place the dataset at ../Data/ALL" >&2
    exit 2
fi
echo "Data root    : ${DATA_ROOT}"

# ---- Step 6: conda env ------------------------------------------------------
module purge
module load conda
if ! conda env list | awk '{print $1}' | grep -qx ethiodance; then
    echo "FATAL: conda env 'ethiodance' not found. Create it once:" >&2
    echo "    module load conda" >&2
    echo "    mamba create -n ethiodance python=3.11 -y" >&2
    echo "    conda activate ethiodance" >&2
    echo "    pip install -r requirements.txt" >&2
    exit 2
fi
# shellcheck disable=SC1091
conda activate ethiodance

# Fail fast if the env is missing critical packages.
python -c "import torch, transformers; print(f'torch={torch.__version__} hf={transformers.__version__} cuda={torch.cuda.is_available()} gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

# From here on, surface any unexpected failure immediately.
set -Euo pipefail

# ---- Step 7: splits (regenerate if missing) ---------------------------------
if [[ ! -f configs/splits.json ]]; then
    echo "[setup] configs/splits.json missing — generating stratified splits."
    python scripts/prepare_splits.py \
        --data-root "${DATA_ROOT}" \
        --out configs/splits.json \
        --seed 42
fi

# ---- Step 8: SIGUSR1 handler (bash side) ------------------------------------
# SLURM sends SIGUSR1 to the batch wrapper 3 min before time limit. We forward
# it to the python child so the in-process RequeueHandler flushes last.pth.
cleanup_and_requeue() {
    echo "[slurm] SIGUSR1 received at $(date -Iseconds) — forwarding to python PID ${PY_PID:-?}."
    if [[ -n "${PY_PID:-}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
        kill -USR1 "${PY_PID}" || true
        wait "${PY_PID}" || true
    fi
    echo "[slurm] python exited; running 'scontrol requeue ${SLURM_JOB_ID}'."
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
}
trap cleanup_and_requeue USR1

# ---- Step 9: train (background so signals are delivered) --------------------
python -u scripts/train.py \
    --config configs/hipergator.yaml \
    --output-dir "${RUN_DIR}" \
    --resume \
    -o data.root="${DATA_ROOT}" \
    "$@" &
PY_PID=$!
echo "[slurm] python launched, PID=${PY_PID}"

# `wait` may exit non-zero on signal; capture exit code without tripping set -e.
set +e
wait "${PY_PID}"
PY_EXIT=$?
set -e

echo "[slurm] python exited with status ${PY_EXIT}."

# ---- Step 10: post-training figures (best-effort) ---------------------------
BEST="${RUN_DIR}/checkpoints/best.pth"
if [[ -f "${BEST}" && ${PY_EXIT} -eq 0 ]]; then
    echo "[slurm] Generating paper figures."
    set +e
    python -u scripts/visualize.py \
        --config configs/hipergator.yaml \
        --checkpoint "${BEST}" \
        --output-dir "${RUN_DIR}/figures" \
        -o data.root="${DATA_ROOT}"
    set -e
fi

echo "Finished at $(date -Iseconds). Outputs in ${RUN_DIR}."
exit ${PY_EXIT}
