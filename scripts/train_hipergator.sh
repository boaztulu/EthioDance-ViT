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
#SBATCH --output=../slurm_logs/ethiodance_%j.out
#SBATCH --error=../slurm_logs/ethiodance_%j.err

# =============================================================================
# EthioDance-ViT — TimeSformer fine-tuning on HiPerGator
# =============================================================================
# Usage:
#   sbatch scripts/train_hipergator.sh                       # full run
#   sbatch scripts/train_hipergator.sh --debug               # smoke test on GPU
#   sbatch scripts/train_hipergator.sh -o train.epochs=120   # override a field
#
# On requeue (SIGUSR1 or --requeue), this script reuses the SAME run dir so
# training resumes from last.pth. The run dir is keyed on SLURM_JOB_ID without
# the array task suffix, so `scontrol requeue` keeps writing to one place.
# =============================================================================

set -euo pipefail

# All heavy artifacts live OUTSIDE the git repo (one level up).
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p ../slurm_logs ../experiments

# ---- Run directory (stable across requeues of the same SLURM job) -----------
BASE_JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:?SLURM_JOB_ID unset}}"
RUN_DIR="${RUN_DIR:-$(pwd)/../experiments/ethiodance_${BASE_JOB_ID}}"
mkdir -p "${RUN_DIR}/checkpoints"

# ---- Environment ------------------------------------------------------------
module purge
module load conda
# Use an env named `ethiodance`; create once with:
#   mamba create -n ethiodance python=3.11 pip && conda activate ethiodance
#   pip install -r requirements.txt
conda activate ethiodance

echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Base job id  : ${BASE_JOB_ID}"
echo "Node         : $(hostname)"
echo "GPU          : ${CUDA_VISIBLE_DEVICES:-none}"
echo "Run dir      : ${RUN_DIR}"
echo "Started      : $(date -Iseconds)"
echo "============================================================"

python -c "import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}, gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

# ---- Splits (regenerate if missing) -----------------------------------------
if [[ ! -f configs/splits.json ]]; then
    echo "[setup] configs/splits.json missing — generating stratified splits."
    python scripts/prepare_splits.py \
        --data-root ../Data/ALL \
        --out configs/splits.json \
        --seed 42
fi

# ---- SIGUSR1 handler: forward to the training process, wait, then requeue ---
# We launch python in the background, capture its PID, and install a trap that
# forwards SIGUSR1 to python so the in-process RequeueHandler flushes a
# checkpoint cleanly. The python process itself calls `scontrol requeue` once
# the resumable checkpoint is on disk.
cleanup_and_requeue() {
    echo "[slurm] SIGUSR1 received at $(date -Iseconds). Forwarding to PID ${PY_PID}."
    if kill -0 "${PY_PID}" 2>/dev/null; then
        kill -USR1 "${PY_PID}" || true
        wait "${PY_PID}" || true
    fi
    echo "[slurm] python exited; running scontrol requeue."
    scontrol requeue "${SLURM_JOB_ID}" || true
    exit 0
}
trap cleanup_and_requeue USR1

# ---- Train ------------------------------------------------------------------
python -u scripts/train.py \
    --config configs/hipergator.yaml \
    --output-dir "${RUN_DIR}" \
    --resume \
    "$@" &
PY_PID=$!
wait "${PY_PID}"
PY_EXIT=$?

echo "[slurm] python exited with status ${PY_EXIT}."

# ---- Post-training: paper figures -------------------------------------------
BEST="${RUN_DIR}/checkpoints/best.pth"
if [[ -f "${BEST}" && ${PY_EXIT} -eq 0 ]]; then
    echo "[slurm] Generating paper figures."
    python -u scripts/visualize.py \
        --config configs/hipergator.yaml \
        --checkpoint "${BEST}" \
        --output-dir "${RUN_DIR}/figures" || true
fi

echo "Finished at $(date -Iseconds). Outputs in ${RUN_DIR}."
exit ${PY_EXIT}
