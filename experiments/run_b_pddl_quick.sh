#!/usr/bin/env bash
set -euo pipefail

cd /home/claudeuser/RL4VLA/PDDL

K_VALUE="${K_VALUE:-50}"
N_EPOCHS="${N_EPOCHS:-10}"
GPU_ID="${GPU_ID:-0}"

LOGDIR="/home/claudeuser/RL4VLA/PDDL/experiments/logs/b_pddl_quick_$(date +%Y%m%d_%H%M%S)"
CACHE="/home/claudeuser/RL4VLA/PDDL/experiments/fewshot_transition_cache_pddl.pt"
EXP_NAME="b_pddl_adjacent_k${K_VALUE}_e${N_EPOCHS}"

mkdir -p "$LOGDIR"
rm -rf "${CACHE}.lock"

echo "B-only quick test"
echo "  condition: adjacent"
echo "  transition_mask_source: pddl"
echo "  K images: ${K_VALUE}"
echo "  epochs: ${N_EPOCHS}"
echo "  GPU: ${GPU_ID}"
echo "  log: ${LOGDIR}/run.log"
echo "  cache: ${CACHE}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 python -u training/run_fewshot_structural.py \
  --device cuda \
  --fewshot-unit image \
  --k-values "${K_VALUE}" \
  --conditions adjacent \
  --transition-mask-source pddl \
  --transition-cache "${CACHE}" \
  --rebuild-transition-cache \
  --n-epochs "${N_EPOCHS}" \
  --batch-size 32 \
  --d-slot 256 \
  --exp-name "${EXP_NAME}" \
  2>&1 | stdbuf -oL sed 's/^/[B-PDDL] /' | tee "${LOGDIR}/run.log"
