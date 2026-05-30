#!/usr/bin/env bash
set -euo pipefail

cd /home/claudeuser/RL4VLA/PDDL

MASK_SOURCE="${MASK_SOURCE:-pddl}"
K_VALUES="${K_VALUES:-20}"
N_EPOCHS="${N_EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONDITIONS="${CONDITIONS:-static,adjacent,full}"
TRANSITION_WARMUP_EPOCHS="${TRANSITION_WARMUP_EPOCHS:-0}"
W_CONTRAST="${W_CONTRAST:-0.0}"
W_EQUIV="${W_EQUIV:-1.0}"
W_CF="${W_CF:-0.3}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-20.0}"
SCORING_HEAD_TYPE="${SCORING_HEAD_TYPE:-film}"
MAX_TRANSITION_SAMPLES="${MAX_TRANSITION_SAMPLES:-1024}"
GPU_ID="${GPU_ID:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="quick_real_dinov3_${MASK_SOURCE}_${STAMP}"
LOGDIR="experiments/logs/${EXP_NAME}"
CACHE="experiments/fewshot_transition_cache_${MASK_SOURCE}_quick.pt"

mkdir -p "${LOGDIR}"
rm -rf "${CACHE}.lock"

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 python -u training/run_fewshot_structural.py \
  --device cuda \
  --n-epochs "${N_EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --k-values "${K_VALUES}" \
  --fewshot-unit image \
  --conditions "${CONDITIONS}" \
  --transition-mask-source "${MASK_SOURCE}" \
  --transition-warmup-epochs "${TRANSITION_WARMUP_EPOCHS}" \
  --w-contrast "${W_CONTRAST}" \
  --w-equiv "${W_EQUIV}" \
  --w-cf "${W_CF}" \
  --pos-weight-max "${POS_WEIGHT_MAX}" \
  --scoring-head-type "${SCORING_HEAD_TYPE}" \
  --max-transition-samples "${MAX_TRANSITION_SAMPLES}" \
  --transition-cache "${CACHE}" \
  --rebuild-transition-cache \
  --exp-name "${EXP_NAME}" \
  2>&1 | stdbuf -oL sed "s/^/[quick-${MASK_SOURCE}] /" | tee "${LOGDIR}/run.log"

python training/summarize_structural_results.py \
  "experiments/${EXP_NAME}/fewshot_structural_results.json" \
  --md "experiments/${EXP_NAME}/summary.md" \
  --csv "experiments/${EXP_NAME}/summary.csv"

echo "Quick summary: experiments/${EXP_NAME}/summary.md"
