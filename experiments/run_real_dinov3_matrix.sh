#!/usr/bin/env bash
set -euo pipefail

cd /home/claudeuser/RL4VLA/PDDL

K_VALUES="${K_VALUES:-20,50,100,200}"
N_EPOCHS="${N_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CONDITIONS="${CONDITIONS:-static,random_pairs,adjacent,full}"
TRANSITION_WARMUP_EPOCHS="${TRANSITION_WARMUP_EPOCHS:-20}"
W_CONTRAST="${W_CONTRAST:-0.0}"
W_EQUIV="${W_EQUIV:-1.0}"
W_CF="${W_CF:-0.3}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-20.0}"
SCORING_HEAD_TYPE="${SCORING_HEAD_TYPE:-film}"
MAX_TRANSITION_SAMPLES="${MAX_TRANSITION_SAMPLES:-0}"
GPU_ID="${GPU_ID:-0}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="experiments/logs/real_dinov3_matrix_${STAMP}"

mkdir -p "${LOGDIR}"

run_one() {
  local mask="$1"
  local exp_name="real_dinov3_${mask}_${STAMP}"
  local cache="experiments/fewshot_transition_cache_${mask}.pt"
  echo "Running ${exp_name}"
  echo "  K=${K_VALUES} epochs=${N_EPOCHS} warmup=${TRANSITION_WARMUP_EPOCHS} max_trans=${MAX_TRANSITION_SAMPLES} conditions=${CONDITIONS} gpu=${GPU_ID}"
  echo "  mask=${mask} cache=${cache}"

  rm -rf "${cache}.lock"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 python -u training/run_fewshot_structural.py \
    --device cuda \
    --n-epochs "${N_EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --k-values "${K_VALUES}" \
    --fewshot-unit image \
    --conditions "${CONDITIONS}" \
    --transition-mask-source "${mask}" \
    --transition-warmup-epochs "${TRANSITION_WARMUP_EPOCHS}" \
    --w-contrast "${W_CONTRAST}" \
    --w-equiv "${W_EQUIV}" \
    --w-cf "${W_CF}" \
    --pos-weight-max "${POS_WEIGHT_MAX}" \
    --scoring-head-type "${SCORING_HEAD_TYPE}" \
    --max-transition-samples "${MAX_TRANSITION_SAMPLES}" \
    --transition-cache "${cache}" \
    --rebuild-transition-cache \
    --exp-name "${exp_name}" \
    2>&1 | stdbuf -oL sed "s/^/[${mask}] /" | tee "${LOGDIR}/${mask}.log"
}

run_one state_diff
run_one pddl
run_one pddl_conservative

python training/summarize_structural_results.py \
  "experiments/real_dinov3_state_diff_${STAMP}/fewshot_structural_results.json" \
  "experiments/real_dinov3_pddl_${STAMP}/fewshot_structural_results.json" \
  "experiments/real_dinov3_pddl_conservative_${STAMP}/fewshot_structural_results.json" \
  --csv "experiments/real_dinov3_matrix_${STAMP}.csv" \
  --md "experiments/real_dinov3_matrix_${STAMP}.md"

echo "Summary:"
echo "  experiments/real_dinov3_matrix_${STAMP}.csv"
echo "  experiments/real_dinov3_matrix_${STAMP}.md"
