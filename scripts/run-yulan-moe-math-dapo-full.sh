#!/usr/bin/env bash

set -euo pipefail

SLIME_ROOT=/mnt/yulan/lvzhihao/PostTrain/slime
POSTTRAIN_ROOT=/mnt/yulan/lvzhihao/PostTrain
YULAN_ROOT=${POSTTRAIN_ROOT}/YuLan-Pretrain
ROUTE_REPLAY_PATCH=${SLIME_ROOT}/patches/route_replay_yulan_pretrain.patch
ROUTE_REPLAY_TARGETS=(
   megatron/core/transformer/moe/moe_utils.py
   megatron/core/transformer/moe/router.py
)

if ! awk '
   { for (i = 1; i <= NF; i++) {
      if ($i == "machine") { active = ($(i + 1) == "api.wandb.ai"); i++; continue }
      if (active && $i == "password" && $(i + 1) != "") found = 1
   } }
   END { exit !found }
' "$HOME/.netrc" 2>/dev/null; then
   echo "A W&B API key must be configured in this user's ~/.netrc before starting full training." >&2
   exit 2
fi

for required_path in \
   "${SLIME_ROOT}/.venv/bin/python" \
   "${YULAN_ROOT}/.git" \
   "${ROUTE_REPLAY_PATCH}" \
   "${POSTTRAIN_ROOT}/models/YuLan-MoE-Base/config.json" \
   "${POSTTRAIN_ROOT}/models/YuLan-MoE-Base/mcore-tp1pp1ep8-sqrtgate-hybrid0625-05/latest_checkpointed_iteration.txt" \
   "${POSTTRAIN_ROOT}/data/hf_data/math_rl_mid_high_20k_20260924/rollout_20k.jsonl" \
   "${POSTTRAIN_ROOT}/data/aime-2024.jsonl" \
   "${POSTTRAIN_ROOT}/data/aime-2025.jsonl"; do
   if [[ ! -e "${required_path}" ]]; then
      echo "Required path does not exist: ${required_path}" >&2
      exit 2
   fi
done

patch_applied_by_launcher=0
if git -C "${YULAN_ROOT}" apply --reverse --check "${ROUTE_REPLAY_PATCH}" >/dev/null 2>&1; then
   echo "YuLan route-replay patch is already applied; leaving it applied after launch."
elif git -C "${YULAN_ROOT}" diff --quiet -- "${ROUTE_REPLAY_TARGETS[@]}"; then
   git -C "${YULAN_ROOT}" apply "${ROUTE_REPLAY_PATCH}"
   patch_applied_by_launcher=1
else
   echo "YuLan route-replay targets have unrelated local changes; refusing to overwrite them." >&2
   git -C "${YULAN_ROOT}" status --short -- "${ROUTE_REPLAY_TARGETS[@]}" >&2
   exit 2
fi

restore_route_replay_patch() {
   if [[ "${patch_applied_by_launcher}" == "1" ]]; then
      git -C "${YULAN_ROOT}" apply --reverse "${ROUTE_REPLAY_PATCH}"
      echo "Restored YuLan route-replay target files; patch remains at ${ROUTE_REPLAY_PATCH}."
   fi
}
trap restore_route_replay_patch EXIT

export SLIME_USE_YULAN_THD_CP=1
export SLIME_USE_YULAN_TRANSFORMER_BLOCK=1
export AUTO_GENERATE_CU_SEQLENS=1
export DESTROY_ROLLOUT_ENGINES=1
export REBUILD_TRAIN_ACTORS=1
export COLOCATE=1
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
export PROMPT_DATA=${PROMPT_DATA:-${POSTTRAIN_ROOT}/data/hf_data/AIME_like_data_judgeable/aime_like_judgeable.jsonl}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${POSTTRAIN_ROOT}/outputs/yulan_moe_math_dapo_full_cp8_20260914}
if [[ -z "${SAVE_HF:-}" ]]; then
   SAVE_HF="${OUTPUT_ROOT}/hf/rollout_{rollout_id}"
fi
export SAVE_HF
export RUN_LOG_FILE=${RUN_LOG_FILE:-${POSTTRAIN_ROOT}/outputs/logs/yulan_moe_math_dapo_full_cp8_20260914.log}
export SAVE_INTERVAL=20
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-1024}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-16384}
export ROLLOUT_MAX_CONTEXT_LEN=32768
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
export N_SAMPLES_PER_PROMPT=8
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
export OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-32}
export NUM_ROLLOUT=${NUM_ROLLOUT:-446}

mkdir -p "$(dirname "${RUN_LOG_FILE}")"

"${SLIME_ROOT}/scripts/run-yulan-moe-base-math-smoke.sh" \
   --micro-batch-size 1 \
   --use-routing-replay \
   --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}" \
   --save-debug-rollout-data "${OUTPUT_ROOT}/debug/rollout_{rollout_id}.pt" \
   --save-debug-rollout-interval 5 \
   --eval-interval 2 \
   --eval-prompt-data aime2024 "${POSTTRAIN_ROOT}/data/aime-2024.jsonl" aime2025 "${POSTTRAIN_ROOT}/data/aime-2025.jsonl" \
   --eval-input-key prompt \
   --eval-label-key label \
   --n-samples-per-eval-prompt 2 \
   --eval-temperature "${EVAL_TEMPERATURE:-${ROLLOUT_TEMPERATURE}}" \
   --eval-max-prompt-len 1024 \
   --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN:-16384}" \
   --eval-max-context-len 32768 \
   --use-wandb \
   --wandb-mode online \
   --wandb-project yulan-moe-math-rl \
   --wandb-group yulan-moe-dapo-full-cp8 \
   --disable-wandb-random-suffix \
   --wandb-dir "${POSTTRAIN_ROOT}/outputs/wandb" \
   "$@"
