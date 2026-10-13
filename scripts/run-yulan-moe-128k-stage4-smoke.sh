#!/usr/bin/env bash

set -euo pipefail

SLIME_ROOT=/mnt/yulan/lvzhihao/PostTrain/slime
POSTTRAIN_ROOT=/mnt/yulan/lvzhihao/PostTrain

export HF_CHECKPOINT=${HF_CHECKPOINT:-${POSTTRAIN_ROOT}/models/YuLan-MoE-128k}
export MCORE_CHECKPOINT=${MCORE_CHECKPOINT:-${HF_CHECKPOINT}/mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix}
export PROMPT_DATA=${PROMPT_DATA:-${POSTTRAIN_ROOT}/data/hf_data/AIME_like_data/aime_8.jsonl}
export OUTPUT_ROOT=${OUTPUT_ROOT:-${POSTTRAIN_ROOT}/outputs/yulan_moe_128k_stage4_avg2_smoke}
export SAVE_HF=${SAVE_HF:-${OUTPUT_ROOT}/hf/rollout_{rollout_id}}
export RUN_LOG_FILE=${RUN_LOG_FILE:-/tmp/yulan_moe_128k_stage4_avg2_smoke.log}

# One rollout with two prompts and two samples per prompt yields one small
# actor update (global batch size 4). Long-context evaluation is configured
# separately from the short training rollout.
export ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS:-8}
export NUM_ROLLOUT=1
export ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
export N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
export GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-4}
export ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
export ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-4096}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-1536}

# Reuse the validated single-node rollout/actor handoff: release SGLang before
# building the actor, then export the updated actor before rebuilding SGLang.
export COLOCATE=1
export DESTROY_ROLLOUT_ENGINES=1
export REBUILD_TRAIN_ACTORS=1
export SLIME_USE_YULAN_THD_CP=1
export SLIME_USE_YULAN_TRANSFORMER_BLOCK=1
export AUTO_GENERATE_CU_SEQLENS=1

"${SLIME_ROOT}/scripts/run-yulan-moe-128k-math-smoke.sh" \
   --micro-batch-size 1 \
   --save-interval 1 \
   --eval-config "${SLIME_ROOT}/configs/eval/aime24_eval.yaml" \
   --eval-interval 1 \
   --eval-max-prompt-len 1024 \
   --eval-max-context-len 32768 \
   "$@"
