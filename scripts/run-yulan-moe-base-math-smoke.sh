#!/bin/bash

set -euo pipefail
set -x

SLIME_ROOT=/mnt/yulan/lvzhihao/PostTrain/slime
POSTTRAIN_ROOT=/mnt/yulan/lvzhihao/PostTrain
YULAN_ROOT=${POSTTRAIN_ROOT}/YuLan-Pretrain
HF_CHECKPOINT=${HF_CHECKPOINT:-${POSTTRAIN_ROOT}/models/YuLan-MoE-Base}
MCORE_CHECKPOINT=${MCORE_CHECKPOINT:-${HF_CHECKPOINT}/mcore-tp1pp1ep8-sqrtgate-hybrid0625-05}
PROMPT_DATA=${PROMPT_DATA:-${POSTTRAIN_ROOT}/data/hf_data/AIME_like_data/aime_8.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-${POSTTRAIN_ROOT}/outputs/yulan_moe_math_smoke_8k}

for required_path in \
   "${SLIME_ROOT}/.venv/bin/python" \
   "${HF_CHECKPOINT}/config.json" \
   "${MCORE_CHECKPOINT}/latest_checkpointed_iteration.txt" \
   "${PROMPT_DATA}"; do
   if [[ ! -e "${required_path}" ]]; then
      echo "Required path does not exist: ${required_path}" >&2
      exit 2
   fi
done

export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_CUDA_ARCH_LIST=8.0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
export MOE_ROUTER_SQRT_GATE=1
export PYTHONPATH="${YULAN_ROOT}:${POSTTRAIN_ROOT}/Triton-Distributed/python:${POSTTRAIN_ROOT}/flash-linear-attention:${POSTTRAIN_ROOT}/linghe:${SLIME_ROOT}:${PYTHONPATH:-}"

source "${SLIME_ROOT}/.venv/bin/activate"
source "${SLIME_ROOT}/scripts/models/yulan-moe-base.sh"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --model-name qwen3_next
   --ref-load "${MCORE_CHECKPOINT}"
   --save "${OUTPUT_ROOT}"
   --save-interval "${SAVE_INTERVAL:-1}"
   --checkpoint-retention-count 2
   --checkpoint-retention-interval 20
   --no-save-optim
   --no-save-rng
)
if [[ -n "${SAVE_HF:-}" ]]; then
   CKPT_ARGS+=(--save-hf "${SAVE_HF}")
fi

ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
RM_TYPE=${RM_TYPE:-dapo}
REWARD_KEY=${REWARD_KEY:-}
EVAL_REWARD_KEY=${EVAL_REWARD_KEY:-}
PROMPT_SUFFIX=${PROMPT_SUFFIX:-$'\n\nPlease reason step by step, and put your final answer within \\boxed{}.'}
ROLLOUT_ARGS=(
   --prompt-data "${PROMPT_DATA}"
   --input-key query
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type "${RM_TYPE}"
   --num-rollout "${NUM_ROLLOUT:-1}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-2}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
   --rollout-max-prompt-len 1024
   --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN:-8192}"
   --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN:-9216}"
   --rollout-temperature "${ROLLOUT_TEMPERATURE}"
   --global-batch-size "${GLOBAL_BATCH_SIZE:-8}"
   --balance-data
   --prompt-suffix "${PROMPT_SUFFIX}"
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std_with_fallback
)

# Debugging can replay a previously dumped rollout while keeping the SGLang
# engines alive. This avoids spending several minutes on generation when the
# change under test is in the actor/optimizer path. The template also reuses
# rollout_eval_<id>.pt for eval when that file exists.
if [[ -n "${REPLAY_ROLLOUT_DATA:-}" ]]; then
   if [[ "${REPLAY_ROLLOUT_DATA}" == *"{rollout_id}"* ]]; then
      replay_probe="${REPLAY_ROLLOUT_DATA//\{rollout_id\}/0}"
   else
      replay_probe="${REPLAY_ROLLOUT_DATA}"
   fi
   if [[ ! -f "${replay_probe}" ]]; then
      echo "REPLAY_ROLLOUT_DATA does not exist: ${replay_probe}" >&2
      exit 2
   fi
   ROLLOUT_ARGS+=(
      --rollout-function-path slime.rollout.forge_load.generate_rollout
      --load-forge-rollout-data "${REPLAY_ROLLOUT_DATA}"
   )
fi
if [[ -n "${REWARD_KEY}" ]]; then
   ROLLOUT_ARGS+=(--reward-key "${REWARD_KEY}")
fi
if [[ -n "${EVAL_REWARD_KEY}" ]]; then
   ROLLOUT_ARGS+=(--eval-reward-key "${EVAL_REWARD_KEY}")
fi

PARALLEL_ARGS=(
   --tensor-model-parallel-size 1
   --pipeline-model-parallel-size 1
   --context-parallel-size 8
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-1536}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
)
if [[ "${USE_DYNAMIC_BATCH_SIZE:-1}" == "1" ]]; then
   PARALLEL_ARGS+=(--use-dynamic-batch-size)
fi

ALGORITHM_ARGS=(
   --advantage-estimator grpo
   --kl-coef 0
   --kl-loss-coef 0
   --entropy-coef "${ENTROPY_COEF:-0}"
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
if [[ "${USE_PRECISION_AWARE_OPTIMIZER:-1}" == "1" ]]; then
   OPTIMIZER_ARGS+=(
      --use-precision-aware-optimizer
      --exp-avg-dtype "${EXP_AVG_DTYPE:-bf16}"
      --exp-avg-sq-dtype "${EXP_AVG_SQ_DTYPE:-bf16}"
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS:-8}"
   # Rollout requests are independent; cache-aware routing can pin them to
   # one long-running engine. Round-robin keeps all TP2/EP2 engines busy.
   --router-policy round_robin
   # The YuLan plugin's validated HF serving layout is TP2/EP2.  Keep four
   # two-GPU engines on the eight rollout GPUs instead of running an
   # unvalidated TP8/EP8 engine, which produces corrupted trajectories here.
   --rollout-num-gpus-per-engine 2
   --sglang-tensor-parallel-size 2
   --sglang-ep-size 2
   --sglang-dtype bfloat16
   # The released HF config omits this YuLan checkpoint semantic.  Keep the
   # checkpoint immutable and pass it to the plugin through SGLang's native
   # model-config override instead.
   --sglang-json-model-override-args '{"moe_router_sqrt_gate":true}'
   --sglang-mem-fraction-static 0.7
   --sglang-cuda-graph-bs 1 2 4 8
   --sglang-max-running-requests 8
)

MISC_ARGS=(
   --attention-dropout 0
   --hidden-dropout 0
   --attention-backend flash
   --moe-token-dispatcher-type alltoall
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --no-create-attention-mask-in-dataloader
   --gdn-cp-impl chunk
   --gdn-cp-run-level-layout
   --gdn-cp-use-token-exchange
   --gdn-backend fla
   --gdn-gated-norm-backend linghe
   --gdn-use-qk-l2norm-in-kernel
   --gdn-use-gate-in-kernel
   --gdn-use-beta-sigmoid-in-kernel
   --reset-attention-mask
   --reset-position-ids
)

# Strict document packing currently requires TP=1 in MCore.  Keep it enabled
# by default, but allow TP-based actor layouts to opt out explicitly.
if [[ "${AUTO_GENERATE_CU_SEQLENS:-1}" == "1" ]]; then
   MISC_ARGS+=(--auto-generate-cu-seqlens)
fi

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_ADDR

if ! curl --fail --silent --max-time 2 http://127.0.0.1:8265/api/version >/dev/null; then
   ray start \
      --head \
      --node-ip-address "${MASTER_ADDR}" \
      --num-gpus 8 \
      --disable-usage-stats \
      --dashboard-host 127.0.0.1 \
      --dashboard-port 8265
fi

TRAIN_LAYOUT_ARGS=(--actor-num-nodes 1 --actor-num-gpus-per-node 8)
if [[ "${COLOCATE:-0}" == "1" ]]; then
   TRAIN_LAYOUT_ARGS+=(--colocate)
   if [[ "${DESTROY_ROLLOUT_ENGINES:-0}" == "1" ]]; then
      TRAIN_LAYOUT_ARGS+=(--destroy-rollout-engines)
   fi
   if [[ "${REBUILD_TRAIN_ACTORS:-0}" == "1" ]]; then
      TRAIN_LAYOUT_ARGS+=(--rebuild-train-actors)
   fi
   if [[ "${ROLLOUT_NUM_GPUS:-8}" == "0" ]]; then
      # Replay-only validation has no SGLang workers. Avoid initializing the
      # TorchMemorySaver offload path in this mode.
      TRAIN_LAYOUT_ARGS+=(--no-offload-train --no-offload-rollout)
   elif [[ "${DESTROY_ROLLOUT_ENGINES:-0}" == "1" ]]; then
      # Engines are destroyed before actor_train and recreated after saving;
      # no TorchMemorySaver offload is needed in this lifecycle.
      TRAIN_LAYOUT_ARGS+=(--no-offload-train --no-offload-rollout)
   else
      if [[ "${REBUILD_TRAIN_ACTORS:-0}" == "1" ]]; then
         TRAIN_LAYOUT_ARGS+=(--no-offload-train --no-offload-rollout)
      else
         # Full RL keeps SGLang workers for rollout, but releases their GPU
         # weights/KV cache while the colocated actor is training.
         TRAIN_LAYOUT_ARGS+=(--offload-train --offload-rollout)
      fi
   fi
fi

# Keep W&B credentials in the Ray worker user's ~/.netrc. Do not put the key in
# Ray runtime_env: Ray exposes runtime environment variables in job metadata.
set +x
RUNTIME_ENV_JSON="{\"env_vars\":{\"SLIME_USE_YULAN_THD_CP\":\"${SLIME_USE_YULAN_THD_CP:-0}\",\"SLIME_USE_YULAN_TRANSFORMER_BLOCK\":\"${SLIME_USE_YULAN_TRANSFORMER_BLOCK:-0}\"}}"

# Keep xtrace disabled through submission: Ray's command echo would otherwise
# print the W&B credential embedded in runtime_env_json.
LOG_FILE=${RUN_LOG_FILE:-${POSTTRAIN_ROOT}/outputs/logs/yulan_moe_math_rl_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "${LOG_FILE}")"
echo "Full Ray job log: ${LOG_FILE}"
ray job submit \
   --address=http://127.0.0.1:8265 \
   --runtime-env-json "${RUNTIME_ENV_JSON}" \
   --working-dir "${SLIME_ROOT}" \
   -- \
   "${SLIME_ROOT}/.venv/bin/python" train.py \
   "${TRAIN_LAYOUT_ARGS[@]}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${PARALLEL_ARGS[@]}" \
   "${ALGORITHM_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "$@" 2>&1 | tee "${LOG_FILE}"
