#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/ssd/lvzhihao/PostTrain"
SLIME_DIR="${ROOT_DIR}/slime"
YULAN_DIR="${ROOT_DIR}/YuLan-Pretrain"
VENV_DIR="${SLIME_DIR}/.venv"

MODEL_PATH="${MODEL_PATH:-/mnt/hdd/Qwen3-1.7B}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-8265}"
SGLANG_PORT="${SGLANG_PORT:-30110}"

WORK_DIR="${WORK_DIR:-${SLIME_DIR}/.tmp/qwen3_1p7b_smoke_rl_flash}"
REF_LOAD="${WORK_DIR}/qwen3_1p7b_torch_dist"
ACTOR_CKPT="${WORK_DIR}/actor_ckpt"
RUN_LOG="${WORK_DIR}/run.log"

NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-256}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-8192}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.35}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"

mkdir -p "${WORK_DIR}" "${ACTOR_CKPT}"

echo "[0/8] Settings"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  WORK_DIR=${WORK_DIR}"
echo "  ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}"
echo "  N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT}"
echo "  GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "  MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU}"
echo "  SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "Prompt data not found: ${PROMPT_DATA}" >&2
    exit 1
fi

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${YULAN_DIR}"
export PYTHONBUFFERED=16
export CUDA_VISIBLE_DEVICES

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

echo "[1/8] Clean stale ray/sglang processes"
pkill -f "python -m sglang.launch_server" || true
ray stop --force || true
pkill -f "ray::" || true
sleep 2

echo "[2/8] Load model config"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-1.7B.sh"

echo "[3/8] Convert HF checkpoint to Megatron torch_dist if needed"
if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
    torchrun --nproc-per-node "${NUM_GPUS}" \
        "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${MODEL_PATH}" \
        --save "${REF_LOAD}"
else
    echo "  Reusing existing converted checkpoint: ${REF_LOAD}"
fi

echo "[4/8] Start ray head"
ray start \
    --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --dashboard-port="${RAY_PORT}"

echo "[5/8] Prepare train arguments"
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${REF_LOAD}"
    --load "${ACTOR_CKPT}"
    --save "${ACTOR_CKPT}"
    --save-interval 1
)

ROLLOUT_ARGS=(
    --prompt-data "${PROMPT_DATA}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type deepscaler
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --balance-data
)

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-host 127.0.0.1
    --sglang-port "${SGLANG_PORT}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${YULAN_DIR}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"MASTER_ADDR\": \"${MASTER_ADDR}\",
    \"PYTHONBUFFERED\": \"16\",
    \"TORCH_CUDA_ARCH_LIST\": \"8.0\"
  }
}"

echo "[6/8] Submit RL job"
set -x
ray job submit --address="http://${MASTER_ADDR}:${RAY_PORT}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_DIR}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" | tee "${RUN_LOG}"
set +x

echo "[7/8] Ray job finished. Logs saved to ${RUN_LOG}"
echo "[8/8] If needed, inspect actor checkpoint in ${ACTOR_CKPT}"
