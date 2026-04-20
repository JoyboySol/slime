#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/ssd/lvzhihao/PostTrain"
SLIME_DIR="${ROOT_DIR}/slime"
YULAN_DIR="${ROOT_DIR}/YuLan-Pretrain"
VENV_DIR="${SLIME_DIR}/.venv"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/mnt/hdd/lvzhihao/hf_models/Dist-mathcode10b-s1randg-sch1-CPT-200b-stage3-r640k-GDN2.9b-A7-12_20_21_23_46_48_49-sl32768bs128lr2e5-2e5/merged_10ckpts_iter_61984-hf_to_iter_71525-hf_mean}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/hdd/Nanbeige4.1-3B}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-4,5,6}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-7}"
NUM_GPUS="${NUM_GPUS:-3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-8265}"
SGLANG_PORT="${SGLANG_PORT:-30110}"
TEACHER_PORT="${TEACHER_PORT:-30221}"

WORK_DIR="${WORK_DIR:-${SLIME_DIR}/.tmp/yulan_cross_tokenizer_opd}"
REF_LOAD="${REF_LOAD:-${WORK_DIR}/yulan_torch_dist}"
ACTOR_CKPT="${ACTOR_CKPT:-${WORK_DIR}/actor_ckpt}"
RUN_LOG="${WORK_DIR}/run.log"
TEACHER_LOG="${WORK_DIR}/teacher.log"

NUM_ROLLOUT="${NUM_ROLLOUT:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-3}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-6}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-256}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"

MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-2048}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.2}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-2048}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.55}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
OPD_DISABLE_SEQUENCE_FALLBACK="${OPD_DISABLE_SEQUENCE_FALLBACK:-false}"

mkdir -p "${WORK_DIR}" "${ACTOR_CKPT}"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${YULAN_DIR}"
export PYTHONBUFFERED=16
export SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
export TORCH_CUDA_ARCH_LIST=8.0
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-float16}"
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-float32}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR}"

if [[ ! -d "${STUDENT_MODEL_PATH}" ]]; then
    echo "Student model path not found: ${STUDENT_MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -d "${TEACHER_MODEL_PATH}" ]]; then
    echo "Teacher model path not found: ${TEACHER_MODEL_PATH}" >&2
    exit 1
fi

if [[ ! -f "${PROMPT_DATA}" ]]; then
    echo "Prompt data not found: ${PROMPT_DATA}" >&2
    exit 1
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

cleanup() {
    set +e
    if [[ -n "${TEACHER_PID:-}" ]]; then
        kill "${TEACHER_PID}" >/dev/null 2>&1 || true
        wait "${TEACHER_PID}" >/dev/null 2>&1 || true
    fi
    ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[0/9] Settings"
echo "  STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH}"
echo "  TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "  TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES}"
echo "  WORK_DIR=${WORK_DIR}"

echo "[1/9] Clean stale ray/sglang processes"
pkill -f "python3? -m sglang.launch_server" || true
ray stop --force || true
pkill -f "ray::" || true
sleep 2

echo "[2/9] Load student model config"
source "${SCRIPT_DIR}/models/yulan-hybrid-gdn-dense-2.9b.sh"

echo "[3/9] Convert student HF checkpoint to Megatron torch_dist if needed"
if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
    CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
    torchrun --nproc-per-node "${NUM_GPUS}" \
        "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${STUDENT_MODEL_PATH}" \
        --save "${REF_LOAD}"
else
    echo "  Reusing existing converted checkpoint: ${REF_LOAD}"
fi

echo "[4/9] Launch teacher SGLang server"
if curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; then
    echo "  Reusing existing teacher server on port ${TEACHER_PORT}"
else
    CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES}" \
    python3 -m sglang.launch_server \
        --model-path "${TEACHER_MODEL_PATH}" \
        --host 127.0.0.1 \
        --port "${TEACHER_PORT}" \
        --tp 1 \
        --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}" \
        >"${TEACHER_LOG}" 2>&1 &
    TEACHER_PID=$!

    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; then
            echo "  Teacher server ready on port ${TEACHER_PORT}"
            break
        fi
        if ! kill -0 "${TEACHER_PID}" >/dev/null 2>&1; then
            echo "Teacher server exited unexpectedly. Tail log:" >&2
            tail -n 80 "${TEACHER_LOG}" >&2 || true
            exit 1
        fi
        sleep 2
    done
fi

echo "[5/9] Prepare train arguments"
CKPT_ARGS=(
    --hf-checkpoint "${STUDENT_MODEL_PATH}"
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
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --balance-data
)

RM_ARGS=(
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
    --rm-url "http://127.0.0.1:${TEACHER_PORT}/generate"
)

OPD_ARGS=(
    --advantage-estimator grpo
    --use-opd
    --opd-type sglang
    --opd-alignment byte_chunk
    --opd-kl-coef 1.0
    --opd-teacher-hf-checkpoint "${TEACHER_MODEL_PATH}"
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
)
if [[ "${OPD_DISABLE_SEQUENCE_FALLBACK}" == "true" ]]; then
    OPD_ARGS+=(--opd-disable-sequence-fallback)
fi

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
    --attention-backend "${ATTENTION_BACKEND}"
)

export MASTER_ADDR
export NUM_GPUS
export RAY_PORT
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"

echo "[6/9] Sanity-check teacher logprob API"
curl -sf "http://127.0.0.1:${TEACHER_PORT}/generate" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"Hello\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":0,\"skip_special_tokens\":false},\"return_logprob\":true,\"logprob_start_len\":0}" \
    >/dev/null

echo "[7/9] Submit cross-tokenizer OPD smoke job"
set -x
python3 "${SLIME_DIR}/tools/run_train_with_ray_init.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${RM_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"
set +x

echo "[8/9] Job finished. Logs: ${RUN_LOG}"
echo "[9/9] Teacher logs: ${TEACHER_LOG}"
