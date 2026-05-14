#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/ssd/lvzhihao/PostTrain"
SLIME_DIR="${ROOT_DIR}/slime"
YULAN_DIR="${ROOT_DIR}/YuLan-Pretrain"
VENV_DIR="${SLIME_DIR}/.venv"

MODEL_PATH="${MODEL_PATH:-/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill}"
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/hdd/lvzhihao/data/MATH-lighteval/data/train.jsonl}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-4,5,6}"
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"
NUM_GPUS="${NUM_GPUS:-3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12365}"
SGLANG_PORT="${SGLANG_PORT:-30120}"

WORK_DIR="${WORK_DIR:-/mnt/hdd/lvzhihao/slime_grpo_train_workdir}"
WORK_DIR_LINK_DIR="${WORK_DIR_LINK_DIR:-${SLIME_DIR}/workdirs}"
WORK_DIR_LINK_PATH="${WORK_DIR_LINK_PATH:-${WORK_DIR_LINK_DIR}/$(basename "${WORK_DIR}")}"
REF_LOAD="${REF_LOAD:-${WORK_DIR}/yulan_torch_dist}"
ACTOR_CKPT="${ACTOR_CKPT:-${WORK_DIR}/actor_ckpt}"
CKPT_STEP="${CKPT_STEP:-}"
LOG_DIR="${WORK_DIR}/logs"
RUN_LOG="${LOG_DIR}/run.log"
RAY_TMPDIR="${RAY_TMPDIR:-/mnt/hdd/lvzhihao/ray_tmp}"
WANDB_DIR="${WORK_DIR}/wandb"
DEBUG_ROLLOUT_DIR="${WORK_DIR}/debug_rollouts"

NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-33}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-132}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-2048}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1}"
START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"

SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
DEBUG_ROLLOUT_SAVE_INTERVAL="${DEBUG_ROLLOUT_SAVE_INTERVAL:-1}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.25}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-24576}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
LR="${LR:-5e-6}"
CLEAN_STALE_PROCESSES="${CLEAN_STALE_PROCESSES:-0}"

WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-grpo}"
WANDB_GROUP="${WANDB_GROUP:-yulan-grpo}"
WANDB_TEAM="${WANDB_TEAM:-}"
WANDB_HOST="${WANDB_HOST:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME_FROM_STEP="${WANDB_RESUME_FROM_STEP:-}"
WANDB_START_FRESH_ON_RESUME="${WANDB_START_FRESH_ON_RESUME:-1}"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${YULAN_DIR}"
export PYTHONBUFFERED=16
export SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
export TORCH_CUDA_ARCH_LIST=8.0
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-float16}"
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-float32}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR}"

count_visible_devices() {
    local devices="$1"
    if [[ -z "${devices}" ]]; then
        printf '0\n'
        return 0
    fi
    awk -F',' '{print NF}' <<<"${devices}"
}

validate_gpu_partition() {
    python - "$TRAIN_CUDA_VISIBLE_DEVICES" "$ROLLOUT_CUDA_VISIBLE_DEVICES" <<'PY'
import sys

def parse_devices(raw: str):
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]

train = parse_devices(sys.argv[1])
rollout = parse_devices(sys.argv[2])
if len(set(train)) != len(train):
    raise SystemExit(f"TRAIN_CUDA_VISIBLE_DEVICES has duplicates: {train}")
if len(set(rollout)) != len(rollout):
    raise SystemExit(f"ROLLOUT_CUDA_VISIBLE_DEVICES has duplicates: {rollout}")
if set(train) & set(rollout):
    raise SystemExit(
        "TRAIN_CUDA_VISIBLE_DEVICES overlaps with ROLLOUT_CUDA_VISIBLE_DEVICES: "
        f"{sorted(set(train) & set(rollout))}"
    )
if rollout and train and min(rollout) <= max(train):
    raise SystemExit(
        "ROLLOUT_CUDA_VISIBLE_DEVICES must use GPU ids greater than TRAIN_CUDA_VISIBLE_DEVICES in this launcher, "
        f"because Ray slices actor GPUs from the lowest visible GPU ids first. train={train}, rollout={rollout}"
    )
PY
}

resolve_model_config_script() {
    if [[ -n "${MODEL_CONFIG_SCRIPT}" ]]; then
        printf '%s\n' "${MODEL_CONFIG_SCRIPT}"
        return 0
    fi

    local model_name
    model_name="$(basename "${MODEL_PATH}")"
    case "${model_name}" in
        Qwen3-4B-Instruct-2507)
            printf '%s\n' "${SLIME_DIR}/scripts/models/qwen3-4B-Instruct-2507.sh"
            return 0
            ;;
        Qwen3-4B|Qwen3-4B-Instruct)
            printf '%s\n' "${SLIME_DIR}/scripts/models/qwen3-4B.sh"
            return 0
            ;;
    esac

    printf '%s\n' "${SLIME_DIR}/scripts/models/yulan-hybrid-gdn-dense-2.9b.sh"
}

get_model_rope_theta() {
    python - "$MODEL_PATH" <<'PY'
from transformers import AutoConfig
import sys

cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
rope_theta = getattr(cfg, "rope_theta", None)
if rope_theta is None:
    raise SystemExit(1)
print(rope_theta)
PY
}

EXPECTED_GLOBAL_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT / NUM_STEPS_PER_ROLLOUT ))
if (( GLOBAL_BATCH_SIZE != EXPECTED_GLOBAL_BATCH_SIZE )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} does not match ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} * N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT} / NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT}." >&2
    echo "Please keep them consistent. Expected GLOBAL_BATCH_SIZE=${EXPECTED_GLOBAL_BATCH_SIZE}." >&2
    exit 1
fi

if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS} for the current data parallel layout." >&2
    exit 1
fi

if (( ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN > SGLANG_CONTEXT_LENGTH )); then
    echo "Invalid rollout context budget: ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN = $((ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN)) must not exceed SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}." >&2
    exit 1
fi

ROLLOUT_NUM_GPUS="$(count_visible_devices "${ROLLOUT_CUDA_VISIBLE_DEVICES}")"
validate_gpu_partition
if (( ROLLOUT_NUM_GPUS < ROLLOUT_NUM_GPUS_PER_ENGINE )); then
    echo "ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES:-<empty>} provides ${ROLLOUT_NUM_GPUS} GPU(s), which is less than ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}." >&2
    exit 1
fi

RAY_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
if [[ -n "${ROLLOUT_CUDA_VISIBLE_DEVICES}" ]]; then
    RAY_CUDA_VISIBLE_DEVICES="${RAY_CUDA_VISIBLE_DEVICES},${ROLLOUT_CUDA_VISIBLE_DEVICES}"
fi
RAY_NUM_GPUS=$(( NUM_GPUS + ROLLOUT_NUM_GPUS ))

mkdir -p "${WORK_DIR}" "${ACTOR_CKPT}" "${WANDB_DIR}" "${DEBUG_ROLLOUT_DIR}" "${LOG_DIR}" "${RAY_TMPDIR}" "${WORK_DIR_LINK_DIR}"
ln -sfn "${WORK_DIR}" "${WORK_DIR_LINK_PATH}"

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model path not found: ${MODEL_PATH}" >&2
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

cleanup() {
    set +e
    if [[ "${CLEAN_STALE_PROCESSES}" == "1" ]]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "[0/7] Settings"
echo "  MODEL_PATH=${MODEL_PATH}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "  ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES}"
echo "  RAY_CUDA_VISIBLE_DEVICES=${RAY_CUDA_VISIBLE_DEVICES}"
echo "  RAY_NUM_GPUS=${RAY_NUM_GPUS}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  SGLANG_PORT=${SGLANG_PORT}"
echo "  WORK_DIR=${WORK_DIR}"
echo "  WORK_DIR_LINK_PATH=${WORK_DIR_LINK_PATH}"
echo "  RAY_TMPDIR=${RAY_TMPDIR}"
echo "  CLEAN_STALE_PROCESSES=${CLEAN_STALE_PROCESSES}"
echo "  WANDB_MODE=${WANDB_MODE}"
if [[ -n "${WANDB_API_KEY}" ]]; then
    echo "  W&B enabled with project=${WANDB_PROJECT} group=${WANDB_GROUP}"
else
    echo "  W&B disabled because WANDB_API_KEY is empty"
fi

echo "[1/7] Clean stale ray/sglang processes"
if [[ "${CLEAN_STALE_PROCESSES}" == "1" ]]; then
    pkill -f "python3? -m sglang.launch_server" || true
    ray stop --force || true
    pkill -f "ray::" || true
    sleep 2
else
    echo "  Skipping global process cleanup; set CLEAN_STALE_PROCESSES=1 to enable it."
fi

echo "[2/7] Load model config"
MODEL_CONFIG_SCRIPT="$(resolve_model_config_script)"
if [[ ! -f "${MODEL_CONFIG_SCRIPT}" ]]; then
    echo "Model config script not found: ${MODEL_CONFIG_SCRIPT}" >&2
    exit 1
fi
echo "  MODEL_CONFIG_SCRIPT=${MODEL_CONFIG_SCRIPT}"
source "${MODEL_CONFIG_SCRIPT}"

if MODEL_ROTARY_BASE="$(get_model_rope_theta)"; then
    MODEL_ARGS+=(--rotary-base "${MODEL_ROTARY_BASE}")
    echo "  Override rotary base from model hf config: ${MODEL_ROTARY_BASE}"
else
    echo "  Could not read rope_theta from model hf config; keep model preset rotary base" >&2
fi

echo "[3/7] Convert HF checkpoint to Megatron torch_dist if needed"
if [[ ! -f "${REF_LOAD}/latest_checkpointed_iteration.txt" ]]; then
    CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}" \
    torchrun --nproc-per-node "${NUM_GPUS}" --master-port "${MASTER_PORT}" \
        "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint "${MODEL_PATH}" \
        --save "${REF_LOAD}"
else
    echo "  Reusing existing converted checkpoint: ${REF_LOAD}"
fi

echo "[4/7] Prepare train arguments"
CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${REF_LOAD}"
    --load "${ACTOR_CKPT}"
    --save "${ACTOR_CKPT}"
    --save-interval "${SAVE_INTERVAL}"
)
if [[ -n "${CKPT_STEP}" ]]; then
    CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")
fi

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
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --rollout-max-context-len "${SGLANG_CONTEXT_LENGTH}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --rollout-top-p "${ROLLOUT_TOP_P}"
    --balance-data
)
if [[ -n "${START_ROLLOUT_ID}" ]]; then
    ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")
fi

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef 0.0
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
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

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
)

SGLANG_ARGS=(
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-host 127.0.0.1
    --sglang-port "${SGLANG_PORT}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
)

DEBUG_ARGS=(
    --save-debug-rollout-data "${DEBUG_ROLLOUT_DIR}/rollout_{rollout_id}.pt"
    --save-debug-rollout-interval "${DEBUG_ROLLOUT_SAVE_INTERVAL}"
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend "${ATTENTION_BACKEND}"
)

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY}" ]]; then
    WANDB_ARGS+=(
        --use-wandb
        --wandb-mode "${WANDB_MODE}"
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --wandb-key "${WANDB_API_KEY}"
        --wandb-dir "${WANDB_DIR}"
        --disable-wandb-random-suffix
    )
    if [[ -n "${WANDB_TEAM}" ]]; then
        WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
    fi
    if [[ -n "${WANDB_HOST}" ]]; then
        WANDB_ARGS+=(--wandb-host "${WANDB_HOST}")
    fi
    if [[ "${WANDB_START_FRESH_ON_RESUME}" == "1" && -n "${WANDB_RESUME_FROM_STEP}" ]]; then
        WANDB_ARGS+=(--wandb-start-fresh)
        if [[ -n "${WANDB_RUN_ID}" ]]; then
            WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
        fi
    elif [[ -n "${WANDB_RUN_ID}" ]]; then
        WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
    fi
    if [[ -n "${WANDB_RESUME_FROM_STEP}" ]]; then
        WANDB_ARGS+=(--wandb-resume-from-step "${WANDB_RESUME_FROM_STEP}")
    fi
fi

export MASTER_ADDR
export NUM_GPUS
export RAY_NUM_GPUS
export RAY_TMPDIR
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export CUDA_VISIBLE_DEVICES="${RAY_CUDA_VISIBLE_DEVICES}"

echo "[5/7] Launch YuLan GRPO training"
set -x
python3 "${SLIME_DIR}/tools/run_train_with_ray_init.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${DEBUG_ARGS[@]}" \
    "${MISC_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set +x

if (( TRAIN_EXIT_CODE != 0 )); then
    echo "[6/7] Training failed (exit=${TRAIN_EXIT_CODE})" >&2
    echo "  Run log: ${RUN_LOG}" >&2
    echo "  Rollout dumps: ${DEBUG_ROLLOUT_DIR}" >&2
    exit "${TRAIN_EXIT_CODE}"
fi

echo "[6/7] Training finished"
echo "  Run log: ${RUN_LOG}"
echo "  Rollout dumps: ${DEBUG_ROLLOUT_DIR}"
echo "  Checkpoints: ${ACTOR_CKPT}"
echo "  W&B dir: ${WANDB_DIR}"

echo "[7/7] Done"
