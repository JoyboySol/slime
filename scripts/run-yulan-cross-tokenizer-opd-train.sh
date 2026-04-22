#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/ssd/lvzhihao/PostTrain"
SLIME_DIR="${ROOT_DIR}/slime"
YULAN_DIR="${ROOT_DIR}/YuLan-Pretrain"
VENV_DIR="${SLIME_DIR}/.venv"
WORK_DIR="${WORK_DIR:-/mnt/hdd/lvzhihao/slime_opd_train_workdir}"
WORK_DIR_LINK_DIR="${WORK_DIR_LINK_DIR:-${SLIME_DIR}/workdirs}"
WORK_DIR_LINK_PATH="${WORK_DIR_LINK_PATH:-${WORK_DIR_LINK_DIR}/$(basename "${WORK_DIR}")}"


STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/hdd/Nanbeige4.1-3B}"
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/AIME2024}"
EVAL_DATASET_NAME="${EVAL_DATASET_NAME:-aime}"
MATH500_DATA_PATH="${MATH500_DATA_PATH:-/mnt/hdd/dongzican/math_500}"
MATH500_EVAL_DATASET_NAME="${MATH500_EVAL_DATASET_NAME:-math500}"
MATH500_EVAL_SIZE="${MATH500_EVAL_SIZE:-500}"
MATH500_EVAL_SEED="${MATH500_EVAL_SEED:-42}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-3}"
ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-}"
NUM_GPUS="${NUM_GPUS:-3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
SGLANG_PORT="${SGLANG_PORT:-30110}"
TEACHER_PORT="${TEACHER_PORT:-30221}"
REUSE_EXISTING_SERVERS="${REUSE_EXISTING_SERVERS:-0}"
SGLANG_ROUTER_IP="${SGLANG_ROUTER_IP:-127.0.0.1}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-${SGLANG_PORT}}"
ROLLOUT_EXTERNAL_ENGINE_ADDRS="${ROLLOUT_EXTERNAL_ENGINE_ADDRS:-}"

# WORK_DIR="${WORK_DIR:-${SLIME_DIR}/.tmp/yulan_cross_tokenizer_opd_train}"
REF_LOAD="${REF_LOAD:-${WORK_DIR}/yulan_torch_dist}"
ACTOR_CKPT="${ACTOR_CKPT:-${WORK_DIR}/actor_ckpt}"
LOG_DIR="${WORK_DIR}/logs"
RUN_LOG="${LOG_DIR}/run.log"
TEACHER_LOG="${LOG_DIR}/teacher.log"
WANDB_DIR="${WORK_DIR}/wandb"
DEBUG_ROLLOUT_DIR="${WORK_DIR}/debug_rollouts"
EVAL_DIR="${WORK_DIR}/eval"
EVAL_CONFIG_PATH="${EVAL_DIR}/eval_config.yaml"
MATH500_SUBSET_PATH="${EVAL_DIR}/${MATH500_EVAL_DATASET_NAME}.jsonl"
FAILING_SAMPLE_JSON_PATH="${FAILING_SAMPLE_JSON_PATH:-${DEBUG_ROLLOUT_DIR}/latest_failing_sample.json}"
FAILING_SAMPLE_SUMMARY_PATH="${FAILING_SAMPLE_SUMMARY_PATH:-${DEBUG_ROLLOUT_DIR}/latest_replay_summary.jsonl}"

NUM_ROLLOUT="${NUM_ROLLOUT:-300}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-33}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-132}"
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-2048}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-16384}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1}"

EVAL_INTERVAL="${EVAL_INTERVAL:-5}"
N_SAMPLES_PER_EVAL_PROMPT="${N_SAMPLES_PER_EVAL_PROMPT:-1}"
EVAL_MAX_PROMPT_LEN="${EVAL_MAX_PROMPT_LEN:-2048}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-16384}"

SAVE_INTERVAL="${SAVE_INTERVAL:-5}"
DEBUG_ROLLOUT_SAVE_INTERVAL="${DEBUG_ROLLOUT_SAVE_INTERVAL:-1}"
OPD_DISABLE_SEQUENCE_FALLBACK="${OPD_DISABLE_SEQUENCE_FALLBACK:-false}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-24576}"
RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-1}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.25}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-24576}"
TEACHER_MEM_FRACTION_STATIC="${TEACHER_MEM_FRACTION_STATIC:-0.55}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
LR="${LR:-5e-6}"

WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_PROJECT="${WANDB_PROJECT:-slime-opd}"
WANDB_GROUP="${WANDB_GROUP:-deepmath103k-cross-tokenizer-opd}"
WANDB_TEAM="${WANDB_TEAM:-}"
WANDB_HOST="${WANDB_HOST:-}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${YULAN_DIR}"
export PYTHONBUFFERED=16
export SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
export TORCH_CUDA_ARCH_LIST=8.0
export SGLANG_MAMBA_CONV_DTYPE="${SGLANG_MAMBA_CONV_DTYPE:-float16}"
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-float32}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="127.0.0.1,localhost,${MASTER_ADDR}"

resolve_eval_path() {
    local raw_path="$1"
    if [[ -f "${raw_path}" ]]; then
        printf '%s\n' "${raw_path}"
        return 0
    fi

    if [[ -d "${raw_path}" ]]; then
        local candidate
        for candidate in \
            "${raw_path}/aime-2024.jsonl" \
            "${raw_path}/test.jsonl" \
            "${raw_path}/test.parquet"; do
            if [[ -f "${candidate}" ]]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        done

        candidate="$(find "${raw_path}" -maxdepth 2 -type f \( -name '*.jsonl' -o -name '*.parquet' \) | sort | head -n 1 || true)"
        if [[ -n "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    fi

    return 1
}

count_visible_devices() {
    local devices="$1"
    if [[ -z "${devices}" ]]; then
        printf '0\n'
        return 0
    fi
    awk -F',' '{print NF}' <<<"${devices}"
}

validate_gpu_partition() {
    python - "$TRAIN_CUDA_VISIBLE_DEVICES" "$TEACHER_CUDA_VISIBLE_DEVICES" "$ROLLOUT_CUDA_VISIBLE_DEVICES" <<'PY'
import sys

def parse_devices(raw: str):
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x.strip()]

train = parse_devices(sys.argv[1])
teacher = parse_devices(sys.argv[2])
rollout = parse_devices(sys.argv[3])

if len(set(train)) != len(train):
    raise SystemExit(f"TRAIN_CUDA_VISIBLE_DEVICES has duplicates: {train}")
if len(set(teacher)) != len(teacher):
    raise SystemExit(f"TEACHER_CUDA_VISIBLE_DEVICES has duplicates: {teacher}")
if len(set(rollout)) != len(rollout):
    raise SystemExit(f"ROLLOUT_CUDA_VISIBLE_DEVICES has duplicates: {rollout}")

train_set = set(train)
teacher_set = set(teacher)
rollout_set = set(rollout)

if train_set & teacher_set:
    raise SystemExit(
        "TRAIN_CUDA_VISIBLE_DEVICES overlaps with TEACHER_CUDA_VISIBLE_DEVICES: "
        f"{sorted(train_set & teacher_set)}"
    )
if train_set & rollout_set:
    raise SystemExit(
        "TRAIN_CUDA_VISIBLE_DEVICES overlaps with ROLLOUT_CUDA_VISIBLE_DEVICES: "
        f"{sorted(train_set & rollout_set)}"
    )
if teacher_set & rollout_set:
    raise SystemExit(
        "TEACHER_CUDA_VISIBLE_DEVICES overlaps with ROLLOUT_CUDA_VISIBLE_DEVICES: "
        f"{sorted(teacher_set & rollout_set)}"
    )

if rollout and train and min(rollout) <= max(train):
    raise SystemExit(
        "ROLLOUT_CUDA_VISIBLE_DEVICES must use GPU ids greater than TRAIN_CUDA_VISIBLE_DEVICES in this launcher, "
        f"because Ray slices actor GPUs from the lowest visible GPU ids first. train={train}, rollout={rollout}"
    )
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
    echo "Requested prompt budget ${ROLLOUT_MAX_PROMPT_LEN} and response budget ${ROLLOUT_MAX_RESPONSE_LEN} cannot fit into the configured rollout server context length." >&2
    exit 1
fi

if (( EVAL_MAX_PROMPT_LEN + EVAL_MAX_RESPONSE_LEN > SGLANG_CONTEXT_LENGTH )); then
    echo "Invalid eval context budget: EVAL_MAX_PROMPT_LEN + EVAL_MAX_RESPONSE_LEN = $((EVAL_MAX_PROMPT_LEN + EVAL_MAX_RESPONSE_LEN)) must not exceed SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}." >&2
    echo "Requested eval prompt budget ${EVAL_MAX_PROMPT_LEN} and response budget ${EVAL_MAX_RESPONSE_LEN} cannot fit into the configured rollout server context length." >&2
    exit 1
fi

ROLLOUT_NUM_GPUS="$(count_visible_devices "${ROLLOUT_CUDA_VISIBLE_DEVICES}")"
validate_gpu_partition
ROLLOUT_NUM_GPUS_EFFECTIVE="${ROLLOUT_NUM_GPUS}"
if [[ "${REUSE_EXISTING_SERVERS}" == "1" ]]; then
    if [[ -n "${ROLLOUT_EXTERNAL_ENGINE_ADDRS}" ]]; then
        IFS=',' read -r -a ROLLOUT_EXTERNAL_ADDR_ARRAY <<<"${ROLLOUT_EXTERNAL_ENGINE_ADDRS}"
        ROLLOUT_NUM_GPUS_EFFECTIVE=$(( ${#ROLLOUT_EXTERNAL_ADDR_ARRAY[@]} * ROLLOUT_NUM_GPUS_PER_ENGINE ))
    fi
fi
RAY_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
if [[ -n "${ROLLOUT_CUDA_VISIBLE_DEVICES}" ]]; then
    RAY_CUDA_VISIBLE_DEVICES="${RAY_CUDA_VISIBLE_DEVICES},${ROLLOUT_CUDA_VISIBLE_DEVICES}"
fi
RAY_NUM_GPUS=$(( NUM_GPUS + ROLLOUT_NUM_GPUS ))
if [[ "${REUSE_EXISTING_SERVERS}" == "1" ]]; then
    RAY_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
    RAY_NUM_GPUS="${NUM_GPUS}"
fi

if [[ "${REUSE_EXISTING_SERVERS}" != "1" ]] && (( ROLLOUT_NUM_GPUS < ROLLOUT_NUM_GPUS_PER_ENGINE )); then
    echo "ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES:-<empty>} provides ${ROLLOUT_NUM_GPUS} GPU(s), which is less than ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE}." >&2
    exit 1
fi

mkdir -p "${WORK_DIR}" "${ACTOR_CKPT}" "${WANDB_DIR}" "${DEBUG_ROLLOUT_DIR}" "${EVAL_DIR}" "${LOG_DIR}"
mkdir -p "${WORK_DIR_LINK_DIR}"
ln -sfn "${WORK_DIR}" "${WORK_DIR_LINK_PATH}"

resolve_model_config_script() {
    if [[ -n "${MODEL_CONFIG_SCRIPT}" ]]; then
        printf '%s\n' "${MODEL_CONFIG_SCRIPT}"
        return 0
    fi

    local model_name
    model_name="$(basename "${STUDENT_MODEL_PATH}")"

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

create_jsonl_subset() {
    local source_path="$1"
    local output_path="$2"
    local subset_size="$3"
    local subset_seed="$4"

    python - "$source_path" "$output_path" "$subset_size" "$subset_seed" <<'PY'
import json
import random
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
subset_size = int(sys.argv[3])
subset_seed = int(sys.argv[4])

with source_path.open("r", encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

if not rows:
    raise SystemExit(f"No rows found in {source_path}")

subset_size = min(subset_size, len(rows))
rng = random.Random(subset_seed)
selected_indices = sorted(rng.sample(range(len(rows)), subset_size))

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    for index in selected_indices:
        f.write(json.dumps(rows[index], ensure_ascii=False) + "\n")
PY
}

write_eval_config() {
    local config_path="$1"
    local aime_path="$2"
    local math500_path="$3"

    cat >"${config_path}" <<EOF
eval:
  datasets:
    - name: ${EVAL_DATASET_NAME}
      path: ${aime_path}
      input_key: prompt
      label_key: reward_model
    - name: ${MATH500_EVAL_DATASET_NAME}
      path: ${math500_path}
      input_key: problem
      label_key: answer
EOF
}

get_student_rope_theta() {
    python - "$STUDENT_MODEL_PATH" <<'PY'
from transformers import AutoConfig
import sys

cfg = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
rope_theta = getattr(cfg, "rope_theta", None)
if rope_theta is None:
    raise SystemExit(1)
print(rope_theta)
PY
}

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

if ! EVAL_PROMPT_DATA="$(resolve_eval_path "${EVAL_DATA_PATH}")"; then
    echo "Eval data path not found or no supported file discovered under: ${EVAL_DATA_PATH}" >&2
    exit 1
fi

if ! MATH500_PROMPT_DATA="$(resolve_eval_path "${MATH500_DATA_PATH}")"; then
    echo "Math-500 data path not found or no supported file discovered under: ${MATH500_DATA_PATH}" >&2
    exit 1
fi

create_jsonl_subset "${MATH500_PROMPT_DATA}" "${MATH500_SUBSET_PATH}" "${MATH500_EVAL_SIZE}" "${MATH500_EVAL_SEED}"
write_eval_config "${EVAL_CONFIG_PATH}" "${EVAL_PROMPT_DATA}" "${MATH500_SUBSET_PATH}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi

cleanup() {
    set +e
    if [[ "${REUSE_EXISTING_SERVERS}" != "1" ]]; then
        if [[ -n "${TEACHER_PID:-}" ]]; then
            kill "${TEACHER_PID}" >/dev/null 2>&1 || true
            wait "${TEACHER_PID}" >/dev/null 2>&1 || true
        fi
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

find_latest_train_rollout() {
    find "${DEBUG_ROLLOUT_DIR}" -maxdepth 1 -type f -name 'rollout_*.pt' ! -name 'rollout_eval_*' \
        -printf '%T@ %p\n' | sort -n | tail -n 1 | cut -d' ' -f2-
}

dump_failing_sample_json() {
    local latest_rollout
    latest_rollout="$(find_latest_train_rollout)"
    if [[ -z "${latest_rollout}" || ! -f "${latest_rollout}" ]]; then
        echo "  No train rollout dump found under ${DEBUG_ROLLOUT_DIR}; skip failing-sample export." >&2
        return 0
    fi

    echo "  Replaying latest rollout dump: ${latest_rollout}" >&2
    rm -f "${FAILING_SAMPLE_JSON_PATH}" "${FAILING_SAMPLE_SUMMARY_PATH}"
    if ./.venv/bin/python "${SLIME_DIR}/scripts/replay_debug_rollout_opd.py" \
        --debug-rollout-data "${latest_rollout}" \
        --hf-checkpoint "${STUDENT_MODEL_PATH}" \
        --teacher-hf-checkpoint "${TEACHER_MODEL_PATH}" \
        --dump-failing-sample "${FAILING_SAMPLE_JSON_PATH}" \
        >"${FAILING_SAMPLE_SUMMARY_PATH}" 2>&1; then
        if [[ -f "${FAILING_SAMPLE_JSON_PATH}" ]]; then
            echo "  Exported failing sample JSON to: ${FAILING_SAMPLE_JSON_PATH}" >&2
        else
            echo "  Replay finished but did not detect a failing sample. Summary: ${FAILING_SAMPLE_SUMMARY_PATH}" >&2
        fi
    else
        echo "  Replay helper itself failed. Summary: ${FAILING_SAMPLE_SUMMARY_PATH}" >&2
        return 0
    fi
}

echo "[0/9] Settings"
echo "  STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH}"
echo "  TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  EVAL_PROMPT_DATA=${EVAL_PROMPT_DATA}"
echo "  MATH500_PROMPT_DATA=${MATH500_PROMPT_DATA}"
echo "  MATH500_SUBSET_PATH=${MATH500_SUBSET_PATH}"
echo "  EVAL_CONFIG_PATH=${EVAL_CONFIG_PATH}"
echo "  TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "  TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES}"
echo "  ROLLOUT_CUDA_VISIBLE_DEVICES=${ROLLOUT_CUDA_VISIBLE_DEVICES:-<empty>}"
echo "  RAY_CUDA_VISIBLE_DEVICES=${RAY_CUDA_VISIBLE_DEVICES}"
echo "  RAY_NUM_GPUS=${RAY_NUM_GPUS}"
echo "  ROLLOUT_NUM_GPUS_EFFECTIVE=${ROLLOUT_NUM_GPUS_EFFECTIVE}"
echo "  WORK_DIR=${WORK_DIR}"
echo "  WORK_DIR_LINK_PATH=${WORK_DIR_LINK_PATH}"
echo "  WANDB_MODE=${WANDB_MODE}"
echo "  REUSE_EXISTING_SERVERS=${REUSE_EXISTING_SERVERS}"
echo "  SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP}"
echo "  SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT}"
if [[ -n "${WANDB_API_KEY}" ]]; then
    echo "  W&B enabled with project=${WANDB_PROJECT} group=${WANDB_GROUP}"
else
    echo "  W&B disabled because WANDB_API_KEY is empty"
fi

echo "[1/9] Clean stale ray/sglang processes"
if [[ "${REUSE_EXISTING_SERVERS}" != "1" ]]; then
    pkill -f "python3? -m sglang.launch_server" || true
    ray stop --force || true
    pkill -f "ray::" || true
    sleep 2
else
    echo "  Reuse mode enabled; keeping existing teacher / rollout / ray services alive"
fi

echo "[2/9] Load student model config"
MODEL_CONFIG_SCRIPT="$(resolve_model_config_script)"
if [[ ! -f "${MODEL_CONFIG_SCRIPT}" ]]; then
    echo "Model config script not found: ${MODEL_CONFIG_SCRIPT}" >&2
    exit 1
fi
echo "  MODEL_CONFIG_SCRIPT=${MODEL_CONFIG_SCRIPT}"
source "${MODEL_CONFIG_SCRIPT}"

if STUDENT_ROTARY_BASE="$(get_student_rope_theta)"; then
    MODEL_ARGS+=(--rotary-base "${STUDENT_ROTARY_BASE}")
    echo "  Override rotary base from student hf config: ${STUDENT_ROTARY_BASE}"
else
    echo "  Could not read rope_theta from student hf config; keep model preset rotary base" >&2
fi

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
    --save-interval "${SAVE_INTERVAL}"
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
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --rollout-max-context-len "${SGLANG_CONTEXT_LENGTH}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE}"
    --rollout-top-p "${ROLLOUT_TOP_P}"
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
    --entropy-coef 0.0
    --eps-clip 0.2
)
if [[ "${OPD_DISABLE_SEQUENCE_FALLBACK}" == "true" ]]; then
    OPD_ARGS+=(--opd-disable-sequence-fallback)
fi

EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-config "${EVAL_CONFIG_PATH}"
    --skip-eval-before-train
    --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT}"
    --eval-max-prompt-len "${EVAL_MAX_PROMPT_LEN}"
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
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
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS_EFFECTIVE}"
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-host 127.0.0.1
    --sglang-port "${SGLANG_PORT}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
)
if [[ "${REUSE_EXISTING_SERVERS}" == "1" ]]; then
    if [[ -z "${ROLLOUT_EXTERNAL_ENGINE_ADDRS}" ]]; then
        echo "ROLLOUT_EXTERNAL_ENGINE_ADDRS must be set when REUSE_EXISTING_SERVERS=1." >&2
        exit 1
    fi
    if (( ${#ROLLOUT_EXTERNAL_ADDR_ARRAY[@]} == 0 )); then
        echo "ROLLOUT_EXTERNAL_ENGINE_ADDRS did not yield any usable addresses." >&2
        exit 1
    fi
    SGLANG_ARGS+=(
        --rollout-external
        --sglang-router-ip "${SGLANG_ROUTER_IP}"
        --sglang-router-port "${SGLANG_ROUTER_PORT}"
        --rollout-external-engine-addrs "${ROLLOUT_EXTERNAL_ADDR_ARRAY[@]}"
    )
fi

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
    if [[ -n "${WANDB_RUN_ID}" ]]; then
        WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
    fi
fi

export MASTER_ADDR
export NUM_GPUS
export RAY_NUM_GPUS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE="${HAS_NVLINK}"
export CUDA_VISIBLE_DEVICES="${RAY_CUDA_VISIBLE_DEVICES}"

echo "[6/9] Sanity-check teacher logprob API"
curl -sf "http://127.0.0.1:${TEACHER_PORT}/generate" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"Hello\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":0,\"skip_special_tokens\":false},\"return_logprob\":true,\"logprob_start_len\":0}" \
    >/dev/null

echo "[7/9] Launch cross-tokenizer OPD training"
set -x
set +e
python3 "${SLIME_DIR}/tools/run_train_with_ray_init.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${OPD_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${DEBUG_ARGS[@]}" \
    "${MISC_ARGS[@]}" \
    "${RM_ARGS[@]}" 2>&1 | tee "${RUN_LOG}"
TRAIN_EXIT_CODE=${PIPESTATUS[0]}
set -e
set +x

if (( TRAIN_EXIT_CODE != 0 )); then
    echo "[8/9] Training failed (exit=${TRAIN_EXIT_CODE})" >&2
    dump_failing_sample_json
    echo "  Run log: ${RUN_LOG}" >&2
    echo "  Teacher log: ${TEACHER_LOG}" >&2
    echo "  Rollout dumps: ${DEBUG_ROLLOUT_DIR}" >&2
    [[ -f "${FAILING_SAMPLE_JSON_PATH}" ]] && echo "  Failing sample JSON: ${FAILING_SAMPLE_JSON_PATH}" >&2
    [[ -f "${FAILING_SAMPLE_SUMMARY_PATH}" ]] && echo "  Replay summary: ${FAILING_SAMPLE_SUMMARY_PATH}" >&2
    exit "${TRAIN_EXIT_CODE}"
fi

echo "[8/9] Training finished"
echo "  Run log: ${RUN_LOG}"
echo "  Teacher log: ${TEACHER_LOG}"
echo "  Rollout dumps: ${DEBUG_ROLLOUT_DIR}"
echo "  Checkpoints: ${ACTOR_CKPT}"
echo "  W&B dir: ${WANDB_DIR}"

echo "[9/9] Done"
