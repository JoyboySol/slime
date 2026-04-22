#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/mnt/ssd/lvzhihao/PostTrain"
SLIME_DIR="${ROOT_DIR}/slime"
YULAN_DIR="${ROOT_DIR}/YuLan-Pretrain"
VENV_DIR="${SLIME_DIR}/.venv"

STUDENT_MODEL_PATH="${STUDENT_MODEL_PATH:-/mnt/hdd/lvzhihao/hf_models/Dist-mathcode10b-s1randg-sch1-CPT-200b-stage3-r640k-GDN2.9b-A7-12_20_21_23_46_48_49-sl32768bs128lr2e5-2e5/merged_10ckpts_iter_61984-hf_to_iter_71525-hf_mean}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-/mnt/hdd/Nanbeige4.1-3B}"
MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-}"
PROMPT_DATA="${PROMPT_DATA:-/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-4,5,6}"
TEACHER_CUDA_VISIBLE_DEVICES="${TEACHER_CUDA_VISIBLE_DEVICES:-7}"
NUM_GPUS="${NUM_GPUS:-3}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
RAY_PORT="${RAY_PORT:-8265}"
SGLANG_PORT="${SGLANG_PORT:-30110}"
TEACHER_PORT="${TEACHER_PORT:-30221}"
REUSE_EXISTING_SERVERS="${REUSE_EXISTING_SERVERS:-0}"
SGLANG_ROUTER_IP="${SGLANG_ROUTER_IP:-127.0.0.1}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-${SGLANG_PORT}}"
ROLLOUT_EXTERNAL_ENGINE_ADDRS="${ROLLOUT_EXTERNAL_ENGINE_ADDRS:-}"

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
ROLLOUT_MAX_PROMPT_LEN="${ROLLOUT_MAX_PROMPT_LEN:-}"
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
ROLLOUT_NUM_GPUS_EFFECTIVE="${ROLLOUT_NUM_GPUS_EFFECTIVE:-}"

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

if [[ -z "${ROLLOUT_MAX_PROMPT_LEN}" ]]; then
    ROLLOUT_MAX_PROMPT_LEN=$(( SGLANG_CONTEXT_LENGTH - ROLLOUT_MAX_RESPONSE_LEN ))
fi

if (( ROLLOUT_MAX_PROMPT_LEN < 1 )); then
    echo "Invalid rollout context budget: ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN} must be at least 1." >&2
    exit 1
fi

if (( ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN > SGLANG_CONTEXT_LENGTH )); then
    echo "Invalid rollout context budget: ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN = $((ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN)) must not exceed SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}." >&2
    echo "Requested prompt budget ${ROLLOUT_MAX_PROMPT_LEN} and response budget ${ROLLOUT_MAX_RESPONSE_LEN} cannot fit into the configured rollout server context length." >&2
    exit 1
fi

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

sync_reuse_external_sglang_config() {
    python - "$SGLANG_CONTEXT_LENGTH" "$ROLLOUT_NUM_GPUS_PER_ENGINE" "${ROLLOUT_EXTERNAL_ADDR_ARRAY[@]}" <<'PY'
import json
import sys
import requests

declared_context_length = int(sys.argv[1])
declared_gpus_per_engine = int(sys.argv[2])
addresses = sys.argv[3:]
if not addresses:
    raise SystemExit("No external rollout engine addresses provided.")

records = []
for addr in addresses:
    response = requests.get(f"http://{addr}/get_server_info", timeout=10)
    response.raise_for_status()
    info = response.json()
    records.append(
        {
            "addr": addr,
            "context_length": info.get("context_length"),
            "tp_size": info.get("tp_size"),
            "pp_size": info.get("pp_size"),
            "dp_size": info.get("dp_size"),
            "ep_size": info.get("ep_size"),
            "base_gpu_id": info.get("base_gpu_id"),
            "enable_memory_saver": info.get("enable_memory_saver"),
        }
    )

first = records[0]
for record in records[1:]:
    for key in ("context_length", "tp_size", "pp_size", "dp_size", "ep_size", "enable_memory_saver"):
        if record.get(key) != first.get(key):
            raise SystemExit(
                "Inconsistent external rollout engines: "
                f"{json.dumps(records, ensure_ascii=False, sort_keys=True)}"
            )

actual_gpus_per_engine = int(first["tp_size"]) * int(first["pp_size"])
if actual_gpus_per_engine != declared_gpus_per_engine:
    raise SystemExit(
        f"ROLLOUT_NUM_GPUS_PER_ENGINE={declared_gpus_per_engine} does not match "
        f"external engine tp_size*pp_size={actual_gpus_per_engine}. records={json.dumps(records, ensure_ascii=False)}"
    )

print(f"SGLANG_CONTEXT_LENGTH={int(first['context_length'])}")
print(
    "EXTERNAL_SGLANG_SUMMARY="
    + json.dumps(records, ensure_ascii=False, sort_keys=True)
)
if int(first["context_length"]) != declared_context_length:
    print(
        f"EXTERNAL_SGLANG_CONTEXT_LENGTH_MISMATCH declared={declared_context_length} actual={int(first['context_length'])}"
    )
PY
}

if [[ "${REUSE_EXISTING_SERVERS}" == "1" ]]; then
    if [[ -z "${ROLLOUT_EXTERNAL_ENGINE_ADDRS}" ]]; then
        echo "ROLLOUT_EXTERNAL_ENGINE_ADDRS must be set when REUSE_EXISTING_SERVERS=1." >&2
        exit 1
    fi
    IFS=',' read -r -a ROLLOUT_EXTERNAL_ADDR_ARRAY <<<"${ROLLOUT_EXTERNAL_ENGINE_ADDRS}"
    if (( ${#ROLLOUT_EXTERNAL_ADDR_ARRAY[@]} == 0 )); then
        echo "ROLLOUT_EXTERNAL_ENGINE_ADDRS did not yield any usable addresses." >&2
        exit 1
    fi
    if [[ -z "${ROLLOUT_NUM_GPUS_EFFECTIVE}" ]]; then
        ROLLOUT_NUM_GPUS_EFFECTIVE=$(( ${#ROLLOUT_EXTERNAL_ADDR_ARRAY[@]} * ROLLOUT_NUM_GPUS_PER_ENGINE ))
    fi
    while IFS='=' read -r key value; do
        case "${key}" in
            SGLANG_CONTEXT_LENGTH)
                SGLANG_CONTEXT_LENGTH="${value}"
                ;;
            EXTERNAL_SGLANG_SUMMARY)
                EXTERNAL_SGLANG_SUMMARY="${value}"
                ;;
            EXTERNAL_SGLANG_CONTEXT_LENGTH_MISMATCH)
                EXTERNAL_SGLANG_CONTEXT_LENGTH_MISMATCH="${value}"
                ;;
        esac
    done < <(sync_reuse_external_sglang_config)
else
    if [[ -z "${ROLLOUT_NUM_GPUS_EFFECTIVE}" ]]; then
        ROLLOUT_NUM_GPUS_EFFECTIVE="${NUM_GPUS}"
    fi
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
    if [[ "${REUSE_EXISTING_SERVERS}" != "1" ]]; then
        if [[ -n "${TEACHER_PID:-}" ]]; then
            kill "${TEACHER_PID}" >/dev/null 2>&1 || true
            wait "${TEACHER_PID}" >/dev/null 2>&1 || true
        fi
        ray stop --force >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT

echo "[0/9] Settings"
echo "  STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH}"
echo "  TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH}"
echo "  PROMPT_DATA=${PROMPT_DATA}"
echo "  TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}"
echo "  TEACHER_CUDA_VISIBLE_DEVICES=${TEACHER_CUDA_VISIBLE_DEVICES}"
echo "  WORK_DIR=${WORK_DIR}"
echo "  REUSE_EXISTING_SERVERS=${REUSE_EXISTING_SERVERS}"
echo "  SGLANG_ROUTER_IP=${SGLANG_ROUTER_IP}"
echo "  SGLANG_ROUTER_PORT=${SGLANG_ROUTER_PORT}"
echo "  ROLLOUT_NUM_GPUS_EFFECTIVE=${ROLLOUT_NUM_GPUS_EFFECTIVE}"
echo "  SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH}"
if [[ -n "${EXTERNAL_SGLANG_SUMMARY:-}" ]]; then
    echo "  EXTERNAL_SGLANG_SUMMARY=${EXTERNAL_SGLANG_SUMMARY}"
fi
if [[ -n "${EXTERNAL_SGLANG_CONTEXT_LENGTH_MISMATCH:-}" ]]; then
    echo "  ${EXTERNAL_SGLANG_CONTEXT_LENGTH_MISMATCH}"
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
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --rollout-max-context-len "${SGLANG_CONTEXT_LENGTH}"
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
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS_EFFECTIVE}"
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-host 127.0.0.1
    --sglang-port "${SGLANG_PORT}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
)
if [[ "${REUSE_EXISTING_SERVERS}" == "1" ]]; then
    SGLANG_ARGS+=(
        --rollout-external
        --sglang-router-ip "${SGLANG_ROUTER_IP}"
        --sglang-router-port "${SGLANG_ROUTER_PORT}"
        --rollout-external-engine-addrs "${ROLLOUT_EXTERNAL_ADDR_ARRAY[@]}"
    )
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend "${ATTENTION_BACKEND}"
)

LAUNCH_MODE_ARGS=(--colocate)
if [[ "${REUSE_EXISTING_SERVERS}" == "1" ]]; then
    LAUNCH_MODE_ARGS=()
fi

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
    "${LAUNCH_MODE_ARGS[@]}" \
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
