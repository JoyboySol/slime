from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_train_launcher_supports_reuse_existing_servers_mode():
    text = _read_script("run-yulan-cross-tokenizer-opd-train.sh")

    assert 'REUSE_EXISTING_SERVERS="${REUSE_EXISTING_SERVERS:-0}"' in text
    assert 'MASTER_PORT="${MASTER_PORT:-12355}"' in text
    assert 'if [[ "${REUSE_EXISTING_SERVERS}" != "1" ]]; then' in text
    assert '--rollout-external' in text
    assert 'ROLLOUT_EXTERNAL_ENGINE_ADDRS' in text
    assert 'SGLANG_ROUTER_IP' in text
    assert 'SGLANG_ROUTER_PORT' in text
    assert '--rollout-max-context-len "${SGLANG_CONTEXT_LENGTH}"' in text
    assert 'ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN' in text
    assert 'must not exceed SGLANG_CONTEXT_LENGTH' in text
    assert 'WANDB_RUN_ID="${WANDB_RUN_ID:-}"' in text
    assert 'WANDB_RESUME_FROM_STEP="${WANDB_RESUME_FROM_STEP:-}"' in text
    assert 'WANDB_START_FRESH_ON_RESUME="${WANDB_START_FRESH_ON_RESUME:-1}"' in text
    assert 'START_ROLLOUT_ID="${START_ROLLOUT_ID:-}"' in text
    assert 'CKPT_STEP="${CKPT_STEP:-}"' in text
    assert 'if [[ "${WANDB_START_FRESH_ON_RESUME}" == "1" && -n "${WANDB_RESUME_FROM_STEP}" ]]; then' in text
    assert 'WANDB_ARGS+=(--wandb-start-fresh)' in text
    assert 'WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")' in text
    assert 'elif [[ -n "${WANDB_RUN_ID}" ]]; then' in text
    assert 'WANDB_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")' in text
    assert 'WANDB_ARGS+=(--wandb-resume-from-step "${WANDB_RESUME_FROM_STEP}")' in text
    assert 'ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID}")' in text
    assert 'CKPT_ARGS+=(--ckpt-step "${CKPT_STEP}")' in text
    assert 'echo "  MASTER_PORT=${MASTER_PORT}"' in text
    assert 'torchrun --nproc-per-node "${NUM_GPUS}" --master-port "${MASTER_PORT}" \\' in text


def test_smoke_launcher_supports_reuse_existing_servers_mode():
    text = _read_script("run-yulan-cross-tokenizer-opd-smoke.sh")

    assert 'REUSE_EXISTING_SERVERS="${REUSE_EXISTING_SERVERS:-0}"' in text
    assert 'if [[ "${REUSE_EXISTING_SERVERS}" != "1" ]]; then' in text
    assert '--rollout-external' in text
    assert 'ROLLOUT_EXTERNAL_ENGINE_ADDRS' in text
    assert 'SGLANG_ROUTER_IP' in text
    assert 'SGLANG_ROUTER_PORT' in text
    assert 'MODEL_CONFIG_SCRIPT="${MODEL_CONFIG_SCRIPT:-}"' in text
    assert 'Override rotary base from student hf config' in text
    assert 'GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by NUM_GPUS=${NUM_GPUS}' in text
    assert 'sync_reuse_external_sglang_config()' in text
    assert 'EXTERNAL_SGLANG_SUMMARY=' in text
    assert '--rollout-max-context-len "${SGLANG_CONTEXT_LENGTH}"' in text
    assert 'ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN' in text
    assert 'must not exceed SGLANG_CONTEXT_LENGTH' in text


def test_yulan_grpo_launcher_uses_idle_gpu_partition_and_qwen3_next_plugin():
    text = _read_script("run-yulan-grpo-train.sh")

    assert 'TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-4,5,6}"' in text
    assert 'ROLLOUT_CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-7}"' in text
    assert "export SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin" in text
    assert 'PROMPT_DATA="${PROMPT_DATA:-/mnt/hdd/lvzhihao/data/MATH-lighteval/data/train.jsonl}"' in text
    assert '--rm-type deepscaler' in text
    assert '--advantage-estimator grpo' in text
    assert '--use-opd' not in text
    assert 'ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN' in text
    assert 'must not exceed SGLANG_CONTEXT_LENGTH' in text
    assert 'WANDB_API_KEY="${WANDB_API_KEY:-}"' in text
    assert 'RAY_TMPDIR="${RAY_TMPDIR:-/mnt/hdd/lvzhihao/ray_tmp}"' in text
    assert 'CLEAN_STALE_PROCESSES="${CLEAN_STALE_PROCESSES:-0}"' in text
    assert "Skipping global process cleanup" in text
    assert 'python3 "${SLIME_DIR}/tools/run_train_with_ray_init.py" \\' in text
