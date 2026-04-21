from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_train_launcher_supports_reuse_existing_servers_mode():
    text = _read_script("run-yulan-cross-tokenizer-opd-train.sh")

    assert 'REUSE_EXISTING_SERVERS="${REUSE_EXISTING_SERVERS:-0}"' in text
    assert 'if [[ "${REUSE_EXISTING_SERVERS}" != "1" ]]; then' in text
    assert '--rollout-external' in text
    assert 'ROLLOUT_EXTERNAL_ENGINE_ADDRS' in text
    assert 'SGLANG_ROUTER_IP' in text
    assert 'SGLANG_ROUTER_PORT' in text


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
