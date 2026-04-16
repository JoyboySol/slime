from argparse import Namespace
import sys
import types

import torch


def _install_fake_megatron():
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_mpu = types.SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        is_pipeline_last_stage=lambda: True,
        get_context_parallel_world_size=lambda: 1,
    )
    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    packed_seq_params.PackedSeqParams = object

    megatron.core = megatron_core
    megatron_core.mpu = megatron_mpu
    sys.modules.setdefault("megatron", megatron)
    sys.modules.setdefault("megatron.core", megatron_core)
    sys.modules.setdefault("megatron.core.mpu", megatron_mpu)
    sys.modules.setdefault("megatron.core.packed_seq_params", packed_seq_params)


_install_fake_megatron()

from slime.backends.megatron_utils.data import log_rollout_data


def test_log_rollout_data_skips_opd_text_fields(monkeypatch):
    captured = {}

    monkeypatch.setattr(
        "slime.backends.megatron_utils.data.mpu.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "slime.backends.megatron_utils.data.mpu.is_pipeline_last_stage",
        lambda: True,
    )
    monkeypatch.setattr(
        "slime.backends.megatron_utils.data.mpu.get_context_parallel_world_size",
        lambda: 1,
    )

    def fake_gather_log_data(metric_name, args, rollout_id, log_dict):
        del metric_name, args, rollout_id
        captured.update(log_dict)
        return {}

    monkeypatch.setattr("slime.backends.megatron_utils.data.gather_log_data", fake_gather_log_data)

    args = Namespace(
        opd_alignment="byte_chunk",
        ci_test=False,
        qkv_format="thd",
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 2])],
        "response_lengths": [1],
        "loss_masks": [torch.tensor([1.0])],
        "total_lengths": [2],
        "raw_reward": [1.0],
        "opd_full_texts": ["A BC"],
        "opd_prompt_texts": ["A"],
        "opd_response_texts": [" BC"],
    }

    log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)

    assert captured["raw_reward"] == 1.0
    assert "opd_full_texts" not in captured
    assert "opd_prompt_texts" not in captured
    assert "opd_response_texts" not in captured
