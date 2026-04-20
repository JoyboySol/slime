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


def test_log_rollout_data_adds_byte_chunk_log_prob_metrics(monkeypatch):
    captured = {}

    class FakeTokenizer:
        def __init__(self, token_map):
            self.token_map = token_map
            self.text_to_ids = {text: [token_id] for token_id, text in token_map.items()}
            self.sorted_tokens = sorted(self.text_to_ids.keys(), key=len, reverse=True)

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            token_ids = []
            index = 0
            while index < len(text):
                matched = False
                for token_text in self.sorted_tokens:
                    if text.startswith(token_text, index):
                        token_ids.extend(self.text_to_ids[token_text])
                        index += len(token_text)
                        matched = True
                        break
                if not matched:
                    raise ValueError(f"Cannot encode text at index {index}: {text!r}")
            return token_ids

        def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            del skip_special_tokens, clean_up_tokenization_spaces
            return "".join(self.token_map[token_id] for token_id in token_ids)

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens
            token_ids = self.encode(text)
            result = {"input_ids": token_ids}
            if return_offsets_mapping:
                offsets = []
                index = 0
                for token_id in token_ids:
                    token_text = self.token_map[token_id]
                    offsets.append((index, index + len(token_text)))
                    index += len(token_text)
                result["offset_mapping"] = offsets
            return result

    student_tokenizer = FakeTokenizer({11: "<", 12: "think", 13: ">"})
    teacher_tokenizer = FakeTokenizer({21: "<think>"})

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

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    def fake_gather_log_data(metric_name, args, rollout_id, log_dict):
        del metric_name, args, rollout_id
        captured.update(log_dict)
        return {}

    monkeypatch.setattr("slime.backends.megatron_utils.data.get_cached_tokenizer", fake_get_cached_tokenizer)
    monkeypatch.setattr("slime.backends.megatron_utils.data.gather_log_data", fake_gather_log_data)

    args = Namespace(
        opd_alignment="byte_chunk",
        ci_test=False,
        qkv_format="thd",
        log_multi_turn=False,
        log_passrate=False,
        log_correct_samples=False,
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )
    rollout_data = {
        "tokens": [torch.tensor([11, 12, 13])],
        "response_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 1.0, 1.0])],
        "total_lengths": [3],
        "rollout_log_probs": [torch.tensor([-0.1, -0.2, -0.3])],
        "teacher_log_probs": [torch.tensor([-0.5])],
        "opd_full_texts": ["<think>"],
    }

    log_rollout_data(rollout_id=0, args=args, rollout_data=rollout_data)

    assert abs(captured["rollout_chunk_log_prob"] - (-0.2)) < 1e-6
    assert abs(captured["teacher_chunk_log_prob"] - (-0.5 / 3.0)) < 1e-6
