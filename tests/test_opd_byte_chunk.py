from argparse import Namespace
import sys
import types

import torch

from slime.rollout.on_policy_distillation import post_process_rewards
from slime.utils.types import Sample
from slime.utils import opd_utils


class FakeTokenizer:
    def __init__(self, token_map):
        self.token_map = token_map
        self.text_to_ids = {}
        for token_id, token_text in token_map.items():
            self.text_to_ids.setdefault(token_text, []).append(token_id)
        self.sorted_tokens = sorted(self.text_to_ids.keys(), key=len, reverse=True)

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        token_ids = []
        index = 0
        while index < len(text):
            matched = False
            for token_text in self.sorted_tokens:
                if text.startswith(token_text, index):
                    token_ids.append(self.text_to_ids[token_text][0])
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


class BoundaryAwareTokenizer(FakeTokenizer):
    def __init__(self, token_map, encode_map, decode_map=None, offsets_map=None):
        super().__init__(token_map)
        self.encode_map = encode_map
        self.decode_map = decode_map or {}
        self.offsets_map = offsets_map or {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text in self.encode_map:
            return list(self.encode_map[text])
        return super().encode(text)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        key = tuple(token_ids)
        if key in self.decode_map:
            return self.decode_map[key]
        return super().decode(token_ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        token_ids = self.encode(text)
        result = {"input_ids": token_ids}
        if return_offsets_mapping:
            if text in self.offsets_map:
                result["offset_mapping"] = list(self.offsets_map[text])
            else:
                offsets = []
                index = 0
                for token_id in token_ids:
                    token_text = self.token_map[token_id]
                    offsets.append((index, index + len(token_text)))
                    index += len(token_text)
                result["offset_mapping"] = offsets
        return result


def test_compute_byte_chunk_reverse_kl_many_teacher_tokens_to_one_student_token():
    student_tokenizer = FakeTokenizer({11: "<think>"})
    teacher_tokenizer = FakeTokenizer({21: "<", 22: "think", 23: ">"})

    reverse_kl = opd_utils.compute_byte_chunk_reverse_kl(
        full_text="<think>",
        prompt_token_count=0,
        student_token_ids=[11],
        response_token_count=1,
        student_log_probs=torch.tensor([-0.2]),
        teacher_log_probs=torch.tensor([-0.1, -0.3, -0.4]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    assert torch.allclose(reverse_kl, torch.tensor([0.6]))


def test_compute_byte_chunk_reverse_kl_averages_chunk_penalty_across_student_tokens():
    student_tokenizer = FakeTokenizer({11: "<", 12: "think", 13: ">"})
    teacher_tokenizer = FakeTokenizer({21: "<think>"})

    reverse_kl = opd_utils.compute_byte_chunk_reverse_kl(
        full_text="<think>",
        prompt_token_count=0,
        student_token_ids=[11, 12, 13],
        response_token_count=3,
        student_log_probs=torch.tensor([-0.1, -0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.5]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    assert torch.allclose(reverse_kl, torch.tensor([-0.033333335, -0.033333335, -0.033333335]))
    assert torch.allclose(reverse_kl.sum(), torch.tensor([-0.1]).sum())


def test_post_process_rewards_uses_teacher_tokenizer_length_for_byte_chunk(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "<think>"})
    teacher_tokenizer = FakeTokenizer({2: "<", 3: "think", 4: ">"})

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    sample = Sample(
        tokens=[1],
        response="<think>",
        response_length=1,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [0.0, 999],
                    [-0.2, 2],
                    [-0.3, 3],
                    [-0.4, 4],
                ]
            }
        },
        opd_full_text="<think>",
    )
    args = Namespace(
        reward_key=None,
        opd_alignment="byte_chunk",
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )

    scalar_rewards, raw_rewards = post_process_rewards(args, [sample])

    assert scalar_rewards == [0.0]
    assert raw_rewards == [0.0]
    assert torch.allclose(sample.teacher_log_probs, torch.tensor([-0.2, -0.3, -0.4]))


def test_post_process_rewards_uses_decoded_response_text_instead_of_sample_response(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "P", 2: "<think>"})
    teacher_tokenizer = FakeTokenizer({3: "P", 4: "<", 5: "think", 6: ">"})

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    sample = Sample(
        tokens=[1, 2],
        response="WRONG_RESPONSE_TEXT",
        response_length=1,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [0.0, 999],
                    [-1.0, 3],
                    [-0.2, 4],
                    [-0.3, 5],
                    [-0.4, 6],
                ]
            }
        },
        opd_full_text="P<think>",
        opd_prompt_text="P",
        opd_response_text="<think>",
    )
    args = Namespace(
        reward_key=None,
        opd_alignment="byte_chunk",
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )

    scalar_rewards, raw_rewards = post_process_rewards(args, [sample])

    assert scalar_rewards == [0.0]
    assert raw_rewards == [0.0]
    assert torch.allclose(sample.teacher_log_probs, torch.tensor([-0.2, -0.3, -0.4]))


def test_post_process_rewards_uses_full_text_boundary_for_teacher_slice(monkeypatch):
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "A", 3: " BC"},
        encode_map={"A BC": [1, 3]},
        decode_map={(1,): "A", (3,): " BC", (1, 3): "A BC"},
        offsets_map={"A BC": [(0, 1), (1, 4)]},
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={10: "A", 11: "BC", 12: "A "},
        encode_map={"A": [10], " BC": [11], "A BC": [12, 11]},
        decode_map={(10,): "A", (11,): "BC", (12,): "A ", (12, 11): "A BC"},
        offsets_map={"A BC": [(0, 2), (2, 4)]},
    )

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    sample = Sample(
        tokens=[1, 3],
        response=" BC",
        response_length=1,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [0.0, 999],
                    [-0.2, 12],
                    [-0.3, 11],
                ]
            }
        },
        opd_full_text="A BC",
    )
    args = Namespace(
        reward_key=None,
        opd_alignment="byte_chunk",
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )

    scalar_rewards, raw_rewards = post_process_rewards(args, [sample])

    assert scalar_rewards == [0.0]
    assert raw_rewards == [0.0]
    assert torch.allclose(sample.teacher_log_probs, torch.tensor([-0.2, -0.3]))


def test_apply_opd_byte_chunk_to_advantages(monkeypatch):
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_mpu = types.SimpleNamespace(
        is_pipeline_last_stage=lambda: True,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
        get_tensor_model_parallel_group=lambda: None,
    )
    megatron.core = megatron_core
    megatron_core.mpu = megatron_mpu
    sys.modules.setdefault("megatron", megatron)
    sys.modules.setdefault("megatron.core", megatron_core)
    sys.modules.setdefault("megatron.core.mpu", megatron_mpu)

    from slime.backends.megatron_utils.loss import apply_opd_kl_to_advantages

    student_tokenizer = FakeTokenizer({99: "Z", 10: "A", 11: "B", 12: "C"})
    teacher_tokenizer = FakeTokenizer({98: "Z", 20: "AB", 21: "C"})

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.backends.megatron_utils.loss.get_cached_tokenizer", fake_get_cached_tokenizer)

    args = Namespace(
        opd_type="sglang",
        opd_kl_coef=2.0,
        opd_alignment="byte_chunk",
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )
    rollout_data = {
        "tokens": [torch.tensor([99, 10, 11, 12])],
        "response_lengths": [3],
        "teacher_log_probs": [torch.tensor([-0.4, -0.5])],
        "opd_full_texts": ["ZABC"],
    }
    advantages = [torch.tensor([1.0, 1.0, 1.0])]
    student_log_probs = [torch.tensor([-0.1, -0.2, -0.3])]

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)

    assert torch.allclose(advantages[0], torch.tensor([0.9, 0.9, 0.6]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([0.05, 0.05, 0.2]))
