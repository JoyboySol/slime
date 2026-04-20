from argparse import Namespace
import importlib.util
import sys
import types

import pytest
import torch

from slime.rollout.on_policy_distillation import (
    _record_student_opd_alignment,
    compute_teacher_log_probs_for_sample,
    post_process_rewards,
)
from slime.utils.types import Sample
from slime.utils import opd_utils
from slime.utils.opd_utils import build_contextual_suffix_token_bytes


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

    def convert_ids_to_tokens(self, token_ids):
        return [self.token_map[token_id] for token_id in token_ids]


def test_build_contextual_suffix_token_bytes_uses_full_text_offsets_instead_of_prefix_decode():
    tokenizer = BoundaryAwareTokenizer(
        token_map={100: "<|im_start|>", 101: " user", 102: "\n", 103: "O", 104: "K"},
        encode_map={
            "<|im_start|> user\n": [100, 101, 102],
            "<|im_start|> user\nOK": [100, 101, 102, 103, 104],
        },
        decode_map={
            (100,): "<|im_start|>",
            (100, 101): "<|im_start|> user",
            (100, 101, 102): "<|im_start|> user\n",
            (100, 101, 102, 103): "<|im_start|>\nO",
            (100, 101, 102, 103, 104): "<|im_start|> user\nOK",
            (103,): "O",
            (103, 104): "OK",
        },
        offsets_map={"<|im_start|> user\nOK": [(0, 12), (12, 17), (17, 18), (18, 19), (19, 20)]},
    )

    token_bytes, token_spans = build_contextual_suffix_token_bytes(
        tokenizer,
        [100, 101, 102, 103, 104],
        suffix_start_index=3,
        prefix_text="<|im_start|> user\n",
        full_text="<|im_start|> user\nOK",
    )

    assert token_bytes == [b"O", b"K"]
    assert token_spans == [(0, 1), (1, 2)]


def test_build_token_byte_spans_falls_back_when_offsets_repeat_same_unicode_span():
    tokenizer = BoundaryAwareTokenizer(
        token_map={10: "A", 11: "B", 21: "\ufffd", 22: "\ufffd", 23: "\ufffd", 24: "\ufffd"},
        encode_map={"A🍪B": [10, 21, 22, 23, 24, 11]},
        decode_map={
            (10,): "A",
            (10, 21): "A",
            (10, 21, 22): "A",
            (10, 21, 22, 23): "A",
            (10, 21, 22, 23, 24): "A🍪",
            (10, 21, 22, 23, 24, 11): "A🍪B",
            (21,): "",
            (21, 22): "",
            (21, 22, 23): "",
            (21, 22, 23, 24): "🍪",
        },
        offsets_map={"A🍪B": [(0, 1), (1, 2), (1, 2), (1, 2), (1, 2), (2, 3)]},
    )

    token_bytes, token_spans = opd_utils.build_token_byte_spans(tokenizer, "A🍪B", [10, 21, 22, 23, 24, 11])

    assert b"".join(token_bytes) == "A🍪B".encode("utf-8")
    assert token_bytes == [b"A", b"", b"", b"", "🍪".encode("utf-8"), b"B"]
    assert token_spans == [(0, 1), (1, 1), (1, 1), (1, 1), (1, 5), (5, 6)]


def test_build_token_byte_spans_falls_back_to_token_strings_when_prefix_decode_is_not_consistent():
    tokenizer = BoundaryAwareTokenizer(
        token_map={10: "ä½ł", 11: "å¥½"},
        encode_map={"你好": [10, 11]},
        decode_map={
            (10,): "你",
            (10, 11): "好",
        },
        offsets_map={"你好": [(0, 1), (0, 1)]},
    )

    token_bytes, token_spans = opd_utils.build_token_byte_spans(tokenizer, "你好", [10, 11])

    assert b"".join(token_bytes) == "你好".encode("utf-8")
    assert token_bytes == ["你".encode("utf-8"), "好".encode("utf-8")]
    assert token_spans == [(0, 3), (3, 6)]


def test_validate_recorded_student_response_alignment_accepts_valid_payload():
    token_bytes, token_spans = opd_utils.validate_recorded_student_response_alignment(
        response_text="AB",
        response_token_count=2,
        response_bytes="AB".encode("utf-8"),
        token_byte_spans=[(0, 1), (1, 2)],
    )

    assert token_bytes == [b"A", b"B"]
    assert token_spans == [(0, 1), (1, 2)]


def test_validate_recorded_student_response_alignment_rejects_wrong_span_count():
    with pytest.raises(ValueError, match="span count"):
        opd_utils.validate_recorded_student_response_alignment(
            response_text="AB",
            response_token_count=2,
            response_bytes="AB".encode("utf-8"),
            token_byte_spans=[(0, 2)],
        )


def test_validate_recorded_student_response_alignment_rejects_non_monotonic_spans():
    with pytest.raises(ValueError, match="monotonic"):
        opd_utils.validate_recorded_student_response_alignment(
            response_text="AB",
            response_token_count=2,
            response_bytes="AB".encode("utf-8"),
            token_byte_spans=[(0, 1), (0, 2)],
        )


def test_validate_recorded_student_response_alignment_rejects_byte_mismatch():
    with pytest.raises(ValueError, match="canonical response text"):
        opd_utils.validate_recorded_student_response_alignment(
            response_text="AB",
            response_token_count=2,
            response_bytes="AX".encode("utf-8"),
            token_byte_spans=[(0, 1), (1, 2)],
        )


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


def test_compute_byte_chunk_aligned_log_probs_returns_chunk_aligned_means():
    student_tokenizer = FakeTokenizer({11: "<", 12: "think", 13: ">"})
    teacher_tokenizer = FakeTokenizer({21: "<think>"})

    student_chunk_log_probs, teacher_chunk_log_probs = opd_utils.compute_byte_chunk_aligned_log_probs(
        full_text="<think>",
        prompt_token_count=0,
        student_token_ids=[11, 12, 13],
        response_token_count=3,
        student_log_probs=torch.tensor([-0.1, -0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.5]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    assert torch.allclose(student_chunk_log_probs, torch.tensor([-0.2, -0.2, -0.2]))
    assert torch.allclose(teacher_chunk_log_probs, torch.tensor([-0.16666667, -0.16666667, -0.16666667]))
    assert torch.allclose(student_chunk_log_probs - teacher_chunk_log_probs, torch.tensor([-0.03333333] * 3))


def test_compute_byte_chunk_aligned_log_probs_prefers_explicit_prompt_text_boundary():
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "A", 2: "B"},
        encode_map={"A B": [1, 2]},
        decode_map={
            (1,): "A",
            (2,): "B",
            (1, 2): "A B",
        },
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={10: "A ", 11: "B"},
        encode_map={"A B": [10, 11]},
        decode_map={
            (10,): "A ",
            (11,): "B",
            (10, 11): "A B",
        },
        offsets_map={"A B": [(0, 2), (2, 3)]},
    )

    with pytest.raises(ValueError, match="selected=2, log_probs=1"):
        opd_utils.compute_byte_chunk_aligned_log_probs(
            full_text="A B",
            prompt_token_count=1,
            student_token_ids=[1, 2],
            response_token_count=1,
            student_log_probs=torch.tensor([-0.3]),
            teacher_log_probs=torch.tensor([-0.4]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            allow_sequence_fallback=False,
        )

    student_chunk_log_probs, teacher_chunk_log_probs = opd_utils.compute_byte_chunk_aligned_log_probs(
        full_text="A B",
        prompt_text="A ",
        prompt_token_count=1,
        student_token_ids=[1, 2],
        response_token_count=1,
        student_log_probs=torch.tensor([-0.3]),
        teacher_log_probs=torch.tensor([-0.4]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        allow_sequence_fallback=False,
    )

    assert torch.allclose(student_chunk_log_probs, torch.tensor([-0.3]))
    assert torch.allclose(teacher_chunk_log_probs, torch.tensor([-0.4]))


def test_replay_summary_marks_teacher_stage_failures(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "replay_debug_rollout_opd",
        "/mnt/ssd/lvzhihao/PostTrain/slime/scripts/replay_debug_rollout_opd.py",
    )
    replay_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(replay_module)

    sample = Sample(
        prompt="P",
        tokens=[1, 2],
        response="R",
        response_length=1,
        rollout_log_probs=[-0.3],
        reward={"meta_info": {"input_token_logprobs": [[0.0, 999], [-0.4, 123]]}},
    )
    args = Namespace(
        hf_checkpoint="student",
        teacher_hf_checkpoint="teacher",
        preview_chars=32,
    )

    def fail_teacher_log_probs(_args, _sample):
        raise ValueError("teacher stage mismatch")

    monkeypatch.setattr(replay_module, "get_cached_tokenizer", lambda _path: object())
    monkeypatch.setattr(replay_module, "compute_teacher_log_probs_for_sample", fail_teacher_log_probs)

    summary = replay_module._summarize_sample(sample, args)

    assert summary["byte_chunk_alignment"] == "failed:ValueError"
    assert summary["error"] == "teacher stage mismatch"
    assert summary["prompt_preview"] == "P"
    assert summary["response_preview"] == "R"


def test_replay_normalize_args_sets_opd_teacher_checkpoint():
    spec = importlib.util.spec_from_file_location(
        "replay_debug_rollout_opd",
        "/mnt/ssd/lvzhihao/PostTrain/slime/scripts/replay_debug_rollout_opd.py",
    )
    replay_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(replay_module)

    args = Namespace(
        hf_checkpoint="student",
        teacher_hf_checkpoint="teacher",
    )

    normalized = replay_module._normalize_args(args)

    assert normalized.opd_teacher_hf_checkpoint == "teacher"


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


def test_post_process_rewards_prefers_explicit_opd_text_fields(monkeypatch):
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


def test_post_process_rewards_uses_prompt_and_response_when_opd_texts_missing(monkeypatch):
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
        prompt="P",
        tokens=[1, 2],
        response="<think>",
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


def test_post_process_rewards_handles_non_roundtrippable_prompt_tokens(monkeypatch):
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={100: "<|im_start|>", 101: " user", 102: "\n", 103: "O", 104: "K", 105: " user\n"},
        encode_map={
            "<|im_start|> user\n": [100, 105],
            "OK": [103, 104],
            "<|im_start|> user\nOK": [100, 105, 103, 104],
        },
        decode_map={
            (100,): "<|im_start|>",
            (100, 101): "<|im_start|> user",
            (100, 101, 102): "<|im_start|> user\n",
            # Crossing the prompt/response boundary is not prefix-consistent
            # for the full token sequence, even though response-only decoding is.
            (100, 101, 102, 103): "<|im_start|>\nO",
            (100, 101, 102, 103, 104): "<|im_start|>\nOK",
            (103,): "O",
            (103, 104): "OK",
        },
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={201: "<|im_start|> user\n", 202: "OK"},
        encode_map={
            "<|im_start|> user\n": [201],
            "OK": [202],
            "<|im_start|> user\nOK": [201, 202],
        },
    )

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    sample = Sample(
        tokens=[100, 101, 102, 103, 104],
        response="OK",
        response_length=2,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [0.0, 999],
                    [-0.1, 201],
                    [-0.2, 202],
                ]
            }
        },
        opd_full_text="<|im_start|> user\nOK",
        opd_prompt_text="<|im_start|> user\n",
        opd_response_text="OK",
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
    assert torch.allclose(sample.teacher_log_probs, torch.tensor([-0.2]))


def test_post_process_rewards_uses_teacher_reward_token_ids_when_local_teacher_encode_mismatches(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "P", 2: "AB"})
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={9: "PAB", 10: "P", 11: "AB"},
        encode_map={"PAB": [9]},
        decode_map={
            (9,): "PAB",
            (10,): "P",
            (11,): "AB",
            (10, 11): "PAB",
        },
        offsets_map={"PAB": [(0, 1), (1, 3)]},
    )

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    sample = Sample(
        tokens=[1, 2],
        response="AB",
        response_length=1,
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [0.0, 999],
                    [-0.1, 10],
                    [-0.2, 11],
                ]
            }
        },
        opd_full_text="PAB",
        opd_prompt_text="P",
        opd_response_text="AB",
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
    assert torch.allclose(sample.teacher_log_probs, torch.tensor([-0.2]))


def test_compute_teacher_log_probs_for_sample_matches_post_process_rewards(monkeypatch):
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={100: "<|im_start|>", 101: " user", 102: "\n", 103: "O", 104: "K", 105: " user\n"},
        encode_map={
            "<|im_start|> user\n": [100, 105],
            "OK": [103, 104],
            "<|im_start|> user\nOK": [100, 105, 103, 104],
        },
        decode_map={
            (100,): "<|im_start|>",
            (100, 101): "<|im_start|> user",
            (100, 101, 102): "<|im_start|> user\n",
            (100, 101, 102, 103): "<|im_start|>\nO",
            (100, 101, 102, 103, 104): "<|im_start|>\nOK",
            (103,): "O",
            (103, 104): "OK",
        },
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={201: "<|im_start|> user\n", 202: "OK"},
        encode_map={
            "<|im_start|> user\n": [201],
            "OK": [202],
            "<|im_start|> user\nOK": [201, 202],
        },
    )

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    sample = Sample.from_dict(
        Sample(
            tokens=[100, 101, 102, 103, 104],
            response="OK",
            response_length=2,
            reward={
                "meta_info": {
                    "input_token_logprobs": [
                        [0.0, 999],
                        [-0.1, 201],
                        [-0.2, 202],
                    ]
                }
            },
            opd_full_text="<|im_start|> user\nOK",
            opd_prompt_text="<|im_start|> user\n",
            opd_response_text="OK",
        ).to_dict()
    )
    args = Namespace(
        reward_key=None,
        opd_alignment="byte_chunk",
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )

    direct_teacher_log_probs = compute_teacher_log_probs_for_sample(args, sample)
    scalar_rewards, raw_rewards = post_process_rewards(args, [sample])

    assert scalar_rewards == [0.0]
    assert raw_rewards == [0.0]
    assert torch.allclose(direct_teacher_log_probs, torch.tensor([-0.2]))
    assert torch.allclose(sample.teacher_log_probs, direct_teacher_log_probs)


def test_training_like_byte_chunk_replay_from_sample_dict(monkeypatch):
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={100: "<|im_start|>", 101: " user", 102: "\n", 103: "O", 104: "K", 105: " user\n"},
        encode_map={
            "<|im_start|> user\n": [100, 105],
            "OK": [103, 104],
            "<|im_start|> user\nOK": [100, 105, 103, 104],
        },
        decode_map={
            (100,): "<|im_start|>",
            (100, 101): "<|im_start|> user",
            (100, 101, 102): "<|im_start|> user\n",
            (100, 101, 102, 103): "<|im_start|>\nO",
            (100, 101, 102, 103, 104): "<|im_start|>\nOK",
            (103,): "O",
            (103, 104): "OK",
        },
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={201: "<|im_start|> user\n", 202: "OK"},
        encode_map={
            "<|im_start|> user\n": [201],
            "OK": [202],
            "<|im_start|> user\nOK": [201, 202],
        },
    )

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    sample = Sample.from_dict(
        Sample(
            tokens=[100, 101, 102, 103, 104],
            response="OK",
            response_length=2,
            rollout_log_probs=[-0.1, -0.3],
            reward={
                "meta_info": {
                    "input_token_logprobs": [
                        [0.0, 999],
                        [-0.1, 201],
                        [-0.2, 202],
                    ]
                }
            },
            opd_full_text="<|im_start|> user\nOK",
            opd_prompt_text="<|im_start|> user\n",
            opd_response_text="OK",
        ).to_dict()
    )
    args = Namespace(
        reward_key=None,
        opd_alignment="byte_chunk",
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )

    teacher_log_probs = compute_teacher_log_probs_for_sample(args, sample)
    student_chunk_log_probs, teacher_chunk_log_probs = opd_utils.compute_byte_chunk_aligned_log_probs(
        full_text=sample.opd_full_text,
        prompt_token_count=len(sample.tokens) - sample.response_length,
        student_token_ids=sample.tokens,
        response_token_count=sample.response_length,
        student_log_probs=torch.tensor(sample.rollout_log_probs, dtype=torch.float32),
        teacher_log_probs=teacher_log_probs,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        allow_sequence_fallback=False,
    )

    assert torch.allclose(teacher_log_probs, torch.tensor([-0.2]))
    assert torch.allclose(student_chunk_log_probs, torch.tensor([-0.2, -0.2]))
    assert torch.allclose(teacher_chunk_log_probs, torch.tensor([-0.1, -0.1]))


def test_compute_byte_chunk_reverse_kl_handles_non_roundtrippable_prompt_tokens():
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={100: "<|im_start|>", 101: " user", 102: "\n", 103: "O", 104: "K", 105: " user\n"},
        encode_map={
            "<|im_start|> user\n": [100, 105],
            "OK": [103, 104],
            "<|im_start|> user\nOK": [100, 105, 103, 104],
        },
        decode_map={
            (100,): "<|im_start|>",
            (100, 101): "<|im_start|> user",
            (100, 101, 102): "<|im_start|> user\n",
            (100, 101, 102, 103): "<|im_start|>\nO",
            (100, 101, 102, 103, 104): "<|im_start|>\nOK",
            (103,): "O",
            (103, 104): "OK",
        },
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={201: "<|im_start|> user\n", 202: "OK"},
        encode_map={
            "<|im_start|> user\n": [201],
            "OK": [202],
            "<|im_start|> user\nOK": [201, 202],
        },
    )

    reverse_kl = opd_utils.compute_byte_chunk_reverse_kl(
        full_text="<|im_start|> user\nOK",
        prompt_token_count=3,
        student_token_ids=[100, 101, 102, 103, 104],
        response_token_count=2,
        student_log_probs=torch.tensor([-0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.4]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    assert torch.allclose(reverse_kl, torch.tensor([-0.05, -0.05]))


def test_compute_byte_chunk_reverse_kl_uses_contextual_student_response_bytes():
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "A", 3: "B"},
        encode_map={
            "P AB": [1, 2, 3],
            "AB": [2, 3],
        },
        decode_map={
            (1,): "P",
            (1, 2): "P A",
            (1, 2, 3): "P AB",
            (2,): "A",
            (2, 3): "AB",
        },
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={10: "P", 11: " A", 12: "B"},
        encode_map={
            "P AB": [10, 11, 12],
        },
        decode_map={
            (10,): "P",
            (10, 11): "P A",
            (10, 11, 12): "P AB",
        },
        offsets_map={"P AB": [(0, 1), (1, 3), (3, 4)]},
    )

    reverse_kl = opd_utils.compute_byte_chunk_reverse_kl(
        full_text="P AB",
        prompt_token_count=1,
        student_token_ids=[1, 2, 3],
        response_token_count=2,
        student_log_probs=torch.tensor([-0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.4, -0.5]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    assert torch.allclose(reverse_kl, torch.tensor([0.2, 0.2]))


def test_compute_byte_chunk_reverse_kl_falls_back_to_sequence_penalty_when_alignment_fails(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "P", 2: "A", 3: "B"})
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})

    def fail_align(_student_token_bytes, _teacher_token_bytes):
        raise ValueError("forced alignment failure")

    monkeypatch.setattr("slime.utils.opd_utils.align_token_byte_chunks", fail_align)

    reverse_kl = opd_utils.compute_byte_chunk_reverse_kl(
        full_text="PAB",
        prompt_token_count=1,
        student_token_ids=[1, 2, 3],
        response_token_count=2,
        student_log_probs=torch.tensor([-0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.7]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    assert torch.allclose(reverse_kl, torch.tensor([0.1, 0.1]))


def test_compute_byte_chunk_reverse_kl_raises_when_sequence_fallback_disabled_on_alignment_failure(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "P", 2: "A", 3: "B"})
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})

    def fail_align(_student_token_bytes, _teacher_token_bytes):
        raise ValueError("forced alignment failure")

    monkeypatch.setattr("slime.utils.opd_utils.align_token_byte_chunks", fail_align)

    with pytest.raises(ValueError, match="sequence fallback disabled"):
        opd_utils.compute_byte_chunk_reverse_kl(
            full_text="PAB",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.7]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            allow_sequence_fallback=False,
        )


def test_compute_byte_chunk_reverse_kl_falls_back_when_student_byte_reconstruction_fails():
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "A", 3: "B"},
        encode_map={"PAB": [1, 2, 3]},
        decode_map={
            (1,): "P",
            (1, 2): "PX",
            (1, 2, 3): "PAB",
            (2,): "A",
            (2, 3): "XB",
        },
    )
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "A", 12: "B"})

    reverse_kl = opd_utils.compute_byte_chunk_reverse_kl(
        full_text="PAB",
        prompt_token_count=1,
        student_token_ids=[1, 2, 3],
        response_token_count=2,
        student_log_probs=torch.tensor([-0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.4, -0.5]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    assert torch.allclose(reverse_kl, torch.tensor([0.2, 0.2]))


def test_compute_byte_chunk_reverse_kl_raises_when_sequence_fallback_disabled_on_student_failure():
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "A", 3: "X"},
        encode_map={"PAB": [1, 2, 3]},
        decode_map={
            (1,): "P",
            (1, 2): "PX",
            (1, 2, 3): "PAB",
            (2,): "A",
            (2, 3): "XB",
        },
        offsets_map={"PAB": [(0, 1), (0, 1), (0, 1)]},
    )
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "A", 12: "B"})
    student_tokenizer.convert_ids_to_tokens = None

    with pytest.raises(ValueError, match="sequence fallback disabled"):
        opd_utils.compute_byte_chunk_reverse_kl(
            full_text="PAB",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.4, -0.5]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            allow_sequence_fallback=False,
        )


def test_compute_byte_chunk_aligned_log_probs_prefers_recorded_student_alignment(monkeypatch):
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "A", 3: "X"},
        encode_map={"PAB": [1, 2, 3]},
        decode_map={
            (1,): "P",
            (1, 2): "PX",
            (1, 2, 3): "PAB",
            (2,): "A",
            (2, 3): "XB",
        },
        offsets_map={"PAB": [(0, 1), (0, 1), (0, 1)]},
    )
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})
    student_tokenizer.convert_ids_to_tokens = None

    def fail_build_contextual(*args, **kwargs):
        raise AssertionError("legacy student reconstruction path should not be used")

    monkeypatch.setattr("slime.utils.opd_utils.build_contextual_suffix_token_bytes", fail_build_contextual)

    student_chunk_log_probs, teacher_chunk_log_probs = opd_utils.compute_byte_chunk_aligned_log_probs(
        full_text="PAB",
        prompt_text="P",
        prompt_token_count=1,
        student_token_ids=[1, 2, 3],
        response_token_count=2,
        student_log_probs=torch.tensor([-0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.4]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        recorded_student_response_bytes="AB".encode("utf-8"),
        recorded_student_token_byte_spans=[(0, 1), (1, 2)],
        allow_sequence_fallback=False,
    )

    assert torch.allclose(student_chunk_log_probs, torch.tensor([-0.25, -0.25]))
    assert torch.allclose(teacher_chunk_log_probs, torch.tensor([-0.2, -0.2]))


def test_record_student_opd_alignment_stores_response_bytes_and_spans():
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="P",
        opd_response_text="AB",
        opd_full_text="PAB",
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "A", 3: "B"},
        encode_map={"PAB": [1, 2, 3], "AB": [2, 3]},
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_response_bytes == [65, 66]
    assert sample.opd_student_token_byte_spans == [[0, 1], [1, 2]]
    assert sample.opd_student_alignment_version == 1
    assert sample.opd_student_alignment_error is None


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


def test_apply_opd_byte_chunk_to_advantages_prefers_recorded_student_alignment(monkeypatch):
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

    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "A", 3: "X"},
        encode_map={"PAB": [1, 2, 3]},
        decode_map={
            (1,): "P",
            (1, 2): "PX",
            (1, 2, 3): "PAB",
            (2,): "A",
            (2, 3): "XB",
        },
        offsets_map={"PAB": [(0, 1), (0, 1), (0, 1)]},
    )
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})
    student_tokenizer.convert_ids_to_tokens = None

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr("slime.backends.megatron_utils.loss.get_cached_tokenizer", fake_get_cached_tokenizer)

    def fail_build_contextual(*args, **kwargs):
        raise AssertionError("legacy student reconstruction path should not be used")

    monkeypatch.setattr("slime.utils.opd_utils.build_contextual_suffix_token_bytes", fail_build_contextual)

    args = Namespace(
        opd_type="sglang",
        opd_kl_coef=2.0,
        opd_alignment="byte_chunk",
        hf_checkpoint="student",
        opd_teacher_hf_checkpoint="teacher",
    )
    rollout_data = {
        "tokens": [torch.tensor([1, 2, 3])],
        "response_lengths": [2],
        "teacher_log_probs": [torch.tensor([-0.4])],
        "opd_full_texts": ["PAB"],
        "opd_prompt_texts": ["P"],
        "opd_student_response_bytes_list": [[65, 66]],
        "opd_student_token_byte_spans_list": [[[0, 1], [1, 2]]],
    }
    advantages = [torch.tensor([1.0, 1.0])]
    student_log_probs = [torch.tensor([-0.2, -0.3])]

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)

    assert torch.allclose(advantages[0], torch.tensor([1.1, 1.1]))
    assert torch.allclose(rollout_data["opd_reverse_kl"][0], torch.tensor([-0.05, -0.05]))
