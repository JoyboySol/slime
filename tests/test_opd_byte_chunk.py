from argparse import Namespace
import importlib.util
import sys
import types

import pytest
import torch

from slime.rollout.on_policy_distillation import (
    _build_canonical_opd_texts,
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


class SentencePieceLikeTokenizer(BoundaryAwareTokenizer):
    def __init__(self, token_map, encode_map, token_piece_map, decode_map=None, offsets_map=None):
        super().__init__(token_map, encode_map, decode_map=decode_map, offsets_map=offsets_map)
        self.token_piece_map = token_piece_map

    def convert_ids_to_tokens(self, token_ids):
        return [self.token_piece_map[token_id] for token_id in token_ids]


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


def test_build_token_byte_spans_token_string_fallback_understands_sentencepiece_space_marker():
    tokenizer = SentencePieceLikeTokenizer(
        token_map={10: "We", 11: "are", 12: "given"},
        encode_map={"We are given": [10, 11, 12]},
        token_piece_map={10: "We", 11: "▁are", 12: "▁given"},
        offsets_map={"We are given": [(0, 2), (0, 0), (0, 0)]},
    )

    token_bytes, token_spans = opd_utils.build_token_byte_spans(tokenizer, "We are given", [10, 11, 12])

    assert b"".join(token_bytes) == b"We are given"
    assert token_bytes == [b"We", b" are", b" given"]
    assert token_spans == [(0, 2), (2, 6), (6, 12)]


def test_build_recorded_student_response_alignment_from_token_ids_recovers_sentencepiece_spaces():
    tokenizer = SentencePieceLikeTokenizer(
        token_map={10: "We", 11: "are", 12: "given"},
        encode_map={"We are given": [10, 11, 12]},
        token_piece_map={10: "We", 11: "▁are", 12: "▁given"},
        offsets_map={"We are given": [(0, 2), (0, 0), (0, 0)]},
    )

    response_bytes, token_spans = opd_utils.build_recorded_student_response_alignment_from_token_ids(
        tokenizer=tokenizer,
        response_text="We are given",
        response_token_ids=[10, 11, 12],
    )

    assert response_bytes == b"We are given"
    assert token_spans == [(0, 2), (2, 6), (6, 12)]


def test_build_recorded_student_response_alignment_from_token_ids_succeeds_when_generation_text_drops_space_markers():
    tokenizer = SentencePieceLikeTokenizer(
        token_map={10: "We", 11: "are", 12: "asked", 13: ":", 14: '"'},
        encode_map={'We are asked: "': [10, 11, 12, 13, 14]},
        token_piece_map={10: "We", 11: "▁are", 12: "▁asked", 13: ":", 14: '▁"'},
        offsets_map={'We are asked: "': [(0, 2), (0, 0), (0, 0), (12, 13), (0, 0)]},
    )

    with pytest.raises(ValueError, match="Generation token texts do not match canonical response text bytes."):
        opd_utils.build_recorded_student_response_alignment_from_token_texts(
            response_text='We are asked: "',
            response_token_texts=["We", "are", "asked", ":", '"'],
        )

    response_bytes, token_spans = opd_utils.build_recorded_student_response_alignment_from_token_ids(
        tokenizer=tokenizer,
        response_text='We are asked: "',
        response_token_ids=[10, 11, 12, 13, 14],
    )

    assert response_bytes == b'We are asked: "'
    assert token_spans == [(0, 2), (2, 6), (6, 12), (12, 13), (13, 15)]


def test_build_recorded_student_response_alignment_from_token_ids_handles_special_token_boundary_space_marker():
    tokenizer = SentencePieceLikeTokenizer(
        token_map={10: "</think>", 11: "", 12: "\n", 13: "To"},
        encode_map={"</think>\n\nTo": [10, 11, 12, 12, 13]},
        token_piece_map={10: "</think>", 11: "▁", 12: "\n", 13: "To"},
        offsets_map={"</think>\n\nTo": [(0, 8), (8, 8), (8, 9), (9, 10), (10, 12)]},
    )

    response_bytes, token_spans = opd_utils.build_recorded_student_response_alignment_from_token_ids(
        tokenizer=tokenizer,
        response_text="</think>\n\nTo",
        response_token_ids=[10, 11, 12, 12, 13],
    )

    assert response_bytes == b"</think>\n\nTo"
    assert token_spans == [(0, 8), (8, 8), (8, 9), (9, 10), (10, 12)]


def test_build_recorded_student_response_alignment_from_token_ids_prefers_utf8_for_unicode_symbols():
    tokenizer = SentencePieceLikeTokenizer(
        token_map={10: "6", 11: "×", 12: "5"},
        encode_map={"6×5": [10, 11, 12]},
        token_piece_map={10: "6", 11: "×", 12: "5"},
        offsets_map={"6×5": [(0, 1), (1, 2), (2, 3)]},
    )

    response_bytes, token_spans = opd_utils.build_recorded_student_response_alignment_from_token_ids(
        tokenizer=tokenizer,
        response_text="6×5",
        response_token_ids=[10, 11, 12],
    )

    assert response_bytes == "6×5".encode("utf-8")
    assert token_spans == [(0, 1), (1, 3), (3, 4)]


def test_build_recorded_student_response_alignment_from_token_ids_supports_hex_byte_tokens():
    tokenizer = SentencePieceLikeTokenizer(
        token_map={10: "", 11: "", 12: "", 13: "", 14: "", 15: ""},
        encode_map={"➡️": [10, 11, 12, 13, 14, 15]},
        token_piece_map={
            10: "<0xE2>",
            11: "<0x9E>",
            12: "<0xA1>",
            13: "<0xEF>",
            14: "<0xB8>",
            15: "<0x8F>",
        },
        offsets_map={"➡️": [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1)]},
    )

    response_bytes, token_spans = opd_utils.build_recorded_student_response_alignment_from_token_ids(
        tokenizer=tokenizer,
        response_text="➡️",
        response_token_ids=[10, 11, 12, 13, 14, 15],
    )

    assert response_bytes == "➡️".encode("utf-8")
    assert token_spans == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]


def test_build_recorded_student_response_alignment_from_token_ids_maps_invalid_hex_byte_to_replacement_char():
    tokenizer = SentencePieceLikeTokenizer(
        token_map={10: "A", 11: "�", 12: "B"},
        encode_map={"A�B": [10, 11, 12]},
        token_piece_map={10: "A", 11: "<0x86>", 12: "B"},
        offsets_map={"A�B": [(0, 1), (1, 2), (2, 3)]},
    )

    response_bytes, token_spans = opd_utils.build_recorded_student_response_alignment_from_token_ids(
        tokenizer=tokenizer,
        response_text="A�B",
        response_token_ids=[10, 11, 12],
    )

    assert response_bytes == "A�B".encode("utf-8")
    assert token_spans == [(0, 1), (1, 4), (4, 5)]


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
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "R"},
        encode_map={"P": [1], "R": [2], "PR": [1, 2]},
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={10: "P", 11: "R"},
        encode_map={"PR": [10, 11]},
    )

    def fail_teacher_log_probs(_args, _sample):
        raise ValueError("teacher stage mismatch")

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr(replay_module, "get_cached_tokenizer", fake_get_cached_tokenizer)
    monkeypatch.setattr(replay_module, "compute_teacher_log_probs_for_sample", fail_teacher_log_probs)

    summary = replay_module._summarize_sample(sample, args)

    assert summary["byte_chunk_alignment"] == "failed:ValueError"
    assert summary["error"] == "teacher stage mismatch"
    assert summary["prompt_preview"] == "P"
    assert summary["response_preview"] == "R"


def test_replay_summary_prefers_recorded_student_alignment(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "replay_debug_rollout_opd",
        "/mnt/ssd/lvzhihao/PostTrain/slime/scripts/replay_debug_rollout_opd.py",
    )
    replay_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(replay_module)

    sample = Sample(
        prompt="P",
        tokens=[1, 2, 3],
        response="AB",
        response_length=2,
        rollout_log_probs=[-0.2, -0.3],
        reward={"meta_info": {"input_token_logprobs": [[0.0, 999], [-0.4, 10]]}},
        opd_full_text="PAB",
        opd_prompt_text="P",
        opd_response_text="AB",
        opd_student_response_bytes=[65, 66],
        opd_student_token_byte_spans=[[0, 1], [1, 2]],
        opd_student_alignment_source="recorded_builder",
        opd_student_alignment_validated=True,
        opd_student_alignment_status="ok_recorded",
    )
    args = Namespace(
        hf_checkpoint="student",
        teacher_hf_checkpoint="teacher",
        preview_chars=32,
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "P", 2: "A", 3: "B"},
        encode_map={"P": [1], "AB": [2, 3], "PAB": [1, 2, 3]},
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={10: "P", 11: "AB"},
        encode_map={"PAB": [10, 11]},
    )

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr(replay_module, "get_cached_tokenizer", fake_get_cached_tokenizer)
    monkeypatch.setattr(replay_module, "compute_teacher_log_probs_for_sample", lambda *_: torch.tensor([-0.4]))

    def fake_build_token_byte_spans(_tokenizer, full_text, token_ids):
        assert full_text == "PAB"
        assert token_ids == [10, 11]
        return [b"P", b"AB"], [(0, 1), (1, 3)]

    monkeypatch.setattr(replay_module, "encode_text", lambda _tokenizer, _text: [10, 11])
    monkeypatch.setattr(replay_module, "build_token_byte_spans", fake_build_token_byte_spans)

    def fake_prepare_byte_chunk_training_entry(**kwargs):
        assert kwargs["recorded_student_response_bytes"] == [65, 66]
        assert kwargs["recorded_student_token_byte_spans"] == [[0, 1], [1, 2]]
        return {
            "status": "ok",
            "student_chunk_log_probs": torch.tensor([-0.25, -0.25]),
            "teacher_chunk_log_probs": torch.tensor([-0.2, -0.2]),
            "reverse_kl": torch.tensor([-0.05, -0.05]),
        }

    monkeypatch.setattr(replay_module, "prepare_byte_chunk_training_entry", fake_prepare_byte_chunk_training_entry)

    summary = replay_module._summarize_sample(sample, args)

    assert summary["byte_chunk_alignment"] == "ok"
    assert summary["student_alignment_source"] == "recorded_payload"
    assert summary["recorded_alignment_source"] == "recorded_builder"
    assert summary["recorded_alignment_validated"] is True
    assert summary["recorded_alignment_status"] == "ok_recorded"
    assert summary["student_chunk_mean"] == pytest.approx(-0.25)


def test_replay_summary_uses_training_canonical_texts_when_rendered_response_is_not_token_consistent(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "replay_debug_rollout_opd",
        "/mnt/ssd/lvzhihao/PostTrain/slime/scripts/replay_debug_rollout_opd.py",
    )
    replay_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(replay_module)

    sample = Sample(
        prompt="PROMPT",
        response="bad rendered response",
        tokens=[1, 2, 3],
        response_length=2,
        rollout_log_probs=[-0.25, -0.35],
        reward={
            "meta_info": {
                "input_token_logprobs": [
                    [0.0, 999],
                    [-0.1, 10],
                    [-0.2, 11],
                    [-0.3, 12],
                ]
            }
        },
        opd_student_response_bytes=[65, 66],
        opd_student_token_byte_spans=[[0, 1], [1, 2]],
        opd_student_alignment_source="recorded_builder",
        opd_student_alignment_validated=True,
        opd_student_alignment_status="ok_recorded",
        opd_student_alignment_complete=True,
        opd_student_alignment_version=1,
    )
    args = Namespace(
        hf_checkpoint="student",
        teacher_hf_checkpoint="teacher",
        opd_teacher_hf_checkpoint="teacher",
        preview_chars=32,
        reward_key=None,
        opd_alignment="byte_chunk",
    )

    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPT": [1], "AB": [2, 3], "PROMPTAB": [1, 2, 3]},
    )
    teacher_tokenizer = BoundaryAwareTokenizer(
        token_map={10: "PROMPT", 11: "A", 12: "B"},
        encode_map={"PROMPTAB": [10, 11, 12]},
    )

    def fake_get_cached_tokenizer(path):
        if path == "student":
            return student_tokenizer
        if path == "teacher":
            return teacher_tokenizer
        raise AssertionError(f"Unexpected tokenizer path: {path}")

    monkeypatch.setattr(replay_module, "get_cached_tokenizer", fake_get_cached_tokenizer)
    monkeypatch.setattr("slime.rollout.on_policy_distillation.get_cached_tokenizer", fake_get_cached_tokenizer)

    def fake_prepare_byte_chunk_training_entry(**kwargs):
        assert kwargs["prompt_text"] == "PROMPT"
        assert kwargs["full_text"] == "PROMPTAB"
        assert torch.allclose(kwargs["teacher_log_probs"], torch.tensor([-0.2, -0.3]))
        return {
            "status": "ok",
            "student_chunk_log_probs": torch.tensor([-0.25, -0.25]),
            "teacher_chunk_log_probs": torch.tensor([-0.2, -0.2]),
            "reverse_kl": torch.tensor([-0.05, -0.05]),
        }

    monkeypatch.setattr(replay_module, "prepare_byte_chunk_training_entry", fake_prepare_byte_chunk_training_entry)

    summary = replay_module._summarize_sample(sample, args)

    assert summary["byte_chunk_alignment"] == "ok"
    assert summary["teacher_log_prob_count"] == 2
    assert summary["teacher_selected_response_count"] == 2
    assert summary["student_chunk_mean"] == pytest.approx(-0.25)


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
    assert normalized.opd_alignment == "byte_chunk"


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


def test_compute_byte_chunk_reverse_kl_requires_recorded_alignment_when_student_bytes_do_not_chunk_align():
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

    with pytest.raises(ValueError, match="alignment_failed"):
        opd_utils.compute_byte_chunk_reverse_kl(
            full_text="P AB",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.4, -0.5]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
        )


def test_compute_byte_chunk_reverse_kl_raises_when_alignment_fails(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "P", 2: "A", 3: "B"})
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})

    def fail_align(_student_token_bytes, _teacher_token_bytes):
        raise ValueError("forced alignment failure")

    monkeypatch.setattr("slime.utils.opd_utils.align_token_byte_chunks", fail_align)

    with pytest.raises(ValueError, match="alignment_failed: forced alignment failure"):
        opd_utils.compute_byte_chunk_reverse_kl(
            full_text="PAB",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.7]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
        )


def test_compute_byte_chunk_reverse_kl_reports_alignment_failure_reason(monkeypatch):
    student_tokenizer = FakeTokenizer({1: "P", 2: "A", 3: "B"})
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})

    def fail_align(_student_token_bytes, _teacher_token_bytes):
        raise ValueError("forced alignment failure")

    monkeypatch.setattr("slime.utils.opd_utils.align_token_byte_chunks", fail_align)

    with pytest.raises(ValueError, match="alignment_failed: forced alignment failure"):
        opd_utils.compute_byte_chunk_reverse_kl(
            full_text="PAB",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.7]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
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


def test_compute_byte_chunk_reverse_kl_reports_student_reconstruction_failure():
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

    with pytest.raises(ValueError, match="student_byte_reconstruction_failed"):
        opd_utils.compute_byte_chunk_reverse_kl(
            full_text="PAB",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.4, -0.5]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
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


def test_compute_byte_chunk_aligned_log_probs_rejects_incomplete_evidence_even_when_fallback_flag_is_set():
    student_tokenizer = FakeTokenizer({1: "P", 2: "A", 3: "B"})
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})

    with pytest.raises(ValueError, match="student_alignment_evidence_incomplete"):
        opd_utils.compute_byte_chunk_aligned_log_probs(
            full_text="PAB",
            prompt_text="P",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.4]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            student_alignment_evidence={
                "version": 2,
                "source": "generation_byte_evidence",
                "response_bytes": [65, 66],
                "token_byte_spans": [[0, 1], [1, 2]],
                "response_token_count": 2,
                "complete": False,
                "validated": True,
                "error": "generation_byte_evidence_incomplete",
                "metadata": {"engine": "sglang"},
            },
            allow_sequence_fallback=True,
        )


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
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_complete is True
    assert sample.opd_student_alignment_validated is True
    assert sample.opd_student_alignment_status == "ok_recorded"
    assert sample.opd_student_alignment_version == 1
    assert sample.opd_student_alignment_error is None
    assert sample.opd_student_alignment_metadata == {
        "response_token_count": 2,
        "response_byte_length": 2,
        "evidence_kind": "recorded_builder_token_ids",
    }


def test_record_student_opd_alignment_marks_failed_recording_metadata(monkeypatch):
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

    monkeypatch.setattr(
        "slime.rollout.on_policy_distillation.build_recorded_student_response_alignment_from_token_ids",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom-token-ids")),
    )
    monkeypatch.setattr(
        "slime.rollout.on_policy_distillation.build_recorded_student_response_alignment",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom-full-sequence")),
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_response_bytes is None
    assert sample.opd_student_token_byte_spans is None
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_complete is False
    assert sample.opd_student_alignment_validated is False
    assert sample.opd_student_alignment_status == "recorded_missing"
    assert sample.opd_student_alignment_error == "boom-full-sequence"
    assert sample.opd_student_alignment_metadata == {
        "response_token_count": 2,
        "generation_token_text_count": None,
        "evidence_kind": "recorded_builder_failed",
    }


def test_record_student_opd_alignment_prefers_token_id_recorded_builder_before_contextual_builder(monkeypatch):
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="PROMPT",
        opd_response_text="AB",
        opd_full_text="PROMPTAB",
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPTAB": [1, 2, 3], "AB": [2, 3]},
    )

    monkeypatch.setattr(
        "slime.rollout.on_policy_distillation.build_recorded_student_response_alignment_from_token_ids",
        lambda **kwargs: (b"AB", [(0, 1), (1, 2)]),
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_response_bytes == [65, 66]
    assert sample.opd_student_token_byte_spans == [[0, 1], [1, 2]]
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_metadata == {
        "response_token_count": 2,
        "response_byte_length": 2,
        "evidence_kind": "recorded_builder_token_ids",
    }


def test_record_student_opd_alignment_marks_failure_when_token_id_builder_fails(monkeypatch):
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="PROMPT",
        opd_response_text="AB",
        opd_full_text="PROMPTAB",
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPTAB": [1, 2, 3], "AB": [2, 3]},
    )

    monkeypatch.setattr(
        "slime.rollout.on_policy_distillation.build_recorded_student_response_alignment_from_token_ids",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("token-id failed")),
    )
    monkeypatch.setattr(
        "slime.rollout.on_policy_distillation.build_recorded_student_response_alignment",
        lambda tokenizer, **kwargs: (_ for _ in ()).throw(ValueError("full-sequence failed")),
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_response_bytes is None
    assert sample.opd_student_token_byte_spans is None
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_complete is False
    assert sample.opd_student_alignment_validated is False
    assert sample.opd_student_alignment_error == "full-sequence failed"
    assert sample.opd_student_alignment_metadata == {
        "response_token_count": 2,
        "generation_token_text_count": None,
        "evidence_kind": "recorded_builder_failed",
    }


def test_record_student_opd_alignment_falls_back_to_contextual_recorded_builder_when_token_id_builder_fails(monkeypatch):
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="PROMPT",
        opd_response_text="AB",
        opd_full_text="PROMPTAB",
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPTAB": [1, 2, 3], "AB": [2, 3]},
    )

    monkeypatch.setattr(
        "slime.rollout.on_policy_distillation.build_recorded_student_response_alignment_from_token_ids",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("token-id failed")),
    )
    monkeypatch.setattr(
        "slime.rollout.on_policy_distillation.build_recorded_student_response_alignment",
        lambda tokenizer, **kwargs: (b"AB", [(0, 1), (1, 2)]),
        raising=False,
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_response_bytes == [65, 66]
    assert sample.opd_student_token_byte_spans == [[0, 1], [1, 2]]
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_complete is True
    assert sample.opd_student_alignment_validated is True
    assert sample.opd_student_alignment_error is None
    assert sample.opd_student_alignment_metadata == {
        "response_token_count": 2,
        "response_byte_length": 2,
        "evidence_kind": "recorded_builder_full_sequence",
    }


def test_canonical_opd_texts_fall_back_to_token_derived_text_when_rendered_response_is_not_token_consistent():
    sample = Sample(
        prompt="PROMPT",
        response="bad rendered response",
        tokens=[1, 2, 3],
        response_length=2,
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPT": [1], "AB": [2, 3], "PROMPTAB": [1, 2, 3]},
    )

    prompt_text, response_text, full_text = _build_canonical_opd_texts(
        sample=sample,
        student_tokenizer=student_tokenizer,
        prompt_token_ids=[1],
    )

    assert prompt_text == "PROMPT"
    assert response_text == "AB"
    assert full_text == "PROMPTAB"


def test_canonical_opd_texts_keep_prompt_boundary_consistent_when_full_text_falls_back_to_token_decode():
    sample = Sample(
        prompt="PROMPT\n",
        response="bad rendered response",
        tokens=[1, 2, 3],
        response_length=2,
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT \n", 2: "A", 3: "B"},
        encode_map={
            "PROMPT\n": [99],
            "AB": [2, 3],
            "PROMPT\nAB": [98],
            "PROMPT \nAB": [1, 2, 3],
        },
    )

    prompt_text, response_text, full_text = _build_canonical_opd_texts(
        sample=sample,
        student_tokenizer=student_tokenizer,
        prompt_token_ids=[1],
    )

    assert prompt_text == "PROMPT \n"
    assert response_text == "AB"
    assert full_text == "PROMPT \nAB"
    assert full_text == f"{prompt_text}{response_text}"


def test_record_student_opd_alignment_uses_token_derived_canonical_text_when_rendered_response_is_not_token_consistent():
    sample = Sample(
        prompt="PROMPT",
        response="bad rendered response",
        tokens=[1, 2, 3],
        response_length=2,
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPT": [1], "AB": [2, 3], "PROMPTAB": [1, 2, 3]},
    )

    sample.opd_prompt_text, sample.opd_response_text, sample.opd_full_text = _build_canonical_opd_texts(
        sample=sample,
        student_tokenizer=student_tokenizer,
        prompt_token_ids=[1],
    )
    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_response_text == "AB"
    assert sample.opd_full_text == "PROMPTAB"
    assert sample.opd_student_response_bytes == [65, 66]
    assert sample.opd_student_token_byte_spans == [[0, 1], [1, 2]]
    assert sample.opd_student_alignment_error is None


def test_record_student_opd_alignment_ignores_generation_token_text_for_production_alignment(monkeypatch):
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="PROMPT",
        opd_response_text="AB",
        opd_full_text="PROMPTAB",
        opd_student_token_texts=["A", "B"],
        opd_student_alignment_metadata={
            "engine": "sglang",
            "detokenizer_protocol": "output_token_logprobs_text",
            "completion_reason": "stop",
            "captured_response_token_count": 2,
        },
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPTAB": [1, 2, 3], "AB": [2, 3]},
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_response_bytes == [65, 66]
    assert sample.opd_student_token_byte_spans == [[0, 1], [1, 2]]
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_complete is True
    assert sample.opd_student_alignment_validated is True
    assert sample.opd_student_alignment_status == "ok_recorded"
    assert sample.opd_student_alignment_error is None
    assert sample.opd_student_alignment_metadata == {
        "engine": "sglang",
        "detokenizer_protocol": "output_token_logprobs_text",
        "completion_reason": "stop",
        "captured_response_token_count": 2,
        "response_token_count": 2,
        "response_byte_length": 2,
        "generation_token_text_count": 2,
        "evidence_kind": "recorded_builder_token_ids",
    }


def test_record_student_opd_alignment_rebuilds_clean_recorded_alignment_from_token_ids(monkeypatch):
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="PROMPT",
        opd_response_text="AB",
        opd_full_text="PROMPTAB",
        opd_student_response_bytes=[65, 66],
        opd_student_token_byte_spans=[[0, 1], [1, 2]],
        opd_student_alignment_version=2,
        opd_student_alignment_source="generation_byte_evidence",
        opd_student_alignment_complete=True,
        opd_student_alignment_validated=True,
        opd_student_alignment_metadata={
            "engine": "sglang",
            "detokenizer_protocol": "output_token_logprobs_text",
            "completion_reason": "stop",
            "captured_response_token_count": 2,
        },
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPTAB": [1, 2, 3], "AB": [2, 3]},
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_alignment_version == 1
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_complete is True
    assert sample.opd_student_alignment_validated is True
    assert sample.opd_student_response_bytes == [65, 66]
    assert sample.opd_student_token_byte_spans == [[0, 1], [1, 2]]


def test_record_student_opd_alignment_uses_token_id_builder_after_invalid_generation_byte_evidence():
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="PROMPT",
        opd_response_text="AB",
        opd_full_text="PROMPTAB",
        opd_student_response_bytes=[65, 90],
        opd_student_token_byte_spans=[[0, 1], [1, 2]],
        opd_student_alignment_version=2,
        opd_student_alignment_source="generation_byte_evidence",
        opd_student_alignment_complete=False,
        opd_student_alignment_validated=False,
        opd_student_token_texts=["A", "B"],
        opd_student_alignment_metadata={
            "engine": "sglang",
            "detokenizer_protocol": "output_token_logprobs_text",
            "completion_reason": "stop",
            "captured_response_token_count": 2,
        },
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPTAB": [1, 2, 3], "AB": [2, 3]},
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_student_alignment_complete is True
    assert sample.opd_student_alignment_validated is True
    assert sample.opd_student_response_bytes == [65, 66]
    assert sample.opd_student_token_byte_spans == [[0, 1], [1, 2]]
    assert sample.opd_student_alignment_metadata["evidence_kind"] == "recorded_builder_token_ids"


def test_compute_byte_chunk_aligned_log_probs_raises_by_default_on_incomplete_evidence():
    student_tokenizer = FakeTokenizer({1: "P", 2: "A", 3: "B"})
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})

    with pytest.raises(ValueError, match="student_alignment_evidence_incomplete"):
        opd_utils.compute_byte_chunk_aligned_log_probs(
            full_text="PAB",
            prompt_text="P",
            prompt_token_count=1,
            student_token_ids=[1, 2, 3],
            response_token_count=2,
            student_log_probs=torch.tensor([-0.2, -0.3]),
            teacher_log_probs=torch.tensor([-0.4]),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            student_alignment_evidence={
                "version": 2,
                "source": "generation_byte_evidence",
                "response_bytes": [65, 66],
                "token_byte_spans": [[0, 1], [1, 2]],
                "response_token_count": 2,
                "complete": False,
                "validated": True,
                "error": "generation_byte_evidence_incomplete",
                "metadata": {"engine": "sglang"},
            },
        )


def test_record_student_opd_alignment_preserves_generation_evidence_failure_context(monkeypatch):
    sample = Sample(
        tokens=[1, 2, 3],
        response_length=2,
        opd_prompt_text="PROMPT",
        opd_response_text="AB",
        opd_full_text="PROMPTAB",
        opd_student_token_texts=["A", ""],
        opd_student_alignment_metadata={
            "engine": "sglang",
            "detokenizer_protocol": "output_token_logprobs_text",
            "completion_reason": "length",
            "captured_response_token_count": 2,
        },
    )
    student_tokenizer = BoundaryAwareTokenizer(
        token_map={1: "PROMPT", 2: "A", 3: "B"},
        encode_map={"PROMPTAB": [1, 2, 3], "AB": [2, 3]},
    )

    _record_student_opd_alignment(sample, student_tokenizer)

    assert sample.opd_student_alignment_status == "ok_recorded"
    assert sample.opd_student_alignment_complete is True
    assert sample.opd_student_alignment_validated is True
    assert sample.opd_student_alignment_source == "recorded_builder"
    assert sample.opd_generation_byte_evidence_attempted is True
    assert sample.opd_generation_byte_evidence_complete is False
    assert sample.opd_generation_byte_evidence_validated is False
    assert sample.opd_generation_byte_evidence_error == (
        "generation_byte_evidence_invalid: Generation token texts do not match canonical response text bytes."
    )
    assert sample.opd_student_alignment_error is None
    assert sample.opd_student_alignment_metadata == {
        "engine": "sglang",
        "detokenizer_protocol": "output_token_logprobs_text",
        "completion_reason": "length",
        "captured_response_token_count": 2,
        "response_token_count": 2,
        "response_byte_length": 2,
        "generation_token_text_count": 2,
        "evidence_kind": "recorded_builder_token_ids",
    }


def test_replay_summary_counts_invalid_generation_byte_evidence_even_after_builder_fallback(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "replay_debug_rollout_opd",
        "/mnt/ssd/lvzhihao/PostTrain/slime/scripts/replay_debug_rollout_opd.py",
    )
    replay_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(replay_module)

    summaries = [
        {
            "recorded_alignment_source": "recorded_builder",
            "recorded_alignment_complete": True,
            "recorded_alignment_validated": True,
            "generation_byte_evidence_attempted": True,
            "generation_byte_evidence_complete": False,
            "generation_byte_evidence_validated": False,
        },
        {
            "recorded_alignment_source": "recorded_builder",
            "recorded_alignment_complete": True,
            "recorded_alignment_validated": True,
            "generation_byte_evidence_attempted": False,
            "generation_byte_evidence_complete": None,
            "generation_byte_evidence_validated": None,
        },
    ]

    metrics = replay_module._summarize_generation_byte_evidence_metrics(summaries)

    assert metrics == {
        "generation_byte_evidence_sample_count": 1,
        "generation_byte_evidence_hit_count": 1,
        "generation_byte_evidence_hit_rate": pytest.approx(0.5),
        "generation_byte_evidence_incomplete_count": 1,
        "generation_byte_evidence_incomplete_rate": pytest.approx(1.0),
        "generation_byte_evidence_invalid_count": 1,
        "generation_byte_evidence_invalid_rate": pytest.approx(1.0),
    }


def test_get_student_alignment_evidence_returns_protocol_object():
    sample = Sample(
        opd_student_response_bytes=[65, 66],
        opd_student_token_byte_spans=[[0, 1], [1, 2]],
        response_length=2,
        opd_student_alignment_version=2,
        opd_student_alignment_source="generation_byte_evidence",
        opd_student_alignment_validated=True,
        opd_student_alignment_complete=True,
        opd_student_alignment_error=None,
        opd_student_alignment_metadata={"engine": "sglang", "captured_response_token_count": 2},
    )

    evidence = opd_utils.get_student_alignment_evidence(sample)

    assert evidence == {
        "version": 2,
        "source": "generation_byte_evidence",
        "response_bytes": [65, 66],
        "token_byte_spans": [[0, 1], [1, 2]],
        "response_token_count": 2,
        "complete": True,
        "validated": True,
        "error": None,
        "metadata": {"engine": "sglang", "captured_response_token_count": 2},
    }


def test_replay_summary_includes_generation_byte_evidence_metrics(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "replay_debug_rollout_opd",
        "/mnt/ssd/lvzhihao/PostTrain/slime/scripts/replay_debug_rollout_opd.py",
    )
    replay_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(replay_module)

    summaries = [
        {
            "recorded_alignment_source": "generation_byte_evidence",
            "recorded_alignment_complete": True,
            "recorded_alignment_validated": True,
        },
        {
            "recorded_alignment_source": "generation_byte_evidence",
            "recorded_alignment_complete": False,
            "recorded_alignment_validated": False,
        },
        {
            "recorded_alignment_source": "recorded_builder",
            "recorded_alignment_complete": True,
            "recorded_alignment_validated": True,
        },
    ]

    metrics = replay_module._summarize_generation_byte_evidence_metrics(summaries)

    assert metrics == {
        "generation_byte_evidence_sample_count": 2,
        "generation_byte_evidence_hit_count": 2,
        "generation_byte_evidence_hit_rate": pytest.approx(2 / 3),
        "generation_byte_evidence_incomplete_count": 1,
        "generation_byte_evidence_incomplete_rate": pytest.approx(0.5),
        "generation_byte_evidence_invalid_count": 1,
        "generation_byte_evidence_invalid_rate": pytest.approx(0.5),
    }


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


def test_apply_opd_byte_chunk_to_advantages_passes_student_alignment_evidence(monkeypatch):
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

    monkeypatch.setattr("slime.backends.megatron_utils.loss.get_cached_tokenizer", lambda _path: object())

    captured = {}

    def fake_compute_byte_chunk_reverse_kl(**kwargs):
        captured["student_alignment_evidence"] = kwargs["student_alignment_evidence"]
        return torch.tensor([-0.05, -0.05], dtype=torch.float32)

    monkeypatch.setattr("slime.backends.megatron_utils.loss.compute_byte_chunk_reverse_kl", fake_compute_byte_chunk_reverse_kl)

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
        "opd_student_alignment_version_list": [2],
        "opd_student_alignment_source_list": ["generation_byte_evidence"],
        "opd_student_alignment_complete_list": [True],
        "opd_student_alignment_validated_list": [True],
        "opd_student_alignment_error_list": [None],
        "opd_student_alignment_metadata_list": [{"engine": "sglang"}],
    }
    advantages = [torch.tensor([1.0, 1.0])]
    student_log_probs = [torch.tensor([-0.2, -0.3])]

    apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)

    assert captured["student_alignment_evidence"] == {
        "version": 2,
        "source": "generation_byte_evidence",
        "response_bytes": [65, 66],
        "token_byte_spans": [[0, 1], [1, 2]],
        "response_token_count": 2,
        "complete": True,
        "validated": True,
        "error": None,
        "metadata": {"engine": "sglang"},
    }
