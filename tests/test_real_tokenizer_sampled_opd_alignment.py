from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from slime.utils.opd_utils import (
    build_recorded_student_response_alignment,
    build_recorded_student_response_alignment_from_token_ids,
    build_token_byte_spans,
    clip_token_bytes_by_region,
    compute_byte_chunk_aligned_log_probs,
    encode_text,
    get_cached_tokenizer,
    validate_recorded_student_response_alignment,
)

STUDENT_MODEL_PATH = "/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill"
TEACHER_MODEL_PATH = "/mnt/hdd/Nanbeige4.1-3B"
SHORT_DATA_PATH = "/mnt/hdd/lvzhihao/output/OpenMathInstruct-2/correct/segments"
LONG_DATA_PATH = "/mnt/hdd/lvzhihao/output/OpenThoughts3-1.2M-math-distill-nanbeige4_1_3b/correct/segments"

_ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "analyze_opd_alignment",
    "/mnt/ssd/lvzhihao/PostTrain/slime/scripts/analyze_opd_alignment.py",
)
assert _ANALYZE_SPEC is not None and _ANALYZE_SPEC.loader is not None
_ANALYZE_MODULE = importlib.util.module_from_spec(_ANALYZE_SPEC)
sys.modules[_ANALYZE_SPEC.name] = _ANALYZE_MODULE
_ANALYZE_SPEC.loader.exec_module(_ANALYZE_MODULE)


def _require_local_path(path: str) -> None:
    if not Path(path).exists():
        pytest.skip(f"Required local path is missing: {path}")


def _collect_rows(path: str, limit: int) -> list[tuple[Path, int, dict]]:
    rows: list[tuple[Path, int, dict]] = []
    for file_path, row_index, row in _ANALYZE_MODULE.iter_rows(Path(path)):
        rows.append((file_path, row_index, row))
        if len(rows) >= limit:
            break
    if not rows:
        pytest.skip(f"No rows discovered under {path}")
    return rows


@pytest.mark.parametrize(
    ("dataset_name", "dataset_path", "sample_limit"),
    [
        ("short", SHORT_DATA_PATH, 128),
        ("long", LONG_DATA_PATH, 128),
    ],
)
def test_sampled_real_tokenizer_byte_alignment_matches_teacher_bytes(dataset_name, dataset_path, sample_limit):
    del dataset_name

    for required_path in (STUDENT_MODEL_PATH, TEACHER_MODEL_PATH, dataset_path):
        _require_local_path(required_path)

    student_tokenizer = get_cached_tokenizer(STUDENT_MODEL_PATH)
    teacher_tokenizer = get_cached_tokenizer(TEACHER_MODEL_PATH)

    for source_file, row_index, row in _collect_rows(dataset_path, sample_limit):
        del source_file, row_index
        messages = _ANALYZE_MODULE._normalize_messages(row)
        prompt_messages = messages[:-1]
        assert prompt_messages, "Expected at least one prompt message before the assistant response"

        raw_tools = row.get("tools")
        tools = _ANALYZE_MODULE._jsonable(raw_tools) if raw_tools is not None else None

        rendered_full_text = _ANALYZE_MODULE._apply_chat_template(
            student_tokenizer, messages, tokenize=False, tools=tools
        )
        rendered_prompt_text = _ANALYZE_MODULE._apply_prompt_template(
            student_tokenizer, prompt_messages, tokenize=False, tools=tools
        )
        assert rendered_full_text.startswith(rendered_prompt_text)
        rendered_response_text = rendered_full_text[len(rendered_prompt_text) :]

        student_full_token_ids = list(
            _ANALYZE_MODULE._apply_chat_template(student_tokenizer, messages, tokenize=True, tools=tools)
        )
        prompt_token_ids = list(
            _ANALYZE_MODULE._apply_prompt_template(student_tokenizer, prompt_messages, tokenize=True, tools=tools)
        )
        response_token_ids = student_full_token_ids[len(prompt_token_ids) :]
        assert response_token_ids, "Expected response tokens in sampled row"

        response_bytes, response_spans = build_recorded_student_response_alignment(
            student_tokenizer,
            full_token_ids=student_full_token_ids,
            prompt_token_count=len(prompt_token_ids),
            prompt_text=rendered_prompt_text,
            response_text=rendered_response_text,
            full_text=rendered_full_text,
        )
        token_bytes, normalized_spans = validate_recorded_student_response_alignment(
            response_text=rendered_response_text,
            response_token_count=len(response_token_ids),
            response_bytes=response_bytes,
            token_byte_spans=response_spans,
        )
        assert b"".join(token_bytes) == rendered_response_text.encode("utf-8")
        assert normalized_spans[-1][1] == len(response_bytes)

        teacher_token_ids = encode_text(teacher_tokenizer, rendered_full_text)
        teacher_token_bytes, teacher_token_spans = build_token_byte_spans(
            teacher_tokenizer, rendered_full_text, teacher_token_ids
        )
        teacher_response_bytes, teacher_response_indices = clip_token_bytes_by_region(
            teacher_token_bytes,
            teacher_token_spans,
            start_byte=len(rendered_prompt_text.encode("utf-8")),
        )

        assert b"".join(teacher_response_bytes) == response_bytes
        assert teacher_response_indices, "Expected teacher response tokens in sampled row"

        student_chunk_log_probs, teacher_chunk_log_probs = compute_byte_chunk_aligned_log_probs(
            full_text=rendered_full_text,
            prompt_text=rendered_prompt_text,
            prompt_token_count=len(prompt_token_ids),
            student_token_ids=student_full_token_ids,
            response_token_count=len(response_token_ids),
            student_log_probs=torch.zeros(len(response_token_ids), dtype=torch.float32),
            teacher_log_probs=torch.zeros(len(teacher_response_indices), dtype=torch.float32),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            recorded_student_response_bytes=response_bytes,
            recorded_student_token_byte_spans=response_spans,
            allow_sequence_fallback=False,
        )

        assert student_chunk_log_probs.shape[0] == len(response_token_ids)
        assert teacher_chunk_log_probs.shape[0] == len(response_token_ids)


@pytest.mark.parametrize(
    ("dataset_name", "dataset_path", "sample_limit"),
    [
        ("short", SHORT_DATA_PATH, 128),
        ("long", LONG_DATA_PATH, 128),
    ],
)
def test_sampled_real_tokenizer_response_token_ids_reconstruct_response_bytes(dataset_name, dataset_path, sample_limit):
    del dataset_name

    for required_path in (STUDENT_MODEL_PATH, dataset_path):
        _require_local_path(required_path)

    student_tokenizer = get_cached_tokenizer(STUDENT_MODEL_PATH)

    for source_file, row_index, row in _collect_rows(dataset_path, sample_limit):
        del source_file, row_index
        messages = _ANALYZE_MODULE._normalize_messages(row)
        prompt_messages = messages[:-1]
        assert prompt_messages, "Expected at least one prompt message before the assistant response"

        raw_tools = row.get("tools")
        tools = _ANALYZE_MODULE._jsonable(raw_tools) if raw_tools is not None else None

        rendered_full_text = _ANALYZE_MODULE._apply_chat_template(
            student_tokenizer, messages, tokenize=False, tools=tools
        )
        rendered_prompt_text = _ANALYZE_MODULE._apply_prompt_template(
            student_tokenizer, prompt_messages, tokenize=False, tools=tools
        )
        rendered_response_text = rendered_full_text[len(rendered_prompt_text) :]

        student_full_token_ids = list(
            _ANALYZE_MODULE._apply_chat_template(student_tokenizer, messages, tokenize=True, tools=tools)
        )
        prompt_token_ids = list(
            _ANALYZE_MODULE._apply_prompt_template(student_tokenizer, prompt_messages, tokenize=True, tools=tools)
        )
        response_token_ids = student_full_token_ids[len(prompt_token_ids) :]
        assert response_token_ids, "Expected response tokens in sampled row"

        response_bytes, response_spans = build_recorded_student_response_alignment_from_token_ids(
            tokenizer=student_tokenizer,
            response_text=rendered_response_text,
            response_token_ids=response_token_ids,
        )

        token_bytes, normalized_spans = validate_recorded_student_response_alignment(
            response_text=rendered_response_text,
            response_token_count=len(response_token_ids),
            response_bytes=response_bytes,
            token_byte_spans=response_spans,
        )

        assert b"".join(token_bytes) == rendered_response_text.encode("utf-8")
        assert normalized_spans[-1][1] == len(response_bytes)
