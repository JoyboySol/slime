import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_opd_alignment.py"
    module_name = "test_analyze_opd_alignment_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ChatTemplateTokenizer:
    def __init__(self):
        self.token_map = {
            100: "<|im_start|>",
            101: " user",
            102: "\n",
            103: "O",
            104: "K",
        }
        self.decode_map = {
            (100,): "<|im_start|>",
            (100, 101): "<|im_start|> user",
            (100, 101, 102): "<|im_start|> user\n",
            (100, 101, 102, 103): "<|im_start|>\nO",
            (100, 101, 102, 103, 104): "<|im_start|> user\nOK",
            (103,): "O",
            (103, 104): "OK",
        }
        self.offsets_map = {
            "<|im_start|> user\n": [(0, 12), (12, 17), (17, 18)],
            "<|im_start|> user\nOK": [(0, 12), (12, 17), (17, 18), (18, 19), (19, 20)],
        }

    def apply_chat_template(self, messages, *, tokenize, return_dict=False, add_generation_prompt=False, tools=None):
        del return_dict, tools
        if tokenize:
            if add_generation_prompt:
                return [100, 101, 102]
            return [100, 101, 102, 103, 104]
        if add_generation_prompt:
            return "<|im_start|> user\n"
        return "<|im_start|> user\nOK"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        mapping = {
            "<|im_start|> user\n": [100, 101, 102],
            "OK": [103, 104],
            "<|im_start|> user\nOK": [100, 101, 102, 103, 104],
        }
        return list(mapping[text])

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return self.decode_map[tuple(token_ids)]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        token_ids = self.encode(text)
        result = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = list(self.offsets_map[text])
        return result


def test_analyze_row_requires_contextual_suffix_reconstruction():
    module = load_module()
    row = {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "OK"}]}

    result = module.analyze_row(
        row,
        source_file=Path("sample.jsonl"),
        row_index=0,
        student_tokenizer=ChatTemplateTokenizer(),
        teacher_tokenizer=ChatTemplateTokenizer(),
    )

    assert result.status == "ok"
    assert result.student_reconstruction_mode == "recorded_builder"


def test_analyze_row_prefers_recorded_payload_when_present():
    module = load_module()
    row = {
        "messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "OK"}],
        "opd_student_response_bytes": [79, 75],
        "opd_student_token_byte_spans": [[0, 1], [1, 2]],
        "opd_student_alignment_source": "recorded_builder",
        "opd_student_alignment_validated": True,
        "opd_student_alignment_status": "ok_recorded",
    }

    result = module.analyze_row(
        row,
        source_file=Path("sample.jsonl"),
        row_index=0,
        student_tokenizer=ChatTemplateTokenizer(),
        teacher_tokenizer=ChatTemplateTokenizer(),
    )

    assert result.status == "ok"
    assert result.student_reconstruction_mode == "recorded_payload"
    assert result.recorded_alignment_source == "recorded_builder"
    assert result.recorded_alignment_validated is True
    assert result.recorded_alignment_status == "ok_recorded"


@pytest.mark.unit
def test_results_to_dataframe_includes_chunk_and_ratio_metrics():
    module = load_module()
    results = [
        module.AnalysisResult(
            sample_id="sample-1",
            source_file="a.jsonl",
            row_index=0,
            status="ok",
            error=None,
            prompt_token_count=10,
            response_token_count=4,
            teacher_response_token_count=6,
            aligned_chunk_count=2,
            one_to_one_aligned_token_count=1,
            student_response_bytes=12,
            teacher_response_bytes=12,
            student_reconstruction_mode="contextual_suffix",
            render_full_matches_decoded=True,
            render_prompt_matches_decoded=True,
            first_diff_byte=None,
            first_diff_student_preview=None,
            first_diff_teacher_preview=None,
            response_preview="'abc'",
        ),
        module.AnalysisResult(
            sample_id="sample-2",
            source_file="b.jsonl",
            row_index=1,
            status="response_bytes_mismatch",
            error="bytes mismatch",
            prompt_token_count=10,
            response_token_count=5,
            teacher_response_token_count=8,
            aligned_chunk_count=None,
            one_to_one_aligned_token_count=None,
            student_response_bytes=18,
            teacher_response_bytes=20,
            student_reconstruction_mode="response_only_fallback",
            render_full_matches_decoded=True,
            render_prompt_matches_decoded=True,
            first_diff_byte=5,
            first_diff_student_preview="'foo'",
            first_diff_teacher_preview="'bar'",
            response_preview="'def'",
        ),
    ]

    frame = module.results_to_frame(results)

    assert list(frame["sample_id"]) == ["sample-1", "sample-2"]
    assert frame.loc[0, "student_tokens_per_chunk"] == pytest.approx(2.0)
    assert frame.loc[0, "teacher_tokens_per_chunk"] == pytest.approx(3.0)
    assert frame.loc[0, "teacher_to_student_token_ratio"] == pytest.approx(1.5)
    assert frame.loc[0, "student_one_to_one_aligned_token_ratio"] == pytest.approx(0.25)
    assert frame.loc[0, "teacher_one_to_one_aligned_token_ratio"] == pytest.approx(1.0 / 6.0)
    assert pd.isna(frame.loc[1, "student_tokens_per_chunk"])
    assert pd.isna(frame.loc[1, "chunk_to_student_token_ratio"])


@pytest.mark.unit
def test_build_report_tables_summarizes_ok_samples_only():
    module = load_module()
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "status": "ok",
                "student_response_token_count": 4,
                "teacher_response_token_count": 6,
                "aligned_chunk_count": 2,
                "one_to_one_aligned_token_count": 1,
                "student_tokens_per_chunk": 2.0,
                "teacher_tokens_per_chunk": 3.0,
                "teacher_to_student_token_ratio": 1.5,
                "chunk_to_student_token_ratio": 0.5,
                "student_one_to_one_aligned_token_ratio": 0.25,
                "teacher_one_to_one_aligned_token_ratio": 1.0 / 6.0,
            },
            {
                "sample_id": "sample-2",
                "status": "ok",
                "student_response_token_count": 8,
                "teacher_response_token_count": 10,
                "aligned_chunk_count": 4,
                "one_to_one_aligned_token_count": 2,
                "student_tokens_per_chunk": 2.0,
                "teacher_tokens_per_chunk": 2.5,
                "teacher_to_student_token_ratio": 1.25,
                "chunk_to_student_token_ratio": 0.5,
                "student_one_to_one_aligned_token_ratio": 0.25,
                "teacher_one_to_one_aligned_token_ratio": 0.2,
            },
            {
                "sample_id": "sample-3",
                "status": "sample_error",
                "student_response_token_count": 100,
                "teacher_response_token_count": 100,
                "aligned_chunk_count": None,
                "one_to_one_aligned_token_count": None,
                "student_tokens_per_chunk": None,
                "teacher_tokens_per_chunk": None,
                "teacher_to_student_token_ratio": 1.0,
                "chunk_to_student_token_ratio": None,
                "student_one_to_one_aligned_token_ratio": None,
                "teacher_one_to_one_aligned_token_ratio": None,
            },
        ]
    )

    summary_frame, status_frame = module.build_report_tables(frame)

    summary = dict(zip(summary_frame["metric"], summary_frame["value"], strict=False))
    statuses = dict(zip(status_frame["status"], status_frame["count"], strict=False))

    assert summary["ok_sample_count"] == 2
    assert summary["student_response_token_count_mean"] == pytest.approx(6.0)
    assert summary["teacher_response_token_count_p95"] == pytest.approx(9.8)
    assert summary["aligned_chunk_count_sum"] == pytest.approx(6.0)
    assert summary["teacher_to_student_token_ratio_mean"] == pytest.approx(1.375)
    assert summary["one_to_one_aligned_token_count_mean"] == pytest.approx(1.5)
    assert summary["student_one_to_one_aligned_token_ratio_mean"] == pytest.approx(0.25)
    assert summary["teacher_one_to_one_aligned_token_ratio_mean"] == pytest.approx((1.0 / 6.0 + 0.2) / 2)
    assert statuses == {"ok": 2, "sample_error": 1}


@pytest.mark.unit
def test_write_report_outputs_tables_and_pngs(tmp_path):
    module = load_module()
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-1",
                "source_file": "a.jsonl",
                "row_index": 0,
                "status": "ok",
                "student_response_token_count": 4,
                "teacher_response_token_count": 6,
                "aligned_chunk_count": 2,
                "one_to_one_aligned_token_count": 1,
                "student_tokens_per_chunk": 2.0,
                "teacher_tokens_per_chunk": 3.0,
                "teacher_to_student_token_ratio": 1.5,
                "chunk_to_student_token_ratio": 0.5,
                "student_one_to_one_aligned_token_ratio": 0.25,
                "teacher_one_to_one_aligned_token_ratio": 1.0 / 6.0,
                "student_response_bytes": 12,
                "teacher_response_bytes": 12,
            },
            {
                "sample_id": "sample-2",
                "source_file": "b.jsonl",
                "row_index": 1,
                "status": "ok",
                "student_response_token_count": 8,
                "teacher_response_token_count": 9,
                "aligned_chunk_count": 4,
                "one_to_one_aligned_token_count": 2,
                "student_tokens_per_chunk": 2.0,
                "teacher_tokens_per_chunk": 2.25,
                "teacher_to_student_token_ratio": 1.125,
                "chunk_to_student_token_ratio": 0.5,
                "student_one_to_one_aligned_token_ratio": 0.25,
                "teacher_one_to_one_aligned_token_ratio": 2.0 / 9.0,
                "student_response_bytes": 22,
                "teacher_response_bytes": 22,
            },
        ]
    )

    module.write_report(frame, tmp_path)

    assert (tmp_path / "sample_metrics.csv").exists()
    assert (tmp_path / "summary_metrics.csv").exists()
    assert (tmp_path / "status_counts.csv").exists()
    assert (tmp_path / "token_count_overview.png").exists()
    assert (tmp_path / "token_alignment_scatter.png").exists()
    assert (tmp_path / "chunk_ratio_distributions.png").exists()
