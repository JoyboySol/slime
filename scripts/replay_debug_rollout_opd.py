#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slime.rollout.on_policy_distillation import compute_teacher_log_probs_for_sample
from slime.utils.opd_utils import (
    build_token_byte_spans,
    clip_token_bytes_by_region,
    compute_byte_chunk_aligned_log_probs,
    encode_text,
    get_cached_tokenizer,
)
from slime.utils.types import Sample


def _canonical_texts(sample: Sample) -> tuple[str, str, str]:
    prompt_text = sample.opd_prompt_text if sample.opd_prompt_text is not None else sample.prompt
    response_text = sample.opd_response_text if sample.opd_response_text is not None else sample.response
    full_text = sample.opd_full_text if sample.opd_full_text is not None else f"{prompt_text}{response_text}"
    return prompt_text, response_text, full_text


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if not hasattr(args, "opd_teacher_hf_checkpoint"):
        args.opd_teacher_hf_checkpoint = args.teacher_hf_checkpoint
    return args


def _summarize_sample(sample: Sample, args: argparse.Namespace) -> dict:
    args = _normalize_args(args)
    try:
        student_tokenizer = get_cached_tokenizer(args.hf_checkpoint)
        teacher_tokenizer = get_cached_tokenizer(args.teacher_hf_checkpoint)
        prompt_text, _response_text, full_text = _canonical_texts(sample)
        prompt_byte_length = len(prompt_text.encode("utf-8"))

        reward_entries = sample.reward["meta_info"]["input_token_logprobs"][1:]
        teacher_log_probs = compute_teacher_log_probs_for_sample(args, sample)

        teacher_token_ids = encode_text(teacher_tokenizer, full_text)
        teacher_token_bytes, teacher_token_spans = build_token_byte_spans(teacher_tokenizer, full_text, teacher_token_ids)
        _teacher_response_bytes, teacher_response_indices = clip_token_bytes_by_region(
            teacher_token_bytes,
            teacher_token_spans,
            start_byte=prompt_byte_length,
        )

        summary = {
            "index": sample.index,
            "group_index": sample.group_index,
            "response_length": sample.response_length,
            "rollout_log_prob_count": len(sample.rollout_log_probs or []),
            "teacher_reward_token_count": len(reward_entries),
            "teacher_selected_response_count": len(teacher_response_indices),
            "teacher_log_prob_count": int(teacher_log_probs.numel()),
            "teacher_full_token_count": len(teacher_token_ids),
            "prompt_byte_length": prompt_byte_length,
            "prompt_char_length": len(prompt_text),
            "response_char_length": len(sample.response or ""),
        }

        if sample.rollout_log_probs is None:
            summary["byte_chunk_alignment"] = "skipped:no_rollout_log_probs"
            return summary

        student_log_probs = torch.tensor(sample.rollout_log_probs, dtype=torch.float32)
        student_chunk_log_probs, teacher_chunk_log_probs = compute_byte_chunk_aligned_log_probs(
            full_text=full_text,
            prompt_text=prompt_text,
            prompt_token_count=len(sample.tokens) - sample.response_length,
            student_token_ids=sample.tokens,
            response_token_count=sample.response_length,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            allow_sequence_fallback=False,
        )
        summary["byte_chunk_alignment"] = "ok"
        summary["student_chunk_mean"] = float(student_chunk_log_probs.mean().item())
        summary["teacher_chunk_mean"] = float(teacher_chunk_log_probs.mean().item())
    except Exception as exc:  # noqa: BLE001
        prompt_text = sample.opd_prompt_text if sample.opd_prompt_text is not None else sample.prompt
        summary = {
            "index": sample.index,
            "group_index": sample.group_index,
            "response_length": sample.response_length,
            "rollout_log_prob_count": len(sample.rollout_log_probs or []),
            "byte_chunk_alignment": f"failed:{type(exc).__name__}",
            "error": str(exc),
            "prompt_preview": (prompt_text or "")[: args.preview_chars],
            "response_preview": (sample.response or "")[: args.preview_chars],
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay byte_chunk OPD on saved debug rollout samples.")
    parser.add_argument("--debug-rollout-data", required=True, help="Path to rollout_*.pt saved by debug_rollout.")
    parser.add_argument("--hf-checkpoint", required=True, help="Student HF checkpoint.")
    parser.add_argument("--teacher-hf-checkpoint", dest="teacher_hf_checkpoint", required=True, help="Teacher HF checkpoint.")
    parser.add_argument("--sample-index", type=int, default=None, help="Only replay one sample index.")
    parser.add_argument("--preview-chars", type=int, default=240, help="Characters to show for prompt/response previews.")
    parser.add_argument("--dump-failing-sample", default=None, help="Optional path to write the first failing sample as JSON.")
    parser.add_argument("--reward-key", default=None, help="Optional nested reward key.")
    args = parser.parse_args()
    args = _normalize_args(args)

    payload = torch.load(args.debug_rollout_data, map_location="cpu")
    samples = [Sample.from_dict(sample_dict) for sample_dict in payload["samples"]]
    if args.sample_index is not None:
        samples = [samples[args.sample_index]]

    first_failure = None
    for sample in samples:
        summary = _summarize_sample(sample, args)
        print(json.dumps(summary, ensure_ascii=False))
        if first_failure is None and str(summary["byte_chunk_alignment"]).startswith("failed:"):
            first_failure = sample

    if first_failure is not None and args.dump_failing_sample is not None:
        output_path = Path(args.dump_failing_sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(first_failure.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
