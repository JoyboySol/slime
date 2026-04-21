#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from slime.rollout.on_policy_distillation import compute_teacher_log_probs_for_sample
from slime.utils.opd_metric_utils import summarize_opd_alignment
from slime.utils.opd_utils import (
    build_token_byte_spans,
    clip_token_bytes_by_region,
    encode_text,
    get_generation_byte_evidence_observability,
    get_cached_tokenizer,
    prepare_byte_chunk_training_entry,
    get_student_alignment_evidence,
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
        entry = prepare_byte_chunk_training_entry(
            full_text=full_text,
            prompt_text=prompt_text,
            student_token_ids=sample.tokens,
            response_token_count=sample.response_length,
            student_log_probs=student_log_probs,
            teacher_log_probs=teacher_log_probs,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            student_alignment_evidence=get_student_alignment_evidence(sample),
            recorded_student_response_bytes=sample.opd_student_response_bytes,
            recorded_student_token_byte_spans=sample.opd_student_token_byte_spans,
        )
        summary["byte_chunk_alignment"] = "ok"
        summary["student_alignment_source"] = (
            "recorded_payload" if sample.opd_student_response_bytes is not None else "legacy_reconstruction"
        )
        summary["recorded_alignment_evidence"] = get_student_alignment_evidence(sample)
        summary["recorded_alignment_source"] = sample.opd_student_alignment_source
        summary["recorded_alignment_version"] = sample.opd_student_alignment_version
        summary["recorded_alignment_complete"] = sample.opd_student_alignment_complete
        summary["recorded_alignment_validated"] = sample.opd_student_alignment_validated
        summary["recorded_alignment_status"] = sample.opd_student_alignment_status
        generation_byte_evidence = get_generation_byte_evidence_observability(sample) or {}
        summary["generation_byte_evidence_attempted"] = generation_byte_evidence.get("attempted")
        summary["generation_byte_evidence_complete"] = generation_byte_evidence.get("complete")
        summary["generation_byte_evidence_validated"] = generation_byte_evidence.get("validated")
        summary["generation_byte_evidence_error"] = generation_byte_evidence.get("error")
        summary["student_chunk_mean"] = float(entry["student_chunk_log_probs"].mean().item())
        summary["teacher_chunk_mean"] = float(entry["teacher_chunk_log_probs"].mean().item())
    except Exception as exc:  # noqa: BLE001
        prompt_text = sample.opd_prompt_text if sample.opd_prompt_text is not None else sample.prompt
        summary = {
            "index": sample.index,
            "group_index": sample.group_index,
            "response_length": sample.response_length,
            "rollout_log_prob_count": len(sample.rollout_log_probs or []),
            "byte_chunk_alignment": f"failed:{type(exc).__name__}",
            "error": str(exc),
            "recorded_alignment_evidence": get_student_alignment_evidence(sample),
            "recorded_alignment_source": sample.opd_student_alignment_source,
            "recorded_alignment_version": sample.opd_student_alignment_version,
            "recorded_alignment_complete": sample.opd_student_alignment_complete,
            "recorded_alignment_validated": sample.opd_student_alignment_validated,
            "recorded_alignment_status": sample.opd_student_alignment_status,
            "generation_byte_evidence_attempted": sample.opd_generation_byte_evidence_attempted,
            "generation_byte_evidence_complete": sample.opd_generation_byte_evidence_complete,
            "generation_byte_evidence_validated": sample.opd_generation_byte_evidence_validated,
            "generation_byte_evidence_error": sample.opd_generation_byte_evidence_error,
            "prompt_preview": (prompt_text or "")[: args.preview_chars],
            "response_preview": (sample.response or "")[: args.preview_chars],
        }

    return summary


def _summarize_generation_byte_evidence_metrics(summaries: list[dict]) -> dict[str, float]:
    metrics, _summary_text = summarize_opd_alignment(
        status_values=None,
        complete_values=[summary.get("recorded_alignment_complete") for summary in summaries],
        source_values=[summary.get("recorded_alignment_source") for summary in summaries],
        validated_values=[summary.get("recorded_alignment_validated") for summary in summaries],
        error_values=None,
        generation_attempted_values=[summary.get("generation_byte_evidence_attempted") for summary in summaries],
        generation_complete_values=[summary.get("generation_byte_evidence_complete") for summary in summaries],
        generation_validated_values=[summary.get("generation_byte_evidence_validated") for summary in summaries],
    )
    return {
        "generation_byte_evidence_sample_count": int(metrics.get("opd_generation_byte_evidence_sample_count", 0.0)),
        "generation_byte_evidence_hit_count": int(metrics["opd_generation_byte_evidence_hit_count"]),
        "generation_byte_evidence_hit_rate": metrics["opd_generation_byte_evidence_hit_rate"],
        "generation_byte_evidence_incomplete_count": int(metrics["opd_generation_byte_evidence_incomplete_count"]),
        "generation_byte_evidence_incomplete_rate": metrics["opd_generation_byte_evidence_incomplete_rate"],
        "generation_byte_evidence_invalid_count": int(metrics["opd_generation_byte_evidence_invalid_count"]),
        "generation_byte_evidence_invalid_rate": metrics["opd_generation_byte_evidence_invalid_rate"],
    }


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
    summaries = []
    for sample in samples:
        summary = _summarize_sample(sample, args)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
        if first_failure is None and str(summary["byte_chunk_alignment"]).startswith("failed:"):
            first_failure = sample

    if summaries:
        print(
            json.dumps(
                {
                    "summary_type": "generation_byte_evidence_metrics",
                    **_summarize_generation_byte_evidence_metrics(summaries),
                },
                ensure_ascii=False,
            )
        )

    if first_failure is not None and args.dump_failing_sample is not None:
        output_path = Path(args.dump_failing_sample)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(first_failure.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
