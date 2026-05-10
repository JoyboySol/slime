from __future__ import annotations

from functools import lru_cache
import logging
import re
from typing import Any

import torch
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from slime.utils.processing_utils import load_tokenizer

logger = logging.getLogger(__name__)
_BYTE_LEVEL_CHAR_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}
_HEX_BYTE_TOKEN_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_UTF8_REPLACEMENT_BYTES = b"\xef\xbf\xbd"


@lru_cache(maxsize=8)
def get_cached_tokenizer(name_or_path: str):
    return load_tokenizer(name_or_path, trust_remote_code=True)


def decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    kwargs = {"skip_special_tokens": False}
    try:
        return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False, **kwargs)
    except TypeError:
        return tokenizer.decode(token_ids, **kwargs)


def encode_text(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _coerce_bytes(value: bytes | bytearray | list[int] | tuple[int, ...]) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    return bytes(value)


def _normalize_token_byte_spans(
    token_byte_spans: list[tuple[int, int]] | list[list[int]] | None,
) -> list[list[int]] | None:
    if token_byte_spans is None:
        return None
    return [[int(start), int(end)] for start, end in token_byte_spans]


def get_student_alignment_evidence(sample_or_row: Any) -> dict[str, Any] | None:
    response_bytes = getattr(sample_or_row, "opd_student_response_bytes", None)
    if response_bytes is None and isinstance(sample_or_row, dict):
        response_bytes = sample_or_row.get("opd_student_response_bytes")

    token_byte_spans = getattr(sample_or_row, "opd_student_token_byte_spans", None)
    if token_byte_spans is None and isinstance(sample_or_row, dict):
        token_byte_spans = sample_or_row.get("opd_student_token_byte_spans")

    source = getattr(sample_or_row, "opd_student_alignment_source", None)
    if source is None and isinstance(sample_or_row, dict):
        source = sample_or_row.get("opd_student_alignment_source")

    version = getattr(sample_or_row, "opd_student_alignment_version", None)
    if version is None and isinstance(sample_or_row, dict):
        version = sample_or_row.get("opd_student_alignment_version")

    validated = getattr(sample_or_row, "opd_student_alignment_validated", None)
    if validated is None and isinstance(sample_or_row, dict):
        validated = sample_or_row.get("opd_student_alignment_validated")

    complete = getattr(sample_or_row, "opd_student_alignment_complete", None)
    if complete is None and isinstance(sample_or_row, dict):
        complete = sample_or_row.get("opd_student_alignment_complete")

    error = getattr(sample_or_row, "opd_student_alignment_error", None)
    if error is None and isinstance(sample_or_row, dict):
        error = sample_or_row.get("opd_student_alignment_error")

    metadata = getattr(sample_or_row, "opd_student_alignment_metadata", None)
    if metadata is None and isinstance(sample_or_row, dict):
        metadata = sample_or_row.get("opd_student_alignment_metadata")

    response_token_count = getattr(sample_or_row, "response_length", None)
    if response_token_count is None and isinstance(sample_or_row, dict):
        response_token_count = sample_or_row.get("response_length")
    if metadata is not None and response_token_count is None:
        response_token_count = metadata.get("response_token_count")

    if (
        response_bytes is None
        and token_byte_spans is None
        and source is None
        and version is None
        and complete is None
        and validated is None
        and error is None
        and metadata is None
    ):
        return None

    return {
        "version": version,
        "source": source,
        "response_bytes": list(_coerce_bytes(response_bytes)) if response_bytes is not None else None,
        "token_byte_spans": _normalize_token_byte_spans(token_byte_spans),
        "response_token_count": response_token_count,
        "complete": complete,
        "validated": validated,
        "error": error,
        "metadata": dict(metadata) if metadata is not None else None,
    }


def get_generation_byte_evidence_observability(sample_or_row: Any) -> dict[str, Any] | None:
    attempted = getattr(sample_or_row, "opd_generation_byte_evidence_attempted", None)
    if attempted is None and isinstance(sample_or_row, dict):
        attempted = sample_or_row.get("opd_generation_byte_evidence_attempted")

    complete = getattr(sample_or_row, "opd_generation_byte_evidence_complete", None)
    if complete is None and isinstance(sample_or_row, dict):
        complete = sample_or_row.get("opd_generation_byte_evidence_complete")

    validated = getattr(sample_or_row, "opd_generation_byte_evidence_validated", None)
    if validated is None and isinstance(sample_or_row, dict):
        validated = sample_or_row.get("opd_generation_byte_evidence_validated")

    error = getattr(sample_or_row, "opd_generation_byte_evidence_error", None)
    if error is None and isinstance(sample_or_row, dict):
        error = sample_or_row.get("opd_generation_byte_evidence_error")

    metadata = getattr(sample_or_row, "opd_generation_byte_evidence_metadata", None)
    if metadata is None and isinstance(sample_or_row, dict):
        metadata = sample_or_row.get("opd_generation_byte_evidence_metadata")

    if attempted is None and complete is None and validated is None and error is None and metadata is None:
        return None

    return {
        "attempted": attempted,
        "complete": complete,
        "validated": validated,
        "error": error,
        "metadata": dict(metadata) if metadata is not None else None,
    }


def get_student_alignment_evidence_from_rollout_data(rollout_data: dict[str, Any], index: int) -> dict[str, Any] | None:
    def _get_list_value(key: str):
        values = rollout_data.get(key)
        if values is None:
            return None
        return values[index]

    evidence = {
        "version": _get_list_value("opd_student_alignment_version_list"),
        "source": _get_list_value("opd_student_alignment_source_list"),
        "response_bytes": _get_list_value("opd_student_response_bytes_list"),
        "token_byte_spans": _get_list_value("opd_student_token_byte_spans_list"),
        "response_token_count": rollout_data["response_lengths"][index],
        "complete": _get_list_value("opd_student_alignment_complete_list"),
        "validated": _get_list_value("opd_student_alignment_validated_list"),
        "error": _get_list_value("opd_student_alignment_error_list"),
        "metadata": _get_list_value("opd_student_alignment_metadata_list"),
    }

    if all(value is None for value in evidence.values()):
        return None
    return evidence


def get_generation_byte_evidence_observability_from_rollout_data(
    rollout_data: dict[str, Any], index: int
) -> dict[str, Any] | None:
    def _get_list_value(key: str):
        values = rollout_data.get(key)
        if values is None:
            return None
        return values[index]

    evidence = {
        "attempted": _get_list_value("opd_generation_byte_evidence_attempted_list"),
        "complete": _get_list_value("opd_generation_byte_evidence_complete_list"),
        "validated": _get_list_value("opd_generation_byte_evidence_validated_list"),
        "error": _get_list_value("opd_generation_byte_evidence_error_list"),
        "metadata": _get_list_value("opd_generation_byte_evidence_metadata_list"),
    }

    if all(value is None for value in evidence.values()):
        return None
    return evidence


def _build_char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def _build_token_byte_spans_via_prefix_decode(tokenizer, token_ids: list[int]) -> tuple[list[bytes], list[tuple[int, int]]]:
    token_bytes = []
    token_byte_spans = []
    prefix_text = ""
    byte_offset = 0
    for index in range(len(token_ids)):
        decoded_prefix = decode_token_ids(tokenizer, token_ids[: index + 1])
        if not decoded_prefix.startswith(prefix_text):
            raise ValueError("Tokenizer decode is not prefix-consistent; cannot reconstruct byte spans.")
        token_text = decoded_prefix[len(prefix_text) :]
        token_text_bytes = token_text.encode("utf-8")
        token_bytes.append(token_text_bytes)
        token_byte_spans.append((byte_offset, byte_offset + len(token_text_bytes)))
        prefix_text = decoded_prefix
        byte_offset += len(token_text_bytes)
    return token_bytes, token_byte_spans


def _token_piece_to_bytes(token_piece: str) -> bytes:
    raw = bytearray()
    for char in token_piece:
        if char == "▁":
            raw.append(0x20)
            continue
        if char in _BYTE_LEVEL_CHAR_TO_BYTE:
            raw.append(_BYTE_LEVEL_CHAR_TO_BYTE[char])
        else:
            raw.extend(char.encode("utf-8"))
    return bytes(raw)


def _token_piece_candidates(token_piece: str) -> list[bytes]:
    candidates: list[bytes] = []

    hex_match = _HEX_BYTE_TOKEN_RE.match(token_piece)
    if hex_match is not None:
        candidates.append(bytes([int(hex_match.group(1), 16)]))

    # SentencePiece-like token surfaces usually want direct UTF-8 bytes, with ▁
    # representing a visible word-boundary space.
    candidates.append(token_piece.replace("▁", " ").encode("utf-8"))
    if token_piece == "▁":
        # Around special-token boundaries YuLan can surface a standalone ▁ that
        # does not materialize as a literal space in the rendered text.
        candidates.append(b"")

    # Preserve compatibility with byte-level token surfaces such as GPT-2 style
    # unicode-mapped bytes and legacy fallbacks.
    candidates.append(_token_piece_to_bytes(token_piece))

    deduped: list[bytes] = []
    seen: set[bytes] = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


def _hex_byte_token_value(token_piece: str) -> int | None:
    hex_match = _HEX_BYTE_TOKEN_RE.match(token_piece)
    if hex_match is None:
        return None
    return int(hex_match.group(1), 16)


def _is_invalid_standalone_utf8_byte(byte_value: int) -> bool:
    return 0x80 <= byte_value <= 0xC1 or byte_value >= 0xF5


def _replacement_candidate_for_isolated_invalid_byte_token(
    token_pieces: list[str], index: int
) -> bytes | None:
    byte_value = _hex_byte_token_value(token_pieces[index])
    if byte_value is None or not _is_invalid_standalone_utf8_byte(byte_value):
        return None
    if index > 0 and _hex_byte_token_value(token_pieces[index - 1]) is not None:
        return None
    if index + 1 < len(token_pieces) and _hex_byte_token_value(token_pieces[index + 1]) is not None:
        return None
    return _UTF8_REPLACEMENT_BYTES


def _build_token_byte_spans_via_token_strings(
    tokenizer, text: str, token_ids: list[int]
) -> tuple[list[bytes], list[tuple[int, int]]]:
    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        raise ValueError("Tokenizer does not expose convert_ids_to_tokens.")

    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)
    if len(token_pieces) != len(token_ids):
        raise ValueError("Tokenizer convert_ids_to_tokens returned unexpected token count.")

    token_bytes = []
    token_byte_spans = []
    target_bytes = text.encode("utf-8")
    byte_offset = 0
    for index, token_piece in enumerate(token_pieces):
        remaining = target_bytes[byte_offset:]
        candidates = _token_piece_candidates(token_piece)
        replacement_candidate = _replacement_candidate_for_isolated_invalid_byte_token(token_pieces, index)
        if replacement_candidate is not None:
            candidates.append(replacement_candidate)
        matching_candidates = [
            candidate for candidate in candidates if remaining.startswith(candidate)
        ]
        if not matching_candidates:
            raise ValueError(
                "Tokenizer token strings do not reconstruct the source text. "
                f"token_index={index} token_piece={token_piece!r} byte_offset={byte_offset}"
            )

        piece_bytes = matching_candidates[0]
        token_bytes.append(piece_bytes)
        token_byte_spans.append((byte_offset, byte_offset + len(piece_bytes)))
        byte_offset += len(piece_bytes)

    if b"".join(token_bytes) != target_bytes:
        raise ValueError("Tokenizer token strings do not reconstruct the source text.")
    return token_bytes, token_byte_spans


def _build_token_byte_spans(
    tokenizer,
    text: str,
    token_ids: list[int],
    *,
    allow_prefix_fallback: bool,
) -> tuple[list[bytes], list[tuple[int, int]]]:
    try:
        tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if tokenized["input_ids"] != token_ids:
            raise ValueError(
                "Token ids do not match tokenizer encoding for response text: "
                f"expected={token_ids}, got={tokenized['input_ids']}"
            )
        if len(tokenized["offset_mapping"]) != len(token_ids):
            raise ValueError("Tokenizer offset mapping count does not match token count.")

        char_to_byte_offsets = _build_char_to_byte_offsets(text)
        token_bytes = []
        token_byte_spans = []
        offset_mapping = list(tokenized["offset_mapping"])
        previous_start_char = 0
        previous_end_char = 0
        for index, (start_char, end_char) in enumerate(offset_mapping):
            if index > 0 and start_char < previous_start_char:
                raise ValueError("Tokenizer offset mapping is not monotonic.")
            start_byte = char_to_byte_offsets[start_char]
            end_byte = char_to_byte_offsets[end_char]
            token_text_bytes = text[start_char:end_char].encode("utf-8")
            next_same_span = index + 1 < len(offset_mapping) and offset_mapping[index + 1] == (start_char, end_char)
            prev_same_span = index > 0 and offset_mapping[index - 1] == (start_char, end_char)
            if token_text_bytes and (prev_same_span or next_same_span):
                if next_same_span:
                    token_bytes.append(b"")
                    token_byte_spans.append((start_byte, start_byte))
                else:
                    token_bytes.append(token_text_bytes)
                    token_byte_spans.append((start_byte, end_byte))
            else:
                token_bytes.append(token_text_bytes)
                token_byte_spans.append((start_byte, end_byte))
            previous_start_char = start_char
            previous_end_char = end_char

        if b"".join(token_bytes) != text.encode("utf-8"):
            raise ValueError("Tokenizer offset mapping bytes do not reconstruct the source text.")
        return token_bytes, token_byte_spans
    except Exception:
        try:
            return _build_token_byte_spans_via_token_strings(tokenizer, text, token_ids)
        except Exception as exc:
            if allow_prefix_fallback:
                return _build_token_byte_spans_via_prefix_decode(tokenizer, token_ids)
            raise ValueError("Tokenizer token/offset reconstruction failed.") from exc


def build_token_byte_spans(tokenizer, text: str, token_ids: list[int]) -> tuple[list[bytes], list[tuple[int, int]]]:
    return _build_token_byte_spans(tokenizer, text, token_ids, allow_prefix_fallback=True)


def build_token_byte_spans_strict(
    tokenizer, text: str, token_ids: list[int]
) -> tuple[list[bytes], list[tuple[int, int]]]:
    return _build_token_byte_spans(tokenizer, text, token_ids, allow_prefix_fallback=False)


def clip_token_bytes_by_region(
    token_bytes: list[bytes],
    token_byte_spans: list[tuple[int, int]],
    start_byte: int,
    end_byte: int | None = None,
) -> tuple[list[bytes], list[int]]:
    clipped_bytes: list[bytes] = []
    kept_indices: list[int] = []
    region_end = end_byte if end_byte is not None else float("inf")

    for index, (token_byte, (token_start, token_end)) in enumerate(zip(token_bytes, token_byte_spans, strict=False)):
        overlap_start = max(token_start, start_byte)
        overlap_end = min(token_end, region_end)
        if overlap_end <= overlap_start:
            continue
        rel_start = overlap_start - token_start
        rel_end = overlap_end - token_start
        clipped_bytes.append(token_byte[rel_start:rel_end])
        kept_indices.append(index)

    return clipped_bytes, kept_indices


def align_token_byte_chunks(
    student_token_bytes: list[bytes], teacher_token_bytes: list[bytes]
) -> list[tuple[slice, slice, bytes]]:
    student_index = 0
    teacher_index = 0
    chunks: list[tuple[slice, slice, bytes]] = []

    while student_index < len(student_token_bytes) and teacher_index < len(teacher_token_bytes):
        student_start = student_index
        teacher_start = teacher_index
        student_acc = b""
        teacher_acc = b""

        while student_acc != teacher_acc or not student_acc:
            if len(student_acc) <= len(teacher_acc):
                if student_index >= len(student_token_bytes):
                    raise ValueError("Student token bytes ended before chunk alignment completed.")
                student_acc += student_token_bytes[student_index]
                student_index += 1
                if student_acc == teacher_acc and student_acc:
                    chunks.append((slice(student_start, student_index), slice(teacher_start, teacher_index), student_acc))
                    break
            if len(teacher_acc) <= len(student_acc):
                if teacher_index >= len(teacher_token_bytes):
                    raise ValueError("Teacher token bytes ended before chunk alignment completed.")
                teacher_acc += teacher_token_bytes[teacher_index]
                teacher_index += 1

                if student_acc == teacher_acc and student_acc:
                    chunks.append((slice(student_start, student_index), slice(teacher_start, teacher_index), student_acc))
                    break

        if not chunks or chunks[-1][0].start != student_start or chunks[-1][1].start != teacher_start:
            raise ValueError("Failed to align student/teacher token bytes into common chunks.")

    if student_index != len(student_token_bytes) or teacher_index != len(teacher_token_bytes):
        raise ValueError("Student/teacher token bytes could not be fully aligned.")

    return chunks


def build_contextual_suffix_token_bytes(
    tokenizer,
    full_token_ids: list[int],
    suffix_start_index: int,
    prefix_text: str,
    full_text: str | None = None,
) -> tuple[list[bytes], list[tuple[int, int]]]:
    if full_text is None:
        full_text = decode_token_ids(tokenizer, full_token_ids)
    token_bytes, token_byte_spans = build_token_byte_spans_strict(tokenizer, full_text, full_token_ids)

    prompt_byte_length = len(prefix_text.encode("utf-8"))
    clipped_bytes: list[bytes] = []
    clipped_spans: list[tuple[int, int]] = []
    for token_byte, (token_start, token_end) in zip(token_bytes[suffix_start_index:], token_byte_spans[suffix_start_index:], strict=False):
        if token_end <= prompt_byte_length:
            continue
        if token_start < prompt_byte_length:
            rel_start = prompt_byte_length - token_start
            clipped_bytes.append(token_byte[rel_start:])
            clipped_spans.append((0, token_end - prompt_byte_length))
            continue
        clipped_bytes.append(token_byte)
        clipped_spans.append((token_start - prompt_byte_length, token_end - prompt_byte_length))

    return clipped_bytes, clipped_spans


def validate_recorded_student_response_alignment(
    *,
    response_text: str,
    response_token_count: int,
    response_bytes: bytes | bytearray | list[int] | tuple[int, ...],
    token_byte_spans: list[tuple[int, int]] | list[list[int]],
) -> tuple[list[bytes], list[tuple[int, int]]]:
    canonical_response_bytes = response_text.encode("utf-8")
    normalized_response_bytes = _coerce_bytes(response_bytes)
    if normalized_response_bytes != canonical_response_bytes:
        raise ValueError("Recorded student response bytes do not match canonical response text.")

    if len(token_byte_spans) != response_token_count:
        raise ValueError("Recorded student response span count does not match response token count.")

    token_bytes: list[bytes] = []
    normalized_spans: list[tuple[int, int]] = []
    cursor = 0
    for raw_start, raw_end in token_byte_spans:
        start = int(raw_start)
        end = int(raw_end)
        if start != cursor or end < start:
            raise ValueError("Recorded student response spans must be monotonic and gap-free.")
        if end > len(normalized_response_bytes):
            raise ValueError("Recorded student response span exceeds response byte length.")
        token_bytes.append(normalized_response_bytes[start:end])
        normalized_spans.append((start, end))
        cursor = end

    if cursor != len(normalized_response_bytes):
        raise ValueError("Recorded student response spans do not cover the full response bytes.")

    return token_bytes, normalized_spans


def build_recorded_student_response_alignment_from_token_texts(
    *,
    response_text: str,
    response_token_texts: list[str],
) -> tuple[bytes, list[tuple[int, int]]]:
    canonical_response_bytes = response_text.encode("utf-8")
    token_bytes = [token_text.encode("utf-8") for token_text in response_token_texts]
    if b"".join(token_bytes) != canonical_response_bytes:
        raise ValueError("Generation token texts do not match canonical response text bytes.")

    cursor = 0
    token_byte_spans: list[tuple[int, int]] = []
    for token_byte in token_bytes:
        end = cursor + len(token_byte)
        token_byte_spans.append((cursor, end))
        cursor = end

    return canonical_response_bytes, token_byte_spans


def build_generation_byte_evidence_observability(
    *,
    response_text: str,
    response_token_texts: list[str],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    observability_metadata = dict(metadata or {})
    observability_metadata.update(
        {
            "response_token_count": len(response_token_texts),
            "generation_token_text_count": len(response_token_texts),
        }
    )

    try:
        response_bytes, _token_byte_spans = build_recorded_student_response_alignment_from_token_texts(
            response_text=response_text,
            response_token_texts=response_token_texts,
        )
        observability_metadata.update(
            {
                "response_byte_length": len(response_bytes),
                "evidence_kind": "generation_byte_evidence",
            }
        )
        return {
            "attempted": True,
            "complete": True,
            "validated": True,
            "error": None,
            "metadata": observability_metadata,
        }
    except Exception as exc:
        observability_metadata["evidence_kind"] = "generation_byte_evidence_invalid"
        return {
            "attempted": True,
            "complete": False,
            "validated": False,
            "error": f"generation_byte_evidence_invalid: {exc}",
            "metadata": observability_metadata,
        }


def build_recorded_student_response_alignment_from_token_ids(
    *,
    tokenizer,
    response_text: str,
    response_token_ids: list[int],
) -> tuple[bytes, list[tuple[int, int]]]:
    canonical_response_bytes = response_text.encode("utf-8")
    token_bytes, token_byte_spans = _build_token_byte_spans_via_token_strings(
        tokenizer,
        response_text,
        response_token_ids,
    )
    if b"".join(token_bytes) != canonical_response_bytes:
        raise ValueError("Tokenizer token strings do not reconstruct the canonical response text.")
    return canonical_response_bytes, token_byte_spans


def build_recorded_student_response_alignment(
    tokenizer,
    *,
    full_token_ids: list[int],
    prompt_token_count: int,
    prompt_text: str,
    response_text: str | None = None,
    full_text: str | None = None,
) -> tuple[bytes, list[tuple[int, int]]]:
    response_token_ids = full_token_ids[prompt_token_count:]
    if response_text is None:
        if full_text is not None and full_text.startswith(prompt_text):
            response_text = full_text[len(prompt_text) :]
        else:
            response_text = decode_token_ids(tokenizer, response_token_ids)
    canonical_response_bytes = response_text.encode("utf-8")

    if full_text is None:
        full_text = f"{prompt_text}{response_text}"

    try:
        token_bytes, token_byte_spans = build_contextual_suffix_token_bytes(
            tokenizer,
            full_token_ids,
            prompt_token_count,
            prompt_text,
            full_text=full_text,
        )
        if b"".join(token_bytes) == canonical_response_bytes:
            return canonical_response_bytes, token_byte_spans
    except Exception:
        pass

    try:
        token_bytes, token_byte_spans = _build_token_byte_spans_via_token_strings(tokenizer, response_text, response_token_ids)
        if b"".join(token_bytes) == canonical_response_bytes:
            return canonical_response_bytes, token_byte_spans
    except Exception:
        pass

    token_bytes, token_byte_spans = build_token_byte_spans_strict(tokenizer, response_text, response_token_ids)
    return canonical_response_bytes, token_byte_spans


def compute_byte_chunk_reverse_kl(
    *,
    full_text: str,
    prompt_text: str | None = None,
    prompt_token_count: int,
    student_token_ids: list[int],
    response_token_count: int,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_tokenizer,
    teacher_tokenizer,
    student_alignment_evidence: dict[str, Any] | None = None,
    recorded_student_response_bytes: bytes | bytearray | list[int] | tuple[int, ...] | None = None,
    recorded_student_token_byte_spans: list[tuple[int, int]] | list[list[int]] | None = None,
    allow_sequence_fallback: bool = False,
) -> torch.Tensor:
    student_chunk_log_probs, teacher_chunk_log_probs = compute_byte_chunk_aligned_log_probs(
        full_text=full_text,
        prompt_text=prompt_text,
        prompt_token_count=prompt_token_count,
        student_token_ids=student_token_ids,
        response_token_count=response_token_count,
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_alignment_evidence=student_alignment_evidence,
        recorded_student_response_bytes=recorded_student_response_bytes,
        recorded_student_token_byte_spans=recorded_student_token_byte_spans,
        allow_sequence_fallback=allow_sequence_fallback,
    )
    return student_chunk_log_probs - teacher_chunk_log_probs


def compute_byte_chunk_aligned_log_probs(
    *,
    full_text: str,
    prompt_text: str | None = None,
    prompt_token_count: int,
    student_token_ids: list[int],
    response_token_count: int,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_tokenizer,
    teacher_tokenizer,
    student_alignment_evidence: dict[str, Any] | None = None,
    recorded_student_response_bytes: bytes | bytearray | list[int] | tuple[int, ...] | None = None,
    recorded_student_token_byte_spans: list[tuple[int, int]] | list[list[int]] | None = None,
    allow_sequence_fallback: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(student_token_ids) != prompt_token_count + response_token_count:
        raise ValueError("Student full token ids length does not match prompt/response split.")
    if response_token_count != student_log_probs.numel():
        raise ValueError("Student response token count and log_probs length mismatch.")

    if response_token_count == 0:
        empty = torch.empty_like(student_log_probs)
        return empty, empty
    prompt_token_ids = student_token_ids[:prompt_token_count]
    response_token_ids = student_token_ids[prompt_token_count:]
    if prompt_text is None:
        prompt_text = decode_token_ids(student_tokenizer, prompt_token_ids)
    if full_text.startswith(prompt_text):
        response_text = full_text[len(prompt_text) :]
    else:
        response_text = decode_token_ids(student_tokenizer, response_token_ids)
    prompt_byte_length = len(prompt_text.encode("utf-8"))

    def _preview(tokenizer, token_ids, token_bytes):
        preview = []
        for token_id, token_byte in zip(token_ids[:8], token_bytes[:8], strict=False):
            token_text = decode_token_ids(tokenizer, [token_id])
            preview.append(
                {
                    "id": token_id,
                    "text": repr(token_text),
                    "bytes_len": len(token_byte),
                    "bytes_hex": token_byte[:16].hex(),
                }
            )
        return preview

    def _sequence_fallback(
        reason: str, *, student_preview_bytes: list[bytes] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del student_preview_bytes
        if allow_sequence_fallback:
            logger.warning(
                "Ignoring allow_sequence_fallback=True because cross-tokenizer OPD now requires strict byte-chunk alignment. "
                "reason=%s",
                reason,
            )
        raise ValueError(reason)

    if student_alignment_evidence is not None:
        if student_alignment_evidence.get("complete") is False:
            return _sequence_fallback(
                f"student_alignment_evidence_incomplete: {student_alignment_evidence.get('error') or 'complete=False'}"
            )
        if student_alignment_evidence.get("validated") is False:
            return _sequence_fallback(
                f"student_alignment_evidence_invalid: {student_alignment_evidence.get('error') or 'validated=False'}"
            )
        recorded_student_response_bytes = student_alignment_evidence.get("response_bytes")
        recorded_student_token_byte_spans = student_alignment_evidence.get("token_byte_spans")

    if recorded_student_response_bytes is not None or recorded_student_token_byte_spans is not None:
        if recorded_student_response_bytes is None or recorded_student_token_byte_spans is None:
            return _sequence_fallback("student_recorded_alignment_incomplete")
        try:
            student_response_bytes, _student_response_spans = validate_recorded_student_response_alignment(
                response_text=response_text,
                response_token_count=response_token_count,
                response_bytes=recorded_student_response_bytes,
                token_byte_spans=recorded_student_token_byte_spans,
            )
        except ValueError as exc:
            return _sequence_fallback(f"student_recorded_alignment_failed: {exc}")
    else:
        try:
            student_response_bytes, _student_response_spans = build_contextual_suffix_token_bytes(
                student_tokenizer, student_token_ids, prompt_token_count, prompt_text
            )
        except ValueError:
            try:
                student_response_bytes, _student_response_spans = build_token_byte_spans(
                    student_tokenizer, response_text, response_token_ids
                )
            except ValueError as exc:
                return _sequence_fallback(f"student_byte_reconstruction_failed: {exc}")

    teacher_token_ids = encode_text(teacher_tokenizer, full_text)
    try:
        teacher_token_bytes, teacher_token_spans = build_token_byte_spans(teacher_tokenizer, full_text, teacher_token_ids)
        teacher_response_bytes, teacher_response_indices = clip_token_bytes_by_region(
            teacher_token_bytes,
            teacher_token_spans,
            start_byte=prompt_byte_length,
        )
        if len(teacher_response_indices) != teacher_log_probs.numel():
            raise ValueError(
                "Teacher response token count and log_probs length mismatch. "
                f"selected={len(teacher_response_indices)}, log_probs={teacher_log_probs.numel()}"
            )
    except ValueError as exc:
        return _sequence_fallback(f"teacher_byte_reconstruction_failed: {exc}", student_preview_bytes=student_response_bytes)
    try:
        chunks = align_token_byte_chunks(student_response_bytes, teacher_response_bytes)
    except ValueError as exc:
        return _sequence_fallback(f"alignment_failed: {exc}", student_preview_bytes=student_response_bytes)

    student_chunk_log_probs = torch.empty_like(student_log_probs)
    teacher_chunk_log_probs = torch.empty_like(student_log_probs)
    for student_slice, teacher_slice, _chunk_bytes in chunks:
        student_chunk_log_prob = student_log_probs[student_slice].sum()
        teacher_chunk_log_prob = teacher_log_probs[teacher_slice].sum()
        student_chunk_length = student_slice.stop - student_slice.start
        student_chunk_log_probs[student_slice] = student_chunk_log_prob / student_chunk_length
        teacher_chunk_log_probs[student_slice] = teacher_chunk_log_prob / student_chunk_length

    return student_chunk_log_probs, teacher_chunk_log_probs


def prepare_byte_chunk_training_entry(
    *,
    full_text: str,
    prompt_text: str | None,
    student_token_ids: list[int],
    response_token_count: int,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_tokenizer,
    teacher_tokenizer,
    student_alignment_evidence: dict[str, Any] | None = None,
    recorded_student_response_bytes: bytes | bytearray | list[int] | tuple[int, ...] | None = None,
    recorded_student_token_byte_spans: list[tuple[int, int]] | list[list[int]] | None = None,
) -> dict[str, Any]:
    prompt_token_count = len(student_token_ids) - response_token_count
    if prompt_token_count < 0:
        raise ValueError("Response token count exceeds full student token count.")

    student_chunk_log_probs, teacher_chunk_log_probs = compute_byte_chunk_aligned_log_probs(
        full_text=full_text,
        prompt_text=prompt_text,
        prompt_token_count=prompt_token_count,
        student_token_ids=student_token_ids,
        response_token_count=response_token_count,
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        student_alignment_evidence=student_alignment_evidence,
        recorded_student_response_bytes=recorded_student_response_bytes,
        recorded_student_token_byte_spans=recorded_student_token_byte_spans,
    )
    return {
        "status": "ok",
        "prompt_token_count": prompt_token_count,
        "response_token_count": response_token_count,
        "student_chunk_log_probs": student_chunk_log_probs,
        "teacher_chunk_log_probs": teacher_chunk_log_probs,
        "reverse_kl": student_chunk_log_probs - teacher_chunk_log_probs,
    }
