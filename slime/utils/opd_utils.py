from __future__ import annotations

from functools import lru_cache

import torch

from slime.utils.processing_utils import load_tokenizer


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


def _build_char_to_byte_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for char in text:
        total += len(char.encode("utf-8"))
        offsets.append(total)
    return offsets


def build_token_byte_spans(tokenizer, text: str, token_ids: list[int]) -> tuple[list[bytes], list[tuple[int, int]]]:
    try:
        tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if tokenized["input_ids"] != token_ids:
            raise ValueError(
                "Token ids do not match tokenizer encoding for response text: "
                f"expected={token_ids}, got={tokenized['input_ids']}"
            )

        char_to_byte_offsets = _build_char_to_byte_offsets(text)
        token_bytes = []
        token_byte_spans = []
        for start_char, end_char in tokenized["offset_mapping"]:
            start_byte = char_to_byte_offsets[start_char]
            end_byte = char_to_byte_offsets[end_char]
            token_bytes.append(text[start_char:end_char].encode("utf-8"))
            token_byte_spans.append((start_byte, end_byte))
        return token_bytes, token_byte_spans
    except Exception:
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


def compute_byte_chunk_reverse_kl(
    *,
    full_text: str,
    prompt_token_count: int,
    student_token_ids: list[int],
    response_token_count: int,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_tokenizer,
    teacher_tokenizer,
) -> torch.Tensor:
    if len(student_token_ids) != prompt_token_count + response_token_count:
        raise ValueError("Student full token ids length does not match prompt/response split.")
    if response_token_count != student_log_probs.numel():
        raise ValueError("Student response token count and log_probs length mismatch.")

    student_token_bytes, student_token_spans = build_token_byte_spans(student_tokenizer, full_text, student_token_ids)
    if response_token_count == 0:
        return torch.empty_like(student_log_probs)
    response_start_index = len(student_token_ids) - response_token_count
    prompt_byte_length = student_token_spans[response_start_index][0]

    student_response_bytes, _student_response_indices = clip_token_bytes_by_region(
        student_token_bytes[response_start_index:],
        student_token_spans[response_start_index:],
        start_byte=prompt_byte_length,
    )

    teacher_token_ids = encode_text(teacher_tokenizer, full_text)
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
    try:
        chunks = align_token_byte_chunks(student_response_bytes, teacher_response_bytes)
    except ValueError as exc:
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

        raise ValueError(
            "Failed to align byte_chunk OPD token bytes. "
            f"full_text={full_text!r}, prompt_token_count={prompt_token_count}, response_token_count={response_token_count}, "
            f"prompt_byte_length={prompt_byte_length}, "
            f"student_token_count={len(student_token_ids)}, teacher_token_count={len(teacher_token_ids)}, "
            f"student_preview={_preview(student_tokenizer, student_token_ids[response_start_index:], student_response_bytes)}, "
            f"teacher_preview={_preview(teacher_tokenizer, [teacher_token_ids[i] for i in teacher_response_indices], teacher_response_bytes)}"
        ) from exc

    reverse_kl = torch.empty_like(student_log_probs)
    for student_slice, teacher_slice, _chunk_bytes in chunks:
        student_chunk_log_prob = student_log_probs[student_slice].sum()
        teacher_chunk_log_prob = teacher_log_probs[teacher_slice].sum()
        student_chunk_penalty = student_chunk_log_prob - teacher_chunk_log_prob
        student_chunk_length = student_slice.stop - student_slice.start
        reverse_kl[student_slice] = student_chunk_penalty / student_chunk_length

    return reverse_kl
