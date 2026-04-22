import aiohttp
import logging
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.opd_utils import (
    build_generation_byte_evidence_observability,
    build_recorded_student_response_alignment_from_token_ids,
    build_token_byte_spans,
    clip_token_bytes_by_region,
    decode_token_ids,
    encode_text,
    get_cached_tokenizer,
    validate_recorded_student_response_alignment,
)
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _is_text_token_consistent(student_tokenizer, text: str | None, token_ids: list[int]) -> bool:
    if not isinstance(text, str):
        return False
    try:
        return encode_text(student_tokenizer, text) == token_ids
    except Exception:
        return False


def _build_canonical_opd_texts(
    *,
    sample: Sample,
    student_tokenizer,
    prompt_token_ids: list[int],
) -> tuple[str, str, str]:
    """Build canonical prompt/response/full texts for byte_chunk OPD.

    Prefer rendered prompt / full strings to preserve chat-template punctuation
    and teacher-facing request text. For response text, require token
    consistency; if the rendered response drifts from the sampled token ids,
    fall back to token-derived response text so recorded student alignment
    remains stable on long cross-tokenizer samples.
    """

    response_token_ids = sample.tokens[len(prompt_token_ids) :]
    decoded_prompt_text = decode_token_ids(student_tokenizer, prompt_token_ids)
    decoded_response_text = decode_token_ids(student_tokenizer, response_token_ids)
    decoded_full_text = decode_token_ids(student_tokenizer, sample.tokens)

    prompt_text = sample.opd_prompt_text
    if prompt_text is None:
        if isinstance(sample.prompt, str):
            prompt_text = sample.prompt
        else:
            prompt_text = decoded_prompt_text

    rendered_response_text = sample.opd_response_text if sample.opd_response_text is not None else sample.response
    response_text = (
        rendered_response_text
        if _is_text_token_consistent(student_tokenizer, rendered_response_text, response_token_ids)
        else decoded_response_text
    )

    full_text = sample.opd_full_text
    if full_text is None:
        full_text = f"{prompt_text}{response_text}"
        if not _is_text_token_consistent(student_tokenizer, full_text, sample.tokens):
            prompt_text = decoded_prompt_text
            response_text = decoded_response_text
            full_text = decoded_full_text
    elif not full_text.startswith(prompt_text) or full_text != f"{prompt_text}{response_text}":
        # Stored rendered prompt/full texts can drift apart on long truncated samples.
        # Recover to one self-consistent token-derived trio so teacher byte clipping
        # uses the exact same prompt/response boundary as student alignment.
        prompt_text = decoded_prompt_text
        response_text = decoded_response_text
        full_text = decoded_full_text

    return prompt_text, response_text, full_text


def _align_teacher_reward_log_probs(
    teacher_tokenizer,
    full_text: str,
    prompt_byte_length: int,
    student_response_length: int,
    reward_token_entries: list[list[float | int | None]],
) -> torch.Tensor:
    local_teacher_token_ids = encode_text(teacher_tokenizer, full_text)
    reward_teacher_token_ids = [int(item[1]) for item in reward_token_entries]
    reward_log_probs = torch.tensor([item[0] for item in reward_token_entries], dtype=torch.float32)

    def _sequence_fallback(reason: str) -> torch.Tensor:
        logger.warning(
            "Teacher byte-span reconstruction failed; falling back to sequence-level teacher log probs. "
            "reason=%s local_count=%s reward_count=%s student_response_length=%s",
            reason,
            len(local_teacher_token_ids),
            len(reward_teacher_token_ids),
            student_response_length,
        )
        sequence_mean = reward_log_probs.mean().item() if reward_log_probs.numel() > 0 else 0.0
        return torch.full(
            (student_response_length,),
            sequence_mean,
            dtype=torch.float32,
        )

    try:
        local_teacher_token_bytes, local_teacher_token_spans = build_token_byte_spans(
            teacher_tokenizer, full_text, local_teacher_token_ids
        )
        _local_response_bytes, local_response_indices = clip_token_bytes_by_region(
            local_teacher_token_bytes,
            local_teacher_token_spans,
            start_byte=prompt_byte_length,
        )
    except Exception as exc:
        try:
            reward_teacher_token_bytes, reward_teacher_token_spans = build_token_byte_spans(
                teacher_tokenizer, full_text, reward_teacher_token_ids
            )
            _reward_response_bytes, reward_response_indices = clip_token_bytes_by_region(
                reward_teacher_token_bytes,
                reward_teacher_token_spans,
                start_byte=prompt_byte_length,
            )
            return reward_log_probs[reward_response_indices]
        except Exception as reward_exc:
            return _sequence_fallback(f"local={exc}; reward={reward_exc}")

    if reward_teacher_token_ids == local_teacher_token_ids:
        return reward_log_probs[local_response_indices]

    if len(reward_teacher_token_ids) == len(local_teacher_token_ids) + 1:
        if reward_teacher_token_ids[1:] == local_teacher_token_ids:
            logger.warning(
                "Teacher reward ids have one extra leading token; trimming it to match local encode. "
                "local_count=%s reward_count=%s local_response_count=%s",
                len(local_teacher_token_ids),
                len(reward_teacher_token_ids),
                len(local_response_indices),
            )
            return reward_log_probs[1:][local_response_indices]
        if reward_teacher_token_ids[:-1] == local_teacher_token_ids:
            logger.warning(
                "Teacher reward ids have one extra trailing token; trimming it to match local encode. "
                "local_count=%s reward_count=%s local_response_count=%s",
                len(local_teacher_token_ids),
                len(reward_teacher_token_ids),
                len(local_response_indices),
            )
            return reward_log_probs[:-1][local_response_indices]

    if len(reward_teacher_token_ids) >= len(local_teacher_token_ids):
        max_start = len(reward_teacher_token_ids) - len(local_teacher_token_ids)
        for start in range(max_start + 1):
            end = start + len(local_teacher_token_ids)
            if reward_teacher_token_ids[start:end] == local_teacher_token_ids:
                logger.warning(
                    "Teacher reward ids differ from local encode; trimming reward ids to local span. "
                    "local_count=%s reward_count=%s trim=(%s,%s)",
                    len(local_teacher_token_ids),
                    len(reward_teacher_token_ids),
                    start,
                    len(reward_teacher_token_ids) - end,
                )
                return reward_log_probs[start:end][local_response_indices]

    try:
        reward_teacher_token_bytes, reward_teacher_token_spans = build_token_byte_spans(
            teacher_tokenizer, full_text, reward_teacher_token_ids
        )
        _reward_response_bytes, reward_response_indices = clip_token_bytes_by_region(
            reward_teacher_token_bytes,
            reward_teacher_token_spans,
            start_byte=prompt_byte_length,
        )
        if len(reward_response_indices) == len(local_response_indices):
            logger.warning(
                "Teacher reward ids differ from local encode; using reward-response byte spans directly. "
                "local_count=%s reward_count=%s local_response_count=%s reward_response_count=%s",
                len(local_teacher_token_ids),
                len(reward_teacher_token_ids),
                len(local_response_indices),
                len(reward_response_indices),
            )
            return reward_log_probs[reward_response_indices]
    except Exception as exc:
        logger.warning(
            "Teacher reward ids could not reconstruct response byte spans directly; "
            "falling back to sequence-level teacher log probs. error=%s",
            exc,
        )

    logger.warning(
        "Teacher reward ids could not be aligned to local encode; falling back to sequence-level teacher log probs. "
        "local_count=%s reward_count=%s local_response_count=%s local_head=%s reward_head=%s local_tail=%s reward_tail=%s",
        len(local_teacher_token_ids),
        len(reward_teacher_token_ids),
        len(local_response_indices),
        local_teacher_token_ids[:8],
        reward_teacher_token_ids[:8],
        local_teacher_token_ids[-8:],
        reward_teacher_token_ids[-8:],
    )
    return _sequence_fallback("unalignable_reward_ids")


def _record_student_opd_alignment(sample: Sample, student_tokenizer) -> None:
    base_alignment_metadata = dict(sample.opd_student_alignment_metadata or {})
    generation_byte_evidence_metadata = dict(sample.opd_generation_byte_evidence_metadata or {})

    if sample.response_length <= 0:
        sample.opd_student_response_bytes = []
        sample.opd_student_token_byte_spans = []
        sample.opd_student_alignment_version = 1
        sample.opd_student_alignment_error = None
        sample.opd_student_alignment_source = "recorded_builder"
        sample.opd_student_alignment_validated = True
        sample.opd_student_alignment_status = "ok_recorded"
        sample.opd_student_alignment_complete = True
        sample.opd_student_alignment_metadata = base_alignment_metadata | {
            "response_token_count": 0,
            "response_byte_length": 0,
            "generation_token_text_count": len(sample.opd_student_token_texts)
            if sample.opd_student_token_texts is not None
            else None,
            "evidence_kind": "empty_response",
        }
        if sample.opd_generation_byte_evidence_attempted is None:
            sample.opd_generation_byte_evidence_attempted = False
        return

    if sample.opd_response_text is None:
        raise ValueError("Canonical OPD response text must be set before recording student alignment.")

    try:
        alignment_source = "recorded_builder"
        generation_token_text_count = len(sample.opd_student_token_texts) if sample.opd_student_token_texts is not None else None
        if sample.opd_student_token_texts is not None:
            if sample.opd_generation_byte_evidence_attempted is None:
                observability = build_generation_byte_evidence_observability(
                    response_text=sample.opd_response_text,
                    response_token_texts=sample.opd_student_token_texts,
                    metadata=generation_byte_evidence_metadata,
                )
                sample.opd_generation_byte_evidence_attempted = observability["attempted"]
                sample.opd_generation_byte_evidence_complete = observability["complete"]
                sample.opd_generation_byte_evidence_validated = observability["validated"]
                sample.opd_generation_byte_evidence_error = observability["error"]
                sample.opd_generation_byte_evidence_metadata = dict(observability["metadata"])
        else:
            if sample.opd_generation_byte_evidence_attempted is None:
                sample.opd_generation_byte_evidence_attempted = False

        response_bytes, response_spans = build_recorded_student_response_alignment_from_token_ids(
            tokenizer=student_tokenizer,
            response_text=sample.opd_response_text,
            response_token_ids=sample.tokens[-sample.response_length :],
        )
        sample.opd_student_response_bytes = list(response_bytes)
        sample.opd_student_token_byte_spans = [list(span) for span in response_spans]
        sample.opd_student_alignment_version = 1
        sample.opd_student_alignment_error = None
        sample.opd_student_alignment_source = alignment_source
        sample.opd_student_alignment_validated = True
        sample.opd_student_alignment_status = "ok_recorded"
        sample.opd_student_alignment_complete = True
        alignment_metadata = base_alignment_metadata | {
            "response_token_count": sample.response_length,
            "response_byte_length": len(response_bytes),
            "evidence_kind": "recorded_builder_token_ids",
        }
        if generation_token_text_count is not None:
            alignment_metadata["generation_token_text_count"] = generation_token_text_count
        sample.opd_student_alignment_metadata = alignment_metadata
    except Exception as exc:
        sample.opd_student_response_bytes = None
        sample.opd_student_token_byte_spans = None
        sample.opd_student_alignment_version = 1
        sample.opd_student_alignment_error = str(exc)
        sample.opd_student_alignment_source = "recorded_builder"
        sample.opd_student_alignment_validated = False
        sample.opd_student_alignment_status = "recorded_missing"
        sample.opd_student_alignment_complete = False
        sample.opd_student_alignment_metadata = base_alignment_metadata | {
            "response_token_count": sample.response_length,
            "generation_token_text_count": generation_token_text_count if "generation_token_text_count" in locals() else None,
            "evidence_kind": "recorded_builder_failed",
        }
        if sample.opd_generation_byte_evidence_attempted is None:
            sample.opd_generation_byte_evidence_attempted = False


def compute_teacher_log_probs_for_sample(args, sample: Sample) -> torch.Tensor:
    """Extract teacher log-probs for one sample using the same path as training.

    This helper keeps tests and debug scripts on the exact same reward-processing
    path as rollout -> post_process_rewards -> train.
    """

    reward = sample.get_reward_value(args)

    if getattr(args, "opd_alignment", "token") == "byte_chunk":
        student_tokenizer = get_cached_tokenizer(args.hf_checkpoint)
        teacher_tokenizer = get_cached_tokenizer(args.opd_teacher_hf_checkpoint)
        prompt_token_ids = sample.tokens[:-sample.response_length] if sample.response_length > 0 else sample.tokens
        prompt_text, _response_text, full_text = _build_canonical_opd_texts(
            sample=sample,
            student_tokenizer=student_tokenizer,
            prompt_token_ids=prompt_token_ids,
        )
        prompt_byte_length = len(prompt_text.encode("utf-8"))

        reward_token_entries = reward["meta_info"]["input_token_logprobs"][1:]
        return _align_teacher_reward_log_probs(
            teacher_tokenizer,
            full_text,
            prompt_byte_length,
            sample.response_length,
            reward_token_entries,
        )

    token_log_probs = torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
    return token_log_probs[-sample.response_length :]


async def reward_func(args, sample, **kwargs):
    payload = {
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    if getattr(args, "opd_alignment", "token") == "byte_chunk":
        student_tokenizer = get_cached_tokenizer(args.hf_checkpoint)
        prompt_token_ids = sample.tokens[:-sample.response_length] if sample.response_length > 0 else sample.tokens
        sample.opd_prompt_text, sample.opd_response_text, sample.opd_full_text = _build_canonical_opd_texts(
            sample=sample,
            student_tokenizer=student_tokenizer,
            prompt_token_ids=prompt_token_ids,
        )
        _record_student_opd_alignment(sample, student_tokenizer)
        payload["text"] = sample.opd_full_text
    else:
        payload["input_ids"] = sample.tokens

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs = []
    for sample in samples:
        teacher_log_probs.append(compute_teacher_log_probs_for_sample(args, sample))

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
