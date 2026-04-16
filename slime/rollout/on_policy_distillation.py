import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.opd_utils import (
    build_token_byte_spans,
    clip_token_bytes_by_region,
    decode_token_ids,
    encode_text,
    get_cached_tokenizer,
)
from slime.utils.types import Sample


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
        response_token_ids = sample.tokens[-sample.response_length:] if sample.response_length > 0 else []
        sample.opd_full_text = decode_token_ids(student_tokenizer, sample.tokens)
        sample.opd_prompt_text = decode_token_ids(student_tokenizer, prompt_token_ids)
        sample.opd_response_text = decode_token_ids(student_tokenizer, response_token_ids)
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
    if getattr(args, "opd_alignment", "token") == "byte_chunk":
        student_tokenizer = get_cached_tokenizer(args.hf_checkpoint)
        teacher_tokenizer = get_cached_tokenizer(args.opd_teacher_hf_checkpoint)
        for reward, sample in zip(raw_rewards, samples, strict=False):
            full_text = sample.opd_full_text if sample.opd_full_text is not None else decode_token_ids(student_tokenizer, sample.tokens)
            student_token_bytes, student_token_spans = build_token_byte_spans(student_tokenizer, full_text, sample.tokens)
            response_start_index = len(sample.tokens) - sample.response_length
            prompt_byte_length = (
                student_token_spans[response_start_index][0] if sample.response_length > 0 else len(full_text.encode("utf-8"))
            )

            teacher_token_ids = encode_text(teacher_tokenizer, full_text)
            teacher_token_bytes, teacher_token_spans = build_token_byte_spans(teacher_tokenizer, full_text, teacher_token_ids)
            _teacher_response_bytes, teacher_response_indices = clip_token_bytes_by_region(
                teacher_token_bytes,
                teacher_token_spans,
                start_byte=prompt_byte_length,
            )
            token_log_probs = torch.tensor(
                [item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32,
            )
            if len(token_log_probs) != len(teacher_token_ids):
                raise ValueError(
                    "Teacher full token count and teacher log_probs length mismatch. "
                    f"tokens={len(teacher_token_ids)}, log_probs={len(token_log_probs)}"
                )
            teacher_log_probs.append(token_log_probs[teacher_response_indices])
    else:
        # Extract teacher log-probs from the sglang response
        teacher_log_probs = [
            torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
            for reward in raw_rewards
        ]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
