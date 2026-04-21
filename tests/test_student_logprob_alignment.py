from argparse import Namespace
import asyncio
import sys
import types

import torch


def _install_fake_megatron():
    megatron_mpu = types.SimpleNamespace(
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        is_pipeline_last_stage=lambda: True,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_context_parallel_group=lambda: None,
    )
    megatron = sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    megatron_core = sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
    packed_seq_params = sys.modules.setdefault(
        "megatron.core.packed_seq_params",
        types.ModuleType("megatron.core.packed_seq_params"),
    )
    packed_seq_params.PackedSeqParams = object

    megatron.core = megatron_core
    megatron_core.mpu = megatron_mpu
    sys.modules["megatron.core.mpu"] = megatron_mpu


_install_fake_megatron()

from slime.backends.megatron_utils.loss import get_log_probs_and_entropy, policy_loss_function
from slime.rollout import sglang_rollout
from slime.utils.types import Sample


def test_generate_keeps_rollout_log_probs_aligned_with_response_tokens_when_truncated(monkeypatch):
    captured_payload = {}

    async def fake_post(url, payload, headers=None):
        del url, headers
        captured_payload.update(payload)
        return {
            "text": "AB",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 11, "A"], [-0.2, 12, "B"]],
                "finish_reason": {"type": "length"},
                "prompt_tokens": 3,
                "completion_tokens": 2,
            },
        }

    monkeypatch.setattr(
        sglang_rollout,
        "GenerateState",
        lambda args: types.SimpleNamespace(tokenizer=None, processor=None),
    )
    monkeypatch.setattr(sglang_rollout, "_prepare_prompt_ids", lambda sample, tokenizer, processor: [101, 102, 103])
    monkeypatch.setattr(sglang_rollout, "post", fake_post)

    args = Namespace(
        ci_test=False,
        use_rollout_routing_replay=False,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        router_policy=None,
        sglang_speculative_algorithm=None,
    )
    sample = Sample(prompt="hello")

    updated = asyncio.run(
        sglang_rollout.generate(
            args,
            sample,
            sampling_params={"max_new_tokens": 2},
        )
    )

    assert updated.tokens == [101, 102, 103, 11, 12]
    assert updated.response == "AB"
    assert updated.response_length == 2
    assert updated.rollout_log_probs == [-0.1, -0.2]
    assert updated.opd_student_token_texts == ["A", "B"]
    assert updated.opd_student_alignment_version is None
    assert updated.opd_student_alignment_source is None
    assert updated.opd_student_alignment_complete is None
    assert updated.opd_student_alignment_validated is None
    assert updated.opd_student_response_bytes is None
    assert updated.opd_student_token_byte_spans is None
    assert updated.opd_student_alignment_metadata is None
    assert updated.opd_generation_byte_evidence_attempted is True
    assert updated.opd_generation_byte_evidence_complete is True
    assert updated.opd_generation_byte_evidence_validated is True
    assert updated.opd_generation_byte_evidence_error is None
    assert updated.opd_generation_byte_evidence_metadata == {
        "engine": "sglang",
        "detokenizer_protocol": "output_token_logprobs_text",
        "completion_reason": "length",
        "captured_response_token_count": 2,
        "response_token_count": 2,
        "response_byte_length": 2,
        "generation_token_text_count": 2,
        "evidence_kind": "generation_byte_evidence",
    }
    assert len(updated.rollout_log_probs) == updated.response_length
    assert updated.tokens[-updated.response_length :] == [11, 12]
    assert captured_payload["return_text_in_logprobs"] is True
    assert updated.status == Sample.Status.TRUNCATED


def test_get_log_probs_and_entropy_extracts_response_window_and_scales_logits(monkeypatch):
    captured = {}

    def fake_calculate_log_probs_and_entropy(logits, full_tokens, tp_group, with_entropy, chunk_size, need_entropy_grad):
        del full_tokens, tp_group, with_entropy, chunk_size, need_entropy_grad
        captured["logits"] = logits.clone()
        seq = torch.arange(logits.size(0), dtype=torch.float32, device=logits.device).unsqueeze(-1)
        entropy = torch.arange(logits.size(0), dtype=torch.float32, device=logits.device)
        return seq, entropy

    monkeypatch.setattr(
        "slime.backends.megatron_utils.loss.calculate_log_probs_and_entropy",
        fake_calculate_log_probs_and_entropy,
    )

    raw_logits = torch.arange(35, dtype=torch.float32).view(1, 7, 5)
    args = Namespace(
        qkv_format="thd",
        rollout_temperature=2.0,
        log_probs_chunk_size=16,
        entropy_coef=0.0,
        allgather_cp=False,
    )

    _, result = get_log_probs_and_entropy(
        raw_logits,
        args=args,
        unconcat_tokens=[
            torch.tensor([10, 11, 12], dtype=torch.long),
            torch.tensor([20, 21, 22, 23], dtype=torch.long),
        ],
        total_lengths=[3, 4],
        response_lengths=[1, 2],
        with_entropy=True,
    )

    assert torch.equal(captured["logits"], raw_logits.squeeze(0) / 2.0)
    assert torch.allclose(result["log_probs"][0], torch.tensor([1.0]))
    assert torch.allclose(result["log_probs"][1], torch.tensor([4.0, 5.0]))
    assert torch.allclose(result["entropy"][0], torch.tensor([1.0]))
    assert torch.allclose(result["entropy"][1], torch.tensor([4.0, 5.0]))


def test_policy_loss_reports_train_rollout_logprob_abs_diff_from_student_paths_only(monkeypatch):
    def fake_get_log_probs_and_entropy(
        logits,
        *,
        args,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        with_entropy,
        max_seq_lens,
    ):
        del logits, args, unconcat_tokens, total_lengths, response_lengths, with_entropy, max_seq_lens
        return torch.empty((0,)), {
            "log_probs": [torch.tensor([-4.0, -6.0])],
            "entropy": [torch.tensor([0.3, 0.7])],
        }

    def fake_compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
        del ppo_kl, advantages, eps_clip, eps_clip_high
        return torch.tensor([0.5, 1.5]), torch.tensor([0.0, 1.0])

    monkeypatch.setattr(
        "slime.backends.megatron_utils.loss.get_log_probs_and_entropy",
        fake_get_log_probs_and_entropy,
    )
    monkeypatch.setattr(
        "slime.backends.megatron_utils.loss.compute_policy_loss",
        fake_compute_policy_loss,
    )

    args = Namespace(
        use_rollout_logprobs=False,
        advantage_estimator="grpo",
        use_opsm=False,
        get_mismatch_metrics=False,
        use_tis=False,
        custom_tis_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        use_kl_loss=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        calculate_per_token_loss=False,
        qkv_format="thd",
    )
    batch = {
        "advantages": [torch.tensor([1.0, 1.0])],
        "log_probs": [torch.tensor([-3.0, -5.0])],
        "rollout_log_probs": [torch.tensor([-1.0, -2.0])],
        "unconcat_tokens": [torch.tensor([0, 1, 2])],
        "response_lengths": [2],
        "total_lengths": [3],
        "loss_masks": [torch.tensor([1.0, 1.0])],
    }

    loss, reported = policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 2, 3), dtype=torch.float32),
        sum_of_sample_mean=lambda x: x.float().mean(),
    )

    assert torch.allclose(loss, torch.tensor(1.0))
    assert torch.allclose(reported["pg_loss"], torch.tensor(1.0))
    assert torch.allclose(reported["ppo_kl"], torch.tensor(1.0))
    assert torch.allclose(reported["entropy_loss"], torch.tensor(0.5))
    assert torch.allclose(reported["train_rollout_logprob_abs_diff"], torch.tensor(2.5))
    assert "opd_reverse_kl" not in reported
