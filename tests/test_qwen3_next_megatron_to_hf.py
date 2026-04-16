from argparse import Namespace

import torch

from slime.backends.megatron_utils.megatron_to_hf.qwen3_next import convert_qwen3_next_to_hf


def _build_args():
    return Namespace(
        hidden_size=6,
        num_attention_heads=6,
        num_query_groups=3,
        kv_channels=1,
    )


def test_convert_qwen3_next_linear_qkv_weight_supports_standard_gqa_layout():
    args = _build_args()
    name = "module.module.decoder.layers.0.self_attention.linear_qkv.weight"

    packed = torch.arange(3 * 4 * 1 * 6, dtype=torch.float32).view(12, 6)
    converted = dict(convert_qwen3_next_to_hf(args, name, packed))

    assert converted["model.layers.0.self_attn.q_proj.weight"].shape == (6, 6)
    assert converted["model.layers.0.self_attn.k_proj.weight"].shape == (3, 6)
    assert converted["model.layers.0.self_attn.v_proj.weight"].shape == (3, 6)


def test_convert_qwen3_next_linear_qkv_weight_supports_gated_layout():
    args = _build_args()
    name = "module.module.decoder.layers.0.self_attention.linear_qkv.weight"

    packed = torch.arange(3 * 6 * 1 * 6, dtype=torch.float32).view(18, 6)
    converted = dict(convert_qwen3_next_to_hf(args, name, packed))

    assert converted["model.layers.0.self_attn.q_proj.weight"].shape == (12, 6)
    assert converted["model.layers.0.self_attn.k_proj.weight"].shape == (3, 6)
    assert converted["model.layers.0.self_attn.v_proj.weight"].shape == (3, 6)
