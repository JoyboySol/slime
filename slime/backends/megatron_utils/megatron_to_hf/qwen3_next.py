import re

import torch


_cached_yulan_gdn_ba_weights = {}
_cached_yulan_gdn_qkvz_weights = {}


def _convert_yulan_gdn_to_hf(args, layer_idx, rest, param):
    """Convert YuLan's native MCore GDN tensors to Qwen3-Next HF tensors."""

    prefix = f"model.layers.{layer_idx}"
    direct_mapping = {
        "self_attention.in_proj.layer_norm_weight": "input_layernorm.weight",
        "self_attention.A_log": "linear_attn.A_log",
        "self_attention.dt_bias": "linear_attn.dt_bias",
        "self_attention.out_norm.weight": "linear_attn.norm.weight",
        "self_attention.out_proj.weight": "linear_attn.out_proj.weight",
    }
    if rest in direct_mapping:
        return [(f"{prefix}.{direct_mapping[rest]}", param)]

    if rest in {
        "self_attention.q_proj.weight",
        "self_attention.k_proj.weight",
        "self_attention.v_proj.weight",
        "self_attention.z_proj.weight",
    }:
        # The RL model uses YuLan's split GDN input projections. Reassemble
        # them in the same per-key-head layout as the HF/SGLang Qwen3-Next
        # in_proj_qkvz tensor.
        projection = rest.removeprefix("self_attention.").removesuffix(".weight")
        _cached_yulan_gdn_qkvz_weights[(layer_idx, projection)] = param
        projection_names = ("q_proj", "k_proj", "v_proj", "z_proj")
        if not all((layer_idx, name) in _cached_yulan_gdn_qkvz_weights for name in projection_names):
            return []

        query, key, value, z = (
            _cached_yulan_gdn_qkvz_weights.pop((layer_idx, name)) for name in projection_names
        )
        num_key_heads = args.linear_num_key_heads
        num_value_heads = args.linear_num_value_heads
        key_head_dim = args.linear_key_head_dim
        value_head_dim = args.linear_value_head_dim
        values_per_key = num_value_heads // num_key_heads
        qkvz = torch.cat(
            [
                query.reshape(num_key_heads, key_head_dim, -1),
                key.reshape(num_key_heads, key_head_dim, -1),
                value.reshape(num_key_heads, values_per_key * value_head_dim, -1),
                z.reshape(num_key_heads, values_per_key * value_head_dim, -1),
            ],
            dim=1,
        ).flatten(0, 1)
        return [(f"{prefix}.linear_attn.in_proj_qkvz.weight", qkvz)]

    if rest in {"self_attention.b_proj.weight", "self_attention.a_proj.weight"}:
        # YuLan stores beta and alpha as separate projections, while the
        # Qwen3-Next HF/SGLang module stores them interleaved per key head in
        # in_proj_ba. Keep this layout in sync with YuLan-Pretrain's m2h
        # synchronizer.
        projection = "b" if rest.endswith("b_proj.weight") else "a"
        cache_key = (layer_idx, projection)
        _cached_yulan_gdn_ba_weights[cache_key] = param
        other_projection = "a" if projection == "b" else "b"
        other_key = (layer_idx, other_projection)
        if other_key not in _cached_yulan_gdn_ba_weights:
            return []

        b_weight = _cached_yulan_gdn_ba_weights.pop((layer_idx, "b"))
        a_weight = _cached_yulan_gdn_ba_weights.pop((layer_idx, "a"))
        num_key_heads = args.linear_num_key_heads
        num_value_heads = args.linear_num_value_heads
        ba_weight = torch.cat(
            [
                b_weight.reshape(num_key_heads, num_value_heads // num_key_heads, -1),
                a_weight.reshape(num_key_heads, num_value_heads // num_key_heads, -1),
            ],
            dim=1,
        ).flatten(0, 1)
        return [(f"{prefix}.linear_attn.in_proj_ba.weight", ba_weight)]

    if rest in {"self_attention.in_proj.weight", "self_attention.conv1d.weight"}:
        num_key_heads = args.linear_num_key_heads
        num_value_heads = args.linear_num_value_heads
        key_head_dim = args.linear_key_head_dim
        value_head_dim = args.linear_value_head_dim
        key_dim = num_key_heads * key_head_dim
        value_dim = num_value_heads * value_head_dim

    if rest == "self_attention.in_proj.weight":
        # YuLan's fused MCore GDN layout is [z, value, query, key, beta,
        # alpha]. Reorder it to the HF/SGLang qkvz+ba layout, matching
        # YuLan-Pretrain's Qwen3-Next MCore-to-HF synchronizer.
        z, value, query, key, beta, alpha = torch.split(
            param,
            [value_dim, value_dim, key_dim, key_dim, num_value_heads, num_value_heads],
            dim=0,
        )
        values_per_key = num_value_heads // num_key_heads
        qkvz = torch.cat(
            [
                query.reshape(num_key_heads, key_head_dim, -1),
                key.reshape(num_key_heads, key_head_dim, -1),
                value.reshape(num_key_heads, values_per_key * value_head_dim, -1),
                z.reshape(num_key_heads, values_per_key * value_head_dim, -1),
            ],
            dim=1,
        ).flatten(0, 1)
        ba = torch.cat(
            [
                beta.reshape(num_key_heads, values_per_key, -1),
                alpha.reshape(num_key_heads, values_per_key, -1),
            ],
            dim=1,
        ).flatten(0, 1)
        return [
            (f"{prefix}.linear_attn.in_proj_qkvz.weight", qkvz),
            (f"{prefix}.linear_attn.in_proj_ba.weight", ba),
        ]

    if rest == "self_attention.conv1d.weight":
        query, key, value = torch.split(param, [key_dim, key_dim, value_dim], dim=0)
        return [(f"{prefix}.linear_attn.conv1d.weight", torch.cat([query, key, value], dim=0))]

    return None


def _convert_mtp_layer(args, name, param, layer_idx):
    """Convert MTP layer parameters from Megatron to HuggingFace format.

    Handles both wrapper layers (enorm, hnorm, final_layernorm, eh_proj) and
    inner transformer layers for any number of MTP layers.
    """
    # MTP wrapper layers (layer index independent in HF format)
    if "enorm.weight" in name:
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        if param.dim() < 2:
            raise ValueError(f"eh_proj weight expects 2D tensor, got {param.shape}")
        first_half, second_half = param.chunk(2, dim=1)
        new_param = torch.cat([second_half, first_half], dim=1)
        return [("mtp.fc.weight", new_param)]

    # MTP inner transformer layers (keep layer index)
    if "transformer_layer" in name:
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_next_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                final_params.append((hf_name, tensor))
        return final_params

    return None


def _convert_gated_norm_to_hf(name, param, *, layer_idx=None):
    """Map YuLan MCore GatedNorm parameters to the HF/SGLang module names."""
    gated_norm_modules = {
        "final_layernorm": "model.norm",
        "input_layernorm": "input_layernorm",
        "pre_mlp_layernorm": "post_attention_layernorm",
    }
    for mcore_module, hf_module in gated_norm_modules.items():
        prefix = f"{mcore_module}."
        if not name.startswith(prefix):
            continue

        suffix = name.removeprefix(prefix)
        hf_suffix = {
            "norm.weight": "rms_norm.weight",
            "gate_down.weight": "w_down.weight",
            "gate_up.weight": "w_up.weight",
            "gate_up.bias": "w_up.bias",
        }.get(suffix)
        if hf_suffix is None:
            return None

        if mcore_module == "final_layernorm":
            hf_name = f"{hf_module}.{hf_suffix}"
        else:
            if layer_idx is None:
                return None
            hf_name = f"model.layers.{layer_idx}.{hf_module}.{hf_suffix}"
        return [(hf_name, param)]

    return None


def convert_qwen3_next_to_hf(args, name, param):
    """Convert Qwen3 Next model parameters from Megatron to HuggingFace format."""
    # Handle MTP layers
    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]
    if name.startswith("module.module.decoder.final_layernorm."):
        rest = name.removeprefix("module.module.decoder.")
        if (converted := _convert_gated_norm_to_hf(rest, param)) is not None:
            return converted

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        if (converted := _convert_gated_norm_to_hf(rest, param, layer_idx=layer_idx)) is not None:
            return converted

        if (converted := _convert_yulan_gdn_to_hf(args, layer_idx, rest, param)) is not None:
            return converted

        # experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
                return outputs
            elif rest == "linear_fc2":
                outputs = [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
                return outputs
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            # Qwen3.5's gated attention stores two query projections per
            # group.  YuLan Qwen3-Next's full-attention layers use ordinary
            # GQA instead (one query projection per group).  The MCore
            # checkpoint determines the layout unambiguously from this fused
            # projection's per-group width.
            if param.shape[1] == value_num_per_group + 2:
                q_param, k_param, v_param = torch.split(
                    param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
                )
                q_param = q_param.reshape(-1, args.hidden_size)
            elif param.shape[1] == 2 * value_num_per_group + 2:
                q_param, k_param, v_param = torch.split(
                    param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
                )
                q_param = (
                    q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                    .transpose(1, 2)
                    .reshape(-1, args.hidden_size)
                )
            else:
                raise ValueError(
                    f"Unexpected QKV layout for {name}: per-group width {param.shape[1]}, "
                    f"expected {value_num_per_group + 2} or {2 * value_num_per_group + 2}"
                )
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            if param.shape[1] == 2 * value_num_per_group * head_dim + 2 * head_dim:
                q_bias, k_bias, v_bias = torch.split(
                    param,
                    split_size_or_sections=[2 * value_num_per_group * head_dim, head_dim, head_dim],
                    dim=1,
                )
                q_bias = (
                    q_bias.reshape(args.num_query_groups, 2, value_num_per_group, head_dim)
                    .transpose(1, 2)
                    .reshape(-1)
                )
            else:
                q_bias, k_bias, v_bias = torch.split(
                    param,
                    split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                    dim=1,
                )
                q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
        elif rest.startswith("self_attention.") and rest[len("self_attention.") :] in [
            "input_layernorm.weight",
            # linear attn
            "linear_attn.A_log",
            "linear_attn.conv1d.weight",
            "linear_attn.dt_bias",
            "linear_attn.in_proj_ba.weight",
            "linear_attn.in_proj_qkvz.weight",
            "linear_attn.norm.weight",
            "linear_attn.out_proj.weight",
            # gated attn
            "self_attn.k_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.q_proj.weight",
            "self_attn.v_proj.weight",
        ]:
            rest = rest[len("self_attention.") :]
            return [(f"model.layers.{layer_idx}.{rest}", param)]

    raise ValueError(f"Unknown parameter name: {name}")
