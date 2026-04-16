import torch
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec

from mbridge.core import register_model
from mbridge.models import Qwen2MoEBridge


@register_model("qwen3_next")
class Qwen3NextBridge(Qwen2MoEBridge):
    _ATTENTION_MAPPING = (
        Qwen2MoEBridge._ATTENTION_MAPPING
        | {
            f"self_attention.{weight_name}": ["model.layers.{layer_number}." + weight_name]
            for weight_name in [
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
            ]
        }
        | {
            "self_attention.linear_qkv.layer_norm_weight": ["model.layers.{layer_number}.input_layernorm.weight"],
            "self_attention.linear_qkv.weight": [
                "model.layers.{layer_number}.self_attn.q_proj.weight",
                "model.layers.{layer_number}.self_attn.k_proj.weight",
                "model.layers.{layer_number}.self_attn.v_proj.weight",
            ],
        }
    )

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc1.layer_norm_weight": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "mlp.linear_fc2.weight": ["model.layers.{layer_number}.mlp.down_proj.weight"],
        "shared_experts.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.shared_expert.gate_proj.weight",
            "model.layers.{layer_number}.mlp.shared_expert.up_proj.weight",
        ],
        "pre_mlp_layernorm": ["model.layers.{layer_number}.post_attention_layernorm.weight"],
        "shared_experts.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.shared_expert.down_proj.weight"
        ],
        "mlp.router.weight": ["model.layers.{layer_number}.mlp.gate.weight"],
        "shared_experts.gate_weight": ["model.layers.{layer_number}.mlp.shared_expert_gate.weight"],
        "mlp.experts.linear_fc1": [
            "model.layers.{layer_number}.mlp.experts.{expert_id}.gate_proj.weight",
            "model.layers.{layer_number}.mlp.experts.{expert_id}.up_proj.weight",
        ],
        "mlp.experts.linear_fc2": ["model.layers.{layer_number}.mlp.experts.{expert_id}.down_proj.weight"],
    }

    def _get_gptmodel_args(self) -> dict:
        """Override to add MTP block spec if needed."""
        ret = super()._get_gptmodel_args()
        if getattr(self.config, "mtp_num_layers", None) is not None:
            transformer_layer_spec = self.config
            mtp_block_spec = get_gpt_mtp_block_spec(self.config, transformer_layer_spec, use_transformer_engine=True)
            ret["mtp_block_spec"] = mtp_block_spec
        return ret

    def _weight_name_mapping_mcore_to_hf(self, mcore_weights_name: str) -> list[str]:
        """Override to handle MTP layer mappings."""
        if "mtp" in mcore_weights_name:
            return self._convert_mtp_param(mcore_weights_name)
        return super()._weight_name_mapping_mcore_to_hf(mcore_weights_name)

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        layer_number = name.split(".")[2]
        convert_names = []
        for keyword, mapping_names in self._MLP_MAPPING.items():
            if keyword in name:
                if "{expert_id}" in mapping_names[0]:
                    expert_id = name.split("weight")[-1]
                    convert_names.extend(
                        [mapped_name.format(layer_number=layer_number, expert_id=expert_id) for mapped_name in mapping_names]
                    )
                else:
                    convert_names.extend([mapped_name.format(layer_number=layer_number) for mapped_name in mapping_names])
                break
        if len(convert_names) == 0:
            raise NotImplementedError(f"Unsupported parameter name: {name}")
        return convert_names

    def _convert_mtp_param(self, name: str) -> list[str]:
        """Convert MTP layer parameters from MCore to HF format."""
        if "mtp.layers." not in name:
            raise NotImplementedError(f"Invalid MTP parameter name: {name}")

        parts = name.split(".")
        mtp_layer_idx = parts[2]  # mtp.layers.{idx}

        direct_name_mapping = {
            f"mtp.layers.{mtp_layer_idx}.eh_proj.weight": "mtp.fc.weight",
            f"mtp.layers.{mtp_layer_idx}.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            f"mtp.layers.{mtp_layer_idx}.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            f"mtp.layers.{mtp_layer_idx}.final_layernorm.weight": "mtp.norm.weight",
        }

        if name in direct_name_mapping:
            return [direct_name_mapping[name]]

        if "transformer_layer" in name:
            proxy_name = name.replace(
                f"mtp.layers.{mtp_layer_idx}.transformer_layer",
                f"decoder.layers.{mtp_layer_idx}",
            )

            if "self_attention" in proxy_name or "input_layernorm.weight" in proxy_name:
                convert_names = super()._weight_name_mapping_attention(proxy_name)
            elif "mlp" in proxy_name or "pre_mlp_layernorm" in proxy_name:
                convert_names = super()._weight_name_mapping_mlp(proxy_name)
            else:
                raise NotImplementedError(f"Unsupported transformer component in MTP: {name}")

            convert_names = [
                cn.replace(f"model.layers.{mtp_layer_idx}", f"mtp.layers.{mtp_layer_idx}") for cn in convert_names
            ]
            return convert_names

        raise NotImplementedError(f"Unsupported MTP parameter name: {name}")

    def _weight_to_mcore_format(
        self, mcore_weights_name: str, hf_weights: list[torch.Tensor]
    ) -> tuple[list[str], list[torch.Tensor]]:
        if "self_attention.linear_qkv." in mcore_weights_name and "layer_norm" not in mcore_weights_name:
            # merge qkv
            assert len(hf_weights) == 3
            hidden_dim = self.hf_config.hidden_size
            num_attention_heads = self.hf_config.num_attention_heads
            num_querys_per_group = num_attention_heads // self.hf_config.num_key_value_heads
            head_dim = getattr(self.hf_config, "head_dim", hidden_dim // num_attention_heads)
            q, k, v = hf_weights

            real_num_key_value_heads = k.shape[0] // head_dim
            if real_num_key_value_heads == 0:
                raise ValueError(f"Invalid k_proj shape for qkv packing: {k.shape}")

            standard_q_rows = real_num_key_value_heads * num_querys_per_group * head_dim
            gated_q_rows = real_num_key_value_heads * 2 * num_querys_per_group * head_dim

            if q.shape[0] == standard_q_rows:
                q = q.view(real_num_key_value_heads, num_querys_per_group, head_dim, -1)
            elif q.shape[0] == gated_q_rows:
                q = (
                    q.view(real_num_key_value_heads, num_querys_per_group, 2, head_dim, -1)
                    .transpose(1, 2)
                    .reshape(real_num_key_value_heads, 2 * num_querys_per_group, head_dim, -1)
                )
            else:
                raise ValueError(
                    "Unexpected q_proj shape for qkv packing: "
                    f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
                )

            k = k.view(real_num_key_value_heads, 1, head_dim, -1)
            v = v.view(real_num_key_value_heads, 1, head_dim, -1)
            out_shape = [-1, hidden_dim] if ".bias" not in mcore_weights_name else [-1]

            qgkv = torch.cat([q, k, v], dim=1).view(*out_shape).contiguous()
            return qgkv

        weight = super()._weight_to_mcore_format(mcore_weights_name, hf_weights)
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = weight.chunk(2, dim=1)
            weight = torch.cat([second_half, first_half], dim=1)
        return weight

    def _weight_to_hf_format(
        self, mcore_weights_name: str, mcore_weights: torch.Tensor
    ) -> tuple[list[str], list[torch.Tensor]]:
        if mcore_weights_name.endswith("eh_proj.weight"):
            first_half, second_half = mcore_weights.chunk(2, dim=1)
            mcore_weights = torch.cat([second_half, first_half], dim=1)
        return super()._weight_to_hf_format(mcore_weights_name, mcore_weights)

    def _build_config(self):
        mtp_args = {}
        if hasattr(self.hf_config, "num_nextn_predict_layers"):
            mtp_args["mtp_num_layers"] = self.hf_config.num_nextn_predict_layers

        base_kwargs = dict(
            use_cpu_initialization=False,
            # Other optimizations
            persist_layer_norm=True,
            bias_activation_fusion=True,
            bias_dropout_fusion=True,
            # Qwen specific
            moe_router_pre_softmax=False,
            qk_layernorm=getattr(self.hf_config, "enable_qk_norm", True),
            # Qwen3 Next specific
            attention_output_gate=getattr(self.hf_config, "attn_output_gate", True),
            **mtp_args,
        )

        if getattr(self.hf_config, "num_experts", 0):
            base_kwargs.update(
                moe_ffn_hidden_size=self.hf_config.moe_intermediate_size,
                moe_router_bias_update_rate=0.001,
                moe_router_topk=self.hf_config.num_experts_per_tok,
                num_moe_experts=self.hf_config.num_experts,
                moe_aux_loss_coeff=self.hf_config.router_aux_loss_coef,
                moe_router_load_balancing_type="none",
                moe_grouped_gemm=True,
                moe_router_score_function=getattr(self.hf_config, "moe_router_score_function", "softmax"),
                moe_shared_expert_gate=True,
            )

        return self._build_base_config(**base_kwargs)
