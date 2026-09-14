"""Megatron model provider for the local YuLan-MoE Qwen3-Next checkpoint."""

import copy
from dataclasses import dataclass

import torch

from megatron.core.models.mamba import MambaModel
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from megatron_patch.model.qwen3_next.layer_specs import _gdn_layer_spec, get_qwen3_next_layer_spec
from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig


@dataclass
class _YuLanGDNSubmodules:
    dense: object
    moe: object


class _YuLanGDNLayer(TransformerLayer):
    """Let MambaStack host a Transformer-style GDN layer."""

    def __init__(
        self,
        config,
        submodules: _YuLanGDNSubmodules,
        layer_number: int = 1,
        residual_in_fp32: bool = False,
        **kwargs,
    ):
        layer_is_moe = config.moe_layer_freq[layer_number - 1]
        super().__init__(
            config=config,
            submodules=submodules.moe if layer_is_moe else submodules.dense,
            layer_number=layer_number,
            **kwargs,
        )


class _YuLanMambaStack(MambaStack):
    """Use the YuLan final-normalization name and implementation."""

    @property
    def final_norm(self):
        """Runtime alias for the checkpoint's ``final_layernorm`` module."""
        return self.final_layernorm

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "final_norm" in self._modules:
            # MambaStack names this module ``final_norm``, but YuLan's MCore
            # checkpoint calls it ``final_layernorm``.  Register the latter so
            # strict checkpoint loading sees the exact expected key; the
            # property above keeps MambaStack's forward path unchanged.
            del self._modules["final_norm"]
            self.final_layernorm = torch.nn.RMSNorm(
                self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
                device=torch.cuda.current_device(),
                dtype=self.config.params_dtype,
            )


class _YuLanMambaModel(MambaModel):
    """Match the converter's omission of the native output layer's null state."""

    def forward(self, *args, packed_seq_params=None, loss_mask=None, **kwargs):
        """Accept slime's packed-batch arguments used by Transformer models.

        MambaModel consumes the packed sequence layout through the normal token
        tensors and does not expose these Transformer-specific keyword arguments.
        They are nevertheless supplied by slime's common Megatron train path.
        """
        del packed_seq_params, loss_mask
        return super().forward(*args, **kwargs)

    def sharded_state_dict(self, *args, **kwargs):
        state_dict = super().sharded_state_dict(*args, **kwargs)
        state_dict.pop("output_layer._extra_state", None)
        return state_dict


def _get_yulan_stack_spec(args):
    stack_spec = get_qwen3_next_layer_spec(args)

    # Recover the full TransformerLayer (GDN + MLP) from the backend helper.
    # MambaStack passes one Mamba-only compatibility kwarg, accepted above.
    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider

    moe_gdn_layer = _gdn_layer_spec(
        args,
        TESpecProvider(normalization=args.normalization),
        args.normalization,
    )
    dense_args = copy.copy(args)
    dense_args.num_experts = None
    dense_gdn_layer = _gdn_layer_spec(
        dense_args,
        TESpecProvider(normalization=args.normalization),
        args.normalization,
    )
    return ModuleSpec(
        module=_YuLanMambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=_YuLanGDNLayer,
                submodules=_YuLanGDNSubmodules(
                    dense=dense_gdn_layer.submodules,
                    moe=moe_gdn_layer.submodules,
                ),
            ),
            attention_layer=stack_spec.submodules.attention_layer,
            mlp_layer=stack_spec.submodules.mlp_layer,
        ),
    )


def model_provider(pre_process: bool = True, post_process: bool = True, vp_stage: int | None = None):
    """Build the YuLan hybrid GDN/attention model with its native MCore layout."""

    args = get_args()
    config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)
    return _YuLanMambaModel(
        config=config,
        mamba_stack_spec=_get_yulan_stack_spec(args),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )
