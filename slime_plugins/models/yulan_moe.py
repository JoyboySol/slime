"""Megatron model provider for the local YuLan-MoE Qwen3-Next checkpoint."""

import copy
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from megatron.core.models.mamba import MambaModel
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols, allocate_layers
from megatron.core.transformer.transformer_block import TransformerBlock, TransformerBlockSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import WrappedTensor, deprecate_inference_params, make_viewless_tensor
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

    def forward(
        self,
        hidden_states,
        attention_mask,
        inference_context: BaseInferenceContext = None,
        rotary_pos_emb=None,
        *,
        inference_params: BaseInferenceContext = None,
    ):
        """Run the native stack while forwarding Slime's packed THD metadata.

        The base MambaStack predates YuLan's packed-THD CP path and drops
        ``packed_seq_params`` before calling TransformerLayer.  The actor uses
        a common Megatron forward API, so keep the compatibility state on this
        local subclass and pass it to every GDN/MoE layer.
        """
        packed_seq_params = getattr(self, "_slime_packed_seq_params", None)
        padding_mask = getattr(self, "_slime_padding_mask", None)
        inference_context = deprecate_inference_params(inference_context, inference_params)
        if not self.pre_process:
            hidden_states = self.input_tensor
        if isinstance(hidden_states, WrappedTensor):
            hidden_states = hidden_states.unwrap()
        if inference_context:
            assert inference_context.is_static_batching(), "Mamba currently does not support dynamic inference batching."
            inference_context.max_seqlen = inference_context.max_sequence_length
            inference_context.seqlen_offset = inference_context.sequence_len_offset
        if (
            ((self.config.cuda_graph_impl == "local" and "full_iteration" not in self.config.cuda_graph_scope)
             or self.config.flash_decode)
            and inference_context
            and inference_context.is_static_batching()
            and not self.training
        ):
            sequence_len_offset = torch.tensor(
                [inference_context.sequence_len_offset] * hidden_states.shape[1],
                dtype=torch.int32,
                device="cuda",
            )
        else:
            sequence_len_offset = None
        use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
        use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
        outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()
        with outer_fp8_context:
            for layer in self.layers:
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context else nullcontext()
                )
                with inner_fp8_context:
                    if isinstance(layer, TransformerLayer):
                        hidden_states, _ = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                            rotary_pos_emb=rotary_pos_emb,
                            sequence_len_offset=sequence_len_offset,
                            packed_seq_params=packed_seq_params,
                            padding_mask=padding_mask,
                        )
                    else:
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            inference_context=inference_context,
                        )
                if isinstance(hidden_states, tuple):
                    hidden_states = hidden_states[0]
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_norm(hidden_states)
        return make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )


class _YuLanTransformerBlock(TransformerBlock):
    """TransformerBlock adapter that preserves the MambaModel constructor ABI.

    YuLan's GDN CP implementation performs layout transitions around the whole
    transformer block.  The normal MambaStack iterates layers itself and cannot
    run that block-level protocol, so this opt-in adapter uses the existing
    TransformerBlock implementation while accepting MambaStack's constructor
    arguments and checkpoint layout.
    """

    def __init__(
        self,
        config,
        submodules,
        residual_in_fp32=False,
        pre_process=True,
        hybrid_attention_ratio=0.0,
        hybrid_mlp_ratio=0.0,
        hybrid_override_pattern=None,
        post_layer_norm=True,
        post_process=True,
        device=None,
        dtype=None,
        pg_collection=None,
        vp_stage=None,
    ):
        del residual_in_fp32, hybrid_attention_ratio, hybrid_mlp_ratio
        del hybrid_override_pattern, device, dtype
        super().__init__(
            config=config,
            spec=submodules,
            post_layer_norm=post_layer_norm,
            pre_process=pre_process,
            post_process=post_process,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

    def forward(self, *args, **kwargs):
        packed_seq_params = getattr(self, "_slime_packed_seq_params", None)
        padding_mask = getattr(self, "_slime_padding_mask", None)
        kwargs.setdefault("packed_seq_params", packed_seq_params)
        kwargs.setdefault("padding_mask", padding_mask)
        return super().forward(*args, **kwargs)


class _YuLanMambaModel(MambaModel):
    """Match the converter's omission of the native output layer's null state."""

    def forward(self, *args, packed_seq_params=None, loss_mask=None, padding_mask=None, **kwargs):
        """Accept slime's packed-batch arguments used by Transformer models.

        MambaModel consumes the packed sequence layout through the normal token
        tensors and does not expose these Transformer-specific keyword arguments.
        They are nevertheless supplied by slime's common Megatron train path.
        """
        del loss_mask
        decoder = self.decoder
        decoder._slime_packed_seq_params = packed_seq_params
        decoder._slime_padding_mask = padding_mask
        try:
            return super().forward(*args, **kwargs)
        finally:
            decoder._slime_packed_seq_params = None
            decoder._slime_padding_mask = None

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
    stack_submodules = MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=_YuLanGDNLayer,
                submodules=_YuLanGDNSubmodules(
                    dense=dense_gdn_layer.submodules,
                    moe=moe_gdn_layer.submodules,
                ),
            ),
            attention_layer=stack_spec.submodules.attention_layer,
            mlp_layer=stack_spec.submodules.mlp_layer,
        )
    if os.environ.get("SLIME_USE_YULAN_TRANSFORMER_BLOCK", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        layer_specs = []
        for layer_type in allocate_layers(
            args.num_layers,
            args.hybrid_attention_ratio,
            args.hybrid_mlp_ratio,
            args.hybrid_override_pattern,
        ):
            layer_specs.append(
                {
                    Symbols.MAMBA: stack_submodules.mamba_layer,
                    Symbols.ATTENTION: stack_submodules.attention_layer,
                    Symbols.MLP: stack_submodules.mlp_layer,
                }[layer_type]
            )
        from megatron.core.extensions.transformer_engine import TENorm

        return ModuleSpec(
            module=_YuLanTransformerBlock,
            submodules=TransformerBlockSubmodules(layer_specs=layer_specs, layer_norm=TENorm),
        )
    return ModuleSpec(module=_YuLanMambaStack, submodules=stack_submodules)


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
