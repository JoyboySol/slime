import sys
import types

import torch.nn as nn


def _install_fake_megatron():
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_mpu = types.SimpleNamespace(
        get_tensor_model_parallel_group=lambda: None,
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
    )
    tensor_parallel = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, **kwargs: x,
        scatter_to_sequence_parallel_region=lambda x, **kwargs: x,
    )

    inference_contexts = types.ModuleType("megatron.core.inference.contexts")
    inference_contexts.BaseInferenceContext = object

    packed_seq_params = types.ModuleType("megatron.core.packed_seq_params")
    packed_seq_params.PackedSeqParams = object

    transformer_module = types.ModuleType("megatron.core.transformer.module")
    transformer_module.MegatronModule = nn.Module

    spec_utils = types.ModuleType("megatron.core.transformer.spec_utils")
    spec_utils.ModuleSpec = object

    transformer_block = types.ModuleType("megatron.core.transformer.transformer_block")
    transformer_block.get_num_layers_to_build = lambda config, vp_stage=None: 0

    transformer_layer = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer.get_transformer_layer_offset = lambda config, vp_stage=None: 0

    gpt_layer_specs = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    gpt_layer_specs.get_gpt_decoder_block_spec = lambda config, **kwargs: types.SimpleNamespace(layer_specs=[])

    megatron.core = megatron_core
    megatron_core.mpu = megatron_mpu
    megatron_core.tensor_parallel = tensor_parallel

    sys.modules.setdefault("megatron", megatron)
    sys.modules.setdefault("megatron.core", megatron_core)
    sys.modules.setdefault("megatron.core.mpu", megatron_mpu)
    sys.modules.setdefault("megatron.core.tensor_parallel", tensor_parallel)
    sys.modules.setdefault("megatron.core.inference.contexts", inference_contexts)
    sys.modules.setdefault("megatron.core.packed_seq_params", packed_seq_params)
    sys.modules.setdefault("megatron.core.transformer.module", transformer_module)
    sys.modules.setdefault("megatron.core.transformer.spec_utils", spec_utils)
    sys.modules.setdefault("megatron.core.transformer.transformer_block", transformer_block)
    sys.modules.setdefault("megatron.core.transformer.transformer_layer", transformer_layer)
    sys.modules.setdefault("megatron.core.models.gpt.gpt_layer_specs", gpt_layer_specs)


_install_fake_megatron()

from slime_plugins.models import qwen3_next, yulan_mini


class _DummyLinearAttention(nn.Module):
    def forward(self, hidden_states, cu_seqlens=None):
        del cu_seqlens
        return hidden_states


def _fake_hf_attention_init(self, args, config, layer_number, cp_comm_type="p2p", pg_collection=None):
    del args, config, cp_comm_type, pg_collection
    nn.Module.__init__(self)
    self.hf_layer_idx = layer_number - 1
    self.hf_config = types.SimpleNamespace(
        hidden_size=16,
        rms_norm_eps=1e-6,
    )


def test_yulan_linear_attention_uses_llama_rmsnorm(monkeypatch):
    monkeypatch.setattr(yulan_mini.HuggingfaceAttention, "__init__", _fake_hf_attention_init)
    monkeypatch.setattr(yulan_mini, "YuLanMiniGatedDeltaNet", lambda config, idx: _DummyLinearAttention())
    monkeypatch.setattr(yulan_mini, "Qwen3NextAttention", object(), raising=False)

    attn = yulan_mini.Attention(args=types.SimpleNamespace(sequence_parallel=False), config=None, layer_number=1)

    assert type(attn.input_layernorm).__name__ == "LlamaRMSNorm"


def test_qwen3_next_linear_attention_uses_llama_rmsnorm(monkeypatch):
    monkeypatch.setattr(qwen3_next.HuggingfaceAttention, "__init__", _fake_hf_attention_init)
    monkeypatch.setattr(qwen3_next, "Qwen3NextGatedDeltaNet", lambda config, idx: _DummyLinearAttention())
    monkeypatch.setattr(qwen3_next, "Qwen3NextAttention", object(), raising=False)

    attn = qwen3_next.Attention(args=types.SimpleNamespace(sequence_parallel=False), config=None, layer_number=1)

    assert type(attn.input_layernorm).__name__ == "LlamaRMSNorm"
