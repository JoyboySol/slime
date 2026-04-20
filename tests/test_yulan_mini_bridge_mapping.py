import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def install_bridge_stubs():
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    gpt_layer_specs_mod.get_gpt_mtp_block_spec = lambda _config, transformer_layer_spec, **_kwargs: (
        "mtp-spec",
        transformer_layer_spec,
    )

    mbridge_mod = types.ModuleType("mbridge")
    mbridge_core_mod = types.ModuleType("mbridge.core")
    mbridge_models_mod = types.ModuleType("mbridge.models")

    def register_model(_names):
        def decorator(cls):
            return cls

        return decorator

    class Qwen2MoEBridge:
        _ATTENTION_MAPPING = {}

        def _weight_to_mcore_format(self, _mcore_weights_name, hf_weights):
            return torch.cat(hf_weights, dim=0)

    mbridge_core_mod.register_model = register_model
    mbridge_models_mod.Qwen2MoEBridge = Qwen2MoEBridge

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["mbridge"] = mbridge_mod
    sys.modules["mbridge.core"] = mbridge_core_mod
    sys.modules["mbridge.models"] = mbridge_models_mod


def load_bridge_module(name: str):
    install_bridge_stubs()
    module_path = Path(__file__).resolve().parents[1] / "slime_plugins" / "mbridge" / f"{name}.py"
    module_name = f"test_{name}_bridge_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_yulan_mini_bridge_matches_qwen3_next_qkv_packing_for_gated_layout():
    yulan_module = load_bridge_module("yulan_mini")
    qwen_module = load_bridge_module("qwen3_next")

    yulan_bridge = yulan_module.YuLanMiniBridge.__new__(yulan_module.YuLanMiniBridge)
    qwen_bridge = qwen_module.Qwen3NextBridge.__new__(qwen_module.Qwen3NextBridge)
    hf_config = types.SimpleNamespace(
        hidden_size=6,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=1,
    )
    yulan_bridge.hf_config = hf_config
    qwen_bridge.hf_config = hf_config

    q = torch.arange(12 * 6, dtype=torch.float32).view(12, 6)
    k = torch.arange(3 * 6, dtype=torch.float32).view(3, 6) + 100
    v = torch.arange(3 * 6, dtype=torch.float32).view(3, 6) + 200

    yulan_packed = yulan_bridge._weight_to_mcore_format(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        [q, k, v],
    )
    qwen_packed = qwen_bridge._weight_to_mcore_format(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        [q, k, v],
    )

    assert torch.equal(yulan_packed, qwen_packed)
