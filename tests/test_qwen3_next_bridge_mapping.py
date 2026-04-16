import importlib.util
import sys
import types
from pathlib import Path

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

        def _weight_name_mapping_mcore_to_hf(self, name: str) -> list[str]:
            if "mlp" in name or "pre_mlp_layernorm" in name:
                return self._weight_name_mapping_mlp(name)
            raise NotImplementedError(f"Unsupported parameter name: {name}")

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

        def _weight_name_mapping_attention(self, name: str) -> list[str]:
            raise NotImplementedError(f"Unexpected attention mapping lookup: {name}")

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


def load_bridge_module():
    install_bridge_stubs()
    module_path = Path(__file__).resolve().parents[1] / "slime_plugins" / "mbridge" / "qwen3_next.py"
    module_name = "test_qwen3_next_bridge_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dense_mlp_layernorm_mapping_uses_post_attention_layernorm():
    module = load_bridge_module()
    bridge = module.Qwen3NextBridge.__new__(module.Qwen3NextBridge)

    names = bridge._weight_name_mapping_mcore_to_hf("decoder.layers.0.mlp.linear_fc1.layer_norm_weight")

    assert names == ["model.layers.0.post_attention_layernorm.weight"]


def test_mtp_dense_mlp_layernorm_mapping_uses_mtp_prefix():
    module = load_bridge_module()
    bridge = module.Qwen3NextBridge.__new__(module.Qwen3NextBridge)

    names = bridge._convert_mtp_param("mtp.layers.0.transformer_layer.mlp.linear_fc1.layer_norm_weight")

    assert names == ["mtp.layers.0.post_attention_layernorm.weight"]


def test_weight_to_mcore_format_supports_standard_gqa_qkv_layout():
    module = load_bridge_module()
    bridge = module.Qwen3NextBridge.__new__(module.Qwen3NextBridge)
    bridge.hf_config = types.SimpleNamespace(
        hidden_size=6,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=1,
    )

    q = torch.arange(6 * 6, dtype=torch.float32).view(6, 6)
    k = torch.arange(3 * 6, dtype=torch.float32).view(3, 6) + 100
    v = torch.arange(3 * 6, dtype=torch.float32).view(3, 6) + 200

    packed = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.linear_qkv.weight", [q, k, v])

    assert packed.shape == (12, 6)


def test_weight_to_mcore_format_supports_gated_qkv_layout():
    module = load_bridge_module()
    bridge = module.Qwen3NextBridge.__new__(module.Qwen3NextBridge)
    bridge.hf_config = types.SimpleNamespace(
        hidden_size=6,
        num_attention_heads=6,
        num_key_value_heads=3,
        head_dim=1,
    )

    q = torch.arange(12 * 6, dtype=torch.float32).view(12, 6)
    k = torch.arange(3 * 6, dtype=torch.float32).view(3, 6) + 100
    v = torch.arange(3 * 6, dtype=torch.float32).view(3, 6) + 200

    packed = bridge._weight_to_mcore_format("decoder.layers.0.self_attention.linear_qkv.weight", [q, k, v])

    assert packed.shape == (18, 6)
