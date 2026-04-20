from argparse import Namespace

from slime.backends.megatron_utils.model_name_utils import infer_update_weight_model_name


def test_infer_update_weight_model_name_prefers_explicit_model_name():
    args = Namespace(model_name="forced_name", spec=["slime_plugins.models.yulan_mini", "get_yulan_mini_spec"])

    inferred = infer_update_weight_model_name(
        args,
        hf_config=type("ForcedConfig", (), {})(),
    )

    assert inferred == "forced_name"


def test_infer_update_weight_model_name_uses_spec_suffix_for_yulan():
    args = Namespace(model_name=None, spec=["slime_plugins.models.yulan_mini", "get_yulan_mini_spec"])

    class Qwen3NextConfig:
        pass

    inferred = infer_update_weight_model_name(args, hf_config=Qwen3NextConfig())

    assert inferred == "yulan_mini"


def test_infer_update_weight_model_name_falls_back_to_hf_config_name():
    args = Namespace(model_name=None, spec=None)

    class Qwen3NextConfig:
        pass

    inferred = infer_update_weight_model_name(args, hf_config=Qwen3NextConfig())

    assert inferred == "qwen3nextconfig"
