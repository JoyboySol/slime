from __future__ import annotations


def infer_update_weight_model_name(args, hf_config) -> str:
    if getattr(args, "model_name", None):
        return args.model_name

    spec = getattr(args, "spec", None)
    if spec:
        spec_module = spec[0] if isinstance(spec, (list, tuple)) else spec
        if isinstance(spec_module, str) and spec_module:
            return spec_module.rsplit(".", 1)[-1]

    return type(hf_config).__name__.lower()
