import os


def _is_megatron_checkpoint_dir(path: str) -> bool:
    return os.path.exists(os.path.join(path, "latest_checkpointed_iteration.txt"))


def _looks_like_hf_checkpoint(path: str) -> bool:
    if os.path.isfile(path):
        return path.endswith((".safetensors", ".bin"))
    if not os.path.isdir(path):
        return False
    return os.path.exists(os.path.join(path, "config.json"))


def resolve_bridge_initial_load_path(load_path: str | None, ref_load: str | None, hf_checkpoint: str | None) -> str | None:
    """Choose the initial actor load path in bridge mode.

    Prefer an explicit valid Megatron checkpoint or HF checkpoint. If the user
    passes an empty/non-checkpoint work directory (for example a freshly
    created `actor_ckpt`), fall back to `ref_load` and then `hf_checkpoint`.
    """

    if load_path is None:
        return ref_load or hf_checkpoint

    if _is_megatron_checkpoint_dir(load_path) or _looks_like_hf_checkpoint(load_path):
        return load_path

    if os.path.isdir(load_path):
        try:
            if not any(os.scandir(load_path)):
                return ref_load or hf_checkpoint
        except FileNotFoundError:
            return ref_load or hf_checkpoint

    return ref_load or hf_checkpoint
