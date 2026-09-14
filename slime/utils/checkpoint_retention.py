"""Checkpoint retention for train-loop snapshots."""

import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)
_ITERATION_RE = re.compile(r"^iter_(\d+)$")


def _checkpoint_id(name: str, prefix: str, suffix: str = "") -> int | None:
    if not name.startswith(prefix) or (suffix and not name.endswith(suffix)):
        return None
    value = name[len(prefix) : len(name) - len(suffix) if suffix else None]
    return int(value) if value.isdigit() else None


def _keep_ids(current_id: int, ids: set[int], latest_count: int, interval: int) -> set[int]:
    if latest_count < 1:
        raise ValueError(f"latest_count must be positive, got {latest_count}")
    if interval < 1:
        raise ValueError(f"interval must be positive, got {interval}")
    latest = sorted(ids)[-latest_count:]
    # rollout_id is zero-based; the user-facing training step is one-based.
    periodic = {checkpoint_id for checkpoint_id in ids if (checkpoint_id + 1) % interval == 0}
    return set(latest) | periodic | ({current_id} & ids)


def _remove_old_directories(
    directories: list[tuple[int, Path]], current_id: int, latest_count: int, interval: int
) -> list[Path]:
    ids = {checkpoint_id for checkpoint_id, _ in directories}
    keep = _keep_ids(current_id, ids, latest_count, interval)
    removed: list[Path] = []
    for checkpoint_id, path in directories:
        if checkpoint_id not in keep:
            shutil.rmtree(path)
            removed.append(path)
    return removed


def prune_checkpoint_storage(
    save_dir: str | Path | None,
    save_hf_template: str | None,
    current_id: int,
    *,
    latest_count: int = 2,
    interval: int = 20,
) -> None:
    """Remove old MCore and HF snapshots while preserving the active snapshot."""
    removed: list[Path] = []

    if save_dir is not None:
        save_path = Path(save_dir)
        if save_path.is_dir():
            mcore_dirs = []
            for path in save_path.iterdir():
                match = _ITERATION_RE.fullmatch(path.name)
                if match and path.is_dir():
                    mcore_dirs.append((int(match.group(1)), path))
            removed.extend(_remove_old_directories(mcore_dirs, current_id, latest_count, interval))

    if save_hf_template is not None and "{rollout_id}" in save_hf_template:
        template_path = Path(save_hf_template)
        current_path = Path(save_hf_template.format(rollout_id=current_id))
        parent = current_path.parent
        prefix, suffix = template_path.name.split("{rollout_id}", 1)
        if parent.is_dir():
            hf_dirs = []
            for path in parent.iterdir():
                checkpoint_id = _checkpoint_id(path.name, prefix, suffix)
                if checkpoint_id is not None and path.is_dir():
                    hf_dirs.append((checkpoint_id, path))
            removed.extend(_remove_old_directories(hf_dirs, current_id, latest_count, interval))

    if removed:
        logger.info("Checkpoint retention removed %d old snapshots: %s", len(removed), ", ".join(map(str, removed)))
