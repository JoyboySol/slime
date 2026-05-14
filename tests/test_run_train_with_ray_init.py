import importlib.util
import os
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "run_train_with_ray_init.py"
SPEC = importlib.util.spec_from_file_location("run_train_with_ray_init", MODULE_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)
_resolve_ray_num_cpus = module._resolve_ray_num_cpus


def test_resolve_ray_num_cpus_uses_explicit_env(monkeypatch):
    monkeypatch.setenv("RAY_NUM_CPUS", "12")

    assert _resolve_ray_num_cpus(num_gpus=5) == 12


def test_resolve_ray_num_cpus_caps_large_hosts_by_gpu_count(monkeypatch):
    monkeypatch.delenv("RAY_NUM_CPUS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 256)

    assert _resolve_ray_num_cpus(num_gpus=5) == 40


def test_resolve_ray_num_cpus_keeps_small_hosts_unchanged(monkeypatch):
    monkeypatch.delenv("RAY_NUM_CPUS", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)

    assert _resolve_ray_num_cpus(num_gpus=5) == 8
