# Cross Tokenizer OPD Generation Evidence Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve generation-native byte-evidence attempt state across fallback so `run.log` and replay summaries report true hit/incomplete/invalid rates even when training falls back to `recorded_builder`.

**Architecture:** Keep final student-alignment selection semantics unchanged for training, but split "generation evidence was attempted" from "final recorded alignment source". Store dedicated generation-byte-evidence observability fields on `Sample`, transport them through rollout/train/replay, and compute metrics from those fields instead of inferring from the final alignment source.

**Tech Stack:** Python, pytest, dataclasses, rollout/train logging, replay diagnostics

---

### Task 1: Lock the intended observability behavior with tests

**Files:**
- Modify: `tests/test_opd_byte_chunk.py`
- Modify: `tests/test_rollout_logging.py`
- Test: `tests/test_opd_byte_chunk.py`
- Test: `tests/test_rollout_logging.py`

- [ ] **Step 1: Write the failing regression tests**

Add a unit test in `tests/test_opd_byte_chunk.py` that builds a sample where generation token texts are invalid, tokenizer reconstruction succeeds, and the final source remains `recorded_builder`, while dedicated generation-byte-evidence fields still record `attempted=True`, `complete=False`, `validated=False`, and the invalid error.

Add a logging test in `tests/test_rollout_logging.py` that passes rollout data with `recorded_builder` as the final source but with generation-byte-evidence attempt lists showing one invalid attempt, then assert the logged metrics report non-zero generation-byte-evidence sample/hit/incomplete/invalid counts.

- [ ] **Step 2: Run the new tests and verify they fail**

Run: `./.venv/bin/pytest tests/test_opd_byte_chunk.py -k "generation_evidence" -v`
Expected: FAIL because generation-byte-evidence attempt fields/metrics are not preserved separately yet.

Run: `./.venv/bin/pytest tests/test_rollout_logging.py -k "generation_byte_evidence" -v`
Expected: FAIL because logging still infers metrics from final alignment source only.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_opd_byte_chunk.py tests/test_rollout_logging.py
git commit -m "test: lock generation evidence observability behavior"
```

### Task 2: Preserve generation-byte-evidence attempt state independently from final alignment source

**Files:**
- Modify: `slime/utils/types.py`
- Modify: `slime/rollout/sglang_rollout.py`
- Modify: `slime/rollout/on_policy_distillation.py`
- Modify: `slime/ray/rollout.py`
- Modify: `slime/utils/opd_utils.py`
- Test: `tests/test_opd_byte_chunk.py`

- [ ] **Step 1: Add explicit Sample fields for generation-byte-evidence attempt state**

Add dataclass fields for:
- attempted / sample_count participation
- complete
- validated
- error
- metadata

These fields must survive `to_dict()` / `from_dict()` without special handling beyond the dataclass field list.

- [ ] **Step 2: Record attempt state in rollout generation**

Update `_update_generation_byte_evidence(...)` in `slime/rollout/sglang_rollout.py` so every attempt records the dedicated observability fields even when byte reconstruction fails.

- [ ] **Step 3: Preserve attempt state through training-side fallback**

Update `_record_student_opd_alignment(...)` in `slime/rollout/on_policy_distillation.py` so fallback to `recorded_builder` or `generation_logprobs_text` does not erase the earlier generation-byte-evidence attempt state.

- [ ] **Step 4: Transport the new fields through rollout/train data helpers**

Update `slime/ray/rollout.py` and `slime/utils/opd_utils.py` so the new generation-byte-evidence fields are exported, reloaded, and visible to diagnostics.

- [ ] **Step 5: Run focused tests and verify green**

Run: `./.venv/bin/pytest tests/test_opd_byte_chunk.py -k "generation_evidence" -v`
Expected: PASS

### Task 3: Recompute logging and replay metrics from dedicated attempt fields

**Files:**
- Modify: `slime/utils/opd_metric_utils.py`
- Modify: `slime/backends/megatron_utils/data.py`
- Modify: `scripts/replay_debug_rollout_opd.py`
- Modify: `tests/test_rollout_logging.py`
- Modify: `tests/test_opd_byte_chunk.py`

- [ ] **Step 1: Update metric helpers**

Change `summarize_opd_alignment(...)` to accept dedicated generation-byte-evidence sample/complete/validated values and compute:
- sample_count
- hit_count / hit_rate
- incomplete_count / incomplete_rate
- invalid_count / invalid_rate

Use final alignment source only for final-source breakdown, not for generation evidence metrics.

- [ ] **Step 2: Update train logging and replay aggregation**

Wire the new lists into `log_rollout_data(...)` and `_summarize_generation_byte_evidence_metrics(...)` so `run.log` and replay summary both report the dedicated attempt-based rates.

- [ ] **Step 3: Run the logging/replay regression tests**

Run: `./.venv/bin/pytest tests/test_rollout_logging.py tests/test_opd_byte_chunk.py -k "generation_byte_evidence or replay_summary" -v`
Expected: PASS

### Task 4: Make launcher reuse mode the default documented path for real smoke validation

**Files:**
- Modify: `scripts/run-yulan-cross-tokenizer-opd-smoke.sh`
- Modify: `scripts/run-yulan-cross-tokenizer-opd-train.sh`
- Modify: `.ai/output/09_跨词表opd经验总结.md`
- Test: `tests/test_cross_tokenizer_opd_launcher_scripts.py`

- [ ] **Step 1: Add or update launcher-script regression coverage**

Extend launcher tests so `REUSE_EXISTING_SERVERS=1` requires explicit external engine/router inputs and does not kill live SGLang services.

- [ ] **Step 2: Tighten launcher messaging**

Make the scripts print the reuse-mode contract clearly, including the exact env vars required to keep teacher / rollout / ray services alive across smoke runs.

- [ ] **Step 3: Run launcher regression tests**

Run: `./.venv/bin/pytest tests/test_cross_tokenizer_opd_launcher_scripts.py -v`
Expected: PASS

### Task 5: Verify end to end with real smoke and preserved services

**Files:**
- Verify only: `workdirs/.../logs/run.log`
- Verify only: `workdirs/.../debug_rollouts/*.pt`

- [ ] **Step 1: Run targeted Python verification**

Run:
`./.venv/bin/pytest tests/test_opd_byte_chunk.py tests/test_rollout_logging.py tests/test_analyze_opd_alignment.py tests/test_student_logprob_alignment.py -v`

Run:
`./.venv/bin/python -m py_compile scripts/replay_debug_rollout_opd.py slime/utils/opd_metric_utils.py slime/rollout/on_policy_distillation.py slime/rollout/sglang_rollout.py slime/ray/rollout.py`

- [ ] **Step 2: Run real smoke with service reuse**

First launch teacher / rollout services once, then run the smoke/train launcher with:
- `REUSE_EXISTING_SERVERS=1`
- `ROLLOUT_EXTERNAL_ENGINE_ADDRS=...`
- `SGLANG_ROUTER_IP=...`
- `SGLANG_ROUTER_PORT=...`

Expected:
- services stay alive after the training command exits
- `run.log` reports non-zero generation-byte-evidence sample count on invalid-attempt batches
- replay summary prints the same invalid/incomplete accounting

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-04-21-cross-tokenizer-opd-generation-evidence-observability.md \
  slime/utils/types.py \
  slime/rollout/sglang_rollout.py \
  slime/rollout/on_policy_distillation.py \
  slime/ray/rollout.py \
  slime/utils/opd_utils.py \
  slime/utils/opd_metric_utils.py \
  slime/backends/megatron_utils/data.py \
  scripts/replay_debug_rollout_opd.py \
  scripts/run-yulan-cross-tokenizer-opd-smoke.sh \
  scripts/run-yulan-cross-tokenizer-opd-train.sh \
  tests/test_opd_byte_chunk.py \
  tests/test_rollout_logging.py \
  tests/test_cross_tokenizer_opd_launcher_scripts.py \
  .ai/output/09_跨词表opd经验总结.md
git commit -m "fix: preserve generation evidence observability across fallback"
```
