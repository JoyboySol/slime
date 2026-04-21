# Cross-Tokenizer OPD Generation Evidence Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add phase-1 observability for recorded student alignment provenance and status across rollout, train, and diagnostics.

**Architecture:** Extend `Sample` with explicit recorded-alignment provenance fields, populate them during rollout-side alignment recording, preserve/transport them through train data, and surface them in replay/analysis outputs. This phase is intentionally observability-first and does not change byte-chunk math or fallback policy.

**Tech Stack:** Python, PyTorch, pytest, existing slime rollout/train diagnostics

---

### Task 1: Add failing tests for alignment metadata on `Sample` recording

**Files:**
- Modify: `tests/test_opd_byte_chunk.py`
- Modify: `slime/rollout/on_policy_distillation.py`

- [ ] **Step 1: Write the failing test**

Add tests that assert:
- successful `_record_student_opd_alignment()` sets:
  - `opd_student_alignment_source`
  - `opd_student_alignment_validated`
  - `opd_student_alignment_status`
- failed `_record_student_opd_alignment()` sets:
  - payload fields to `None`
  - `validated=False`
  - status indicating recorded payload missing/failed

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_opd_byte_chunk.py -k 'alignment_status or alignment_validated or alignment_source' -v`
Expected: FAIL because fields are not populated yet.

- [ ] **Step 3: Write minimal implementation**

Update rollout-side alignment recording to populate the new metadata.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_opd_byte_chunk.py -k 'alignment_status or alignment_validated or alignment_source' -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_opd_byte_chunk.py slime/rollout/on_policy_distillation.py slime/utils/types.py
git commit -m "feat: add recorded student alignment provenance metadata"
```

### Task 2: Transport alignment metadata through rollout/train structures

**Files:**
- Modify: `slime/utils/types.py`
- Modify: `slime/ray/rollout.py`
- Modify: `slime/backends/megatron_utils/data.py`
- Test: existing `tests/test_rollout_logging.py`

- [ ] **Step 1: Write the failing test**

Add or extend a test to assert rollout logging skips the new non-scalar
alignment metadata transport fields instead of trying to aggregate them.

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/bin/pytest tests/test_rollout_logging.py -v`
Expected: FAIL if new transport keys are introduced but not skipped.

- [ ] **Step 3: Write minimal implementation**

Propagate alignment metadata lists in train data and ensure logging skips them.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/bin/pytest tests/test_rollout_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add slime/ray/rollout.py slime/backends/megatron_utils/data.py tests/test_rollout_logging.py slime/utils/types.py
git commit -m "feat: transport alignment provenance through rollout data"
```

### Task 3: Surface alignment provenance in replay and analysis

**Files:**
- Modify: `scripts/replay_debug_rollout_opd.py`
- Modify: `scripts/analyze_opd_alignment.py`
- Modify: `tests/test_analyze_opd_alignment.py`
- Modify: `tests/test_opd_byte_chunk.py`

- [ ] **Step 1: Write the failing test**

Add tests asserting replay/analysis outputs include:
- alignment source
- alignment validated flag
- alignment status

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_opd_byte_chunk.py -k 'replay_summary' -v
```
Expected: FAIL because outputs do not yet include the new fields.

- [ ] **Step 3: Write minimal implementation**

Update replay and analysis scripts to expose the new metadata without changing
their existing status behavior.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_opd_byte_chunk.py -k 'replay_summary' -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/replay_debug_rollout_opd.py scripts/analyze_opd_alignment.py tests/test_analyze_opd_alignment.py tests/test_opd_byte_chunk.py
git commit -m "feat: report alignment provenance in opd diagnostics"
```

### Task 4: Run regression verification

**Files:**
- No code changes required

- [ ] **Step 1: Run byte-chunk regression**

Run: `./.venv/bin/pytest tests/test_opd_byte_chunk.py -v`
Expected: PASS.

- [ ] **Step 2: Run diagnostics regression**

Run: `./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v`
Expected: PASS.

- [ ] **Step 3: Run real tokenizer sampled regression**

Run: `./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v`
Expected: PASS for short 128 and long 128.

- [ ] **Step 4: Optional manual evidence check**

Run a replay/analyze command on an existing long rollout dump and verify the new
metadata appears in output.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: verify phase-1 alignment observability regression coverage"
```
