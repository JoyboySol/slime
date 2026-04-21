# Cross Tokenizer OPD Hard Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove fallback-driven ambiguity from the cross-tokenizer OPD pipeline so training, replay, and diagnostics all rely on one strict recorded-alignment protocol, while also reorganizing OPD launcher and test coverage into a cleaner long-term shape.

**Architecture:** Treat `token ids + canonical text + validated recorded payload` as the only production path. Generation token text stays observability-only and never upgrades into strong byte evidence or recovery logic. Byte-chunk preparation/alignment failures become hard errors in strict paths, and a new extracted pre-training validation path lets us prove rollout data can enter training without running a full training job.

**Tech Stack:** Python, pytest, bash launcher scripts, dataclasses, rollout/train logging, diagnostic scripts

---

### Task 1: Lock strict no-fallback semantics with failing tests

**Files:**
- Modify: `tests/test_opd_byte_chunk.py`
- Modify: `tests/test_student_logprob_alignment.py`
- Modify: `tests/test_analyze_opd_alignment.py`
- Test: `tests/test_opd_byte_chunk.py`
- Test: `tests/test_student_logprob_alignment.py`
- Test: `tests/test_analyze_opd_alignment.py`

- [ ] **Step 1: Write the failing tests for strict alignment semantics**

Add focused tests that assert:
- invalid `generation_byte_evidence` never becomes final alignment input for training
- `generation_logprobs_text` is not selected as a strong alignment source in the strict production path
- byte-chunk consumers raise instead of sequence-fallback when recorded payload is incomplete or invalid
- diagnostics still report generation-text observability but do not rely on it for reconstruction

- [ ] **Step 2: Run the strict-alignment tests and verify they fail**

Run: `./.venv/bin/pytest tests/test_opd_byte_chunk.py tests/test_student_logprob_alignment.py tests/test_analyze_opd_alignment.py -k "generation or strict or recorded" -v`
Expected: FAIL because current code still allows generation-text reconstruction and sequence fallback behavior.

- [ ] **Step 3: Commit the red tests**

```bash
git add tests/test_opd_byte_chunk.py tests/test_student_logprob_alignment.py tests/test_analyze_opd_alignment.py
git commit -m "test: lock strict cross-tokenizer opd semantics"
```

### Task 2: Remove fallback-driven production paths from OPD alignment

**Files:**
- Modify: `slime/rollout/sglang_rollout.py`
- Modify: `slime/rollout/on_policy_distillation.py`
- Modify: `slime/utils/opd_utils.py`
- Modify: `slime/utils/types.py`
- Modify: `slime/ray/rollout.py`
- Test: `tests/test_opd_byte_chunk.py`
- Test: `tests/test_student_logprob_alignment.py`

- [ ] **Step 1: Demote generation token text to observability-only**

Keep collecting `output_token_logprobs[*][2]` metadata when available, but stop treating it as production byte evidence. Remove code that promotes generation token texts into `opd_student_response_bytes` / `opd_student_token_byte_spans`.

- [ ] **Step 2: Make training-side student alignment selection strict**

Update `_record_student_opd_alignment(...)` so the only valid production builder is:
- preserved recorded payload already validated
- or token-id / canonical-text reconstruction via tokenizer token pieces

Remove `generation_logprobs_text` as a final alignment source.

- [ ] **Step 3: Remove sequence fallback from strict byte-chunk consumers**

Update `compute_byte_chunk_aligned_log_probs(...)` and `compute_byte_chunk_reverse_kl(...)` so strict paths raise on:
- incomplete recorded payload
- invalid recorded payload
- student/teacher byte reconstruction failure
- chunk alignment failure

If compatibility toggles must remain for other callers, isolate them behind explicit non-default compatibility helpers rather than the production path.

- [ ] **Step 4: Run focused unit tests and verify green**

Run: `./.venv/bin/pytest tests/test_opd_byte_chunk.py tests/test_student_logprob_alignment.py -v`
Expected: PASS

### Task 3: Extract a pre-training pipeline validator that does not run full training

**Files:**
- Modify: `slime/backends/megatron_utils/data.py`
- Modify: `slime/backends/megatron_utils/loss.py`
- Modify: `scripts/replay_debug_rollout_opd.py`
- Create: `tests/test_cross_tokenizer_opd_training_entry.py`
- Test: `tests/test_cross_tokenizer_opd_training_entry.py`

- [ ] **Step 1: Add a narrow helper for “can this rollout data enter byte-chunk training?”**

Extract a helper that:
- loads student/teacher tokenizers
- validates recorded student alignment evidence
- computes byte-chunk aligned student/teacher log-prob tensors
- returns a compact structured summary suitable for tests and diagnostics

This helper must not require launching Megatron training or stepping an optimizer.

- [ ] **Step 2: Write the red end-to-end entry test**

Add a test that builds one realistic rollout-data sample and proves the extracted helper can validate the exact pre-training data path without running training.

- [ ] **Step 3: Run the new entry test and verify red then green**

Run: `./.venv/bin/pytest tests/test_cross_tokenizer_opd_training_entry.py -v`
Expected before implementation: FAIL because no extracted helper exists yet.
Expected after implementation: PASS

- [ ] **Step 4: Wire replay diagnostics to the helper where appropriate**

Reuse the extracted helper in `scripts/replay_debug_rollout_opd.py` when it reduces duplicated training-entry logic.

### Task 4: Reorganize OPD-related tests and launcher-script coverage

**Files:**
- Modify: `tests/test_cross_tokenizer_opd_launcher_scripts.py`
- Modify: `tests/test_rollout_logging.py`
- Modify: `tests/test_rollout_logging_non_scalar_rewards.py`
- Modify: `tests/test_real_tokenizer_sampled_opd_alignment.py`
- Modify: `scripts/run-yulan-cross-tokenizer-opd-smoke.sh`
- Modify: `scripts/run-yulan-cross-tokenizer-opd-train.sh`
- Test: `tests/test_cross_tokenizer_opd_launcher_scripts.py`

- [ ] **Step 1: Define a cleaner split between unit, launcher, and real-environment coverage**

Keep:
- strict byte-chunk/unit behavior in `tests/test_opd_byte_chunk.py`
- rollout generation-text observability in `tests/test_student_logprob_alignment.py`
- launcher contract tests in `tests/test_cross_tokenizer_opd_launcher_scripts.py`
- real tokenizer regression in `tests/test_real_tokenizer_sampled_opd_alignment.py`

Move or trim assertions so each file owns one responsibility instead of mixed concerns.

- [ ] **Step 2: Simplify launcher script tests around stable contracts**

Focus launcher tests on invariant contract checks:
- reuse/external-server mode env contract
- no unintended server lifecycle management in reuse mode
- required routing/env vars
- optional debug rollout/replay hooks

Avoid overly brittle string assertions when a smaller contract assertion is enough.

- [ ] **Step 3: Tighten script messaging and helper structure**

Refactor the two YuLan OPD launcher scripts so shared behavior is grouped into clearly named shell helpers and the reuse-mode contract is easy to scan.

- [ ] **Step 4: Run launcher/logging/real-tokenizer tests**

Run: `./.venv/bin/pytest tests/test_cross_tokenizer_opd_launcher_scripts.py tests/test_rollout_logging.py tests/test_rollout_logging_non_scalar_rewards.py -v`
Expected: PASS

### Task 5: Update diagnostics and docs to match the cleaned protocol

**Files:**
- Modify: `scripts/analyze_opd_alignment.py`
- Modify: `.ai/output/08_跨词表opd重构.md`
- Modify: `.ai/output/09_跨词表opd经验总结.md`
- Modify: `.ai/output/11_YuLan_tokenizer行为分析.md`
- Test: `tests/test_analyze_opd_alignment.py`

- [ ] **Step 1: Align diagnostic wording with strict protocol**

Make diagnostics distinguish clearly between:
- recorded payload used
- token-id builder used
- generation text observed but ignored for production alignment
- hard failure due to invalid/missing payload

- [ ] **Step 2: Refresh working notes**

Update the OPD notes to reflect the new rule:
- no production fallback
- generation text is observability only
- training-entry validation can be run without full training

- [ ] **Step 3: Run diagnostics tests**

Run: `./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v`
Expected: PASS

### Task 6: Verify the cleaned pipeline end to end

**Files:**
- Verify only: `scripts/replay_debug_rollout_opd.py`
- Verify only: `workdirs/.../run.log`

- [ ] **Step 1: Run targeted regression suite**

Run:
`./.venv/bin/pytest tests/test_opd_byte_chunk.py tests/test_student_logprob_alignment.py tests/test_analyze_opd_alignment.py tests/test_cross_tokenizer_opd_training_entry.py tests/test_cross_tokenizer_opd_launcher_scripts.py tests/test_rollout_logging.py tests/test_rollout_logging_non_scalar_rewards.py -v`

Run:
`./.venv/bin/python -m py_compile scripts/analyze_opd_alignment.py scripts/replay_debug_rollout_opd.py slime/utils/opd_utils.py slime/rollout/on_policy_distillation.py slime/rollout/sglang_rollout.py slime/backends/megatron_utils/data.py slime/backends/megatron_utils/loss.py`

- [ ] **Step 2: Run extracted training-entry validation on saved rollout data**

Use the new helper or replay script against a saved debug rollout artifact and confirm it can validate the training-entry byte-chunk path without launching full training.

- [ ] **Step 3: Commit the hard cleanup**

```bash
git add docs/superpowers/plans/2026-04-21-cross-tokenizer-opd-hard-cleanup.md \
  slime/rollout/sglang_rollout.py \
  slime/rollout/on_policy_distillation.py \
  slime/utils/opd_utils.py \
  slime/utils/types.py \
  slime/ray/rollout.py \
  slime/backends/megatron_utils/data.py \
  slime/backends/megatron_utils/loss.py \
  scripts/analyze_opd_alignment.py \
  scripts/replay_debug_rollout_opd.py \
  scripts/run-yulan-cross-tokenizer-opd-smoke.sh \
  scripts/run-yulan-cross-tokenizer-opd-train.sh \
  tests/test_opd_byte_chunk.py \
  tests/test_student_logprob_alignment.py \
  tests/test_analyze_opd_alignment.py \
  tests/test_cross_tokenizer_opd_training_entry.py \
  tests/test_cross_tokenizer_opd_launcher_scripts.py \
  tests/test_rollout_logging.py \
  tests/test_rollout_logging_non_scalar_rewards.py \
  tests/test_real_tokenizer_sampled_opd_alignment.py \
  .ai/output/08_跨词表opd重构.md \
  .ai/output/09_跨词表opd经验总结.md \
  .ai/output/11_YuLan_tokenizer行为分析.md
git commit -m "refactor: harden cross-tokenizer opd mainline"
```
