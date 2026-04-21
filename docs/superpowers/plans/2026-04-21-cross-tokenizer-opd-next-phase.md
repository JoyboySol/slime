# Cross-Tokenizer OPD Next Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make cross-tokenizer OPD observability visible in the main `run.log`, define the next generation-native byte evidence protocol slice, and preserve real-environment regression coverage while implementing the transition.

**Architecture:** Split the work into three linked tracks. First, route rollout/train OPD summaries into a driver-visible logging channel instead of relying on Ray worker logs. Second, introduce a protocol object for generation-native student byte evidence that can coexist with current fallback paths. Third, codify the regression baseline around unit tests, sampled real-tokenizer alignment tests, and long-response smoke runs so each protocol step is validated against real behavior.

**Tech Stack:** Python, Ray, PyTorch, pytest, slime rollout/train pipeline, markdown specs/plans

---

### Task 1: Route OPD summaries into the main `run.log`

**Files:**
- Modify: `slime/ray/rollout.py`
- Modify: `slime/backends/megatron_utils/data.py`
- Modify: `slime/utils/logging_utils.py`
- Test: `tests/test_rollout_logging.py`
- Test: `tests/test_rollout_logging_non_scalar_rewards.py`

- [ ] **Step 1: Write the failing test**

Add tests that assert OPD rollout/train summaries are emitted through a driver-visible structured channel, not only through local worker `logger.info`.

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
./.venv/bin/pytest tests/test_rollout_logging.py -k 'driver_visible or summary' -v
./.venv/bin/pytest tests/test_rollout_logging_non_scalar_rewards.py -k 'summary' -v
```
Expected: FAIL because summaries are only local logger lines today.

- [ ] **Step 3: Write minimal implementation**

Implement a shared helper that:
- computes OPD summary counters once
- injects them into the structured metrics dict sent through `logging_utils.log`
- preserves concise `logger.info` output locally

Ensure summary keys are scalar and stable so they can propagate through the existing logging path.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
./.venv/bin/pytest tests/test_rollout_logging.py -v
./.venv/bin/pytest tests/test_rollout_logging_non_scalar_rewards.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add slime/ray/rollout.py slime/backends/megatron_utils/data.py slime/utils/logging_utils.py tests/test_rollout_logging.py tests/test_rollout_logging_non_scalar_rewards.py
git commit -m "feat: route opd summaries through driver-visible logging"
```

### Task 2: Write the generation-native byte evidence protocol spec

**Files:**
- Create or modify: `docs/superpowers/specs/2026-04-21-cross-tokenizer-opd-generation-byte-evidence-protocol.md`
- Reference: `docs/superpowers/specs/2026-04-21-cross-tokenizer-opd-generation-evidence-design.md`
- Reference: `docs/superpowers/specs/2026-04-21-cross-tokenizer-opd-generation-token-text-bridge.md`

- [ ] **Step 1: Write the protocol draft**

Document:
- the authoritative evidence object shape
- required fields for bytes, spans, completeness, and provenance
- versioning and backward compatibility
- how train/diagnostics consume the object

- [ ] **Step 2: Review against current code paths**

Verify the spec matches current rollout/train boundaries in:
- `slime/rollout/sglang_rollout.py`
- `slime/rollout/on_policy_distillation.py`
- `slime/ray/rollout.py`
- `slime/backends/megatron_utils/data.py`

- [ ] **Step 3: Save the spec**

Write the finalized protocol doc and ensure terminology is consistent with existing `recorded_builder`, `generation_logprobs_text`, and fallback semantics.

- [ ] **Step 4: Sanity-check the document**

Run:
```bash
sed -n '1,260p' docs/superpowers/specs/2026-04-21-cross-tokenizer-opd-generation-byte-evidence-protocol.md
```
Expected: clear protocol doc with explicit field semantics and transition plan.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-21-cross-tokenizer-opd-generation-byte-evidence-protocol.md
git commit -m "docs: define generation-native byte evidence protocol"
```

### Task 3: Add protocol-level regression tests before implementation

**Files:**
- Modify: `tests/test_opd_byte_chunk.py`
- Modify: `tests/test_student_logprob_alignment.py`
- Modify: `tests/test_real_tokenizer_sampled_opd_alignment.py`

- [ ] **Step 1: Write the failing tests**

Add tests that pin down:
- evidence completeness / incompleteness behavior
- preference order between authoritative generation evidence and reconstruction fallback
- no regression for 128 short + 128 long sampled alignment

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -k 'generation evidence or completeness' -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
```
Expected: FAIL before the new protocol object is implemented.

- [ ] **Step 3: Write minimal implementation hooks**

Add the smallest data-model and builder hooks needed so tests can target protocol semantics without yet requiring a full engine API change.

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_opd_byte_chunk.py tests/test_student_logprob_alignment.py tests/test_real_tokenizer_sampled_opd_alignment.py slime/utils/types.py slime/utils/opd_utils.py slime/rollout/on_policy_distillation.py
git commit -m "test: pin generation evidence protocol semantics"
```

### Task 4: Implement the first protocol slice in rollout-side evidence production

**Files:**
- Modify: `slime/utils/types.py`
- Modify: `slime/utils/opd_utils.py`
- Modify: `slime/rollout/sglang_rollout.py`
- Modify: `slime/rollout/on_policy_distillation.py`
- Modify: `scripts/replay_debug_rollout_opd.py`
- Modify: `scripts/analyze_opd_alignment.py`

- [ ] **Step 1: Implement the protocol object**

Introduce the new evidence fields or wrapper object with:
- response bytes
- token byte spans
- completeness flag
- provenance
- version

- [ ] **Step 2: Prefer protocol evidence in rollout-side recording**

Update rollout-side alignment building so authoritative generation evidence is preferred over local tokenizer reconstruction when available and complete.

- [ ] **Step 3: Propagate protocol evidence through diagnostics**

Ensure replay and analyze scripts display the new protocol state in a way that matches train semantics.

- [ ] **Step 4: Run targeted verification**

Run:
```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add slime/utils/types.py slime/utils/opd_utils.py slime/rollout/sglang_rollout.py slime/rollout/on_policy_distillation.py scripts/replay_debug_rollout_opd.py scripts/analyze_opd_alignment.py tests/test_opd_byte_chunk.py tests/test_student_logprob_alignment.py tests/test_analyze_opd_alignment.py
git commit -m "feat: add first slice of generation byte evidence protocol"
```

### Task 5: Re-run the real-environment regression baseline

**Files:**
- No code changes required

- [ ] **Step 1: Run core unit regression**

Run:
```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_rollout_logging.py -v
```
Expected: PASS.

- [ ] **Step 2: Run sampled real-tokenizer regression**

Run:
```bash
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
```
Expected: PASS for short 128 and long 128.

- [ ] **Step 3: Run small long-response smoke**

Run the long smoke launcher with:
- `NUM_ROLLOUT=5`
- `ROLLOUT_BATCH_SIZE=3`
- `N_SAMPLES_PER_PROMPT=2`
- `GLOBAL_BATCH_SIZE=6`
- `ROLLOUT_MAX_RESPONSE_LEN=8192`

Expected:
- `rollout_0.pt` is produced
- OPD summaries are visible in the main `run.log`
- no new regression in recorded alignment success rate

- [ ] **Step 4: Capture evidence in `.ai/output`**

Append or create a result note summarizing:
- summary log visibility
- protocol slice behavior
- smoke outcomes

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test: verify next-phase cross-tokenizer opd regression baseline"
```
