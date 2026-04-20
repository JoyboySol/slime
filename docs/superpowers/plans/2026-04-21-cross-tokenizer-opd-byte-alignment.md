# Cross-Tokenizer OPD Byte Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragile student-side byte-span reconstruction with rollout-produced alignment payloads and verify the new path using pinned regressions plus sampled real-environment checks.

**Architecture:** Add a response-only student byte alignment payload to samples during rollout construction, validate and consume that payload in OPD post-processing, and keep the old reconstruction path only as a compatibility fallback. Extend diagnostics and tests to exercise the exact training-side codepaths with the real student and teacher tokenizers.

**Tech Stack:** Python, PyTorch, HuggingFace tokenizers, Slime rollout pipeline, existing OPD utilities, pytest

---

### Task 1: Define the sample-level alignment contract

**Files:**
- Create: `docs/superpowers/specs/2026-04-21-cross-tokenizer-opd-byte-alignment-design.md`
- Create: `docs/superpowers/plans/2026-04-21-cross-tokenizer-opd-byte-alignment.md`
- Modify: `slime/utils/types.py`
- Test: `tests/test_opd_byte_chunk.py`

- [ ] **Step 1: Add failing tests for recorded student alignment fields**

Add tests that construct a sample carrying:

- `opd_response_text`
- `opd_student_response_bytes`
- `opd_student_token_byte_spans`
- `response_length`

and assert the validation helper accepts the payload and rejects:

- wrong span count
- non-monotonic spans
- bytes that do not match `opd_response_text`

- [ ] **Step 2: Run the targeted tests and verify failure**

Run: `pytest tests/test_opd_byte_chunk.py -k recorded_alignment -v`
Expected: FAIL because the payload fields and validation helper do not exist yet.

- [ ] **Step 3: Extend the sample type with the new optional fields**

Modify `slime/utils/types.py` so the sample object can carry the new response-only
alignment payload and preserve it through `to_dict()` / `from_dict()`.

- [ ] **Step 4: Add a shared validation helper**

Implement a helper in `slime/utils/opd_utils.py` that validates the recorded
student response byte payload against canonical OPD response text and returns a
normalized representation for downstream use.

- [ ] **Step 5: Re-run the targeted tests**

Run: `pytest tests/test_opd_byte_chunk.py -k recorded_alignment -v`
Expected: PASS

### Task 2: Build rollout-side student response alignment

**Files:**
- Modify: `slime/rollout/sglang_rollout.py`
- Modify: `slime/ray/rollout.py`
- Modify: `slime/rollout/on_policy_distillation.py`
- Modify: `slime/utils/opd_utils.py`
- Test: `tests/test_opd_byte_chunk.py`
- Test: `tests/test_student_logprob_alignment.py`

- [ ] **Step 1: Add failing tests for rollout-produced alignment payloads**

Write tests that pass realistic `tokens`, `response_length`, `opd_prompt_text`,
and `opd_response_text` through the same helper used by rollout assembly, then
assert:

- one span per response token
- concatenated bytes equal the response text bytes
- zero-width spans are handled explicitly when needed

- [ ] **Step 2: Run those tests and verify failure**

Run: `pytest tests/test_opd_byte_chunk.py -k rollout_alignment_builder -v`
Expected: FAIL because no rollout-side builder exists.

- [ ] **Step 3: Implement a linear-time response alignment builder**

Add a helper in `slime/utils/opd_utils.py` that scans response token ids once
and produces:

- response UTF-8 bytes
- one byte span per response token
- an explicit error object when canonical response bytes cannot be proven

Avoid repeated whole-prefix decoding.

- [ ] **Step 4: Wire the builder into rollout sample assembly**

Modify the sample construction path so every OPD-capable sample records the new
payload during training-side rollout assembly.

- [ ] **Step 5: Re-run the focused tests**

Run:
- `pytest tests/test_opd_byte_chunk.py -k "recorded_alignment or rollout_alignment_builder" -v`
- `pytest tests/test_student_logprob_alignment.py -v`

Expected: PASS

### Task 3: Switch OPD byte-chunk alignment to prefer recorded payloads

**Files:**
- Modify: `slime/utils/opd_utils.py`
- Modify: `slime/rollout/on_policy_distillation.py`
- Test: `tests/test_opd_byte_chunk.py`

- [ ] **Step 1: Add failing tests for payload-first behavior**

Add tests asserting `compute_byte_chunk_aligned_log_probs()`:

- consumes recorded student alignment when present
- skips the fragile student reconstruction path in that case
- still supports the legacy path when the new payload is absent
- preserves `allow_sequence_fallback=False` semantics

- [ ] **Step 2: Run the targeted tests and verify failure**

Run: `pytest tests/test_opd_byte_chunk.py -k payload_first -v`
Expected: FAIL because OPD still reconstructs student bytes from text.

- [ ] **Step 3: Implement payload-first student byte loading**

Refactor OPD utilities to:

- validate and normalize recorded student payloads first
- use legacy reconstruction only for compatibility
- improve failure reasons so logs distinguish payload validation,
  teacher reconstruction, and chunk alignment failures

- [ ] **Step 4: Re-run the OPD byte-chunk tests**

Run: `pytest tests/test_opd_byte_chunk.py -v`
Expected: PASS

### Task 4: Lock in the real regressions and diagnostics

**Files:**
- Modify: `scripts/replay_debug_rollout_opd.py`
- Modify: `scripts/analyze_opd_alignment.py`
- Modify: `tests/test_analyze_opd_alignment.py`
- Modify: `tests/test_inspect_rollout_script.py`
- Test: `tests/test_opd_byte_chunk.py`

- [ ] **Step 1: Add regression coverage for the known fallback samples**

Create tests that load the pinned failing samples under
`workdirs/slime_opd_train_workdir_full_night/debug_rollouts/manual_fallback_samples/`
and assert the new payload-aware path no longer requires the old student
reconstruction heuristic to analyze them.

- [ ] **Step 2: Update diagnostic scripts to inspect recorded payloads**

Refactor the scripts so their primary analysis path reads recorded alignment
payloads when present, and only falls back to legacy reconstruction for
backward-compatible analysis.

- [ ] **Step 3: Run the diagnostic-script tests**

Run:
- `pytest tests/test_analyze_opd_alignment.py -v`
- `pytest tests/test_inspect_rollout_script.py -v`

Expected: PASS

### Task 5: Add sampled real-environment verification

**Files:**
- Modify: `scripts/analyze_opd_alignment.py`
- Create: `tests/test_real_tokenizer_sampled_opd_alignment.py`

- [ ] **Step 1: Add a sampled verification harness**

Create a test module or script-backed harness that:

- uses student tokenizer `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`
- uses teacher tokenizer `/mnt/hdd/Nanbeige4.1-3B`
- samples bounded deterministic rows from:
  - `/mnt/hdd/lvzhihao/output/OpenMathInstruct-2/`
  - `/mnt/hdd/lvzhihao/output/OpenThoughts3-1.2M-math-distill-nanbeige4_1_3b/`

- [ ] **Step 2: Reuse training-side alignment code in the sampled tests**

Ensure the sampled verification path imports the same rollout/alignment helpers
used by training, not a duplicate analyzer implementation.

- [ ] **Step 3: Assert real-environment success criteria**

Check at minimum:

- student payload construction succeeds for sampled rows
- payload bytes equal canonical response bytes
- teacher response bytes equal the same canonical bytes
- byte-chunk alignment completes without student reconstruction failures

- [ ] **Step 4: Run the sampled verification**

Run: `pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v`
Expected: PASS on the local environment where the referenced models and data are present.

### Task 6: Final verification and checkpoint

**Files:**
- Modify: `slime/utils/opd_utils.py`
- Modify: `slime/rollout/on_policy_distillation.py`
- Modify: `slime/rollout/sglang_rollout.py`
- Modify: `slime/ray/rollout.py`
- Modify: `slime/utils/types.py`
- Modify: `tests/test_opd_byte_chunk.py`
- Modify: `tests/test_student_logprob_alignment.py`
- Modify: `tests/test_analyze_opd_alignment.py`
- Create: `tests/test_real_tokenizer_sampled_opd_alignment.py`

- [ ] **Step 1: Run the focused unit and integration suite**

Run:
- `pytest tests/test_opd_byte_chunk.py -v`
- `pytest tests/test_student_logprob_alignment.py -v`
- `pytest tests/test_analyze_opd_alignment.py -v`
- `pytest tests/test_inspect_rollout_script.py -v`
- `pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v`

Expected: PASS

- [ ] **Step 2: Run a syntax sanity check for touched scripts**

Run:
- `python -m py_compile scripts/replay_debug_rollout_opd.py scripts/analyze_opd_alignment.py`

Expected: no output, exit code 0

- [ ] **Step 3: Commit the implementation checkpoint**

```bash
git add docs/superpowers/specs/2026-04-21-cross-tokenizer-opd-byte-alignment-design.md \
        docs/superpowers/plans/2026-04-21-cross-tokenizer-opd-byte-alignment.md \
        slime/utils/types.py \
        slime/utils/opd_utils.py \
        slime/rollout/on_policy_distillation.py \
        slime/rollout/sglang_rollout.py \
        slime/ray/rollout.py \
        scripts/replay_debug_rollout_opd.py \
        scripts/analyze_opd_alignment.py \
        tests/test_opd_byte_chunk.py \
        tests/test_student_logprob_alignment.py \
        tests/test_analyze_opd_alignment.py \
        tests/test_inspect_rollout_script.py \
        tests/test_real_tokenizer_sampled_opd_alignment.py
git commit -m "refactor: record rollout-side opd byte alignment"
```
