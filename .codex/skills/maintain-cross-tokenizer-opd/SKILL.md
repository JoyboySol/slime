---
name: maintain-cross-tokenizer-opd
description: Maintain the cross-tokenizer OPD training, alignment, and diagnostics pipeline in this slime repo. Use when changing byte-chunk OPD behavior, student/teacher byte alignment, rollout-side recorded alignment payloads, replay/analyze scripts, or regression tests for cross-tokenizer OPD with the YuLan student and Nanbeige teacher models.
---

# Maintain Cross Tokenizer Opd

## Overview

Use this skill when working on the cross-tokenizer OPD path that aligns student and teacher log-probs by byte chunks. Follow the rollout-produced recorded-alignment design first; only use the legacy student reconstruction path for compatibility or debugging.

## Workflow

1. Start from the current architecture:
- Student alignment payload is produced on the training side and stored on `Sample`.
- OPD consumers should prefer recorded payloads over student text reconstruction.
- Diagnostic scripts should match training behavior and prefer the same payloads.

2. Touch the right layer:
- Sample fields: `slime/utils/types.py`
- Student alignment and OPD utilities: `slime/utils/opd_utils.py`
- Teacher reward extraction and sample preparation: `slime/rollout/on_policy_distillation.py`
- Rollout/train-data transport: `slime/ray/rollout.py`
- Train-time metrics and KL application: `slime/backends/megatron_utils/data.py`, `slime/backends/megatron_utils/loss.py`
- Diagnostics: `scripts/replay_debug_rollout_opd.py`, `scripts/analyze_opd_alignment.py`
- Core regression tests: `tests/test_opd_byte_chunk.py`

3. Preserve the main invariant:
- Do not make OPD depend primarily on reconstructing student byte spans from rendered response text.
- Prefer recorded payload fields:
  - `opd_student_response_bytes`
  - `opd_student_token_byte_spans`
  - `opd_student_alignment_version`
  - `opd_student_alignment_error`
- If recorded payloads are missing, keep compatibility behavior explicit and well logged.

## Change Patterns

### When changing alignment production

- Update the rollout-side builder in `slime/utils/opd_utils.py`.
- Prefer full-sequence + prompt-boundary aware logic over response-only logic.
- Validate against canonical prompt/response/full texts, not raw student decode alone.
- Keep empty spans explicit instead of silently dropping tokens.

### When changing OPD consumption

- Update `compute_byte_chunk_aligned_log_probs()` / `compute_byte_chunk_reverse_kl()`.
- Payload-first behavior must remain the default when recorded alignment is present.
- Keep sequence fallback behavior intact and test both enabled and disabled modes.

### When changing diagnostics

- `scripts/replay_debug_rollout_opd.py` and `scripts/analyze_opd_alignment.py` should agree with training semantics.
- If training uses recorded payloads first, diagnostics must do the same.
- Distinguish these cases in summaries and logs:
  - recorded payload used
  - builder-generated payload used
  - legacy reconstruction used
  - teacher-side reconstruction failure
  - chunk alignment failure

## Testing

Use TDD for behavior changes: add or update the narrowest failing test first, run it red, then implement.

Minimum local test stack after code changes:

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
```

If you changed diagnostics or payload selection, also run:

```bash
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
./.venv/bin/python -m py_compile \
  scripts/analyze_opd_alignment.py \
  scripts/replay_debug_rollout_opd.py \
  slime/utils/opd_utils.py \
  slime/rollout/on_policy_distillation.py \
  slime/ray/rollout.py \
  slime/backends/megatron_utils/loss.py \
  slime/backends/megatron_utils/data.py
```

For a compact file map and copy-paste command list, read
`references/quick-reference.md`.

## Real-Environment Regression

Use these exact real-environment assets when verifying cross-tokenizer alignment behavior:

- Student model: `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`
- Teacher model: `/mnt/hdd/Nanbeige4.1-3B`
- Short corpus: `/mnt/hdd/lvzhihao/output/OpenMathInstruct-2/correct/segments`
- Long corpus: `/mnt/hdd/lvzhihao/output/OpenThoughts3-1.2M-math-distill-nanbeige4_1_3b/correct/segments`

The repository already includes sampled regression coverage in:
- `tests/test_real_tokenizer_sampled_opd_alignment.py`

Current expectation:
- Verify at least 128 sampled rows from each corpus when making risky alignment changes.
- Treat a passing 128/128 short + 128/128 long run as the baseline evidence standard.

## Debugging Order

When investigating a regression, check in this order:

1. Does the sample carry recorded student alignment payloads?
2. Do recorded response bytes equal canonical response text bytes?
3. Does teacher response byte slicing still match the same canonical bytes?
4. Does byte-chunk alignment fail only after both sides are valid?
5. Are diagnostics scripts reproducing the same path as training?

Do not jump straight to patching `align_token_byte_chunks()` if the actual breakage is in payload production or canonical text construction.
