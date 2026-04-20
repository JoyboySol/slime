# Cross-Tokenizer OPD Quick Reference

## Core Files

- `slime/utils/types.py`
  Holds `Sample` fields for recorded student alignment payloads.

- `slime/utils/opd_utils.py`
  Holds:
  - recorded payload validation
  - recorded student alignment builder
  - byte-chunk alignment
  - payload-first OPD chunk log-prob computation

- `slime/rollout/on_policy_distillation.py`
  Holds:
  - canonical OPD text construction
  - teacher reward-logprob extraction
  - `_record_student_opd_alignment()`

- `slime/ray/rollout.py`
  Moves recorded alignment payloads from `Sample` into `train_data`.

- `slime/backends/megatron_utils/data.py`
  Uses payload-aware OPD utilities for metrics.

- `slime/backends/megatron_utils/loss.py`
  Uses payload-aware OPD utilities for reverse-KL application.

- `scripts/replay_debug_rollout_opd.py`
  Replays a saved rollout sample and should prefer recorded payloads.

- `scripts/analyze_opd_alignment.py`
  Analyzes dataset rows and should match training-side alignment behavior.

## Key Tests

- `tests/test_opd_byte_chunk.py`
  Main regression file for byte-chunk OPD behavior.

- `tests/test_student_logprob_alignment.py`
  Protects logprob and response-window alignment assumptions.

- `tests/test_analyze_opd_alignment.py`
  Protects diagnostic-script behavior.

- `tests/test_real_tokenizer_sampled_opd_alignment.py`
  Real-environment sampled verification using the actual student and teacher tokenizers.

## Real Environment Assets

- Student tokenizer/model:
  `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`

- Teacher tokenizer/model:
  `/mnt/hdd/Nanbeige4.1-3B`

- Short corpus:
  `/mnt/hdd/lvzhihao/output/OpenMathInstruct-2/correct/segments`

- Long corpus:
  `/mnt/hdd/lvzhihao/output/OpenThoughts3-1.2M-math-distill-nanbeige4_1_3b/correct/segments`

## Standard Verification Commands

### Focused OPD regressions

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
```

### Real-environment sampled regression

This test currently uses 128 sampled rows from each corpus.

```bash
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
```

### End-to-end verification set

```bash
./.venv/bin/pytest \
  tests/test_opd_byte_chunk.py \
  tests/test_student_logprob_alignment.py \
  tests/test_analyze_opd_alignment.py \
  tests/test_real_tokenizer_sampled_opd_alignment.py \
  -v
```

### Syntax checks

```bash
./.venv/bin/python -m py_compile \
  slime/utils/opd_utils.py \
  slime/rollout/on_policy_distillation.py \
  slime/ray/rollout.py \
  slime/backends/megatron_utils/loss.py \
  slime/backends/megatron_utils/data.py \
  scripts/analyze_opd_alignment.py \
  scripts/replay_debug_rollout_opd.py \
  tests/test_real_tokenizer_sampled_opd_alignment.py
```

## Debugging Checklist

1. Confirm the sample carries:
   - `opd_student_response_bytes`
   - `opd_student_token_byte_spans`
2. Confirm recorded response bytes equal canonical response text bytes.
3. Confirm teacher response bytes slice to the same canonical response bytes.
4. Confirm failures happen after both byte streams are validated.
5. Confirm diagnostics scripts and training code choose the same alignment path.

## Current Architecture Notes

- Preferred path: rollout-produced recorded student alignment payload
- Compatibility path: legacy student reconstruction only when recorded payloads are absent
- Evidence standard for risky changes: 128 short samples + 128 long samples passing with real tokenizers
