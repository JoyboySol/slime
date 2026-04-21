# Cross-Tokenizer OPD Generation Evidence Design

## Summary

The current cross-tokenizer OPD path still falls back on long responses because
student-side recorded alignment is often produced by local tokenizer
reconstruction rather than generation-native evidence. This design formalizes a
long-term architecture where rollout produces authoritative student byte
alignment evidence and all downstream consumers prefer that evidence over local
reconstruction.

This spec also defines an immediately implementable phase-1 slice:

- make student alignment provenance explicit
- propagate provenance through rollout -> train -> diagnostics
- emit structured status for recorded-payload use vs fallback

That phase does not yet require rollout-engine protocol changes, but it creates
the observability needed to implement and validate generation-native evidence in
the next phase.

## Problem Statement

Today, byte-chunk OPD prefers recorded student alignment payloads when present,
but long real-world samples still frequently hit:

- `student_byte_reconstruction_failed`
- `Tokenizer token/offset reconstruction failed.`

Root cause investigation on `rollout_0.pt` from
`slime_opd_train_workdir_full_night_new` shows:

- the train side is correctly consuming recorded payloads when available
- payload transport is intact
- the failure occurs because rollout-side recorded payload production itself
  often fails on long responses
- current payload production still depends on tokenizer-based reconstruction,
  which is not stable enough for all long cross-tokenizer samples

As a result, the main path is still too dependent on local tokenizer
reconstruction even after the earlier refactor.

## Goals

### Primary goals

- Define generation-side student alignment as the canonical truth source
- Make payload provenance and validation status explicit
- Ensure train, replay, and analysis tools use the same payload semantics
- Make it easy to distinguish:
  - recorded payload used successfully
  - recorded payload missing
  - recorded payload invalid
  - legacy reconstruction fallback used
  - teacher-side failure
  - chunk alignment failure

### Non-goals for phase 1

- No rollout-engine protocol expansion yet
- No removal of legacy fallback yet
- No strict-mode enforcement that recorded payload must exist

## Architecture

### Long-term model

There are two conceptual layers:

1. **Generation / rollout layer**
   - produces student alignment evidence
   - validates it locally
   - records provenance

2. **Train / diagnostics layer**
   - consumes recorded evidence
   - performs lightweight consistency checks
   - uses legacy reconstruction only as compatibility fallback

The critical architectural rule is:

> Student token-to-byte alignment truth should come from the generation side,
> not from downstream reconstruction.

### Phase-1 model

Until generation-native token byte evidence is available, rollout will continue
to construct recorded payloads locally, but it must explicitly label:

- where the payload came from
- whether it was validated
- why it failed when absent

Downstream code will propagate and report those labels.

## Data Model

The existing `Sample` fields remain:

- `opd_student_response_bytes`
- `opd_student_token_byte_spans`
- `opd_student_alignment_version`
- `opd_student_alignment_error`

Add these fields:

- `opd_student_alignment_source: str | None`
  - examples:
    - `recorded_builder`
    - `generation_trace`
    - `legacy_reconstruction_fallback`
- `opd_student_alignment_validated: bool | None`
  - `True` when rollout considers the payload trustworthy
  - `False` when payload construction failed
- `opd_student_alignment_status: str | None`
  - examples:
    - `ok_recorded`
    - `recorded_missing`
    - `recorded_invalid`
    - `fallback_legacy`

These new fields are diagnostic and transport metadata. They are not the payload
itself, but they describe the payload lifecycle.

## Canonical Flow

### Rollout / reward stage

When byte-chunk OPD is enabled:

1. Build canonical prompt/response/full texts.
2. Attempt to record student alignment payload.
3. On success:
   - set bytes + spans
   - set `source`
   - set `validated=True`
   - set `status="ok_recorded"`
4. On failure:
   - leave bytes/spans as `None`
   - set `validated=False`
   - set `status="recorded_missing"`
   - store structured error text in `opd_student_alignment_error`

### Train stage

When applying byte-chunk OPD:

1. If recorded payload exists:
   - validate payload against canonical response text
   - on success, use it
   - on failure, set effective status `recorded_invalid`
2. If recorded payload is missing:
   - use legacy reconstruction only as compatibility fallback
   - set effective status `fallback_legacy`

### Diagnostics stage

Replay and analysis tools should report both:

- sample-level recorded alignment metadata
- effective alignment mode used during reconstruction / analysis

This allows the user to separate “payload unavailable” from “payload available
but invalid”.

## Implementation Plan Scope

This spec authorizes **phase 1 only**:

- add provenance/status fields
- populate them on rollout-side recording
- transport them through train data where useful
- surface them in diagnostics/replay
- add tests for the new semantics

Future work for generation-native byte evidence is intentionally left to a
follow-up spec after phase-1 observability is in place.

## Files In Scope

Core:

- `slime/utils/types.py`
- `slime/rollout/on_policy_distillation.py`
- `slime/ray/rollout.py`
- `slime/utils/opd_utils.py`
- `slime/backends/megatron_utils/loss.py`
- `slime/backends/megatron_utils/data.py`

Diagnostics:

- `scripts/replay_debug_rollout_opd.py`
- `scripts/analyze_opd_alignment.py`

Tests:

- `tests/test_opd_byte_chunk.py`
- `tests/test_analyze_opd_alignment.py`

## Validation Strategy

### Required tests

- targeted unit tests for new sample metadata behavior
- targeted unit tests for diagnostics summaries
- existing byte-chunk regression suite

### Required commands

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
```

### Manual evidence check

Use an existing long-response debug rollout dump and verify that replay /
analysis now surfaces:

- recorded alignment source
- recorded alignment validation state
- recorded alignment status

This phase does not require the long-response payload success rate to increase.
That belongs to the next protocol-level phase.

## Risks

### Risk: metadata drift between rollout and diagnostics

Mitigation:

- define provenance/status names centrally in rollout logic
- write tests asserting replay and analysis display the same meanings

### Risk: overloading status strings with mixed meanings

Mitigation:

- keep `source`, `validated`, and `status` separate
- do not encode every failure dimension into a single string

### Risk: accidental behavior change in train-time OPD

Mitigation:

- phase 1 is observability-first
- do not change byte-chunk math or fallback policy in this phase

## Future Phase

The next phase should replace rollout-side local reconstruction as the primary
recorded-payload source with generation-native byte evidence. That work will
likely require rollout-engine protocol changes and a follow-up design.
