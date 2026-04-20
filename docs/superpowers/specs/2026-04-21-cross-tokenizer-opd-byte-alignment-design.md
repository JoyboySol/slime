# Cross-Tokenizer OPD Byte Alignment Design

**Date:** 2026-04-21

## Goal

Replace the current student-side byte-span reconstruction path for cross-tokenizer
OPD with a rollout-produced alignment record that is linear-time, robust to
non-idempotent tokenization, and directly reusable by training-side OPD code and
diagnostic tools.

## Problem Statement

The current byte-chunk OPD path reconstructs student token byte spans after the
fact from `student_token_ids`, `prompt_text`, and `full_text`. That design is
fragile for real tokenizers and expensive for long responses:

- it assumes the student tokenizer can be inverted from rendered text back to
  the original token sequence
- it falls back through `offset_mapping`, `convert_ids_to_tokens`, and
  prefix-decoding heuristics
- the prefix-decoding fallback becomes effectively O(n^2) for long responses
- it fails on real student outputs where detokenization is not prefix-consistent
  or where `decode -> encode` is not idempotent

Real failures were already observed with:

- student model: `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`
- teacher model: `/mnt/hdd/Nanbeige4.1-3B`

The observed failure mode is not primarily a student/teacher byte mismatch. The
student-side token-to-byte reconstruction itself is failing before byte-chunk
alignment begins.

## Non-Goals

- Redesign the OPD objective itself
- Change teacher log-prob extraction semantics
- Remove sequence-level fallback entirely
- Introduce a new standalone alignment-only codepath that diverges from training
  data structures
- Perform full-dataset validation over the user-provided corpora in one pass

## Recommended Approach

Move student byte-span construction from OPD post-processing to rollout/sample
construction.

The training pipeline should record a student-side alignment artifact during
rollout assembly, then OPD post-processing should consume that artifact instead
of reconstructing spans from rendered text.

In effect:

- old design: infer student token byte spans later
- new design: record student token byte spans once, verify them later

This turns the student alignment problem from a brittle reverse-engineering task
into a forward data capture task.

## Architecture Overview

### Student-Side Source of Truth

Each sample should carry an OPD alignment payload generated from the same
training-side codepath that already assembles `tokens`, `response_length`,
`prompt`, `response`, `opd_prompt_text`, `opd_response_text`, and
`opd_full_text`.

The alignment payload should include:

- `opd_student_response_bytes`
- `opd_student_token_byte_spans`
- `opd_student_alignment_version`
- optional diagnostic metadata when generation-time alignment cannot be proven

The payload must describe the response region only, because OPD penalties are
applied over response tokens.

### Teacher-Side Source of Truth

Teacher bytes can continue to be derived from the canonical `full_text` at
post-processing time. The teacher tokenizer is used as a forward tokenizer only:

- tokenize canonical `full_text`
- derive token bytes and spans once
- clip to the response byte region
- align teacher response bytes to recorded student response bytes

### OPD Post-Processing Contract

`compute_byte_chunk_aligned_log_probs()` should prefer the rollout-produced
student alignment payload. The existing reconstruction logic should remain only
as a compatibility fallback for legacy samples or incomplete debug payloads.

Sequence fallback remains available, but should become an exception path rather
than a normal robustness mechanism.

## Data Model

### New Sample Fields

The implementation should add fields equivalent to the following semantics:

- `opd_student_response_bytes`: the exact UTF-8 bytes for the student response
  text used by OPD
- `opd_student_token_byte_spans`: one entry per response token, each a
  half-open byte span `[start, end)` into `opd_student_response_bytes`
- `opd_student_alignment_version`: version string or integer for future
  migration and compatibility handling
- `opd_student_alignment_error`: optional structured reason if rollout-side
  alignment could not be proven

The concrete storage type can be adapted to the existing `Sample` conventions,
but the semantic contract should stay stable.

### Alignment Invariants

The new payload must satisfy all of the following:

1. The number of spans equals `response_length`
2. Spans are monotonic and non-overlapping
3. Empty spans are allowed, but only as explicit zero-width spans
4. Concatenating per-token byte slices from the spans reproduces
   `opd_student_response_bytes`
5. `opd_student_response_bytes` equals
   `opd_response_text.encode("utf-8")` for canonical OPD text
6. If canonical bytes do not match recorded bytes, the sample must carry a
   structured diagnostic instead of silently proceeding as if alignment were
   trustworthy

## Rollout-Side Alignment Strategy

### Preferred Strategy

Generate the student response byte/spans from the same finalized response token
sequence that is saved into the sample, using a linear-time incremental
detokenization pass.

This pass should operate over response token ids only and build:

- a cumulative byte buffer
- one byte span per response token

The implementation should avoid repeated full-prefix decoding.

### Why This Is Better Than the Current Prefix Fallback

The current fallback repeatedly decodes token prefixes and compares full strings.
That is expensive and still depends on prefix-consistency assumptions.

The new pass should be organized as a single scan over the response token
sequence, with bounded per-token work and explicit validation against canonical
response bytes.

### Boundaries

Prompt/response boundary handling should happen before the alignment payload is
saved, using the canonical OPD prompt and response texts already attached to the
sample. The payload should represent response tokens only so OPD post-processing
does not need to reason about prompt trimming.

## Runtime Behavior

### Success Path

On samples with a valid alignment payload:

- student alignment is read directly from the sample
- teacher response bytes are constructed from canonical `full_text`
- byte chunks are aligned
- chunk-level student and teacher log-probs are computed without student-side
  text reconstruction

### Compatibility Path

On older samples or on samples missing the new fields:

- keep the existing student reconstruction path
- emit explicit logging that the legacy path is being used

### Failure Path

If rollout-side alignment cannot be proven for a sample:

- save a structured error field on the sample or in diagnostics
- OPD post-processing either falls back to sequence-level penalty or raises,
  depending on `allow_sequence_fallback`
- logging should state whether the failure happened during rollout-side
  alignment production, student payload validation, teacher reconstruction, or
  byte-chunk alignment

## Testing Strategy

Testing should reuse training-side codepaths as much as possible and should stay
close to the real environment.

### 1. Unit Tests

Extend the OPD byte-chunk tests to cover:

- successful consumption of recorded student byte spans
- payload validation failures
- compatibility fallback for samples without the new fields
- preservation of sequence-level fallback semantics

### 2. Fixed Regression Samples

Use the already observed fallback cases as pinned regression inputs:

- `workdirs/slime_opd_train_workdir_full_night/debug_rollouts/manual_fallback_samples/rollout0_sample_20_index_20.json`
- `workdirs/slime_opd_train_workdir_full_night/debug_rollouts/manual_fallback_samples/rollout0_sample_21_index_21.json`

These tests should verify that the new path no longer depends on the failing
student reconstruction heuristic for those samples.

### 3. Real-Environment Sampled Integration Tests

Run sampled checks using:

- short-text rollout data:
  `/mnt/hdd/lvzhihao/output/OpenMathInstruct-2/`
- long-text rollout data:
  `/mnt/hdd/lvzhihao/output/OpenThoughts3-1.2M-math-distill-nanbeige4_1_3b/`
- student tokenizer:
  `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`
- teacher tokenizer:
  `/mnt/hdd/Nanbeige4.1-3B`

The sampled tests should be deterministic and bounded. They should report:

- student payload construction success rate
- payload validation success rate
- teacher reconstruction success rate
- byte-stream equality rate
- byte-chunk alignment success rate

### 4. Diagnostic Script Coverage

Update the diagnostic scripts so they can inspect the new payload directly,
rather than reconstructing student bytes as their primary job.

## Observability

Add metrics or structured logs for:

- new-payload path usage count
- legacy reconstruction path usage count
- rollout-side payload validation failures
- teacher-side reconstruction failures
- byte-chunk alignment failures after validated payloads

These signals should make it obvious whether future failures are caused by data
production or by alignment logic.

## Risks

- The rollout stack may not expose as much token-level detokenization structure
  as desired, requiring a carefully implemented local incremental scanner.
- Some samples may still fail the new payload validation if the training-side
  canonical text fields are inconsistent with the actual generated token stream.
- Carrying byte arrays in debug or rollout payloads may modestly increase sample
  size; validation should focus on response-only bytes to limit overhead.

## Notes

The normal brainstorming workflow calls for a review subagent. In this session,
delegated subagents were not used because the user did not request delegated
agent work. The design is being written and implemented directly in the current
workspace.
