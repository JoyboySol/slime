# Cross-Tokenizer OPD Generation Byte Evidence Protocol

## Summary

This document defines the long-term protocol for student-side alignment
evidence in cross-tokenizer OPD. The goal is to stop treating byte alignment as
something reconstructed downstream from text and instead make generation output
an authoritative byte-level evidence object that training and diagnostics can
consume directly.

This protocol is intentionally backward-compatible with the current
recorded-payload path:

- existing fields such as `opd_student_response_bytes` and
  `opd_student_token_byte_spans` remain usable
- rollout-side fallback paths remain available during the transition
- diagnostics and train-time code can gradually prefer the new protocol object

## Problem Statement

The current bridge design improved robustness, but it is still not the ideal
long-term architecture because student alignment truth can still depend on:

- rendered response text
- token-derived canonical text
- `generation_logprobs_text`
- tokenizer reconstruction fallback

This creates two maintenance problems:

1. there is no single authoritative source of student byte truth
2. long-response failures still collapse into fallback chains instead of being
   represented as explicit evidence completeness states

The protocol defined here addresses both issues.

## Goals

### Primary goals

- Define a single authoritative generation-side evidence object
- Represent bytes, spans, provenance, and completeness explicitly
- Allow train, replay, and analysis tools to consume the same evidence object
- Make partial / truncated / invalid evidence first-class states
- Preserve backward compatibility during rollout-engine migration

### Non-goals

- No teacher-side protocol redesign in this phase
- No immediate removal of existing fallback code
- No requirement that every rollout engine already exposes byte offsets

## Authoritative Evidence Object

The generation side should eventually produce one evidence object per sample
response:

```python
{
    "version": 2,
    "source": "generation_byte_evidence",
    "response_bytes": list[int],
    "token_byte_spans": list[list[int]],
    "response_token_count": int,
    "complete": bool,
    "validated": bool,
    "error": str | None,
    "metadata": {
        "engine": str | None,
        "detokenizer_protocol": str | None,
        "completion_reason": str | None,
    },
}
```

## Field Semantics

### `version`

- Protocol version for the evidence object
- Start the new protocol at `2`
- Version `1` remains the legacy recorded-alignment payload semantics

### `source`

Allowed values should be explicit and stable:

- `generation_byte_evidence`
- `generation_logprobs_text`
- `recorded_builder`
- `legacy_reconstruction_fallback`

Long-term expectation:

- `generation_byte_evidence` becomes the normal success path
- other values are transitional or diagnostic

### `response_bytes`

- UTF-8 byte sequence for the canonical student response region only
- Must correspond to the same response region used for teacher OPD scoring
- This is the authoritative byte payload consumed by train and diagnostics

### `token_byte_spans`

- Byte spans for each response token, relative to `response_bytes`
- Must be monotonic, gap-free, and bounded within `response_bytes`
- Span count must equal `response_token_count`

### `response_token_count`

- Number of student response tokens covered by the evidence object
- Must equal the number of spans when `complete=True`

### `complete`

- `True` means the evidence covers the entire response token sequence
- `False` means evidence is partial or truncated

This field is critical because long responses may otherwise look like generic
reconstruction failures when the real issue is simply incomplete evidence
capture.

### `validated`

- `True` means rollout already verified the evidence against canonical response
  bytes
- `False` means the evidence object exists but cannot be trusted yet

### `error`

- Structured reason when evidence is partial, invalid, or absent
- Prefer explicit staged reasons such as:
  - `generation_byte_evidence_incomplete`
  - `generation_byte_evidence_invalid`
  - `generation_logprobs_text_failed`
  - `tokenizer_reconstruction_failed`

### `metadata`

Optional protocol details that help future debugging without affecting train
semantics.

Recommended keys:

- `engine`
- `detokenizer_protocol`
- `completion_reason`
- `captured_response_token_count`

## Canonical Flow

### Rollout / generation stage

1. Generate student output.
2. Collect authoritative byte evidence from the generation engine if available.
3. Build the evidence object.
4. Validate:
   - `response_bytes` match canonical response text bytes
   - span count matches response token count when complete
   - spans are monotonic and gap-free
5. Store the object on `Sample`.

### Train stage

1. Prefer the evidence object if present.
2. If `complete=True` and `validated=True`, use it directly.
3. If present but incomplete or invalid:
   - record explicit status
   - choose the configured fallback
4. If absent:
   - fall back to legacy reconstruction only for compatibility

### Diagnostics stage

Replay and analysis should report:

- protocol version
- source
- complete / validated flags
- effective alignment mode used
- fallback reason when not using authoritative evidence

## Compatibility Mapping

The current payload fields map naturally into the protocol:

- `opd_student_response_bytes` -> `response_bytes`
- `opd_student_token_byte_spans` -> `token_byte_spans`
- `opd_student_alignment_version` -> `version`
- `opd_student_alignment_source` -> `source`
- `opd_student_alignment_validated` -> `validated`
- `opd_student_alignment_error` -> `error`

Recommended additive fields for the next implementation slice:

- `opd_student_alignment_complete: bool | None`
- `opd_student_alignment_metadata: dict | None`

These can be added without breaking existing consumers.

## Transition Plan

### Phase A: observability and driver-visible summaries

- ensure summary metrics reach the main `run.log`
- continue using current builder paths
- distinguish incomplete evidence from invalid evidence

### Phase B: protocol object on `Sample`

- add explicit completeness and metadata fields
- propagate them through rollout -> train -> diagnostics
- prefer the protocol object over local reconstruction

### Phase C: generation-engine authoritative bytes

- modify rollout-engine interface so generation returns byte evidence directly
- stop relying on `decode([id])` style token text as the primary source

### Phase D: deprecate legacy reconstruction

- retain fallback for debugging only
- make authoritative evidence the normal path

## Required Invariants

For evidence to be considered usable in byte-chunk OPD:

- `response_bytes` exactly equals canonical response text bytes
- `token_byte_spans` count equals response token count when `complete=True`
- spans are monotonic and gap-free
- final span end equals response byte length

If any invariant fails:

- set `validated=False`
- keep the failure reason in `error`
- fall back explicitly instead of silently repairing the object

## Validation Strategy

Minimum validation for each protocol change:

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
```

And at least one long-response smoke run should confirm:

- OPD summaries are visible in the main `run.log`
- `complete` / `validated` states are observable
- fallback reasons stay explicit

## Recommended Next Implementation Slice

The next concrete code step should not attempt a full engine rewrite. It should:

1. Add `complete` and optional `metadata` fields to `Sample`
2. Thread them through rollout/train/diagnostics
3. Continue using current bridge evidence where necessary
4. Distinguish:
   - complete authoritative evidence
   - incomplete authoritative evidence
   - bridge evidence
   - legacy fallback

That gives the codebase a stable transition shape while the true
generation-engine byte API is being prepared.
