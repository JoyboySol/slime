# Cross-Tokenizer OPD Generation Token-Text Bridge

## Summary

This phase introduces the smallest protocol slice that moves recorded student
alignment closer to generation-native evidence without requiring a new rollout
engine byte-span API.

The rollout path now requests `return_text_in_logprobs=True` from SGLang and
records the per-token detokenized response texts emitted by the generation
engine. When these token texts are available and match the canonical response
text bytes, rollout-side recorded student alignment is built directly from that
generation-native evidence instead of reconstructing token boundaries through
the local tokenizer.

## Motivation

Phase 1 added observability and confirmed that long-response failures still
come from rollout-side student alignment production. The existing builder path
still depends on tokenizer reconstruction, which remains fragile on long
cross-tokenizer samples.

SGLang 0.5.9 already supports `return_text_in_logprobs`, so we can bridge to a
more robust source immediately:

- generated token ids still come from rollout
- generated token texts now also come from rollout
- recorded response bytes and spans can be derived from those token texts

This removes one major class of local re-decode assumptions while preserving
backward compatibility.

## Scope

### In scope

- request token texts from SGLang generate responses
- store generation-native student token texts on `Sample`
- prefer generation token texts when building recorded student response bytes
  and token byte spans
- retain the existing tokenizer-based builder as fallback
- cover the new behavior with rollout and OPD regression tests

### Out of scope

- no train-side protocol change
- no teacher-side byte evidence changes
- no rollout-engine byte offset API yet
- no removal of tokenizer reconstruction fallback

## Data Model

Add to `Sample`:

- `opd_student_token_texts: list[str] | None`

This field stores generation-native response token texts in response-token
order. It is diagnostic and rollout-produced evidence, not the final recorded
alignment payload itself.

## Canonical Flow

### Rollout generate

1. Send `return_text_in_logprobs=True` to SGLang.
2. Read `output_token_logprobs`.
3. If each entry contains a token text, append it to
   `sample.opd_student_token_texts`.

### Rollout alignment recording

1. Build canonical OPD prompt/response/full texts as before.
2. If `opd_student_token_texts` exists and length matches `response_length`:
   - encode each token text to UTF-8 bytes
   - require concatenated bytes to equal canonical response bytes
   - derive monotonic byte spans by cumulative byte length
   - mark `opd_student_alignment_source="generation_logprobs_text"`
3. Otherwise, fall back to the existing tokenizer-based recorded-alignment
   builder and keep `opd_student_alignment_source="recorded_builder"`.

## Why This Helps

- It removes the need to rediscover student response token boundaries from a
  local tokenizer in the success path.
- It keeps canonical byte validation against teacher-facing response text.
- It is a small change with low blast radius because train-time byte-chunk OPD
  still consumes the same recorded payload fields as before.

## Limits

This is not the final architecture. Token texts are stronger than local decode,
but they are still one step short of authoritative byte-span evidence emitted
directly by the generation engine. The next phase should capture generation
byte evidence or equivalent detokenization offsets directly from rollout.

## Validation

Required regression commands:

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
```
