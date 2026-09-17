---
name: yulan-math-rl-runner
description: Run, debug, and reproduce the YuLan MoE Math RL pipeline in slime, including HF/MCore loading, CP8/EP8 actor training, SGLang rollout lifecycle, routing replay, DAPO filtering, verifier/eval, and checkpoint retention.
metadata:
  short-description: Operate the YuLan MoE Math RL pipeline
---

# YuLan MoE Math RL runner

Use this skill when a task concerns the YuLan MoE Math RL launch scripts, rollout/training consistency, SGLang/actor GPU orchestration, route replay, Math/DAPO rewards, or reproducing the stabilized pipeline in this repository.

## Baseline and scope

The stabilized branch was compared against `origin/main` at:

```text
origin/main: 4c193f1
current commit: c08c2cf (stabilize Math RL rollout and replay diagnostics)
```

The main entrypoints are:

- `scripts/run-yulan-moe-math-dapo-full.sh`: full 8-GPU launch wrapper, YuLan route-replay patch management, W&B/log/output defaults.
- `scripts/run-yulan-moe-base-math-smoke.sh`: Ray submission and model/rollout argument assembly.
- `scripts/models/yulan-moe-base.sh`: Qwen3-Next/YuLan hybrid MoE model definition.

Preserve unrelated working-tree changes. Do not permanently edit YuLan-Pretrain for a slime experiment: use `.ai/route_replay_yulan_pretrain.patch`; the full launcher applies it only when needed and reverses it on exit.

## What differs from origin/main

The relevant changes in `c08c2cf` are grouped as follows:

1. **YuLan model and HF/MCore conversion**
   - Adds `slime_plugins/models/yulan_moe.py` and the complete 56-layer hybrid GDN/MoE model configuration.
   - Extends `slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py` for YuLan GDN tensors and both ordinary-GQA and gated-attention QKV layouts.
   - Passes `padding_mask` through the MCore forward path.

2. **Routing replay and CP/actor initialization**
   - Adds explicit actor-local `TopKRouter` replay-slot registration in `megatron_utils/actor.py`.
   - Adds route replay and YuLan transformer/THD-CP environment propagation.
   - Defers actor creation when `--destroy-rollout-engines` is active, and recreates actors after rollout engines are destroyed.

3. **SGLang lifecycle and colocated memory safety**
   - Adds `ServerGroup.destroy`, `RolloutServer.destroy_engines`, and `recreate_engines`.
   - The intended lifecycle is: create HF-serving engines → rollout → save rollout → destroy engines → create/train actor → save MCore/HF → recreate engines with the new HF snapshot.
   - With engine destruction, pass `--no-offload-train --no-offload-rollout`; do not simultaneously keep SGLang weights/KV cache and actor training weights on the same GPUs.
   - Serving layout is four engines, each `TP=2, EP=2`, across eight rollout GPUs. Actor layout is `TP=1, PP=1, CP=8, EP=8`, with `--auto-generate-cu-seqlens` retained.

4. **Math verifier and eval metrics**
   - `math_dapo_utils.py` accepts `\\boxed`/`\\fbox`, broader final-answer forms, TeX canonicalization, math_verify variants, degree/radian variants, and prompt-aware multiple-choice labels.
   - DAPO training rewards remain `+1/-1`; eval converts positive rewards to `0/1` accuracy.
   - Eval reward dictionaries must use the configured reward key or the conventional `score` key.

5. **Rollout data, filtering, and observability**
   - Dataset loading supports directories of JSONL files.
   - Prompt suffix/system prompt are applied before chat templating.
   - Debug rollout saving is interval-controlled for training and always retained for eval.
   - Dynamic filtering is group-level: `rollout_batch_size=32`, `n_samples_per_prompt=8` means 32 accepted prompt groups / 256 samples per rollout. `drop_*` counters are rejected candidate groups, not the final training size.

6. **Checkpoint and W&B operations**
   - Checkpoint retention keeps the newest two snapshots plus periodic snapshots every 20 steps.
   - W&B random suffix generation is compatible with newer W&B versions.

## Recommended full-run configuration

Unless the user explicitly requests another layout, use the launcher defaults:

```text
HF checkpoint: /mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-Base
MCore reference: .../mcore-tp1pp1ep8-sqrtgate-hybrid0625-05
actor: 8 GPUs, TP1/PP1/CP8/EP8
rollout: 8 GPUs, four TP2/EP2 SGLang engines
global batch: 256
micro batch: 1
rollout batch: 32 groups
rollout N: 8
rollout max response: 16384
eval max response: 16384
temperature: 0.7
over-sampling batch: 48
max tokens per GPU: 1024
save interval: 20
eval interval: 2
debug rollout interval: 5
advantage estimator: GRPO
reward: DAPO
route replay: enabled
dynamic filter: check_reward_nonzero_std_with_fallback
```

The prompt suffix is:

```text
Please reason step by step, and put your final answer within \\boxed{}.
```

## Launch procedure

Before launch:

1. Check `git status`; do not erase user changes.
2. Verify the HF config, MCore reference, merged judgeable JSONL, AIME 2024/2025 files, and route-replay patch exist.
3. Verify the W&B key without printing it. A W&B HTTP 401 is an authentication failure, not a training failure; do not repeatedly restart the same invalid key.
4. Use a new output/log directory for a fresh experiment. Do not mix a failed run's partial checkpoint with a new base run.
5. Run `bash -n scripts/run-yulan-moe-math-dapo-full.sh` before submission.

For a fresh run, submit the wrapper under `nohup`/`setsid` with secrets supplied only through the environment:

```bash
nohup setsid env \
  WANDB_API_KEY="$WANDB_API_KEY" \
  OUTPUT_ROOT=/mnt/yulan/lvzhihao/PostTrain/outputs/<run> \
  RUN_LOG_FILE=/mnt/yulan/lvzhihao/PostTrain/outputs/logs/<run>.log \
  SAVE_HF=/mnt/yulan/lvzhihao/PostTrain/outputs/<run>/hf/rollout_{rollout_id} \
  ROLLOUT_NUM_GPUS=8 \
  ROLLOUT_MAX_RESPONSE_LEN=16384 \
  EVAL_MAX_RESPONSE_LEN=16384 \
  OVER_SAMPLING_BATCH_SIZE=48 \
  bash scripts/run-yulan-moe-math-dapo-full.sh \
  > /mnt/yulan/lvzhihao/PostTrain/outputs/logs/<run>.launcher.log 2>&1 < /dev/null &
```

Immediately record the Ray submission ID from the launcher log and verify:

```bash
ray job status <submission-id>
rg -n -- '--save-hf|--rollout-max-response-len|--eval-max-response-len|--over-sampling-batch-size|Rollout generation' <run>.log
```

Do not use an unquoted Bash default expression containing `{rollout_id}`. The safe launcher form is the explicit `if [[ -z ... ]]` initialization currently used in `run-yulan-moe-math-dapo-full.sh`; otherwise a supplied path can become `rollout_{rollout_id}}` and fail later in `str.format` during HF save.

## Replay and no-engine debugging

To reuse a saved training rollout, set:

```text
REPLAY_ROLLOUT_DATA=/path/debug/rollout_{rollout_id}.pt
```

The matching eval file should be `/path/debug/rollout_eval_0.pt`. For replay-only actor validation, make `ROLLOUT_NUM_GPUS=0`; the smoke wrapper then adds `--no-offload-train --no-offload-rollout` and does not need SGLang workers. Use a new output directory and set `NUM_ROLLOUT` to the number of replay files actually available.

Do not assume `--load-forge-rollout-data` alone disables SGLang: the forge loader intentionally preserves the normal engine lifecycle unless rollout GPUs are explicitly set to zero.

## Diagnosis checklist

- **W&B `CommError` with HTTP 401:** invalid/revoked key; validate the credential and restart only after replacing it.
- **`Single '}' encountered in format string`:** malformed `SAVE_HF`; inspect the submitted command and fix the launcher initialization, not the checkpoint converter.
- **OOM at actor train:** verify SGLang engines were destroyed before actor creation; check for `--destroy-rollout-engines`, `--rebuild-train-actors`, and no simultaneous engine/actor residency.
- **Large train/rollout logprob difference:** first confirm temperature conventions and route replay, then compare CP1 versus CP8. Do not infer an optimizer bug from this metric alone.
- **Low DAPO effective signal:** `zero_std/count_*` describes accepted zero-variance groups; the final group count is still `rollout_batch_size`. With the fallback filter, zero-variance groups may be retained when needed to fill the batch.
- **Very slow rollout:** inspect response truncation and max length first. At length 16384, oversampling 48 and four TP2/EP2 engines are expected to be much slower than a short smoke run.

## Validation

Run the targeted checks after code changes:

```bash
bash -n scripts/run-yulan-moe-base-math-smoke.sh
bash -n scripts/run-yulan-moe-math-dapo-full.sh
pytest -q tests/test_rm_math.py tests/test_rm_math_dapo.py tests/test_hf_to_megatron.py tests/test_checkpoint_retention.py
```

For a launch change, validate the actual Ray entrypoint in the log; shell syntax alone is insufficient.
