# Cross-Tokenizer OPD Train Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone training launcher script for the validated cross-tokenizer OPD workflow with default train/eval/W&B/debug-rollout configuration.

**Architecture:** Build a new Bash launcher on top of the existing smoke script pattern. Keep the smoke script unchanged, reuse the existing Yulan model preset, and centralize all output paths under a single work directory so training artifacts are easy to inspect.

**Tech Stack:** Bash, Ray, SGLang, Slime training entrypoint, existing model preset scripts

---

### Task 1: Document the launcher contract

**Files:**
- Create: `docs/superpowers/specs/2026-04-16-cross-tokenizer-opd-train-launcher-design.md`
- Create: `docs/superpowers/plans/2026-04-16-cross-tokenizer-opd-train-launcher.md`

- [ ] **Step 1: Record the approved defaults**

Capture the requested train data, eval data, teacher model, default hyperparameters, W&B behavior, and rollout dump location.

- [ ] **Step 2: Record the non-goals**

Explicitly keep the existing smoke script unchanged and avoid broader training-runtime refactors.

### Task 2: Add the launcher script

**Files:**
- Create: `scripts/run-yulan-cross-tokenizer-opd-train.sh`

- [ ] **Step 1: Start from the smoke launcher structure**

Reuse the environment setup, student conversion flow, teacher server startup, and train entrypoint pattern already proven by the smoke script.

- [ ] **Step 2: Replace smoke-sized defaults with training defaults**

Set:

- `GLOBAL_BATCH_SIZE=128`
- `ROLLOUT_BATCH_SIZE=32`
- `N_SAMPLES_PER_PROMPT=4`
- `ROLLOUT_MAX_PROMPT_LEN=2048`
- `ROLLOUT_MAX_RESPONSE_LEN=16384`
- `ROLLOUT_TEMPERATURE=1`
- `LR=1e-5`
- `EVAL_MAX_RESPONSE_LEN=16384`

- [ ] **Step 3: Wire eval in by default**

Point `--eval-prompt-data` at the AIME2024 parquet file and set eval parameters through environment-variable-backed defaults.

- [ ] **Step 4: Wire W&B in with a safe fallback**

Build `WANDB_ARGS` only when `WANDB_API_KEY` is non-empty; otherwise keep training runnable without W&B.

- [ ] **Step 5: Enable rollout debug dumps**

Pass `--save-debug-rollout-data "${WORK_DIR}/debug_rollouts/rollout_{rollout_id}.pt"` so each rollout is preserved for later inspection.

### Task 3: Verify the launcher

**Files:**
- Test: `scripts/run-yulan-cross-tokenizer-opd-train.sh`

- [ ] **Step 1: Run Bash syntax validation**

Run: `bash -n scripts/run-yulan-cross-tokenizer-opd-train.sh`
Expected: no output, exit code 0

- [ ] **Step 2: Review the generated defaults**

Check the script contents to verify train/eval paths, W&B env vars, rollout dump path, and OPD arguments all match the approved design.

- [ ] **Step 3: Summarize run instructions**

Prepare the final user-facing guidance showing the default invocation and the main environment variables they may want to override.
