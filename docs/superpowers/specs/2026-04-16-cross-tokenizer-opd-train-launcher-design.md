# Cross-Tokenizer OPD Train Launcher Design

**Date:** 2026-04-16

## Goal

Add a dedicated full-training launcher script for cross-tokenizer OPD runs in `slime`, using the validated byte-chunk OPD path and the user's local train/eval assets.

## Context

The repository already contains a smoke script for cross-tokenizer OPD in
`scripts/run-yulan-cross-tokenizer-opd-smoke.sh`. That script is intentionally
minimal and tuned for debugging. The user now needs a fuller launcher with:

- larger default training hyperparameters
- eval enabled by default
- W&B configuration exposed via environment variables
- rollout debug dumps saved to disk by default
- the teacher model fixed to `/mnt/hdd/Nanbeige4.1-3B`
- the training data fixed to
  `/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl`
- the eval data fixed to
  `/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/AIME2024/test.parquet`

## Design

Create a new script instead of modifying the smoke launcher.

Reasons:

- The smoke script remains a stable minimal reproduction path for debugging.
- The training launcher can safely diverge in defaults without weakening smoke reproducibility.
- Users get a cleaner operational entrypoint for day-to-day training.

The new script will:

- reuse the existing Yulan model argument preset from `scripts/models/yulan-hybrid-gdn-dense-2.9b.sh`
- convert the student HuggingFace checkpoint into `torch_dist` if needed
- start an SGLang teacher server for the teacher model
- launch training through `tools/run_train_with_ray_init.py`
- enable byte-chunk OPD with the teacher HF checkpoint path
- enable eval by default with the AIME2024 parquet file
- save rollout debug dumps under `${WORK_DIR}/debug_rollouts/rollout_{rollout_id}.pt`

## Defaults

The launcher will default to:

- `GLOBAL_BATCH_SIZE=128`
- `N_SAMPLES_PER_PROMPT=4`
- `ROLLOUT_BATCH_SIZE=32`
- `ROLLOUT_MAX_PROMPT_LEN=2048`
- `ROLLOUT_MAX_RESPONSE_LEN=16384`
- `ROLLOUT_TEMPERATURE=1`
- `LR=1e-5`
- `EVAL_MAX_RESPONSE_LEN=16384`

These values match the requested training profile while keeping the script easy
to override through environment variables.

## Data Handling

Training data uses the existing slime-style JSONL format with:

- `--input-key prompt`
- `--label-key label`

The AIME eval parquet already follows the repo's common eval pattern and will be
wired with:

- `--eval-prompt-data aime <path>`

No extra eval label override is required in the launcher.

## W&B

W&B will be optional but first-class. The script will expose:

- `WANDB_API_KEY`
- `WANDB_PROJECT`
- `WANDB_GROUP`
- `WANDB_TEAM`
- `WANDB_HOST`
- `WANDB_MODE`

If `WANDB_API_KEY` is empty, the launcher will disable W&B instead of failing.

## Output Layout

All operational outputs will live under `${WORK_DIR}`:

- checkpoints: `${WORK_DIR}/actor_ckpt`
- converted checkpoint: `${WORK_DIR}/yulan_torch_dist`
- train log: `${WORK_DIR}/run.log`
- teacher log: `${WORK_DIR}/teacher.log`
- W&B files: `${WORK_DIR}/wandb`
- rollout debug dumps: `${WORK_DIR}/debug_rollouts/rollout_{rollout_id}.pt`

## Verification

Because this change is a launcher script rather than core runtime logic, the
main verification steps are:

- `bash -n` on the new script
- manual review of the assembled command-line arguments
- path validation in the script before training starts

## Notes

The brainstorming workflow normally calls for a spec review loop and a dedicated
worktree. In this session, subagent review was not used because the user did not
request delegated agent work, and the script is being implemented directly in
the current workspace.
