---
name: build-hybrid-slime-env
description: Build or repair a local non-Docker slime environment in slime/.venv using uv, while supporting both a fast migration flow from YuLan-Pretrain/.venv and a clean from-scratch flow that rebuilds CUDA extensions such as apex, flash-attn, and transformer-engine. Use when slime must run with the custom Hybrid GDN Dense stack instead of the upstream Megatron/SGLang backend.
---

# Hybrid slime Environment Build

Use this skill when the task is to build, repair, or document the local environment in `/mnt/ssd/lvzhihao/PostTrain/slime/.venv`.

## Goal

Produce a usable local environment for slime with:

- Megatron backend from `/mnt/ssd/lvzhihao/PostTrain/YuLan-Pretrain`
- SGLang plugin from `/mnt/ssd/lvzhihao/PostTrain/sglang_qwen3_next_plugin`
- slime installed from the current workspace
- package installation and upgrades driven by `uv`

## Important framing

Do not explain this setup only through `/mnt/ssd/lvzhihao/PostTrain/build.sh`.

That script is a convenience wrapper, but the real documentation must be split into manual,
executable steps so someone can:

- rerun a single failed step
- repair one package without rebuilding the whole environment
- understand exactly where compatibility broke

When maintaining the docs, always present two environment paths:

- Path A: copy `YuLan-Pretrain/.venv` into `slime/.venv`, then repair incompatibilities
- Path B: create `slime/.venv` from scratch, then compile the heavy CUDA packages explicitly

## Common assumptions

These paths are the defaults used throughout the steps below:

```bash
ROOT=/mnt/ssd/lvzhihao/PostTrain
SLIME_DIR=$ROOT/slime
YULAN_DIR=$ROOT/YuLan-Pretrain
PLUGIN_DIR=$ROOT/sglang_qwen3_next_plugin
VENV_DIR=$SLIME_DIR/.venv
PYTHON_BIN=python3.10
UV_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
TORCH_INDEX=https://download.pytorch.org/whl/cu128
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=16
```

Recommended runtime exports:

```bash
export PYTHONPATH=$YULAN_DIR
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
export TORCH_CUDA_ARCH_LIST=8.0
```

## Path A: Migrate YuLan-Pretrain/.venv and repair

Use this path when:

- the YuLan Megatron environment is already known-good
- the fastest practical outcome matters more than environment purity
- you expect to inherit working `apex` or `transformer_engine` builds

### 1. Copy the virtualenv

```bash
cd $ROOT
rm -rf $VENV_DIR
rsync -a --delete $YULAN_DIR/.venv/ $VENV_DIR/
```

### 2. Rewrite embedded paths

```bash
OLD="$YULAN_DIR/.venv"
NEW="$VENV_DIR"

sed -i "s|$OLD|$NEW|g" "$NEW/pyvenv.cfg"
sed -i "s|$OLD|$NEW|g" "$NEW/bin/activate"
sed -i "s|$OLD|$NEW|g" "$NEW/bin/activate.csh"
sed -i "s|$OLD|$NEW|g" "$NEW/bin/activate.fish"

for file in "$NEW/bin/"*; do
    if [ -f "$file" ] && head -n 1 "$file" | grep -q "$OLD"; then
        sed -i "1 s|$OLD|$NEW|" "$file"
    fi
done
```

### 3. Ensure uv exists

```bash
$VENV_DIR/bin/pip install -i $UV_INDEX uv
$VENV_DIR/bin/uv --version
```

### 4. Upgrade the torch stack

```bash
$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $TORCH_INDEX \
  --extra-index-url $UV_INDEX \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1
```

### 5. Install runtime dependencies

```bash
$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  'ray[default]' \
  'sglang-router>=0.2.3' \
  sglang==0.5.9 \
  blobfile \
  omegaconf \
  pylatexenc \
  'mcp[cli]' \
  memray \
  numba \
  qwen_vl_utils \
  ring_flash_attn \
  mbridge
```

### 6. Install local editable packages

```bash
$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  --editable $SLIME_DIR \
  --no-deps

$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  --editable $PLUGIN_DIR
```

### 7. Repair incompatible CUDA extensions

The most common failure after a torch upgrade is `flash_attn` ABI breakage.

First remove the stale wheel:

```bash
source $VENV_DIR/bin/activate

uv pip uninstall \
  --python $VENV_DIR/bin/python \
  flash-attn flash_attn -y
```

Rebuild against the current torch:

```bash
uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  flash-attn==2.7.4.post1 \
  --no-build-isolation
```

If `transformer_engine` or `apex` also break, rebuild them using the commands from Path B.

## Path B: Build slime/.venv from scratch

Use this path when:

- you need the cleanest and most reproducible environment
- migrated binary wheels are too fragile
- you want explicit control over CUDA extension compilation

### 1. Create the virtualenv and install uv

```bash
cd $SLIME_DIR
rm -rf $VENV_DIR
$PYTHON_BIN -m venv $VENV_DIR
$VENV_DIR/bin/pip install -i $UV_INDEX --upgrade pip setuptools wheel uv ninja packaging
```

### 2. Install the torch stack first

```bash
$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $TORCH_INDEX \
  --extra-index-url $UV_INDEX \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1
```

### 3. Install generic Python dependencies

```bash
$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  cmake \
  pybind11 \
  pyyaml \
  numpy \
  scipy \
  sentencepiece \
  tiktoken \
  datasets \
  transformers \
  accelerate \
  safetensors \
  einops \
  blobfile \
  omegaconf \
  pylatexenc \
  'ray[default]' \
  'sglang-router>=0.2.3' \
  sglang==0.5.9 \
  numba \
  qwen_vl_utils \
  ring_flash_attn \
  mbridge
```

### 4. Build apex from source

```bash
source $VENV_DIR/bin/activate
cd $YULAN_DIR/3rdparty/apex

APEX_CPP_EXT=1 \
APEX_CUDA_EXT=1 \
uv pip install \
  --python $VENV_DIR/bin/python \
  --no-build-isolation \
  --no-deps \
  .
```

If the repo layout differs, search for apex first:

```bash
find $YULAN_DIR -maxdepth 4 -type d -name apex
```

### 5. Build flash-attn from source

```bash
source $VENV_DIR/bin/activate

uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  flash-attn==2.7.4.post1 \
  --no-build-isolation
```

If it looks stuck, inspect compiler activity:

```bash
pgrep -af 'flash-attn|nvcc|c\+\+|g\+\+|ninja|cmake|ptxas|cicc'
```

### 6. Build transformer-engine from source

Preferred method when `transformer_engine` source exists in YuLan:

```bash
source $VENV_DIR/bin/activate
cd $YULAN_DIR
find . -maxdepth 4 -type d -name transformer_engine
```

Then install from the actual source root:

```bash
cd <transformer_engine_source_root>
NVTE_FRAMEWORK=pytorch \
uv pip install \
  --python $VENV_DIR/bin/python \
  --no-build-isolation \
  --no-deps \
  .
```

If the local source is unavailable, fall back to the PyPI package that matches the current torch/CUDA combination:

```bash
source $VENV_DIR/bin/activate
uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  transformer-engine[pytorch]
```

### 7. Install slime and plugin

```bash
$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  --editable $SLIME_DIR \
  --no-deps

$VENV_DIR/bin/uv pip install \
  --python $VENV_DIR/bin/python \
  --index-url $UV_INDEX \
  --editable $PLUGIN_DIR
```

## Validation flow

### 1. Core import validation

```bash
source $VENV_DIR/bin/activate
export PYTHONPATH=$YULAN_DIR
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin

python - <<'PY'
import importlib

mods = [
    "torch",
    "slime",
    "sglang",
    "sglang_router",
    "sglang_qwen3_next_plugin",
    "megatron",
    "transformer_engine",
    "flash_attn",
    "apex",
]

for name in mods:
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__version__", "<no __version__>"))
    except Exception as exc:
        print(name, "FAIL", type(exc).__name__, exc)
PY
```

### 2. Plugin takeover validation

```bash
source $VENV_DIR/bin/activate
export PYTHONPATH=$YULAN_DIR
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin

python $PLUGIN_DIR/scripts/check_plugin_import.py
```

### 3. Real SGLang service validation

```bash
source $VENV_DIR/bin/activate
export PYTHONPATH=$YULAN_DIR
export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin

python -m sglang.launch_server \
  --model-path <hf_model_path> \
  --host 127.0.0.1 \
  --port 30110 \
  --tp 1
```

In another shell:

```bash
source $VENV_DIR/bin/activate
python $PLUGIN_DIR/scripts/run_acceptance.py \
  --host 127.0.0.1 \
  --port 30110
```

### 4. RL smoke validation

For Qwen3-1.7B on GPUs `0,1,2,3`, use:

```bash
source $VENV_DIR/bin/activate
export PYTHONPATH=$YULAN_DIR
export CUDA_VISIBLE_DEVICES=0,1,2,3

WORK_DIR=$SLIME_DIR/.tmp/qwen3_1p7b_thrpt_32x8 \
MODEL_PATH=/mnt/hdd/Qwen3-1.7B \
PROMPT_DATA=/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl \
ROLLOUT_BATCH_SIZE=32 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=256 \
MAX_TOKENS_PER_GPU=8192 \
SGLANG_MEM_FRACTION_STATIC=0.35 \
$SLIME_DIR/scripts/run-qwen3-1.7B-smoke-rl.sh
```

Success criteria:

- rollout finishes and collects `256` samples
- `ref_log_probs` finishes
- `actor_train` finishes
- `step 0` is printed
- checkpoint is saved under `actor_ckpt`
- Ray job ends with `SUCCEEDED`

## Known issues

### flash-attn undefined symbol after torch upgrade

This means the extension was built against the wrong torch ABI. Rebuild `flash-attn` against the
current torch.

### Long compile times

Heavy CUDA builds can spend a long time in `cicc`, `ptxas`, `nvcc`, `ninja`, or `g++`. High CPU
usage during that phase is normal and is not itself a hang.

### Runtime requirement discovered in practice

The runtime environment used by Ray jobs should include:

```bash
export TORCH_CUDA_ARCH_LIST=8.0
```

Without it, some CUDA extension behavior may differ from the interactive shell environment.

## Output to maintain

Whenever you build or repair this environment, update:

- `/mnt/ssd/lvzhihao/PostTrain/slime/.ai/output/00_环境构建.md`
- `/mnt/ssd/lvzhihao/PostTrain/slime/.ai/output/01_最小化测试_qwen.md`

These files should capture:

- the exact commands used
- which build path was used
- any package conflicts and how they were fixed
- what was finally validated successfully
