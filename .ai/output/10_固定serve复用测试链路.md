# 固定 Serve 复用测试链路

## 目标

让 teacher SGLang server 和 rollout router/workers 常驻，只重复启动训练进程，从而避免每次 smoke 或小规模训练都重新加载模型。

当前两个 launcher 已支持：

- `REUSE_EXISTING_SERVERS=1`
- `SGLANG_ROUTER_IP`
- `SGLANG_ROUTER_PORT`
- `ROLLOUT_EXTERNAL_ENGINE_ADDRS`
- `--rollout-external`

## 一次性启动 Teacher

```bash
cd /mnt/ssd/lvzhihao/PostTrain/slime
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=3 \
python -m sglang.launch_server \
  --model-path /mnt/hdd/Nanbeige4.1-3B \
  --host 127.0.0.1 \
  --port 30221 \
  --tp 1 \
  --mem-fraction-static 0.55
```

健康检查：

```bash
curl -sf http://127.0.0.1:30221/health_generate
```

## 一次性启动 Rollout Workers

三个 worker 的示例，每个 worker 占 1 张卡：

```bash
cd /mnt/ssd/lvzhihao/PostTrain/slime
source .venv/bin/activate
```

worker 1:

```bash
CUDA_VISIBLE_DEVICES=4 \
python -m sglang.launch_server \
  --model-path /mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill \
  --host 127.0.0.1 \
  --port 31001 \
  --tp 1 \
  --mem-fraction-static 0.25 \
  --context-length 24576
```

worker 2:

```bash
CUDA_VISIBLE_DEVICES=5 \
python -m sglang.launch_server \
  --model-path /mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill \
  --host 127.0.0.1 \
  --port 31002 \
  --tp 1 \
  --mem-fraction-static 0.25 \
  --context-length 24576
```

worker 3:

```bash
CUDA_VISIBLE_DEVICES=6 \
python -m sglang.launch_server \
  --model-path /mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill \
  --host 127.0.0.1 \
  --port 31003 \
  --tp 1 \
  --mem-fraction-static 0.25 \
  --context-length 24576
```

## 一次性启动 Router

```bash
cd /mnt/ssd/lvzhihao/PostTrain/slime
source .venv/bin/activate

python -m sglang_router.launch_router \
  --host 127.0.0.1 \
  --port 30110 \
  --worker-urls \
    http://127.0.0.1:31001 \
    http://127.0.0.1:31002 \
    http://127.0.0.1:31003
```

## 重复跑 Smoke

这里训练只占 `0,1,2`，teacher 和 rollout 服务都直接复用，不会被脚本清掉。

```bash
cd /mnt/ssd/lvzhihao/PostTrain/slime

WORK_DIR=/mnt/hdd/lvzhihao/slime_opd_train_workdir_reuse_smoke \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
TEACHER_CUDA_VISIBLE_DEVICES=3 \
NUM_GPUS=3 \
REUSE_EXISTING_SERVERS=1 \
SGLANG_ROUTER_IP=127.0.0.1 \
SGLANG_ROUTER_PORT=30110 \
ROLLOUT_EXTERNAL_ENGINE_ADDRS=127.0.0.1:31001,127.0.0.1:31002,127.0.0.1:31003 \
TEACHER_PORT=30221 \
NUM_ROLLOUT=4 \
ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=2 \
GLOBAL_BATCH_SIZE=4 \
ROLLOUT_MAX_RESPONSE_LEN=1024 \
bash scripts/run-yulan-cross-tokenizer-opd-smoke.sh
```

## 重复跑 Train

```bash
cd /mnt/ssd/lvzhihao/PostTrain/slime

WORK_DIR=/mnt/hdd/lvzhihao/slime_opd_train_workdir_reuse_train \
TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 \
TEACHER_CUDA_VISIBLE_DEVICES=3 \
ROLLOUT_CUDA_VISIBLE_DEVICES= \
NUM_GPUS=3 \
ROLLOUT_NUM_GPUS_PER_ENGINE=1 \
REUSE_EXISTING_SERVERS=1 \
SGLANG_ROUTER_IP=127.0.0.1 \
SGLANG_ROUTER_PORT=30110 \
ROLLOUT_EXTERNAL_ENGINE_ADDRS=127.0.0.1:31001,127.0.0.1:31002,127.0.0.1:31003 \
TEACHER_PORT=30221 \
SAVE_INTERVAL=1 \
DEBUG_ROLLOUT_SAVE_INTERVAL=1 \
OPD_DISABLE_SEQUENCE_FALLBACK=false \
MATH500_EVAL_SIZE=500 \
bash scripts/run-yulan-cross-tokenizer-opd-train.sh
```

## 备注

- `ROLLOUT_EXTERNAL_ENGINE_ADDRS` 用逗号分隔，脚本会自动转成 `--rollout-external-engine-addrs ...`
- `REUSE_EXISTING_SERVERS=1` 时，launcher 不再执行 `pkill sglang`、`ray stop --force`
- train launcher 在复用模式下会把 `rollout_num_gpus` 解释为外部 engines 数量乘以 `ROLLOUT_NUM_GPUS_PER_ENGINE`
- smoke launcher 在复用模式下会自动取消 `--colocate`，避免把外部 rollout 当成本地 colocate rollout
