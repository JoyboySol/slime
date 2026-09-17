# Math RL 全量训练启动说明（2026-09-16）

## 当前任务

```text
Ray job: raysubmit_Mrt5iZsxHMW3mwpF
状态: RUNNING
```

启动脚本：

```text
/mnt/yulan/lvzhihao/PostTrain/slime/scripts/run-yulan-moe-math-dapo-full.sh
```

## 启动命令

在 Slime 环境中执行以下命令。W&B key 需要提前通过环境变量传入，不要写入脚本或日志：

```bash
export WANDB_API_KEY='<your-wandb-api-key>'
export GLOBAL_BATCH_SIZE=256
export ROLLOUT_BATCH_SIZE=32
export OVER_SAMPLING_BATCH_SIZE=64
export NUM_ROLLOUT=446
export OUTPUT_ROOT=/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_math_dapo_full_cp8_rebuild_20260916
export RUN_LOG_FILE=/mnt/yulan/lvzhihao/PostTrain/outputs/logs/yulan_moe_math_dapo_full_cp8_rebuild_20260916.log

cd /mnt/yulan/lvzhihao/PostTrain/slime
./scripts/run-yulan-moe-math-dapo-full.sh
```

后台启动示例：

```bash
setsid bash -c '
  export WANDB_API_KEY="<your-wandb-api-key>"
  export GLOBAL_BATCH_SIZE=256
  export ROLLOUT_BATCH_SIZE=32
  export OVER_SAMPLING_BATCH_SIZE=64
  export NUM_ROLLOUT=446
  export OUTPUT_ROOT=/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_math_dapo_full_cp8_rebuild_20260916
  export RUN_LOG_FILE=/mnt/yulan/lvzhihao/PostTrain/outputs/logs/yulan_moe_math_dapo_full_cp8_rebuild_20260916.log
  cd /mnt/yulan/lvzhihao/PostTrain/slime
  exec ./scripts/run-yulan-moe-math-dapo-full.sh
' </dev/null > /mnt/yulan/lvzhihao/PostTrain/outputs/logs/yulan_moe_math_dapo_full_cp8_rebuild_20260916_launcher.log 2>&1 &
```

## 关键配置

```text
训练：GRPO + DAPO reward
global_batch_size=256
rollout_batch_size=32
n_samples_per_prompt=8
over_sampling_batch_size=64
num_rollout=446

actor：TP=1, PP=1, CP=8, EP=8
max_tokens_per_gpu=1024
auto-generate-cu-seqlens 开启
动态 microbatch 开启

rollout：8 张 GPU，4 个 engine，每个 engine TP=2、EP=2
temperature=0.7
max_prompt_len=1024
max_response_len=31744

eval：每 2 个 rollout step 一次
AIME2024：/mnt/yulan/lvzhihao/PostTrain/data/aime-2024.jsonl
AIME2025：/mnt/yulan/lvzhihao/PostTrain/data/aime-2025.jsonl
n_samples_per_eval_prompt=2

routing replay：--use-routing-replay
rollout routing replay：关闭
train actor：每轮训练后释放并重建
```

## 重要启动逻辑

```text
rollout 生成 64 组候选
→ DAPO 动态过滤 zero-reward-std group
→ 保留 32 组进入训练（256 条 response）
→ 销毁 SGLang engine
→ 创建 train actor 并训练
→ 保存 MCore/HF checkpoint
→ 释放 train actor
→ 重建 SGLang engine
→ 下一轮 rollout
```

当前正式训练不使用 `--use-rollout-routing-replay`，因为已验证该参数会造成 SGLang 与 MCore 路由不一致，使训推 logprob 差异显著增大。

## 输出与日志

```text
输出：/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_math_dapo_full_cp8_rebuild_20260916
主日志：/mnt/yulan/lvzhihao/PostTrain/outputs/logs/yulan_moe_math_dapo_full_cp8_rebuild_20260916.log
W&B 目录：/mnt/yulan/lvzhihao/PostTrain/outputs/wandb
```

HF checkpoint 模板：

```text
${OUTPUT_ROOT}/hf/rollout_{rollout_id}
```

MCore/HF checkpoint 保留策略：最近 2 个，以及每 20 个 rollout 的周期性 checkpoint。

## 监控与停止

```bash
/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/ray job status raysubmit_Mrt5iZsxHMW3mwpF
tail -f /mnt/yulan/lvzhihao/PostTrain/outputs/logs/yulan_moe_math_dapo_full_cp8_rebuild_20260916.log
/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/ray job stop raysubmit_Mrt5iZsxHMW3mwpF
```

重点观察：

```text
train/train_rollout_logprob_abs_diff
rollout/dynamic_filter/drop_*
rollout/zero_std/count_*
train/global_batch_size
```
