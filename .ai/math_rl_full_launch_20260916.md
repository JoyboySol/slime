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

## SGLang rollout 并发与 503 排查记录（2026-09-25）

### 现象与结论

2026-09-25 的 mid/high 难度数据全量任务出现 rollout 请求长时间不返回、SGLang Router 反复重试并最终返回 HTTP 503。排查后发现，客户端请求并发上限与 SGLang engine 的处理并发不匹配，是导致请求堆积和重试放大的主要原因：

```text
旧配置：sglang_server_concurrency 默认 512
rollout engines：4
slime semaphore 上限：512 × 4 = 2048 个并发请求
SGLang：每个 engine max_running_requests=8
总 running slots：4 × 8 = 32
```

rollout 会为每组 prompt 并行提交 `n_samples_per_prompt=8` 个生成请求；客户端 semaphore 的 2048 上限远高于 32 个 engine running slots，缺少有效的客户端背压。旧任务日志中 engine 的 `#queue-req` 曾升至数十，Router 出现大量 `Retry backoff`；最终客户端记录到 101 个 HTTP 503。GPU 仍在工作，因此现象不是 GPU 空闲或 rollout actor 未启动。现有日志能确认并发不匹配及排队/重试现象，但不足以断定 Router 内部首先触发的具体超时原因。

### 处理与复查

重启任务时显式加入：

```bash
--sglang-server-concurrency 8
```

此时 slime semaphore 上限为 `8 × 4 = 32`，与 4 个 engine 各 8 个 running slots 对齐；`over_sampling_batch_size=32` 保持不变。2026-09-25 新任务早期日志已观察到 HTTP 200、eval 进度前进，engine 队列基本为 0，且观察窗口内未见 503 或 Router retry。慢请求仍有约 138 秒延迟，所以这属于早期验证，不能仅凭启动阶段认定长期吞吐问题彻底解决。

```text
旧任务：raysubmit_fidQ4zFTTtQ291Xk（已停止）
新任务：raysubmit_fxWhr97st1pQXLAz（启动时仍在运行）
新任务输出：/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_math_dapo_mid_high_20k_full_cp8_20260925_reqdiag_conc8
新任务日志：/mnt/yulan/lvzhihao/PostTrain/outputs/logs/yulan_moe_math_dapo_mid_high_20k_full_cp8_20260925_reqdiag_conc8.log
```

后续排查优先同时检查 `sglang_server_concurrency`、每个 engine 的 `max_running_requests`、`#queue-req`、Router retry/503 和成功请求延迟。不要只看 GPU 利用率判断 rollout 是否健康。
