# Qwen3-1.7B slime 最小化测试结果

## 测试目标

验证下面这条真实链路在 slime 中可用：

- 模型：`/mnt/hdd/Qwen3-1.7B`
- 数据：`/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl`
- GPU：`0,1,2,3`
- 环境：`/mnt/ssd/lvzhihao/PostTrain/slime/.venv`
- Megatron 后端：`/mnt/ssd/lvzhihao/PostTrain/YuLan-Pretrain`
- SGLang 插件：`/mnt/ssd/lvzhihao/PostTrain/sglang_qwen3_next_plugin`

目标不是只验证 import，而是验证 slime 的 RL 流程真实能跑起来。

## 测试命令

实际使用的是吞吐量更高、并且最终确认可跑通的一组参数。

启动前环境：

```bash
source /mnt/ssd/lvzhihao/PostTrain/slime/.venv/bin/activate
export PYTHONPATH=/mnt/ssd/lvzhihao/PostTrain/YuLan-Pretrain
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

实际命令：

```bash
WORK_DIR=/mnt/ssd/lvzhihao/PostTrain/slime/.tmp/qwen3_1p7b_thrpt_32x8 \
MODEL_PATH=/mnt/hdd/Qwen3-1.7B \
PROMPT_DATA=/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl \
ROLLOUT_BATCH_SIZE=32 \
N_SAMPLES_PER_PROMPT=8 \
GLOBAL_BATCH_SIZE=256 \
MAX_TOKENS_PER_GPU=8192 \
SGLANG_MEM_FRACTION_STATIC=0.35 \
/mnt/ssd/lvzhihao/PostTrain/slime/scripts/run-qwen3-1.7B-smoke-rl.sh
```

## 关键参数组合

这次最终确认可用的参数组合是：

- `--actor-num-gpus-per-node 4`
- `--attention-backend flash`
- `--tensor-model-parallel-size 1`
- `--pipeline-model-parallel-size 1`
- `--sequence-parallel`
- `--use-dynamic-batch-size`
- `--max-tokens-per-gpu 8192`
- `--rollout-batch-size 32`
- `--n-samples-per-prompt 8`
- `--global-batch-size 256`
- `--rollout-max-response-len 256`
- `--rollout-num-gpus-per-engine 1`
- `--sglang-mem-fraction-static 0.35`
- `--sglang-context-length 2048`

对应 Ray job runtime env 里还带了：

```bash
PYTHONPATH=/mnt/ssd/lvzhihao/PostTrain/YuLan-Pretrain
CUDA_DEVICE_MAX_CONNECTIONS=1
NCCL_NVLS_ENABLE=1
MASTER_ADDR=127.0.0.1
PYTHONBUFFERED=16
TORCH_CUDA_ARCH_LIST=8.0
```

## 实际结果

这次不是只启动成功，而是完整完成了一个 RL 训练 step。

日志中确认到的事实如下：

- Ray job `raysubmit_j6F7PJ1JL3cFK7mk` 最终状态为 `SUCCEEDED`
- 4 个 SGLang engine 成功拉起并 ready
- rollout 成功收集 `256` 条样本
- `ref_log_probs` 成功结束
- `log_probs` 成功结束
- `actor_train` 成功结束
- 输出了 `step 0`
- checkpoint 成功保存

日志中关键成功信号包括：

```text
Final collected 256 samples from rollout to train
Timer ref_log_probs end
step 0: {... 'train/step': 0}
successfully saved checkpoint from iteration 0 to .../actor_ckpt
Job 'raysubmit_j6F7PJ1JL3cFK7mk' succeeded
```

## 性能指标

本次跑通时日志中的关键性能数据：

- `perf/tokens_per_gpu_per_sec: 2846.5417231343463`
- `perf/actor_train_tok_per_s: 10515.766205082902`
- `perf/step_time: 41.80731272697449`
- `perf/rollout_time: 5.755756139755249`
- `perf/ref_log_probs_time: 22.90457534790039`
- `perf/actor_train_time: 8.505324125289917`

## 产物位置

运行日志：

- `/mnt/ssd/lvzhihao/PostTrain/slime/.tmp/qwen3_1p7b_thrpt_32x8/run.log`

Ray driver 日志：

- `/mnt/ssd/cache_tmp/tmp/ray/session_latest/logs/job-driver-raysubmit_j6F7PJ1JL3cFK7mk.log`

checkpoint 输出目录：

- `/mnt/ssd/lvzhihao/PostTrain/slime/.tmp/qwen3_1p7b_thrpt_32x8/actor_ckpt`

其中已经确认存在：

- `iter_0000000/common.pt`
- `iter_0000000/*.distcp`
- `latest_checkpointed_iteration.txt`
- `rollout/global_dataset_state_dict_0.pt`

## 测试结论

结论是：

- `slime + YuLan-Pretrain + sglang_qwen3_next_plugin + Qwen3-1.7B + DeepMath-103K slime_style_train_data.jsonl`
- 在 `0,1,2,3` 四张卡上
- 使用上述高吞吐参数组合
- 已经真实跑通 RL 流程

这说明当前环境已经满足“最小化测试可跑”的要求，而且不是保守小参数，而是吞吐量更高的一组参数。

## 补充说明

这次 rollout 中 reward 仍然是 `0.0`，所以这里验证的是：

- 训练链路可跑
- 环境和后端接入可用
- 参数组合可稳定执行一个真实 step

它不等价于奖励设计已经达到业务目标。

如果后续要继续推进，下一阶段应关注：

- reward 是否符合预期
- 数据是否需要改造
- 是否要把当前高吞吐参数组合固化为正式训练脚本
