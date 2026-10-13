# Stage 4：小 batch 单步 RL smoke 与起始模型 AIME24 eval

## 目标和范围

Stage 4 验证 YuLan-MoE-128k 的短序列 RL 链路：先记录起始模型的 AIME 2024 avg@2 基线，再用少量 rollout 完成一个 actor train step，保存权重并重建 SGLang rollout engines。这里的训练 rollout 使用 4096 context；AIME24 评测单独使用 32768 context。该阶段不是 32k 训练压力测试，也不是正式训练。

## 配置和信源

- 启动入口：[`scripts/run-yulan-moe-128k-stage4-smoke.sh`](../../scripts/run-yulan-moe-128k-stage4-smoke.sh)。它在独立的 128k 启动入口 [`scripts/run-yulan-moe-128k-math-smoke.sh`](../../scripts/run-yulan-moe-128k-math-smoke.sh) 上增加一轮小 batch、checkpoint 导出和 eval 参数。
- HF 初始权重：`/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k`。
- Slime 初始训练权重：`/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k/mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix`，加载为 iteration 0。
- 训练 prompt：`/mnt/yulan/lvzhihao/PostTrain/data/hf_data/AIME_like_data/aime_8.jsonl`。
- AIME24 eval：[`configs/eval/aime24_eval.yaml`](../../configs/eval/aime24_eval.yaml)，数据文件为 `/mnt/yulan/lvzhihao/PostTrain/data/aime-2024.jsonl`，共 30 道题。
- Slime 执行日志：`/tmp/yulan_moe_128k_stage4_avg2_smoke.log`。
- 输出目录：`/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage4_avg2_smoke`。

Eval 配置为每个题目采样 2 次、temperature 0.7、top-p 1.0、最大 response 31744。启动参数 `--eval-max-prompt-len 1024 --eval-max-context-len 32768` 使 prompt 与 response 的总上限为 32768。没有启用 pass@2 聚合；`eval/aime24` 是 60 个生成结果上的平均二值 reward，也就是这里要求的 avg@2。

训练 smoke 设为 `NUM_ROLLOUT=1`、`ROLLOUT_BATCH_SIZE=2`、`N_SAMPLES_PER_PROMPT=2`、`GLOBAL_BATCH_SIZE=4`、`--micro-batch-size 1`。训练 rollout 使用 response cap 1024、context cap 4096。采用单机 8 卡、TP1/PP1/CP8/EP8，rollout 和 actor 按脚本配置拆卸/重建，以适配单机资源。

## 执行结果

### 起始模型 AIME24 基线

Eval 在训练前运行完毕，日志中的 `eval 0` 结果为：

| 指标 | 结果 |
| --- | ---: |
| AIME24 avg@2 | 0.6667（66.7/100） |
| 判对生成数 | 40 / 60 |
| response length mean / median | 18,605.6 / 17,429 tokens |
| response length min / max | 3,598 / 31,744 tokens |
| truncated ratio | 0.20 |

因此，“AIME 2024 评测在 65 分左右”的基线验收通过。20% 的输出触及 31744 上限，这会影响可解释性；记录该限制，不把这个分数当成无截断的模型能力估计。

### 训练单步和权重保存

- 真实 SGLang rollout 完成 2 个 prompt × 2 个 sample，rollout 时间约 13.4 秒。
- 两个训练样本都生成到 1024 token 上限并被截断；两者 reward 都是 0，组内 advantage 都是 0。
- actor 执行了 1 个 microbatch，actor train 时间约 48.4 秒，训练等待和预处理正常结束。
- MCore checkpoint 写入 `outputs/yulan_moe_128k_stage4_avg2_smoke/iter_0000000`；HF rollout checkpoint 写入 `outputs/yulan_moe_128k_stage4_avg2_smoke/hf/rollout_0`，包含 128 个 safetensors 分片。
- 训练后 SGLang engines 已重建，updated HF 模型开始接受 eval 请求，说明 checkpoint 导出后的服务重载路径已启动。

这次 smoke 证明 rollout→GRPO 数据处理→actor train step→checkpoint/HF 导出→SGLang 重建的执行链路能走通。由于两个样本 reward 和 advantage 均为零，不能据此声称模型获得了有效学习更新。下一轮应使用 response cap 内可完成、且同一 prompt 的两个 sample 有 reward 差异的样本；确认非零 advantage、梯度和参数变化后，再把“有效更新”标为通过。

## 停止边界和未完成项

训练后 eval 由 `--eval-interval 1` 自动触发，对更新后模型再次运行 AIME24 avg@2。用户随后要求停止进程，Ray job `raysubmit_Sd3siRn3tRSDCEVb` 已停止；此次更新后 eval 只启动了一部分请求，没有完整结果。不要把它和训练前完成的 66.7 分基线混淆。启动 checkpoint、训练 checkpoint 和 HF 导出均已落盘；此次停止没有停止共享 Ray 集群。

## 后续建议

1. 另准备短答案、可判分且组内 reward 有差异的训练样本，让 1024 response cap 不会把整个 batch 截断。
2. 复跑一个小 batch，检查 reward、advantage、grad norm、参数变化和 checkpoint 保存。
3. 若要做 train 前后 AIME24 对照，单独安排更新后 eval；当前未完成的 post-update eval 不作为验收数据。
