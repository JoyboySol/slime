# 02 最小化测试 YuLan Hybrid GDN Dense

## 测试目标

在 `slime` 中使用以下资源完成一次最小化 RL 冒烟测试，并确认自定义 `Hybrid GDN Dense` 架构能够走通：

- Model Path: `/mnt/hdd/lvzhihao/hf_models/Dist-mathcode10b-s1randg-sch1-CPT-200b-stage3-r640k-GDN2.9b-A7-12_20_21_23_46_48_49-sl32768bs128lr2e5-2e5/merged_10ckpts_iter_61984-hf_to_iter_71525-hf_mean`
- Data Path: `/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl`
- GPUs: `0,1,2,3`
- SGLang plugin: `sglang_qwen3_next_plugin`

## 最终结论

本次 `YuLan Hybrid GDN Dense` 的 slime RL 最小化测试已经跑通。

这里的“跑通”指的是：

1. Ray 作业成功启动，并完成 `slime.train` 主流程。
2. SGLang rollout engine 成功拉起，并通过 health check。
3. rollout 真正执行，生成样本并回传训练端。
4. actor 侧完成 `step 0` 训练。
5. `save_model` 成功执行，并写出 actor checkpoint。
6. driver 正常退出，终端打印 `[driver] train(args) done`。

## 成功运行时的关键命令

最终跑通时使用的是 `scripts/run-yulan-hybrid-gdn-dense-smoke-rl.sh`，其中关键环境如下：

```bash
source /mnt/ssd/lvzhihao/PostTrain/slime/.venv/bin/activate
export PYTHONPATH=/mnt/ssd/lvzhihao/PostTrain/YuLan-Pretrain
export CUDA_VISIBLE_DEVICES=0,1,2,3
export SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float32
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost

/mnt/ssd/lvzhihao/PostTrain/slime/scripts/run-yulan-hybrid-gdn-dense-smoke-rl.sh
```

说明：

- `SGLANG_EXTERNAL_MODEL_PACKAGE` 不能全局直接导出给整个 driver / Ray 初始化过程。
- 应通过 `SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE` 传给 slime，再由 rollout actor 的 runtime env 注入为真正的 `SGLANG_EXTERNAL_MODEL_PACKAGE`。
- `SGLANG_MAMBA_CONV_DTYPE=float16` 与 `SGLANG_MAMBA_SSM_DTYPE=float32` 是本次 Hybrid GDN Dense 模型在 SGLang 0.5.9 上稳定跑通的必要条件。
- 必须取消代理环境变量，否则当前机器上的 `ray.init()` / Ray 本地链路有明显概率卡住。

## 实际使用的 smoke 参数

本次为了优先验证链路可用性，使用的是保守但真实可运行的一组参数：

```text
--num-rollout 1
--rollout-batch-size 2
--n-samples-per-prompt 2
--num-steps-per-rollout 1
--global-batch-size 4
--rollout-max-response-len 256
--rollout-temperature 0.8
--max-tokens-per-gpu 2048
--rollout-num-gpus-per-engine 1
--sglang-mem-fraction-static 0.2
--sglang-context-length 2048
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--context-parallel-size 1
--expert-model-parallel-size 1
```

这组参数并不是“最大吞吐”的最终配置，但它已经能稳定完成一轮 rollout + train + save，是后续继续调吞吐的可靠基线。

## 关键修复与根因

### 1. Ray 会被代理环境变量干扰

现象：

- `ray.init()` 曾经出现“本地实例已经启动，但 driver 长时间不返回”的情况。
- `ray job submit` / 本地 driver 链路都表现出异常。

根因：

- 当前机器的 `HTTP_PROXY` / `HTTPS_PROXY` 会干扰 Ray 的本地控制面通信。

解决方式：

```bash
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=127.0.0.1,localhost
```

### 2. 不能全局导出 `SGLANG_EXTERNAL_MODEL_PACKAGE`

现象：

- 如果在 driver 级别直接 `export SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin`，Ray 初始化链路会表现异常甚至挂住。

解决方式：

- 改为使用 `SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE` 作为 slime 层的输入环境变量。
- 在 rollout actor 创建时，把它转换成 runtime env 中的 `SGLANG_EXTERNAL_MODEL_PACKAGE`。

对应文件：

- `slime/ray/rollout.py`

### 3. SGLang Hybrid Mamba 路径存在 conv dtype 不匹配

历史报错：

```text
RuntimeError: Expected conv_states_.scalar_type() == input_type to be true
```

根因：

- Hybrid GDN Dense 在当前 SGLang 0.5.9 的 mamba/linear attention 路径中，conv state dtype 与输入 dtype 不一致。

解决方式：

```bash
export SGLANG_MAMBA_CONV_DTYPE=float16
export SGLANG_MAMBA_SSM_DTYPE=float32
```

### 4. Megatron -> HF 的 QKV 拆分逻辑需要适配当前 checkpoint

历史报错：

```text
split_with_sizes expects split_sizes to sum exactly to 7 ... got [10, 1, 1]
```

根因：

- `slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py` 中，对 `self_attention.linear_qkv.weight` 的拆分逻辑沿用了不适用于当前 Hybrid GDN Dense checkpoint 的假设。

实际修复：

- 将 `linear_qkv.weight` 的拆分从错误的双倍 Q 打包假设，修正为标准 GQA packed 全注意力布局。
- 当前代码使用：

```python
q_param, k_param, v_param = torch.split(
    param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
)
```

对应文件：

- `slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py`

## 跑通证据

### 1. driver 正常完成

`run.log` 末尾出现：

```text
[driver] parse_args done
[driver] train(args) begin
[driver] train(args) done
```

文件位置：

- `/mnt/ssd/lvzhihao/PostTrain/slime/.tmp/yulan_hybrid_gdn_dense_smoke_rl/run.log`

### 2. rollout 真实发生

Ray worker 日志中已经出现真实 rollout 样本收集和吞吐指标，之前排查时可见：

- `First rollout sample: ...`
- `Finish rollout: ...`
- `Final collected 4 samples from rollout to train`
- `perf/tokens_per_gpu_per_sec: 42.14848422494363`

说明不是“空跑”或只停留在初始化阶段。

### 3. actor 侧完成训练 step 0

`MegatronTrainRayActor` 日志中有明确训练指标：

```text
step 0: {
  'train/loss': 0.0,
  'train/pg_loss': 0.0,
  'train/entropy_loss': 4.846604347229004,
  'train/ppo_kl': 0.0,
  'train/grad_norm': 0.0,
  'train/step': 0
}
```

同时有性能指标：

```text
perf 0: {
  'perf/ref_log_probs_time': 62.21775937080383,
  'perf/actor_train_time': 65.66773915290833,
  'perf/train_time': 128.96854257583618,
  'perf/actor_train_tok_per_s': 21.34990511452544,
  'perf/step_time': 138.49948954582214
}
```

### 4. save_model 成功完成

日志中已经出现：

```text
Timer save_model start
Timer save_model end (elapsed: 31.1s)
```

并且 checkpoint 真实落盘。

### 5. actor checkpoint 已写出

checkpoint 输出目录：

- `/mnt/ssd/lvzhihao/PostTrain/slime/.tmp/yulan_hybrid_gdn_dense_smoke_rl/actor_ckpt`

实测结果：

- 目录大小约 `35G`
- 文件数 `14`

可见文件包括：

- `latest_checkpointed_iteration.txt`
- `iter_0000000/common.pt`
- `iter_0000000/metadata.json`
- `iter_0000000/__0_0.distcp`
- `iter_0000000/__0_1.distcp`
- `iter_0000000/__1_0.distcp`
- `iter_0000000/__1_1.distcp`
- `iter_0000000/__2_0.distcp`
- `iter_0000000/__2_1.distcp`
- `iter_0000000/__3_0.distcp`
- `iter_0000000/__3_1.distcp`

## 本次运行涉及的关键文件

### 运行脚本

- `scripts/run-yulan-hybrid-gdn-dense-smoke-rl.sh`
- `scripts/models/yulan-hybrid-gdn-dense-2.9b.sh`
- `tools/run_train_with_ray_init.py`

### 适配与修复

- `slime/ray/rollout.py`
- `slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py`
- `slime_plugins/mbridge/qwen3_next.py`

## 运行后的残留说明

日志末尾有一条：

```text
WARNING: destroy_process_group() was not called before program exit
```

这条 warning 没有阻止本次训练、保存和 driver 正常退出；当前将其视为退出清理不够优雅的告警，而不是本次 smoke 的 blocker。

## 后续建议

1. 先以这套 smoke 参数作为稳定基线，不要在没有保留它的情况下直接继续加大吞吐。
2. 下一步如果要追求“吞吐量最大的参数组合”，建议只单变量调整以下参数：
   - `MAX_TOKENS_PER_GPU`
   - `ROLLOUT_BATCH_SIZE`
   - `N_SAMPLES_PER_PROMPT`
   - `SGLANG_MEM_FRACTION_STATIC`
3. 在所有后续实验中，继续保留以下前置条件：
   - `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy`
   - `export SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin`
   - `export SGLANG_MAMBA_CONV_DTYPE=float16`
   - `export SGLANG_MAMBA_SSM_DTYPE=float32`

## 状态摘要

截至 `2026-04-09`，`YuLan Hybrid GDN Dense` 在 `slime` 中已经完成一次真实的最小化 RL 流程冒烟测试，链路状态为：

- 环境可用
- 插件接管可用
- rollout 可用
- train step 可用
- checkpoint save 可用
- driver 正常退出
