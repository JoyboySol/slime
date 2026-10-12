# YuLan-MoE-128k RL 阶段三：SGLang 权重与 logprob 验证

## 目标与结论

验证 SGLang 插件能否解析模型、服务原始 HF checkpoint 和 Slime 导出的 actor checkpoint，并检查 actor 更新是否反映在服务端结果中。

阶段三已验证插件解析、两份 checkpoint 的 SGLang 加载、固定输入生成和 prompt logprob sanity check。初始 HF→MCore→HF tensor roundtrip 逐位相同，验证了初始权重导出映射。阶段二 actor replay 对初始 SGLang rollout 的 `train/train_rollout_logprob_abs_diff` 为 `0.085617`，低于运行配置的 CI 阈值 `0.1`。

已完成同一更新后 checkpoint 的 MCore actor 与 SGLang 逐 token logprob 对照，但平均绝对差为 `0.112996`，超过 0.1 阈值。初始 tensor roundtrip 无差异，因此这个结果只说明训练后两边的前向 logprob 有偏差；它本身不能证明 exporter 转坏了权重。阶段三仍需定位这项前向差异的来源。

## 环境与模型

- 仓库分支：`lzh/yulan-moe-gpu-dev`
- Slime Python：`/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/python`
- SGLang 插件：`/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin`
- 插件包：`sglang_qwen3_next_plugin`
- 原始 HF checkpoint：`/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k`
- Slime 更新后 MCore checkpoint：`/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4`
- Slime 导出的 actor HF checkpoint：`/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4/hf/rollout_0`
- SGLang serving layout：TP2/EP2，BF16，context length 131072，A100 80GB 两卡。

## 执行记录

### 1. 插件解析

使用 Slime 虚拟环境和插件文档提供的 import check：

```bash
SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin XDG_CACHE_HOME=/tmp HOME=/tmp \
  /mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/python scripts/check_plugin_import.py
```

模型类成功解析为：

```text
sglang_qwen3_next_plugin.variants.sglang_0_5_9.Qwen3NextForCausalLM
```

### 2. 原始 HF checkpoint

首次启动暴露出服务进程需要 `ninja` 在 `PATH` 中。将 Slime venv 的 bin 目录加入 `PATH` 后，插件模型可加载并完成 warmup。服务参数如下，模型路径设为原始 HF checkpoint：

```bash
PATH=/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin:$PATH \
CUDA_VISIBLE_DEVICES=0,1 \
SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin \
XDG_CACHE_HOME=/tmp HOME=/tmp \
/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/python -m sglang.launch_server \
  --model-path /mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k \
  --host 127.0.0.1 --port 30110 --context-length 131072 \
  --tp-size 2 --ep-size 2 --dtype bfloat16 --trust-remote-code \
  --disable-cuda-graph --mem-fraction-static 0.85
```

插件仓库中的 `scripts/validate_generation.py` 两个固定输入均通过：中文自我介绍 prompt 和英文 `What is 2 + 3? Answer briefly.`。`scripts/validate_logprob.py` 也通过：

| Prompt | 原始 HF 平均 NLL | token 数 | 结果 |
|---|---:|---:|---|
| `请用两句话解释什么是机器学习。` | 3.5608 | 8 | PASS |
| `What is 2 + 3? Answer briefly.` | 3.1270 | 12 | PASS |

### 3. Slime 导出的 actor HF checkpoint

使用与原始 checkpoint 完全相同的 TP2/EP2、BF16 和 serving 参数，只把 `--model-path` 改为：

```text
/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4/hf/rollout_0
```

服务成功加载全部 128 个 safetensors shard，模型类为 `Qwen3NextForCausalLM`，warmup 成功。重复运行插件仓库中的固定输入检查：

| Prompt | actor 导出 HF 平均 NLL | token 数 | 结果 |
|---|---:|---:|---|
| `请用两句话解释什么是机器学习。` | 3.5278 | 8 | PASS |
| `What is 2 + 3? Answer briefly.` | 3.1068 | 12 | PASS |

两条 prompt 的 NLL 都与原始 HF checkpoint 不同，且生成请求通过。这证明 SGLang 确实载入了经过 actor 更新并导出的权重，而不是仍使用原始 checkpoint。此项是固定文本的服务端 sanity check，不是模型间精确 logprob 对照。

### 4. 初始 HF→MCore→HF tensor roundtrip

源 checkpoint 的 `release/common.pt` 元数据显示 `iteration=1`、`consumed_train_samples=0`；YuLan-Pretrain 保存的 SFT 启动命令把它作为 `--load` 输入。基于这些记录，此处将它作为未训练的初始 MCore 权重与原始 HF 做 roundtrip。

Slime 通用离线工具 `tools/convert_torch_dist_to_hf.py` 直接读取该 checkpoint 时，在 MoE 独立 expert tensor 布局上触发 `assert param.shape[0] == num_experts`，无法处理这份 checkpoint。于是改走 Slime 正常 actor load/export 路径：加载原始 MCore checkpoint，用阶段二固定 rollout 做一次 actor 前向，额外传 `--lr 0 --weight-decay 0`，日志确认 `train/lr=0.0`，然后使用正常 `--save-hf` exporter 输出到：

```text
/tmp/yulan_moe_128k_initial_roundtrip_actor/hf/rollout_0
```

逐 safetensors 分片对照原始 HF `/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k` 与 roundtrip HF：

| 检查项 | 结果 |
|---|---:|
| HF tensor keys | 32,226 vs 32,226；missing 0，unexpected 0 |
| shape mismatch | 0 |
| dtype | BF16 32,128 个，FP32 98 个；两边逐 tensor 一致 |
| tensor 元素数 | 34,107,921,536 |
| bitwise equal tensors | 32,226 / 32,226 |
| global mean abs diff / RMSE / max abs diff | 0 / 0 / 0 |
| finite values | 全部 finite |

这证明初始 HF 经这份 MCore checkpoint 再由 Slime exporter 输出后，所有权重逐位还原。它验证的是初始权重转换；训练后 checkpoint 的 SGLang/MCore 前向差异仍需要单独分析。

对于“导出是否改变 tensor”这个问题，逐 tensor roundtrip 是直接检查：它不依赖采样、tokenization、logprob kernel 或模型并行数值误差。这里得到的零差异说明初始权重的 HF→MCore→HF 映射正确；它不能单独证明经过训练更新后的每个 tensor 也正确导出，因此更新后 `0.112996` 的前向差异仍保留为独立待查项。

### 5. 更新后 actor 与 SGLang 的同状态 logprob 对照

为避免只验证模型能加载、却没有验证导出的 actor 分数，采用阶段二保存的 8 条 rollout。每条包含完整输入 token IDs，其中 prompt 长度为 58 或 68，response 长度为 1024。

1. 用 SGLang 加载更新后 actor HF checkpoint，serving layout 为 TP2/EP2、BF16；加入与 RL launcher 相同的 `--json-model-override-args '{"moe_router_sqrt_gate":true}'`。
2. 对每条完整 `input_ids` 调用 `/generate`，设置 `return_logprob=true`、`logprob_start_len=0`。SGLang 返回的输入 token IDs 与原 rollout 逐 token 相同；取 prompt 之后的 1024 项 logprob，替换固定 rollout 中对应的 `rollout_log_probs`。
3. 关闭 SGLang 服务，Slime 从更新后的 MCore checkpoint `/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4` 加载 actor，重放这组 scored rollout。

第一次对照时 serving 没传该 override，结果也是 `0.1129957512`。随后按 RL launcher 补上 override，再打分和回放，结果仍为 `0.1129957512`；两次 SGLang 针对固定 token 序列返回的 logprob 数值也完全相同。因此 sqrt-gate 配置差异已作为该偏差的原因排除，真正原因尚未定位。

Slime 日志 `/tmp/yulan_moe_128k_stage3_actor_logprob_compare_sqrtgate.log` 的 step 0 指标为：

```text
train/train_rollout_logprob_abs_diff = 0.11299575120210648
ci_train_rollout_logprob_abs_diff_threshold = 0.1
```

每条 response 的平均绝对差依次为 `0.10502`、`0.10538`、`0.08885`、`0.10634`、`0.12287`、`0.11764`、`0.12875`、`0.12911`。差值整体略高于 0.1 阈值；单 token 绝对差峰值在 2.4–6.1 范围。由此确认训练后 MCore actor 与 SGLang 服务对相同输入的前向 logprob 仍有偏差；但初始权重 roundtrip 已证明初始导出转换没有 tensor 差异，因此不能仅凭此指标断定训练后 exporter 错误。本阶段发现了运行时 logprob 不一致，尚未定位其来源。

固定 token 序列及 SGLang 新打分数据保存在 `/tmp/yulan_moe_128k_stage3_exported_actor_scored_rollout_sqrtgate.pt`，actor debug 数据保存在 `/tmp/yulan_moe_128k_stage3_actor_train_sqrtgate_0_rank0.pt`。

### 6. Actor 与初始 rollout logprob 的已有基线

阶段二真实 rollout 后用固定 rollout 重放 actor。其训练日志 `/tmp/yulan_moe_128k_stage2_export_retry4.log` 在 step 0 记录：

```text
train/train_rollout_logprob_abs_diff = 0.08561718463897705
ci_train_rollout_logprob_abs_diff_threshold = 0.1
```

该值是阶段二 actor 对初始 SGLang rollout 概率的比较，低于阈值；它不能替代更新后 checkpoint 的一致性结果。更新后同状态对照见上节，结果为 `0.112996`，超过阈值。

## 本阶段信源

| 信源 | 用途 |
|---|---|
| `/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/README.md` | 插件依赖、128k serving layout 和验证入口。 |
| `/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/scripts/check_plugin_import.py` | 确认 external model package 的解析结果。 |
| `/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/scripts/validate_generation.py` | 固定输入生成检查。 |
| `/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/scripts/validate_logprob.py` | prompt token logprob 与平均 NLL 检查。 |
| `.ai/adapt_rl_for_yulan_moe_128k/stage2_execution_and_sources.md` | 最终 actor MCore/HF checkpoint 路径及导出过程。 |
| `/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k/mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix/release/common.pt` | 检查初始 MCore checkpoint 的 iteration 与 consumed samples 元数据。 |
| `slime/backends/megatron_utils/hf_checkpoint_saver.py`、`slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py` | 初始 MCore 权重的 Slime HF exporter 路径及 Qwen3-Next/YuLan tensor 映射。 |
| `tools/convert_torch_dist_to_hf.py` | 尝试离线导出时因当前 MoE expert shard layout 不兼容而失败；改用 Slime actor saver 导出。 |
| `/tmp/yulan_moe_128k_initial_roundtrip_actor/hf/rollout_0` | 初始 MCore checkpoint 的 Slime HF roundtrip 结果；与原始 HF 全 tensor 逐位一致。 |
| `/tmp/yulan_moe_128k_stage2_export_retry4.log` | actor rollout logprob 差异、CI 阈值和训练 step 指标。 |
| `/tmp/yulan_moe_128k_stage3_actor_logprob_compare_sqrtgate.log` | 同一更新后 MCore actor 与导出 HF/SGLang 的固定序列 logprob 对照。 |
| `/tmp/yulan_moe_128k_initial_roundtrip_actor.log` | 初始 MCore checkpoint 的零学习率 actor replay 与 HF 导出日志。 |
| `scripts/run-yulan-moe-128k-math-smoke.sh` | 阶段二 MCore actor 和 rollout 配置来源。 |

服务测试后已停止 SGLang 进程，端口 30110 不再监听；随后 actor 对照任务也已退出。

## 后续待办

定位造成 `0.112996` 平均绝对差、以及最高约 `6.1` token 差异的具体运行时原因。由于初始 roundtrip 所有 tensor 逐位相同，后续应优先对同一个更新后 MCore checkpoint 做 tensor 级别 MCore/HF 对照，并核对 actor 与 SGLang 的 logprob kernel、精度、token 边界和 mask；再用同一固定序列复测。
