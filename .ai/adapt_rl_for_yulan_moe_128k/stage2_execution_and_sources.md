# YuLan-MoE-128k RL 阶段二：执行过程与信源

## 范围和结果

阶段二验证 Slime actor 的短序列前向、loss、反向、optimizer update、MCore checkpoint 保存/重载，以及更新后权重导出为 HF 格式。目标模型是 `/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k`。本阶段没有验证 SGLang 加载导出权重后的 logprob 一致性；该项属于阶段三。

在 8 卡 A100 上使用 TP1/PP1/CP8/EP8 训练 actor，rollout 使用 SGLang 的 TP2/EP2 layout。短测抽取 2 个 prompt，每个 prompt 采样 4 个回答；response 上限 1024，context 上限 4096。所有生成结果都触及 response 上限并被标记为 `truncated`，因此这些样本只证明训练管线能够运行，不用于评价回答质量。

最终验收结果：

- actor 完成真实前向、loss、反向和 optimizer update。最后一次成功运行的 `train/loss` 为 `-1.39698e-08`，`train/grad_norm` 为 `0.70580`，均为有限值；训练日志有 `actor train: 100%`。
- 更新后的 MCore torch_dist checkpoint 保存成功，并能由后续运行重新加载。
- HF 导出成功写出 128 个 safetensors 权重分片。对全部分片中的 32,226 个张量进行扫描后，所有浮点值均为有限值。
- 转换器覆盖了 YuLan GatedNorm、GDN fused/split 投影和 attention-output-gate QKV bias 的布局。

## 执行过程

### 1. 从 SGLang 生成固定 rollout

先用真实 SGLang rollout 得到小批样本，然后将样本落盘，供 actor 重放。这样能保留真实 rollout 和 reward 结果，同时在反复排查 exporter 时避免每次都重新生成回答。

```bash
OUTPUT_ROOT=/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_rollout_only \
ROLLOUT_NUM_GPUS=8 NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 \
ROLLOUT_MAX_RESPONSE_LEN=1024 ROLLOUT_MAX_CONTEXT_LEN=4096 \
RUN_LOG_FILE=/tmp/yulan_moe_128k_stage2_rollout.log \
bash scripts/run-yulan-moe-128k-math-smoke.sh \
  --debug-rollout-only \
  --save-debug-rollout-data '/tmp/yulan_moe_128k_stage2/rollout_{rollout_id}.pt' \
  --save-debug-rollout-interval 1
```

产物为 `/tmp/yulan_moe_128k_stage2/rollout_0.pt`，包含 8 个样本。两组样本各自有正负 reward，raw reward 均值为 0.25。8 个回答都达到 1024-token response 上限并带有 `truncated` 状态。

### 2. 用固定 rollout 做 actor 更新并保存 MCore checkpoint

关闭 rollout GPU，读入上一步的 rollout dump。启动脚本从指定的 MCore checkpoint 初始化模型，actor 在 8 卡上做一个训练 step，并将更新后的 checkpoint 写入新的输出目录。

```bash
OUTPUT_ROOT=/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_actor_smoke \
SAVE_HF='/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_actor_smoke/hf/rollout_{rollout_id}' \
ROLLOUT_NUM_GPUS=0 NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 \
ROLLOUT_MAX_RESPONSE_LEN=1024 ROLLOUT_MAX_CONTEXT_LEN=4096 \
RUN_LOG_FILE=/tmp/yulan_moe_128k_stage2_actor_train.log \
bash scripts/run-yulan-moe-128k-math-smoke.sh \
  --load-debug-rollout-data '/tmp/yulan_moe_128k_stage2/rollout_{rollout_id}.pt' \
  --save-debug-train-data '/tmp/yulan_moe_128k_stage2/train_{rollout_id}_rank{rank}.pt'
```

这一步完成 forward/backward/update，`train/grad_norm` 为有限值并保存了 `iter_0000000`。首次 HF 导出未完成：报错指出 exporter 没有识别 `module.module.decoder.final_layernorm.gate_down.weight`。

### 3. 独立验证 MCore checkpoint 可以重载

在 actor smoke 输出完成后运行 load-only 检查：

```bash
HF_CHECKPOINT=/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k \
MCORE_CHECKPOINT=/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_actor_smoke \
OUTPUT_ROOT=/tmp/yulan_moe_128k_stage2_reload_check \
CHECKPOINT_LOAD_ONLY=1 \
RUN_LOG_FILE=/tmp/yulan_moe_128k_stage2_reload_check.log \
bash scripts/run-yulan-moe-128k-math-smoke.sh
```

日志确认 5.57B actor 从 `yulan_moe_128k_stage2_actor_smoke` 的 iteration 0 成功加载，Ray 作业成功退出。

### 4. 按实际 checkpoint 命名补齐 HF exporter

排查过程以 Ray exporter 抛出的第一个未知参数为入口，并对照 YuLan-Pretrain 的 MCore↔HF synchronizer 和 SGLang 插件模块名确定目标 HF 名称及 tensor 排列：

1. `final_layernorm.gate_down.weight`：MCore GatedNorm 参数原来没有对应映射。新增 final norm、每层 input norm 和 pre-MLP norm 的映射，将 `norm.weight`、`gate_down.weight`、`gate_up.weight/bias` 分别映射到 HF/SGLang 的 `rms_norm`、`w_down`、`w_up`。
2. `self_attention.a_proj.weight`：MCore 将 GDN beta/alpha 分开保存，HF/SGLang 使用 `in_proj_ba.weight`。新增按 key-head 交错 beta/alpha 后合并的路径。
3. `self_attention.k_proj.weight`：运行时模型采用 split GDN q/k/v/z 投影，而非一个 fused `in_proj.weight`。新增缓存四个投影并按每个 key head 合并为 `in_proj_qkvz.weight` 的路径。
4. fused GDN 路径也修正为 YuLan 实际的 `[z, value, query, key, beta, alpha]` 顺序，再转换成 HF 的 qkvz/ba 排列。
5. `linear_qkv.bias` 的实际 per-group 宽度包含两组 query（query 与 output gate），原转换器按普通 GQA 的单组 query 拆分会 shape mismatch。新增双 query 布局的拆分和交错，与 YuLan 源 synchronizer 一致；普通 GQA 分支保留。

每种布局都使用合成张量做定向映射检查，并执行 `git diff --check`。最终 HF 导出由真实训练 checkpoint 验证通过。

### 5. 最终训练、保存、导出和有限值扫描

最后一轮 exporter 验收以此前成功保存的 MCore checkpoint 为起点，重放同一 rollout、再做一个短 actor update，并写入独立的最终目录：

```bash
HF_CHECKPOINT=/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k \
MCORE_CHECKPOINT=/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry3 \
OUTPUT_ROOT=/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4 \
SAVE_HF='/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4/hf/rollout_{rollout_id}' \
ROLLOUT_NUM_GPUS=0 NUM_ROLLOUT=1 ROLLOUT_BATCH_SIZE=2 \
N_SAMPLES_PER_PROMPT=4 GLOBAL_BATCH_SIZE=8 \
ROLLOUT_MAX_RESPONSE_LEN=1024 ROLLOUT_MAX_CONTEXT_LEN=4096 \
RUN_LOG_FILE=/tmp/yulan_moe_128k_stage2_export_retry4.log \
bash scripts/run-yulan-moe-128k-math-smoke.sh \
  --load-debug-rollout-data '/tmp/yulan_moe_128k_stage2/rollout_{rollout_id}.pt' \
  --save-debug-train-data '/tmp/yulan_moe_128k_stage2/train_retry4_{rollout_id}_rank{rank}.pt'
```

成功产物：

- MCore checkpoint：`/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4/iter_0000000`。
- HF checkpoint：`/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_128k_stage2_export_retry4/hf/rollout_0`，含 `model.safetensors.index.json`、128 个 safetensors 分片、模型配置和 tokenizer 文件。
- Ray 训练/导出日志：`/tmp/yulan_moe_128k_stage2_export_retry4.log`，Ray job `raysubmit_ewzYn6B1HSqmRpyt` 状态为 succeeded。
- finite scan：逐个打开全部 128 个 safetensors 分片，读取并检查 32,226 个张量中的浮点值；结果均有限。

## 排障迭代记录

每次尝试都保留独立输出目录；HF exporter 失败发生在 MCore checkpoint 已成功写出的阶段，所以后一次尝试可以从上次 MCore checkpoint 重新加载。

| 输出目录 | 导出结果 | 发现与处理 |
|---|---|---|
| `yulan_moe_128k_stage2_actor_smoke` | 失败 | 缺少 final GatedNorm 的 HF 名称映射；补齐 GatedNorm 映射。 |
| `yulan_moe_128k_stage2_export_retry` | 失败 | 缺少 GDN `a_proj.weight`；依据 YuLan synchronizer 合并分开的 beta/alpha。 |
| `yulan_moe_128k_stage2_export_retry2` | 失败 | 缺少 split GDN `k_proj.weight`；补齐 split q/k/v/z 合并，并校准 fused GDN section 顺序。 |
| `yulan_moe_128k_stage2_export_retry3` | 失败 | 带 output gate 的 QKV bias 每组宽度大于普通 GQA；补齐双 query bias 的拆分重排。 |
| `yulan_moe_128k_stage2_export_retry4` | 成功 | 128 个 HF 权重分片全部写完，浮点张量 finite scan 通过。 |

## 信源目录

以下是本阶段决策和验证直接使用的本地信源。临时日志和输出保存在 `/tmp` 与 PostTrain outputs 下；正式实现与计划记录在当前 Slime 仓库。

| 信源 | 用途 |
|---|---|
| `.ai/adapt_rl_for_yulan_moe_128k/plan.md` | 阶段目标、验收条件、阶段顺序和本阶段结果。 |
| `.ai/adapt_rl_for_yulan_moe_128k/stage1_model_structure_and_checkpoint_loading.md` | 阶段一操作过程、HF/MCore 模型结构对照和 checkpoint 加载记录。 |
| `scripts/run-yulan-moe-128k-math-smoke.sh` | 本阶段 actor、rollout-only、debug replay、checkpoint 和 HF export 的启动入口。 |
| `scripts/models/yulan-moe-128k.sh` | 128k 专属模型参数，含 GatedNorm、GDN、MoE、attention output gate 和 router 参数。 |
| `slime_plugins/models/yulan_moe.py` | 128k YuLan 模型 provider 和模型层 pattern 入口。 |
| `slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py` | 此阶段修复的 Qwen3-Next/YuLan 权重名称和 tensor 布局转换。 |
| `slime/backends/megatron_utils/megatron_to_hf/__init__.py`、`slime/backends/megatron_utils/hf_checkpoint_saver.py` | exporter 分派与 HF checkpoint 写出流程。 |
| `/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k/config.json`、`model.safetensors.index.json` | HF 架构配置、模型层次、权重名称和 shard 信息。 |
| `/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k/mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix` | 初始 MCore torch_dist checkpoint；阶段一加载和阶段二训练初始化所用。 |
| `/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/outputs/yulan_moe_128k_alltoall_smoke_5steps_qgbiasfix/pretrain_gpt_command.sh` | SFT 训练参数来源，与 HF config 和 checkpoint 一起核对结构。 |
| `/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/megatron/core/ssm/gated_delta_net.py` | GDN split projection 参数定义、名称以及 q/k/v/z/b/a 语义。 |
| `/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/scripts/distributed_checkpoints_convertor/impl/qwen3_next/m2h_synchronizer.py` | YuLan fused MCore GDN 到 HF/SGLang qkvz/ba 的权威排列参考。 |
| `/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/scripts/distributed_checkpoints_convertor/impl/qwen3_next/h2m_synchronizer.py` | HF 到 MCore 的反向布局参考；确认 fused 顺序为 z/v/q/k/b/a。 |
| `/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/megatron/core/transformer/gated_norm.py` | YuLan GatedNorm 的 MCore 参数名和结构。 |
| `/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/sglang_qwen3_next_plugin/variants/sglang_0_5_9.py` | SGLang GatedNorm/GDN HF 模块名、`in_proj_qkvz` 与 `in_proj_ba` 权重布局。 |
| `/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/README.md` | 插件依赖、TP2/EP2 模型加载与短生成背景；本阶段未据此宣称 actor 导出权重已通过 SGLang 验收。 |
| `/mnt/yulan/lvzhihao/PostTrain/data/hf_data/AIME_like_data/aime_8.jsonl` | smoke rollout 数据来源。 |
| `/tmp/yulan_moe_128k_stage2/rollout_0.pt`、`/tmp/yulan_moe_128k_stage2/train_retry4_0_rank0.pt` | 固定 rollout 和 actor debug batch 的中间产物。 |
| `/tmp/yulan_moe_128k_stage2_rollout.log`、`/tmp/yulan_moe_128k_stage2_actor_train.log`、`/tmp/yulan_moe_128k_stage2_reload_check.log`、`/tmp/yulan_moe_128k_stage2_export_retry4.log` | rollout、actor 更新、MCore reload 和最终 HF 导出的运行证据。 |

## 当前工作树与后续

本阶段代码修改位于 `slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py`，阶段验收更新位于 `plan.md`。这些改动随本过程文档一起提交。阶段三仍需单独启动目标 SGLang 插件，加载最终 HF 导出，并比较固定输入上的 actor/SGLang logprob；阶段四以后再验证完整 RL 多轮同步、colocate 生命周期和 32k context。
