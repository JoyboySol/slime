# 阶段 1 教程：对齐 YuLan-MoE-128k 模型结构与 checkpoint 加载

本文记录阶段 1 的目标、信源、参数对齐过程、实际加载验证方式、结果和已知限制。目标读者是后续需要接入另一个 YuLan 模型的同学；可以把文末的“新模型信源清单”当成新任务的资料收集模板。

## 1. 阶段目标和完成边界

阶段 1 要回答两个问题：

1. Slime/Megatron 构造出来的模型，是否按 128k 模型自己的结构和语义配置，而不是误用了 Base 的结构？
2. Slime 能否从这次 RL 计划复用的 MCore checkpoint 构造 actor 并加载权重？

本阶段做了结构参数核对和 checkpoint 初始化加载。加载测试使用 8 张 GPU，TP1/PP1/CP8/EP8，模型最大位置长度为 131072；RL 的 rollout context 目标仍为 32768。`max_position_embeddings=131072` 是模型能力配置，不表示本次要在 128k token 上做 RL。

本阶段没有执行真实 actor 前向/反向、optimizer 更新、保存重载、SGLang 权重同步、rollout、reward、端到端 RL，也没有验证 32k 下的显存和吞吐。这些留给后续阶段。尤其是模型能够构造并加载 checkpoint，不等于 GDN CP、packed THD/cu_seqlens 的训练前向已经验证。

## 2. 这次使用的环境和输入

主要工作目录：

```text
/mnt/yulan/lvzhihao/PostTrain/slime
```

模型和训练相关目录：

```text
HF 模型：       /mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k
MCore 初始权重：/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k/mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix
SFT 命令快照：  /mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/outputs/yulan_moe_128k_alltoall_smoke_5steps_qgbiasfix/pretrain_gpt_command.sh
Slime venv：    /mnt/yulan/lvzhihao/PostTrain/slime/.venv
Pretrain 源码：  /mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain
SGLang 插件：   /mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin
```

训练目标约束来自需求确认：Base 和 128k 虽然相近但并不相同，因此新建 128k 专用参数/启动入口，不改 Base；本次单机 RL 的 context 用 32k；优先评估复用 SFT 产出的 MCore checkpoint。

## 3. 信源核对顺序

不要从旧启动脚本复制一份后就认为结构已对齐。对本次模型采用了以下交叉核对：

| 信源 | 用途 | 本次核对到的内容 |
| --- | --- | --- |
| HF `config.json` | 模型结构的外部描述 | `model_type=qwen3_next`、56 层、隐藏维度/头数、层类型排列、词表、最大位置长度、GatedNorm、attention output gate 和 MoE 配置 |
| YuLan-Pretrain 的 SFT 完整命令快照 | 训练实际采用的 MCore 参数 | HF config 未表达或表达不全的 GDN、MoE、QKV bias、clamp、parallelism、checkpoint format 等参数 |
| MCore checkpoint 的 `latest_checkpointed_iteration.txt`、`release/metadata.json`、`release/common.pt` 和 shard | 确认 checkpoint 是否存在、格式和保存时的架构语义 | `latest` 指向 `release`；metadata 标识 `torch_dist`；common args 和 shard 中的实际模型 key/shape 用于与待构造模型核对 |
| Slime 现有 YuLan Base 参数与 full launcher | 了解 Slime 参数写法、训练流程和已有实现 | 只作为启动/算法/资源配置参考；没有把 Base 参数文件当成 128k 的权威结构定义 |
| Slime YuLan provider、Megatron/YuLan 模型实现 | 确认参数如何落实到 module/state dict | 共享 `slime_plugins.models.yulan_moe.model_provider`，开启 YuLan TransformerBlock 路径，以容纳 GatedNorm 和 GDN block 级 CP 行为 |
| SGLang 插件 README/实现 | 后续 serving 和 rollout 集成的依据 | 128k HF serving 已有 TP2/EP2 的已知布局；这不能替代 Slime checkpoint/load 或权重同步验证 |
| 实际 Ray actor load-only 日志 | 最终加载证据 | actor 构造参数量、拓扑、checkpoint 路径、strictness warning、加载 iteration 和 Ray job 状态 |

本次具体查看的实现文件包括 Slime 的 `slime_plugins/models/yulan_moe.py`（provider/module 选择）、`slime/backends/megatron_utils/data.py` 和 `cp_utils.py`（YuLan THD/CP 开关），YuLan-Pretrain 的 `megatron/core/ssm/gated_delta_net.py`（split projection 与 checkpoint key 映射）、`megatron/core/models/gpt/gpt_layer_specs.py`（linear-attention/MoE layer pattern）和 `megatron/training/checkpointing.py`/`megatron/core/dist_checkpointing/validation.py`（strictness 行为）。SGLang 的 `scripts/serve_yulan_moe_128k.sh`、`tests/test_yulan_moe_128k.py` 与模型实现也作为 serving 支持线索；这次阶段 1 没有重新跑 SGLang 测试。

若信源有冲突，优先保留冲突并查明，不能静默选一个值。本次一个重要例子是：SFT 命令使用 `--gdn-split-in-proj`，而 MCore checkpoint 的 common args 显示保存参数中的 split 标志为 false。通过 MCore split GDN 的 virtual sharded-state key 转换实际加载后，确认该 checkpoint 可与 split runtime module 对接；今后遇到相似差异也要靠实际状态字典转换/加载来验收，不应只相信单份 args dump。

## 4. 结构参数如何对齐

### 4.1 基础架构和维度

HF config 与 SFT 命令共同确认了以下结构：

| 项目 | 128k 值 | 来源/Slime 配置 |
| --- | ---: | --- |
| model type | `qwen3_next` | HF config；smoke 中 `--model-name qwen3_next` |
| transformer 层数 | 56 | HF `num_hidden_layers`；`--num-layers 56` |
| hidden size | 2048 | HF `hidden_size`；`--hidden-size 2048` |
| dense FFN intermediate | 4800 | HF `intermediate_size`；`--ffn-hidden-size 4800` |
| attention heads | 32 | `--num-attention-heads 32` |
| query groups / KV heads | 8 | HF `num_key_value_heads`；`--group-query-attention --num-query-groups 8` |
| head dim | 64 | HF `head_dim`；`--kv-channels 64`，且 32×64=2048、8×64=512 |
| vocab size | 99072 | HF `vocab_size`；`--vocab-size 99072` |
| max positions | 131072 | HF `max_position_embeddings`；`--max-position-embeddings 131072` |
| norm epsilon | 1e-6 | HF `rms_norm_eps`；`--norm-epsilon 1e-6` |
| position embedding | none | SFT `--position-embedding-type none` |

SFT 命令的 `--seq-length 49152` 是这条 SFT 任务的数据/训练序列长度；它与模型位置上限 131072 是不同概念。当前 RL 总 context 按需求使用 32768，不把模型的最大位置长度误当成训练长度。

### 4.2 Hybrid attention 与 GDN

HF `layer_types` 给出 56 层里 linear attention 和 full attention 的逐层排列。Slime 的 YuLan provider 通过 `MambaStack.allocate_layers` 构造 GDN/attention 层，因此用 hybrid override pattern 精确表达该逐层排列。两个 hybrid ratio 设为 0 的作用是让 allocator 不对 override pattern 强制执行自动分配的 layer-count 一致性检查；override pattern 仍会实际决定每层是 GDN（`M`）还是 full attention（`*`）。当前值：

```text
--hybrid-override-pattern MMMMMMMMMMMM\*MMMMMMM\*\*M\*MMMMMMMMMMMMMMMMMMMMMM\*M\*\*MMMMMM
--linear-attention-type gated_delta_net
--linear-attention-freq "[1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,1,1,1,1]"
```

上面的 `M`/`*` 是当前 provider 使用的 pattern 表示，应该和 HF `layer_types` 按每层逐项核对；更改模型层类型时，不能只复制总数相同但次序不同的 pattern。该 pattern 长度为 56，其中 49 个 `M` 对应 HF linear-attention/GDN 层，7 个 `*` 对应 full-attention 层，没有 `-`（独立 MLP-only layer）。各层内部使用 dense MLP 还是 MoE，由独立的 `moe_layer_freq` 决定。

SFT 命令中的 `--hybrid-attention-ratio 0.0625`、`--hybrid-mlp-ratio 0.5` 不能原样搬进当前 Slime 模型参数。该 SFT 启动的是 `pretrain_gpt.py`，经 `gpt_builder` 构造 GPT decoder block；逐层 attention 类型和 dense/MoE 类型由 `linear_attention_freq` 与 `moe_layer_freq` 生成，hybrid ratios 不负责该 GPT builder 的逐层结构分配。Slime 的 `yulan_moe.model_provider` 则构造 YuLan MambaStack，并把 hybrid ratios 和 override pattern 一起传给 `allocate_layers`。对 56 层而言，SFT 的 ratio 自动计数约为 4 个 attention、28 个 MLP-only、24 个 GDN/Mamba；这与 HF 的 7 个 full attention、49 个 GDN pattern 数量不同。若把这两个 ratio 与当前 override pattern 同时使用，allocator 会因 override 的符号计数不匹配而报错；若没有 override，则会构造出错误的混合层类型。因此这里的 0/0 是适配 Slime allocator + 精确 override 的配置方式，不代表把模型 attention/MLP 设为 0，也不改变 `moe_layer_freq` 所定义的逐层 MoE 结构。

GDN 的结构参数来自 SFT 命令：conv kernel 4，key/value head dim 64，key heads 8，value heads 32。配置启用 split projection、fused split forward，并按 SFT 设置 alpha weight scale 和 LR multiplier。split projection 在该模型上要求 TP=1，这是本次 TP1 布局与该配置相容的原因之一。GDN 的完整运行时 kernel/CP 路径还要在阶段 2 做短序列 forward/backward 验证。

### 4.3 Attention、Norm、输出和数值语义

以下参数是不能只靠 hidden size 等常规形状推断出来的模型语义：

```text
--disable-bias-linear
--add-qkv-bias
--attention-output-gate
--gated-norm
--gated-norm-lora-rank 16
--normalization RMSNorm
--final-logits-clamp-value 30
--activation-func-clamp-value 7
--activation-func-clamp-mode soft
```

HF config 显示 `attn_output_gate=true`、`norm_type=gated_rms`；SFT 命令补足 gated norm rank 16、QKV bias 和 clamp 行为。参数维度可以“成功加载”但 gate/norm 语义配错，因此这些需要显式配置。

### 4.4 MoE

HF/SFT 信息显示 192 个 experts、top-k 4、expert intermediate 512、shared expert intermediate 512。Slime 参数还显式对齐 softmax router、fp32 router、sqrt gate、grouped GEMM、shared expert gate、global aux loss 和 aux loss coefficient 0.05。MoE 层逐层分布由 SFT 的 `--moe-layer-freq` 列表映射，不应依据“多数层是 MoE”而自己重建。

其中 `moe_router_sqrt_gate` 不一定存在于发布的 HF config，但已作为 YuLan checkpoint 语义出现在 SFT args/Pretrain 代码中。Slime 训练侧用 `--moe-router-sqrt-gate` 并设置 `MOE_ROUTER_SQRT_GATE=1`；SGLang 侧通过 `--sglang-json-model-override-args` 注入 `{"moe_router_sqrt_gate":true}`，不修改原始 HF config 文件。

### 4.5 并行和 context

本次 load-only 使用：

```text
TP=1, PP=1, CP=8, EP=8, expert TP=1
```

这与 MCore checkpoint 名称 `tp1pp1ep8` 和 SFT 命令里的 TP1/PP1/CP8/EP8 相符。load-only 需要 8 张 GPU，因为它实际构造完整 actor；它不是 CPU-only 的元数据检查。

全量 RL launcher 的预设是 `rollout-max-context-len=32768`、默认 response 上限 16384、`MAX_TOKENS_PER_GPU=1024`。这些是全量训练候选值，32k 资源稳定性还未在阶段 1 验证。smoke launcher 的 response 默认 8192，`MAX_TOKENS_PER_GPU` 默认 1536，且可通过环境变量覆盖。

## 5. 为何使用独立参数文件和 provider 环境开关

新文件是 `scripts/models/yulan-moe-128k.sh`。它定义 128k 自己的 `MODEL_ARGS`，复用 Slime 的 YuLan provider：

```text
slime_plugins.models.yulan_moe.model_provider
```

这样 Base 参数与 128k 参数各自维护；共享 provider 只表示可复用同一套模型构造代码，不表示 Base 与 128k 结构相同。

smoke 启动前设置：

```bash
export SLIME_USE_YULAN_TRANSFORMER_BLOCK=1
export SLIME_USE_YULAN_THD_CP=1
export MOE_ROUTER_SQRT_GATE=1
```

`SLIME_USE_YULAN_TRANSFORMER_BLOCK=1` 是必要的：独立启动时不能假定当前 shell 已经预置了 Base full launcher 的环境。该路径让 provider 使用 YuLan 的 TransformerBlock adapter，从而承载 GatedNorm final norm 和 GDN block-level CP 协议。若走旧 MambaStack，加载时会期望普通 `final_layernorm.weight`，而实际 checkpoint 是 `final_layernorm.norm.*` 和 gate 参数，曾因此报 `KeyError`。

`SLIME_USE_YULAN_THD_CP=1` 让 Slime 的数据和 CP 辅助路径使用 YuLan 的 THD/CP 适配。阶段 1 只确认这个环境能随 Ray runtime env 到达 actor，并成功初始化/加载；packed THD、cu_seqlens 的实际前向行为仍未验收。

smoke 脚本会构造 Ray runtime env JSON，把这些开关以及 `PYTHONPATH`、插件路径等传给 Ray worker。只在提交进程 `export` 变量而没有传播给 worker，是常见的“driver 看得到、actor 看不到”的配置错误。

## 6. Checkpoint 选择和加载语义

### 6.1 先检查 checkpoint 目录

这次采用的 MCore 权重候选是：

```text
/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k/mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix
```

检查至少包括：

1. `latest_checkpointed_iteration.txt` 存在，内容为 `release`。
2. `release/metadata.json` 存在，`sharded_backend` 为 `torch_dist`。
3. `release/common.pt`/common args 与目标结构参数相符。
4. rank shard 存在且可读；如果能从 shard metadata 查看 QKV、norm、GDN、MoE 权重 shape/key，也要核对关键项。
5. 了解该 checkpoint 是否保存 optimizer/RNG，以及你是“用作初始权重”还是“恢复原训练”。

SFT 命令使用 `--no-save-optim`，本次把 checkpoint 当 RL 的初始模型权重，不恢复 SFT optimizer 状态。Slime 启动参数使用 `--ref-load`，在没有单独 `--load` 时由 Slime 用它初始化训练权重；checkpoint iteration 0 的日志表示这次作为新 RL run 的初始化，不表示源文件的训练历史被覆写。

### 6.2 load-only smoke 命令

从 Slime repo 根目录运行：

```bash
cd /mnt/yulan/lvzhihao/PostTrain/slime

CHECKPOINT_LOAD_ONLY=1 \
RUN_LOG_FILE=/tmp/yulan_moe_128k_checkpoint_load_log_all.log \
bash scripts/run-yulan-moe-128k-math-smoke.sh
```

必要时可显式改路径：

```bash
HF_CHECKPOINT=/path/to/hf-model \
MCORE_CHECKPOINT=/path/to/mcore-checkpoint \
CHECKPOINT_LOAD_ONLY=1 \
RUN_LOG_FILE=/tmp/model_checkpoint_load.log \
bash scripts/run-yulan-moe-128k-math-smoke.sh
```

注意，脚本当前仍要求 `PROMPT_DATA` 存在，即使 load-only 不生成 rollout。这是 launcher 的路径 preflight，不代表 load-only 会训练或读入 prompt 做生成。当前默认 prompt 文件是 `data/hf_data/AIME_like_data/aime_8.jsonl`。

`CHECKPOINT_LOAD_ONLY=1` 的效果：

- 设置 rollout GPU 数为 0、rollout 数为 0；
- 增加 `--debug-train-only`，只构建训练 actor 并加载模型；
- 对 distributed checkpoint 使用 `DIST_CKPT_STRICTNESS`，默认 `log_all`，让所有 mismatch 被记录但继续尝试加载；
- 仍会启动 Ray，并给 actor 分配 8 张 GPU；因此机器需要空闲的 8 卡和足够显存；
- 不执行 optimizer step，不启动本次验证所需的 SGLang rollout engines。

第一次在此脚本执行时，如果本机 Ray dashboard 尚未运行，脚本会用 8 GPU 启动 Ray head。任务结束后可查看 GPU/Ray 状态；在确认没有别的工作使用当前 Ray cluster 后，可执行：

```bash
/mnt/yulan/lvzhihao/PostTrain/slime/.venv/bin/ray stop --force
```

不要在共享机器上未确认其他用户/任务的情况下停止共用 Ray cluster。

### 6.3 strictness 的解释和这次实际结果

Megatron 的 DCP strictness 不同于普通 `load_state_dict(strict=True)`。本次使用 `log_all` 时，加载日志报告了 `decoder.layers.*.self_attention.in_proj._extra_state` 这一组 GDN split projection 的 metadata key mismatch；它没有报告参数 tensor 的 missing/unexpected key 或 shape mismatch，随后日志报告 checkpoint 成功加载，Ray job succeeded。

将 `DIST_CKPT_STRICTNESS=raise_all` 用于同一 load-only 流程会在 DCP metadata 检查处因为这组 `_extra_state` key 提前报错。Pretrain 的 GDN split-in-proj `_load_from_state_dict` 逻辑会将 checkpoint 的 virtual `in_proj.*` 项转换到拆分 runtime projection，并为不存在的空 Transformer Engine `_extra_state` 恢复空默认值。该行为解释了为什么模型加载能完成，但不能把 `raise_all` 说成通过。

复现有 warning 的成功加载：

```bash
CHECKPOINT_LOAD_ONLY=1 \
DIST_CKPT_STRICTNESS=log_all \
RUN_LOG_FILE=/tmp/yulan_moe_128k_checkpoint_load_log_all.log \
bash scripts/run-yulan-moe-128k-math-smoke.sh
```

若未来 checkpoint 出现其他 mismatch，不能因为本次 `_extra_state` 可兼容就一概忽略。应逐个确认 key 所属模块、tensor/metadata 类型、转换钩子和实际模型输出；若是任何模型参数 tensor 的 mismatch，应停止后续训练并定位结构或转换问题。

## 7. 验收时要看什么

日志可用以下关键词快速定位：

```bash
rg -n 'number of parameters|loading release distributed checkpoint|successfully loaded checkpoint|Missing keys|unexpected|shape|Job .* succeeded' \
  /tmp/yulan_moe_128k_checkpoint_load_log_all.log
```

阶段 1 本次观察到：

- Slime actor 参数量约 `5,569,877,120`（5.57B）。
- checkpoint 类型为 `release` distributed checkpoint，路径指向上面的 `mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix`。
- 拓扑为 TP1/PP1/CP8/EP8。
- 日志报告 `successfully loaded checkpoint ... at iteration 0`。
- strict `log_all` 中唯一 mismatch 类别是 split GDN `in_proj._extra_state` metadata；无模型参数张量的 missing/unexpected 或 shape 错误。
- Ray job 最终为 succeeded。

日志路径：

```text
/tmp/yulan_moe_128k_checkpoint_load_log_all.log
```

也运行了 shell 语法检查：

```bash
bash -n scripts/models/yulan-moe-128k.sh \
  scripts/run-yulan-moe-128k-math-smoke.sh \
  scripts/run-yulan-moe-128k-math-dapo-full.sh
git diff --check
```

## 8. 本次变更文件和不要误判的事项

阶段 1 新建/更新的 repo 文件：

- `scripts/models/yulan-moe-128k.sh`：128k 专用 Megatron 模型参数。
- `scripts/run-yulan-moe-128k-math-smoke.sh`：128k smoke 和 checkpoint load-only 入口。
- `scripts/run-yulan-moe-128k-math-dapo-full.sh`：32k context 的单机 full RL 入口候选。
- `.ai/adapt_rl_for_yulan_moe_128k/plan.md`：全任务阶段计划和验收状态。
- `.ai/adapt_rl_for_yulan_moe_128k/stage1_model_structure_and_checkpoint_loading.md`：本文。

Base 的模型参数和 Base full launcher 保持原样。新 launcher 目前只是后续阶段的入口，不代表阶段 4/5/6 的真实 RL、单机生命周期和 32k 资源验收已经通过。

## 9. 后续接入新模型：需要收集哪些信源

每个新模型开始前建议建立一个模型资料目录，至少收集以下资料。尽量提供文件、命令快照、日志和对应 Git revision，而不只给一个模型目录路径。

### A. 模型权威定义

- HF checkpoint 完整路径和模型 revision/来源；如果是本地转换产物，说明转换人/命令/日期。
- `config.json`、tokenizer 文件、generation config、模型权重清单；记录文件是否齐全、是否有远程自定义代码。
- 模型结构说明：层数、hidden/intermediate、attention head/KV head、norm、position/rope、embedding/head 是否 tied、MoE 层 pattern、expert/top-k、共享专家、特殊 gate/clamp、GDN/SSM 参数。
- checkpoint hash、版本 tag 或 manifest；尤其要避免同一路径内容被原地替换却无法追溯。

### B. 实际训练和转换的依据

- 产出目标 checkpoint 的完整 SFT/Pretrain/MCore 命令（最好保留可执行脚本和训练时 args dump）。
- 命令对应代码仓库及 commit；用户给的命令快照若引用了另一个版本，必须把那个版本也记录下来。
- 训练日志中实际打印的关键 args、模型构造参数量、并行拓扑和 checkpoint 保存成功信息。
- HF→MCore 转换路径、转换脚本/配置、是否在 conversion 或训练阶段做了 key rename/shape transform。
- 使用的 dtype、norm/gate/router 语义、GDN split/fused 参数、attention backend、MoE dispatcher、special patches/custom kernels。

### C. 初始权重/checkpoint

- 路径、`latest_checkpointed_iteration.txt`、`release/metadata.json`、`common.pt`/args dump、分片清单。
- checkpoint format（例如 `torch_dist`）、Megatron-Core 版本/commit、是否有 optimizer/RNG/scheduler/dataloader state。
- 预期用途是继续训练恢复，还是只加载模型权重后重建 RL optimizer。
- 原 TP/PP/CP/EP/ETP 布局、checkpoint 能否 reshard、目标布局和 strictness 策略。
- 若 checkpoint 不是从对应 HF 权重直接训练得到，要给两者的转换/对应关系和已做的校验。

### D. Slime 与 serving 实现

- 目标 Slime branch/commit、可复用的 model provider、已有相近模型参数文件。
- 有无 custom model/provider、转换 hook、state dict alias、额外环境变量、patch、补丁适用的依赖版本。
- SGLang model plugin 的仓库/commit、HF serving 方式、已验证的 TP/EP layout、JSON config override、生成和 logprob 测试记录。
- 如果训练模块和 serving 模块分属不同 repo，提供双方版本组合和已验证的权重同步/导出链路。

### E. RL 目标和运行环境

- 明确 RL 目标 context，而非仅给模型 max position；prompt/response 预算也分别给出。
- 单机/多机 GPU 型号、卡数、显存、互联；目标 TP/PP/CP/EP 布局和 rollout/actor 资源生命周期。
- 算法、reward model/function、训练和评测数据字段、tokenizer/chat template、batch/samples、保存/评估频率。
- Python/PyTorch/CUDA/NCCL/Megatron/SGLang 和自定义 kernel 版本；关键环境变量和启动容器/venv。
- 可复现的 load log、短前向/反向/更新结果、固定输入的 actor-vs-serving logprob 对照结果，以及已知不兼容项。

### F. 新模型的最小开工包

如果希望快速开始，至少一次性提供：

1. HF 模型目录与 `config.json`/tokenizer。
2. 目标 MCore checkpoint 目录和产出它的完整 SFT/Pretrain 命令快照。
3. 对应训练代码 commit，以及如果有的话模型实现/转换代码路径。
4. SGLang plugin 路径和已验证 serving layout。
5. RL context、GPU 拓扑、数据/reward 目标。
6. 一份当前 Slime Base/full launcher 或明确允许参考的模型启动脚本。

拿到这些信源后，先列一张“HF config ↔ 训练命令 ↔ checkpoint args/keys ↔ Slime provider ↔ SGLang plugin”的对照表，把每个值标注来源和冲突，再新建独立模型参数文件并做 load-only smoke。不要在层 pattern、router/gate、norm、GDN split、vocab、bias 或 parallelism 有未解释冲突时直接启动正式 RL。
