# YuLan-MoE Math-RL 适配代码导读

## 1. 范围与基线

本文解释 slime 分支 `lzh/yulan-moe-gpu` 相对直接分叉基线 `c08c2cf` 的变更，HEAD 为 `fa988d3`。分支包含 4 个提交，整体差异约 2,088 行新增、39 行删除。配套 patch 为 `yulan_math_rl_branch.patch`，由 `git diff --binary --full-index c08c2cf..fa988d3` 导出。

核心适配在首个提交 `7b4dff1`；之后的提交主要整理文档/patch 文件，完善数学与代码评测、启动配置，最终修正启动脚本的 Ray 环境变量传递和 route-replay patch 应用方式。当前 HEAD 本身只改了两个启动脚本。

## 2. 先看端到端链路

一次 Math-RL 训练大致经过：

1. 从 JSONL 读取题目、标签，使用 tokenizer/chat template 构造 prompt。
2. SGLang 根据 prompt 生成一组回答，执行 DAPO 数学 reward。
3. 将回答整理成训练 batch；对 YuLan 的 GDN/MoE、packed sequence 和 context parallel 做布局适配。
4. Megatron actor 加载 MCore checkpoint 并执行 GRPO 更新。
5. 将更新后的 Megatron 权重映射回 HF/SGLang 可使用的参数布局，供下一轮 rollout 使用。
6. 按配置保存 checkpoint、运行 AIME 等评估，并记录指标。

新架构适配的关键不是改 GRPO，而是让模型构建、权重格式、训练 batch、推理服务和 MoE 路由元数据对同一模型定义保持一致。

## 3. YuLan 架构接入点

### 3.1 Megatron model provider 与层结构

[`slime_plugins/models/yulan_moe.py`](../slime_plugins/models/yulan_moe.py) 是模型 provider。启动参数通过 `--custom-model-provider-path slime_plugins.models.yulan_moe.model_provider` 加载它。

Provider 复用 Qwen3-Next 的 Megatron 配置和 layer spec，再按 YuLan 的混合结构组装 GDN、Attention、MLP/MoE 层：

- `_YuLanGDNLayer` 根据 `moe_layer_freq` 为当前层选 dense 或 MoE spec。
- `_get_yulan_stack_spec` 组合 GDN、full attention 和 MLP 的子模块。
- `model_provider` 用配置参数构造 `_YuLanMambaModel`。
- [`scripts/models/yulan-moe-base.sh`](../scripts/models/yulan-moe-base.sh) 指定层数、维度、GDN 频率、MoE expert 数、top-k、shared expert 等结构参数。层 pattern 和频率列表必须与目标 checkpoint 一致。

Slime 的通用 Megatron 训练接口会传 `packed_seq_params`、`loss_mask`、`padding_mask`。YuLan 的 `_YuLanMambaModel.forward` 接收这些参数，把 packed metadata 暂存到 decoder，再由 `_YuLanMambaStack.forward` 传给每个 TransformerLayer。`loss_mask` 由上层 loss 使用，因此模型本身不消费它。

还有两处 checkpoint/运行时兼容处理：

- MambaStack 默认注册 `final_norm`，而这个 checkpoint 期待 `final_layernorm`。adapter 在注册模块时匹配 checkpoint key，并保留 forward 所需的 `final_norm` 属性。
- `sharded_state_dict` 删除转换器不会提供的空 `output_layer._extra_state`，避免严格 checkpoint 对账时出现多余 state。

### 3.2 处理 block 级 GDN/CP 协议

默认路径由 `_YuLanMambaStack` 逐层执行。当启用 `SLIME_USE_YULAN_TRANSFORMER_BLOCK=1` 时，provider 改用 `_YuLanTransformerBlock`，让整个 TransformerBlock 处理 YuLan GDN 的 layout 转换协议。这个分支说明：遇到新架构时，要先辨别差异发生在单层算子还是整个 block 的执行/布局协议，再选择扩展点。

### 3.3 MCore 到 HF 的参数映射

[`slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py`](../slime/backends/megatron_utils/megatron_to_hf/qwen3_next.py) 负责 Megatron 权重转换到 HF/SGLang 参数格式。YuLan GDN 的 fused `in_proj.weight` 被拆成 query/key/value/z/beta/alpha，再按 HF 的 QKVZ 与 BA 分组布局重新拼接；`conv1d`、norm、out projection 也有相应映射。MoE expert 和 shared expert 权重则映射到 HF 的 gate/up/down projection 命名。

重要教训：**只检查参数名和 shape 不够**。不同实现可能有相同 shape，却采用不同的 fused projection 分块顺序。应从模型实现或权重转换代码确认每一块的语义、head 维度和排列顺序，否则模型能加载、生成也可能不报错，但 logits/训练轨迹是错的。

此外，启动脚本同时指定 HF checkpoint（服务端模型/config/tokenizer）与 MCore checkpoint（训练 actor 初始化权重）。做新架构时要分别验证这两条路径，并验证训练后权重能正确同步到 rollout 引擎。

## 4. Packed sequence、CP 与 MoE routing replay

### 4.1 YuLan THD CP batch

[`slime/backends/megatron_utils/data.py`](../slime/backends/megatron_utils/data.py) 在启用 `SLIME_USE_YULAN_THD_CP=1` 且 CP 大于 1 时，为每条样本单独 padding 到 `2 * CP size` 对齐，然后构造 `cu_seqlens`、`padding_mask` 和 `PackedSeqParams`，最后调用 Megatron 的 THD CP 分片 helper。

loss mask 对 padding 位置置零，padding mask 则标记这些物理填充 token。这样可以满足 kernel 的静态分片形状，同时避免填充 token 参与有效训练目标。该路径显式拒绝与 `allgather_cp` 同时使用，因为两者采用不同的分片布局。

### 4.2 MoE routing replay 对齐

Routing replay 记录 rollout 中 router 对 token 选择的 expert，以便训练时重放。YuLan THD CP 路径要求 replay 元数据使用和 token 完全相同的逐样本 padding 与 CP slice；对应逻辑在 [`cp_utils.py`](../slime/backends/megatron_utils/cp_utils.py)。actor 模型建好后还会遍历 `TopKRouter` 并注册 replay slots，见 [`actor.py`](../slime/backends/megatron_utils/actor.py)。

因此适配 MoE 新模型时，要检查：

- rollout 记录的路由数量是否对应每个预测 token；
- BOS/响应边界、padding 和 packed sequence 是否做了相同偏移；
- CP/TP 分片顺序是否和训练输入一致；
- actor 每个本地 router 是否都能找到 replay metadata。

如果这些维度错位，可能表现为 shape 错误，也可能静默地把 token 路由到错误 expert。

## 5. Math-RL 配置与训练生命周期

[`scripts/run-yulan-moe-base-math-smoke.sh`](../scripts/run-yulan-moe-base-math-smoke.sh) 是参数组装入口：选择 prompt/label key、chat template、DAPO reward、采样组、GRPO 算法参数、优化器、并行策略、SGLang engine 和 Megatron 模型 provider。它也是先跑小批次 smoke 的入口。

[`scripts/run-yulan-moe-math-dapo-full.sh`](../scripts/run-yulan-moe-math-dapo-full.sh) 在此基础上设置完整训练参数、AIME 评估、W&B、debug rollout 和 route replay。当前方案 actor 使用 CP=8、EP=8；SGLang 使用 TP=2/EP=2 的 engine 配置。这些是本模型/集群验证过的部署参数，不应直接照搬到其他架构或硬件。

训练与 rollout 同卡时，模型和 KV cache 会争抢显存。完整启动配置启用 colocate，并按生命周期销毁 rollout engine、释放/重建训练 actor；相应训练循环在 [`train.py`](../train.py)。新模型的 smoke 配方应该先验证一轮 rollout 与一次 actor update，再调整并行度和显存策略。

Reward 路由在 [`slime/rollout/rm_hub/__init__.py`](../slime/rollout/rm_hub/__init__.py)，DAPO 细节在 [`math_dapo_utils.py`](../slime/rollout/rm_hub/math_dapo_utils.py)。这部分处理 boxed 答案、数学等价形式、选择题标签等数据约定。新增的 code reward/代码评测支持属于分支中的通用评测增强，不是 YuLan 架构接入的必要条件。

## 6. 当前 HEAD 的两个启动修复

`fa988d3` 做了两项运行时修复：

1. Ray runtime environment 不再只传两个 YuLan 开关，而是从预设白名单中收集需要的 `PYTHONPATH`、动态库路径、CUDA/SGLang 配置和架构开关，再用 JSON 序列化，避免 Ray worker 环境与 launcher 环境不一致。
2. Full launcher 通过 `git apply --reverse --check` 判断 route-replay patch 是否已应用；若未应用且目标文件没有其他改动，则临时应用，并通过 `EXIT` trap 撤销。若检测到目标文件存在不相关修改，则停止，避免覆盖。

这类修复属于实验可复现性和运行环境管理，不改变模型结构或 reward 算法。

## 7. 将这套方法迁移到另一个新架构

建议按依赖顺序推进，每一步都固定一个小的验收目标：

1. **描述结构**：列出每层类型、维度、norm、RoPE/位置编码、MoE 频率、router top-k 和 fused weight 排列。
2. **打通模型构建**：实现或复用 Megatron model provider，先核对模型层数、层型 pattern 和构造参数。
3. **对齐 checkpoint**：实现需要的 state-dict key/weight 转换，逐项核验映射和 tensor 排列；检查加载时缺失、多余 key。
4. **对齐 Slime forward ABI**：确认模型是否接受 packed sequence、loss/padding mask、多模态输入等通用参数，并把元数据传给真正消费它的层。
5. **打通 serving 和 weight sync**：确认 HF/SGLang 能加载并生成；验证 Megatron 更新权重后转换结果与服务端结构一致。
6. **对齐 batch 和并行布局**：实现 CP/TP/EP 对应的分片与 padding；MoE 还需对齐 rollout/training routing replay。
7. **再接 Math-RL**：选定数据 schema、prompt template、答案格式和 reward；确定 GRPO/DAPO、采样组大小、优化器、评估集和保存策略。
8. **逐级扩大规模**：先做生成 smoke，再做单轮 rollout 与单次更新，检查 reward 和 debug rollout，最后扩展到完整训练与周期评估。

推荐的定位顺序是：生成不对先查 serving/config/tokenizer；actor 加载失败先查 model provider 与 checkpoint key；权重更新后生成异常查 MCore→HF 转换；loss/梯度异常查 loss mask 与 packed/CP 布局；MoE replay 错位查 router metadata；reward 不对查 label schema 和答案抽取。

## 8. Patch 应用方式

Patch 是从基线到分支 HEAD 的完整 Git diff，适合在 `c08c2cf` 基线上应用：

```bash
git apply --check .ai/yulan_math_rl_branch.patch
git apply .ai/yulan_math_rl_branch.patch
```

patch 内包含该分支所有受版本控制的改动，不包含当前工作区未提交的文件。若目标 checkout 已经包含这些提交，不要重复应用；可在对应基线上用 `git apply --reverse --check` 检查反向可应用性。

## sglang_qwen3_next_plugin：Rollout 引擎里的推理代码

调用链大致是：

train.py
  └─ 创建 RolloutManager
       └─ start_rollout_servers
            └─ 启动 SGLangEngine / SGLang server 子进程
                 └─ Python 启动时加载 sitecustomize
                      └─ 将 sglang.srt.models.qwen3_next 指向插件实现
                           └─ 插件按 SGLang 版本加载模型代码

脚本通过 SGLANG_EXTERNAL_MODEL_PACKAGE 开关插件，并把变量传入 Ray job 环境；SGLang server 是由 RolloutManager 启动的。设置插件开关 (scripts/run-yulan-moe-base-math-smoke.sh:30) 传入 Ray 环境 (scripts/run-yulan-moe-base-math-smoke.sh:234) 创建 rollout server (slime/backends/sglang_utils/deployment.py:80) 启动 SGLang 子进程 (slime/backends/sglang_utils/sglang_engine.py:140)

本环境安装的 SGLang 是 0.5.9。这个版本下，插件的入口模块根据安装版本选择 sglang_qwen3_next_plugin.variants.sglang_0_5_9，并执行对应的兼容补丁。当前环境的 sitecustomize.py 会把插件模块注册为 SGLang 的 qwen3_next 模块。Python 启动钩子 (/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/sitecustomize.py:48) 版本分发入口 (/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/sglang_qwen3_next_plugin/qwen3_next.py:9) 0.5.9 兼容补丁 (/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin/sglang_qwen3_next_plugin/variants/sglang_0_5_9.py:68)

因此，第一次 rollout 时，SGLang 用 HF checkpoint 加载模型；之后每轮销毁并重建引擎时，同一条插件代码路径会从该轮生成的 HF 快照重新加载。插件负责的是 SGLang 里的 Qwen3-Next 模型实现。

## YuLan-Pretrain：Megatron actor 里的训练代码

训练代码链是：

train.py 创建训练 actor
  └─ MegatronTrainRayActor.init
       └─ initialize_model_and_optimizer
            └─ get_model_provider_func
                 └─ 加载 slime_plugins.models.yulan_moe.model_provider
                      ├─ 导入 YuLan 的 megatron_patch.model.qwen3_next
                      └─ 构造模型，使用 YuLan Megatron fork 中的 megatron.core
                           └─ Megatron forward / backward / optimizer

这里要区分：model_provider 文件本身在 slime 仓库，即 slime_plugins/models/yulan_moe.py (slime_plugins/models/yulan_moe.py:22)；但启动脚本把 YuLan-Pretrain 根目录放在 PYTHONPATH 最前面，因此它导入的 megatron_patch... 和 megatron.core... 来自 YuLan 仓库。PYTHONPATH 顺序 (scripts/run-yulan-moe-base-math-smoke.sh:32) Slime 加载自定义 provider (slime/backends/megatron_utils/model_provider.py:96)

本脚本还设置了 SLIME_USE_YULAN_TRANSFORMER_BLOCK=1。在 provider 中，这个条件会选择 _YuLanTransformerBlock 适配器；它继承当前导入的 megatron.core.transformer.TransformerBlock。由于 PYTHONPATH 优先级，实际基类来自 YuLan-Pretrain。训练 forward 调用到该基类的实现时，会执行 YuLan 的 GDN 与 Context Parallel 布局逻辑。适配器选择 (slime_plugins/models/yulan_moe.py:255) YuLan TransformerBlock 的 GDN CP 路径 (/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/megatron/core/transformer/transformer_block.py:1412)

输入数据准备则是 Slime 的代码：data.py 检查 SLIME_USE_YULAN_THD_CP=1，按 YuLan 的 packed THD 方式整理数据，然后调用 megatron.core.utils.get_thd_batch_on_this_cp_rank。这个 megatron.core 同样会从 YuLan-Pretrain 导入。Slime 的 THD CP 数据分支 (slime/backends/megatron_utils/data.py:57)

路由 replay 还有一条直接连接：启动脚本会把 route_replay_yulan_pretrain.patch 应用到 YuLan 的 router.py 和 moe_utils.py；训练 actor 构造 MoE router 时，这些改动会调用 Slime 的 routing_replay 代码。补丁文件 (patches/route_replay_yulan_pretrain.patch) 训练 actor 注册 replay (slime/backends/megatron_utils/actor.py:102)

并行参数是 Slime 的启动脚本传的，TP/PP/CP/EP 的底层并行机制主要由 YuLan-Pretrain 提供的 Megatron 实现。 Slime 负责启动 Ray actor、传参数、准备数据并调用训练；它不是另写了一套 TP/EP/PP 通信后端。

这个脚本的配置是 TP=1、PP=1、CP=8、EP=8、ETP=1，并指定了 alltoall MoE dispatcher。并行参数 (scripts/run-yulan-moe-base-math-smoke.sh:103) 所以这次实际有 CP 和 EP 的并行通信；TP、PP 配置虽然存在，但大小为 1，没有跨 rank 切分。

底层调用顺序是：Slime 的训练 actor 初始化时调用 Megatron 的并行初始化；这段代码调用 megatron.core.mpu.initialize_model_parallel(...)。Slime 的初始化入口 (slime/backends/megatron_utils/initialize.py:33) 启动脚本把 YuLan-Pretrain 放在 PYTHONPATH 最前面，所以这里导入的 megatron.core 实际来自 YuLan-Pretrain 的代码。PYTHONPATH 顺序 (scripts/run-yulan-moe-base-math-smoke.sh:32)

各部分可以这样看：

| 并行部分 | 这个脚本里的情况 | 代码归属 |
|---|---|---|
| TP | size=1，无张量切分 | Slime 传参数；底层 Megatron API/模型组件来自 YuLan 的 megatron.core |
| PP | size=1，无流水线切分 | 同上 |
| CP | size=8，实际启用 | Megatron 的 CP 进程组和 YuLan 的 GDN CP forward 在 YuLan 代码中；Slime 另有 THD batch 整理和 routing replay 对齐逻辑 |
| EP | size=8，实际启用 | Megatron 的 EP 进程组、MoE 路由和 dispatch 走 YuLan 的 Megatron fork；Slime 负责 actor/rank 启动及 rollout 路由信息处理 |

所以，说“CP 是 YuLan 的”只说了一半：CP 的并行组和模型 forward 实现来自 YuLan；而训练数据如何按这条 CP 路径打包、切分，以及 routing replay 如何对齐，是 Slime 自己适配的。Slime 的 THD CP 数据分支 (slime/backends/megatron_utils/data.py:54) YuLan TransformerBlock 中的 GDN CP 路径 (/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/megatron/core/transformer/transformer_block.py:1412)

最简化地说：并行度由 Slime 配置，Megatron 并行机制由 YuLan-Pretrain 实现，Slime 补上 RL 训练所需的数据和调度适配