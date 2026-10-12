# YuLan-MoE-128k RL 接入计划

## 目标

在 Slime 中完成 `/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k` 的 RL 训练闭环：从 MCore checkpoint 加载模型，执行 rollout、reward 和 actor 更新，正确同步更新后的权重，并在单机上按需销毁和重建 rollout engines/train actors，最终支持 32k context 下的稳定训练。

工作分阶段推进。每阶段先满足验收条件，再扩大输入长度、训练规模或运行时复杂度。

## 已知基线与关键待定项

- 当前 Slime 全量脚本：[`scripts/run-yulan-moe-math-dapo-full.sh`](../../scripts/run-yulan-moe-math-dapo-full.sh)。它以 YuLan-MoE-Base 为目标，并通过 rollout manager、`--destroy-rollout-engines` 和 `--rebuild-train-actors` 在单机上切换 rollout/actor 资源。
- Base 与 128k 结构相近但有差异。保留 [`scripts/models/yulan-moe-base.sh`](../../scripts/models/yulan-moe-base.sh) 和 Base 启动入口不变，为 128k 新建独立模型参数文件及 smoke/full 启动入口；共享现有 provider 时也要通过 128k checkpoint 加载验收。
- 128k HF 模型：`/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k`。
- 推荐优先验证并复用的 MCore 初始权重：`/mnt/yulan/lvzhihao/PostTrain/models/YuLan-MoE-128k/mcore-tp1pp1ep8-alltoall-smoke-5steps-qgbiasfix`。目录包含 `latest_checkpointed_iteration.txt`、`release/` torch_dist checkpoint 和 HF assets；SFT 参数快照显示保存格式为 `torch_dist`。
- SFT 启动参数快照：`/mnt/yulan/lvzhihao/PostTrain/YuLan-Pretrain/outputs/yulan_moe_128k_alltoall_smoke_5steps_qgbiasfix/pretrain_gpt_command.sh`。参数源应与 HF config 和 MCore checkpoint 一起核对。
- SGLang 插件：`/mnt/yulan/lvzhihao/PostTrain/sglang_qwen3_next_plugin`。其 README 记录了 128k 模型在 TP2/EP2 下的加载、生成和 logprob 验收；Slime 侧还要验证同步过来的训练权重。
- SFT 命令的 `--seq-length` 是 `49152`，`--max-position-embeddings` 是 `131072`。这两个值含义不同；本次 RL 的总 context 已确定为 `32768`，不能仅因模型名含 128k 就把训练长度设为 131072 token。
- 本任务的单机全量 RL context 目标确定为 `32768`；模型仍保留 `max_position_embeddings=131072`，但本次全量训练不以 128k context 为目标。现有全量脚本的 context 上限也是 `32768`，response 上限 `16384`、`MAX_TOKENS_PER_GPU=1024` 等值仍需在 32k 目标下做资源验收。
- Slime 在没有单独 `--load` 时会把 `--ref-load` 作为初始训练 checkpoint；该参数可用于这份 MCore checkpoint。但需要用 128k provider/参数验证 `torch_dist` checkpoint 可加载且模型语义一致。该 checkpoint 用作初始权重，不代表能恢复训练 optimizer 状态。

## 阶段与验收


### 1. 对齐 128k 模型结构与 checkpoint 加载

- [x] 新建独立的 `scripts/models/yulan-moe-128k.sh`，复用现有 YuLan provider；Base 参数文件和 Base 启动脚本保持不变。
- [x] 新建独立的 128k smoke/full 启动入口，明确设置 HF checkpoint、MCore `--ref-load`、模型参数和 32k RL context。
- [x] 对照 HF config、SFT 命令和 MCore checkpoint 核对层数、层 pattern、维度、GDN/MoE 参数、vocab、QKV bias 和 router 参数。
- [x] 配置并验证 128k 特有语义：attention output gate、gated norm rank 16、activation soft clamp 7、final logits clamp 30、router sqrt gate。
- [x] 在 8 卡 TP1/PP1/CP8/EP8 下构建 Slime actor，并从 `release` distributed checkpoint 成功加载；无参数张量 shape 错误。
- [ ] 在 actor 前向中验证 packed THD、cu_seqlens 和 GDN CP 路径；这项放到阶段 2 的短序列 actor 验证。

**验收：**Slime 能构建模型并加载指定 MCore checkpoint；参数张量没有 shape mismatch，参数量和关键模块符合预期。split GDN 的 `_extra_state` metadata 差异单独记录并确认由模型加载钩子兼容。

**关于 MCore checkpoint 复用：**可以将它作为 Slime 的初始权重候选。其 `torch_dist` 格式、`release/` 和 latest iteration 标记与 Slime 的 Megatron checkpoint 加载路径相符；实际兼容性以阶段 1 的 Slime 加载验收为准。由于 SFT 命令使用了 `--no-save-optim`，RL 从模型权重开始新的 optimizer 状态，不是从 SFT optimizer 状态续训。

**阶段 1 已验证：**执行 `CHECKPOINT_LOAD_ONLY=1 bash scripts/run-yulan-moe-128k-math-smoke.sh` 后，Slime 日志报告从该路径成功加载 `release` checkpoint，TP1/PP1/CP8/EP8，iteration 0。`log_all` 检查只报告 split GDN 层 `self_attention.in_proj._extra_state` metadata key 不匹配；没有参数张量的 missing/unexpected 或 shape 错误，任务成功。`raise_all` 会因这组 metadata key 提前失败；YuLan 的 GDN split 投影加载钩子会为缺失的空 Transformer Engine extra state 恢复默认值。模型构建约 5.57B 参数。独立 128k smoke 必须设置 `SLIME_USE_YULAN_TRANSFORMER_BLOCK=1`：checkpoint 的 final norm 是 GatedNorm（`final_layernorm.norm.*` 和 gate 参数）；走旧 MambaStack 时会错误地期待普通 `final_layernorm.weight`。

### 2. 验证 Slime actor 的前向、反向和保存

- [ ] 用短序列和固定样本执行 actor 前向、loss、反向及一次 optimizer update。
- [ ] 检查 loss、梯度和更新后参数均为有限值；确认不是只完成初始化而没有实际更新。
- [ ] 保存更新后的 MCore checkpoint，并验证可重新加载。
- [ ] 验证 HF 权重导出/Slime 权重转换路径覆盖 128k 模型特有模块。

**验收：**actor 完成一次真实更新；更新前后 checkpoint 可保存、重载，关键参数没有丢失。

### 3. 验证 Slime 到 SGLang 的权重和 logprob 一致性

- [ ] 在 Slime 训练环境中确认 `SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin` 生效，模型类解析到目标插件实现。
- [ ] 按插件已记录的 TP2/EP2 serving layout 启动服务，先用原始 HF checkpoint 做短 prompt 生成。
- [ ] 加载/同步 Slime 导出的 actor 权重；对固定 prompt 比较 actor 与 SGLang 的 logprob，定位配置、转换或模块语义不一致。
- [ ] 检查权重更新后 SGLang 确实拿到新版本，而非仍在使用初始权重。

**验收：**原始 checkpoint 和 actor 导出权重均可由 SGLang 加载；固定输入上的 logprob 差异在约定容差内，权重更新可见。

### 4. 短序列 RL 闭环 smoke

- [ ] 使用少量可判分样本、短 prompt/response 和小 batch，执行少量 rollout 与训练 step。
- [ ] 覆盖 tokenizer/chat template、reward、dynamic sampling/filter、GRPO batch 构造、actor update、保存和 rollout 权重同步。
- [ ] 初次排障时可使用已保存 rollout replay 隔离 actor 问题；之后仍需至少一次真实 SGLang rollout。
- [ ] 暂不启用长上下文压力、长周期评估和正式训练规模。

**验收：**一轮完整 RL 闭环成功，reward 和训练指标合理，actor 确有更新，下一轮 rollout 使用更新后的权重。

### 5. 单机 rollout/actor 生命周期验收

- [ ] 在 8 卡单机上接入现有 rollout manager 与 colocate 配置。
- [ ] 核实 route-replay 补丁是否适用于目标 checkpoint 和 YuLan-Pretrain 代码版本；沿用现有 launcher 的安全检查/恢复机制，不覆盖无关本地改动。
- [ ] 验证 rollout engines 销毁、train actors 重建、actor 更新/保存、rollout engines 再创建的完整顺序。
- [ ] 检查 Ray runtime env 传播、插件导入路径、checkpoint 路径、显存释放和重建后的权重版本。
- [ ] 连续跑多轮，观察显存是否泄漏、GPU 是否正确分配、训练是否能继续。

**验收：**至少连续完成多个 rollout/train 周期；没有 GPU 资源死锁、持续显存增长或 rollout 使用旧权重的问题。

### 6. 验证 32k context 与 token budget

- [ ] 从短序列逐级升到约 8k、16k、32k；每一级都记录峰值显存、吞吐、失败位置和可恢复性。
- [ ] 在 32k 总 context 下验证 SGLang KV cache、MCore CP/GDN、packing、rollout 时长和单机资源生命周期；本次不要求 49k 或 128k RL。
- [ ] 根据实测结果调整 `rollout-max-prompt-len`、`rollout-max-response-len`、`rollout-max-context-len`、`MAX_TOKENS_PER_GPU`、batch 和 samples 数。

**验收：**32k context 下 rollout、actor 更新、权重同步和 checkpoint 保存都能完成；显存与吞吐数据支持目标训练规模。

### 7. 正式训练配置与运行

- [ ] 将通过验收的模型参数、checkpoint、数据、reward、评估和 W&B 设置组合成独立的 128k RL launcher/config。
- [ ] 验证断点恢复、定期 checkpoint、HF 导出、评估结果和日志可追踪。
- [ ] 先执行短周期正式配置复核，再启动目标训练轮数。

**验收：**正式 launcher 可从目标 checkpoint 启动并恢复；评估、保存、日志和 rollout/actor 生命周期均持续正常。

## 当前建议的执行顺序

先完成阶段 1 的独立 128k 配置和 checkpoint 加载验收；阶段 0 的 RL context 已定为 32k。之后按 2 → 3 → 4 → 5 → 6 → 7 推进。SGLang 插件已有的模型支持视为可复用基础，但不能替代 Slime 导出权重、logprob 和端到端 RL 的验收。
