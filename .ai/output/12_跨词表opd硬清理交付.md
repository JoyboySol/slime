# 12 跨词表 OPD 硬清理交付

## 背景

这一轮工作的目标，不是继续给跨词表 OPD 叠加更多兼容分支，而是把主链路收敛成一个长期可维护、协议清晰、训练与诊断一致的实现。

约束也很明确：

- 主训练链路不接受“出了问题再 fallback 到另一套逻辑”的设计。
- `generation token text` 只能保留为观测信息，不能再伪装成强证据。
- 训练、replay、诊断脚本要尽量复用同一套 byte-chunk 准备逻辑。
- 需要能在“不真正跑训练”的前提下，验证 rollout 样本是否能够正确进入训练前的 OPD byte-chunk 路径。

## 本轮已完成

### 1. 先做了基线备份提交

已提交：

- `6999050` `backup: snapshot cross-tokenizer opd baseline before hard cleanup`

这个提交的意义是先把当时可运行但存在兼容分支和链路混杂的问题状态完整冻结，方便后续做硬清理时有回退参考，也方便对照本轮行为变化。

### 2. 完成了主线硬清理提交

已提交：

- `33ee694` `refactor: harden cross-tokenizer opd mainline`

这一提交是本轮核心交付，重点不是“再支持更多 case”，而是把生产路径收束为单一协议。

### 3. 收紧了跨词表 OPD 的生产协议

现在的主线规则可以概括为：

1. `generation token text` 仍然会记录，但它只用于 observability。
2. 训练侧 student 对齐只接受两类来源：
   - 已记录且校验通过的 alignment payload
   - 基于 `token ids + canonical text + tokenizer token pieces` 的重建结果
3. 不再把 `generation_logprobs_text` 当成生产级 student byte evidence。
4. byte-chunk 主路径出错时，默认直接报错，不再走 sequence fallback 来“尽量训下去”。

换句话说，这一轮的核心不是让更多异常样本静默通过，而是让错误暴露得更早、更可定位。

### 4. 提取了“进入训练前”的独立校验入口

新增了：

- `slime/utils/opd_utils.py` 中的 `prepare_byte_chunk_training_entry(...)`

这个 helper 的职责是把训练真正依赖的前置步骤抽出来：

- 根据 `full_text / prompt_text / student_token_ids / response_token_count` 确定 prompt-response 边界
- 基于 strict byte-chunk 规则准备 student/teacher chunk log probs
- 返回训练前真正需要的结构化结果，例如：
  - `student_chunk_log_probs`
  - `teacher_chunk_log_probs`
  - `reverse_kl`

这意味着现在已经可以在不启动完整训练、不走优化器 step 的情况下，验证一条 rollout 样本是否能正确进入 byte-chunk OPD 训练入口。

### 5. 让 replay 诊断脚本复用训练入口逻辑

`scripts/replay_debug_rollout_opd.py` 不再自己维护一套平行的“近似训练逻辑”，而是复用上面抽出来的 `prepare_byte_chunk_training_entry(...)`。

这件事很重要，因为它减少了两类长期风险：

- 训练和 replay 看起来都“对”，但实际上走的是两套不同语义
- 后续修一处逻辑时，另一处脚本没同步，导致诊断结论和真实训练行为脱节

### 6. 补上了面向训练入口的测试

新增了：

- `tests/test_cross_tokenizer_opd_training_entry.py`

这个测试覆盖的是“给一条样本，验证它能不能正确进入 strict byte-chunk 训练前置路径”，它的定位不是 full e2e 训练测试，而是一个更高层的 pre-training entry validator。

这已经回答了之前那个关键问题：

> 是否能把完整链路从训练中抽出来，避免真正训练，只验证是否能够正确进入训练？

答案是：可以，而且这一轮已经先把最关键的那一段抽出来了。

不过它目前还是“训练入口校验 helper + 单测”的形态，还不是一个独立、标准化的 CLI / CI smoke target，这一点在“待完成事项”里还会再说。

## 关键行为变化

### 1. rollout 侧

- 仍记录 generation 相关 observability 信息。
- 对 `generation_byte_evidence` 做状态统计，但不再把它提升为训练强依赖证据。

### 2. training 侧

- `recorded_builder` 成为主路径。
- student 对齐更加依赖已记录 payload 或 token-id/token-piece 重建。
- 主 byte-chunk 对齐失败会显式报错，而不是默默切换到 sequence fallback。

### 3. diagnosis / replay 侧

- replay 现在尽量贴近真实训练入口语义。
- 诊断输出仍会展示 generation evidence 命中率、invalid rate 等观测指标。
- 但这些指标表达的是“观测质量”，不再表达“生产 fallback 是否可用”。

## 本轮验证证据

本轮已经跑过一组与跨词表 OPD 主线直接相关的回归验证。

### 1. pytest

执行过：

```bash
./.venv/bin/pytest \
  tests/test_rollout_logging.py \
  tests/test_rollout_logging_non_scalar_rewards.py \
  tests/test_analyze_opd_alignment.py \
  tests/test_opd_byte_chunk.py \
  tests/test_student_logprob_alignment.py \
  tests/test_cross_tokenizer_opd_training_entry.py \
  tests/test_real_tokenizer_sampled_opd_alignment.py \
  -v
```

结果：

- `79 passed`

### 2. 语法编译检查

执行过：

```bash
./.venv/bin/python -m py_compile \
  slime/utils/opd_utils.py \
  slime/rollout/on_policy_distillation.py \
  slime/rollout/sglang_rollout.py \
  scripts/replay_debug_rollout_opd.py \
  scripts/analyze_opd_alignment.py
```

结果：

- 通过

## 与最初目标的对应关系

### 1. “先做一个 git commit 提交一下”

已完成。

而且不是只做了一个提交，而是分成了：

- 基线备份提交 `6999050`
- 硬清理主提交 `33ee694`

这样后续继续整理时，历史边界会更清楚。

### 2. “为了最长期维护，不接受任何 fallback”

主链路层面，这个方向已经落实了。

尤其是在 byte-chunk 生产路径上，当前实现已经明显偏向：

- 严格校验
- 明确失败
- 不用 sequence fallback 掩盖协议问题

不过要严格说“全仓库所有相关测试、脚本、历史工具链都完全没有 fallback 影子”，现在还不能这么下结论，原因见下文待完成事项。

### 3. “把不干净的代码链路清理干净”

已经做了一轮有实质效果的清理：

- 训练与 replay 的 byte-chunk 入口开始复用
- generation text 被降级为 observability-only
- student 对齐证据来源更清晰

但“链路彻底干净”还没有完全收尾，主要剩在测试组织与部分历史文档/脚本层面。

### 4. “是否能抽出高级测试，只验能否正确进入训练”

已部分完成，而且是高价值的一步。

当前状态是：

- 已经抽出了 `prepare_byte_chunk_training_entry(...)`
- 已经有了对应测试 `tests/test_cross_tokenizer_opd_training_entry.py`

尚未完成的是：

- 还没有把它进一步包成统一的 CLI / 脚本化 smoke check
- 还没有把它纳入更明确的 CI 或常规回归入口

## 仍待完成的任务

下面这些是我认为下一阶段最值得继续做的事情，也是这次交付里最需要诚实保留的未完成项。

### 1. 拆分 `tests/test_opd_byte_chunk.py`

这仍然是当前最明显的不干净点之一。

这个文件现在承担了太多职责，混杂了：

- byte-chunk 对齐基础行为
- recorded payload / builder 行为
- replay summary 行为
- 部分训练接入相关断言

建议拆成若干更稳定的测试文件，例如：

- `tests/test_opd_byte_chunk_alignment_core.py`
- `tests/test_opd_recorded_alignment_protocol.py`
- `tests/test_opd_replay_summary.py`
- `tests/test_opd_training_bridge.py`

如果这一步不做，后续继续维护时，测试会越来越难读，也会更难判断某个失败到底是在测哪一层。

### 2. 继续整理 launcher / smoke 脚本

这轮已经补了一部分 launcher 覆盖和协议收紧，但脚本层面的可读性、职责切分、统一入口命名，仍然有继续整理空间。

更理想的状态应该是：

- “真正训练”
- “只验证 rollout 样本能否进入训练入口”
- “复用 serve 做最小冒烟”

这三类入口各自有明确脚本，不混写在一份大脚本里。

### 3. 把训练入口校验 helper 再向前推进成标准工具

现在已经有：

- 训练入口 helper
- 对应单测
- replay 脚本复用

但还缺：

- 一个用户可直接运行的标准 CLI
- 更清晰的输入约定，例如直接吃 `debug_rollout_data`
- 更明确的失败分类输出
- 更适合挂到 CI / 回归链路里的 exit code 语义

这会直接决定它能不能成为“最高级但不真正训练”的标准验证工具。

### 4. 历史工作记录还没有全部回写同步

当前 `.ai/output` 里已有的历史记录，至少这些还可以考虑同步更新：

- `08_跨词表opd重构.md`
- `09_跨词表opd经验总结.md`
- `11_YuLan_tokenizer行为分析.md`

原因不是“为了好看”，而是为了避免后面回看记录时，把旧结论误当成当前协议。

### 5. 进一步确认是否要继续清理兼容分支

从主线协议上看，这一轮已经完成了最关键的 no-fallback 收敛。

但如果目标是“仓库里与跨词表 OPD 有关的所有分支都不再保留旧兼容语义”，那还需要再做一次全局扫描，重点关注：

- 历史诊断脚本
- 低频调用路径
- 旧测试中的 compatibility wording

这一步我认为是下一轮值得专门做的一次“扫尾型 cleanup”。

## 当前结论

如果只看主训练链路，这一轮已经把跨词表 OPD 明显推进到了一个更适合长期维护的状态：

- 协议更单一
- 失败更显式
- 训练与 replay 更一致
- 也已经具备“不真正训练，只验证能否进入训练入口”的基础能力

但如果标准提高到“整个相关工具链已经完全整理干净”，那现在还不能算全部收尾，最主要的剩余工作就是：

- 测试文件继续拆分
- 训练入口校验工具进一步产品化
- 历史记录与脚本入口继续收口

## 相关提交

- `6999050` `backup: snapshot cross-tokenizer opd baseline before hard cleanup`
- `33ee694` `refactor: harden cross-tokenizer opd mainline`

## 额外说明

当前工作树里还存在与本轮交付无关的其他变更痕迹，我没有擅自处理它们，避免把这次跨词表 OPD 清理与别的工作混在一起。

## 2026-04-21 追加验证

这轮又补了一次“真实主链路 + 真实 rollout 重放”的双重确认。

### 1. 修掉了 replay 脚本的一个假失败来源

问题不在训练主链路本身，而在：

- `scripts/replay_debug_rollout_opd.py`

之前这个脚本在没有显式传入 `opd_alignment` 时，会让

- `compute_teacher_log_probs_for_sample(...)`

退回默认的 `token` 语义，直接按 student `response_length` 裁 teacher logprobs。

但脚本后半段又继续按 `byte_chunk` 规则重建 teacher response token 区间，所以会制造出这种“看起来像训练坏了”的假失败：

- `teacher_selected_response_count=122`
- `teacher_log_prob_count=128`

现在已经把 replay 默认语义固定为：

- `opd_alignment=byte_chunk`

因此 replay 与训练侧 teacher logprob 抽取路径保持一致。

### 2. 新一轮真实严格 smoke 已重新跑通 5 step

本轮真实训练验证时间：

- `2026-04-21`

工作目录：

- 实际目录：`/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current`
- 本地软链：`/mnt/ssd/lvzhihao/PostTrain/slime/workdirs/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current`

关键日志：

- `run.log`: `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/logs/run.log`
- `teacher.log`: `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/logs/teacher.log`

训练产出的 rollout dump：

- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/debug_rollouts/rollout_0.pt`
- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/debug_rollouts/rollout_1.pt`
- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/debug_rollouts/rollout_2.pt`
- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/debug_rollouts/rollout_3.pt`
- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/debug_rollouts/rollout_4.pt`

本次 smoke 使用的是严格配置：

- `OPD_DISABLE_SEQUENCE_FALLBACK=true`
- `NUM_ROLLOUT=5`
- 单卡 train + 单卡 rollout + 单卡 teacher

`run.log` 中 `rollout/train opd summary 0..4` 全部为：

- `status={'ok_recorded': 1}`
- `source={'recorded_builder': 1}`
- `evidence_kind={'recorded_builder_token_ids': 1}`
- `validated={True: 1}`
- `top_errors=[]`

### 3. 新产出的 5 个 rollout 也已经全部严格 replay 成功

人工审阅汇总文件：

- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step5_strict_v2_current/debug_rollouts/latest_replay_summary.jsonl`

该文件里 5 个样本全部为：

- `byte_chunk_alignment="ok"`

也就是说，到这一轮为止，可以同时确认：

- 训练主链路已经能稳定进入并跑过 5 个严格 step
- rollout 侧 recorded student alignment 是完整且可验证的
- teacher 侧 logprob 抽取与 replay 语义已经一致
- 这 5 个真实样本在训练后 replay 时没有出现 teacher/student byte-chunk 对齐错误

### 4. 当前还剩下的更小尾巴

主问题已经收敛，但还有一个“脚本可维护性”层面的尾巴值得下一轮处理：

- `run-yulan-cross-tokenizer-opd-train.sh` 里 `WANDB_API_KEY="${WANDB_API_KEY:-...}"` 会让“显式传空字符串”也回退到默认 key

这不影响 OPD 严格对齐本身，但会让“想彻底关掉 wandb”的行为不够直观，后续可以单独清一下。

## 2026-04-21 追加验证（二）

这一轮又补了一次比前面更强的真实 smoke，目的不是只看“能不能跑起来”，而是确认跨词表 OPD 主链路在更长一点的真实训练过程中，是否持续保持无对齐错误。

### 1. 新的更长 smoke 已稳定跑过 8 个 rollout/train step

本次工作目录：

- 实际目录：`/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_step8_longer_v1`
- 本地软链：`/mnt/ssd/lvzhihao/PostTrain/slime/workdirs/slime_opd_train_workdir_cross_tokenizer_step8_longer_v1`

关键日志：

- `run.log`：`/mnt/ssd/lvzhihao/PostTrain/slime/workdirs/slime_opd_train_workdir_cross_tokenizer_step8_longer_v1/logs/run.log`
- `latest_replay_summary.jsonl`：`/mnt/ssd/lvzhihao/PostTrain/slime/workdirs/slime_opd_train_workdir_cross_tokenizer_step8_longer_v1/debug_rollouts/latest_replay_summary.jsonl`

`run.log` 中从 `rollout/train opd summary 0` 一直到 `rollout/train opd summary 7` 都是同一组干净统计：

- `status={'ok_recorded': 4}`
- `complete={True: 4}`
- `source={'recorded_builder': 4}`
- `evidence_kind={'recorded_builder_token_ids': 4}`
- `validated={True: 4}`
- `top_errors=[]`

最后日志也正常结束在：

- `[driver] train(args) done`

这说明这次不是“碰巧前几个 step 没炸”，而是连续 8 个 rollout/train 周期都维持了 recorded payload 主路径的严格对齐。

### 2. 对应的 replay 汇总也已经是 32/32 样本全绿

这次更长 smoke 一共产出了：

- 8 个 rollout dump
- 每个 rollout 4 个样本
- 合计 32 个真实样本

对 `latest_replay_summary.jsonl` 做过滤后，可以确认：

- `sample_rows=32`
- `ok_rows=32`
- `non_ok_values=[]`
- `max_index=31`

也就是说，这 32 个真实样本在训练后 replay 时，全部都是：

- `byte_chunk_alignment="ok"`

这里没有再出现 teacher/student byte-chunk 对齐失败样本。

### 3. 这次验证覆盖了更长 response，不只是很短样本

这一轮 replay 汇总里，32 个样本的：

- `response_length=512`

同时 teacher 侧被选中的 response token 数量虽然小于 student `response_length`，但它在 byte-chunk 语义下是自洽的，而且所有样本都满足：

- `teacher_selected_response_count == teacher_log_prob_count`

实际观测到的 teacher 计数覆盖区间是：

- 最小 `424`
- 最大 `500`

这说明当前主链路已经不只是“短样本不报错”，而是在 `response_length=512` 这一档真实输出长度下，仍然能够稳定完成 byte-chunk 对齐与 teacher logprob 抽取。

### 4. 当前“看起来慢”的部分主要不是 train 前 OPD 对齐

如果问“train 前的 OPD 对齐是否非常缓慢”，基于这次更长 smoke 的日志，结论仍然是：

- 不是

更明显的耗时主要集中在初始化阶段，例如：

- Ray 启动
- rollout / SGLang 侧初始化
- 训练组件装配

这次日志里：

- `ray.init(local) begin` 在 `2026-04-21 16:04:37`
- 第一条 `driver rollout opd summary 0` 在 `2026-04-21 16:07:49`

这段时间更接近“服务与运行时启动成本”。

而一旦进入 step 循环之后，后续的 `rollout/train opd summary` 基本都是几秒级持续推进，并没有表现出“OPD byte-chunk 对齐本身是主要瓶颈”的特征。

### 5. 到当前为止，更准确的状态判断

如果只看“跨词表 OPD 主训练链路是否已经达到无对齐错误的稳定 smoke 状态”，那么当前可以更有把握地说：

- 严格 recorded payload 主路径已经连续跑过 8 个真实 step
- 训练日志没有出现 OPD 对齐错误
- replay 的 32 个真实样本全部 byte-chunk 对齐成功

仍然保留的待办，不再是“主链路还在频繁对齐失败”，而更多是：

- 脚本与测试继续整理
- 历史兼容痕迹继续收口
- 把训练入口校验工具进一步标准化

## 2026-04-21 追加验证（三）

这一次不是新的短 smoke，而是更接近真实负载的 full run：

- 工作目录：`/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_full_7gpu_len16384`
- 本地软链：`/mnt/ssd/lvzhihao/PostTrain/slime/workdirs/slime_opd_train_workdir_cross_tokenizer_full_7gpu_len16384`
- 启动方式：
  `WORK_DIR=/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_full_7gpu_len16384 TRAIN_CUDA_VISIBLE_DEVICES=0,1,2 TEACHER_CUDA_VISIBLE_DEVICES=3 ROLLOUT_CUDA_VISIBLE_DEVICES=4,5,7 NUM_GPUS=3 ROLLOUT_NUM_GPUS_PER_ENGINE=1 OPD_DISABLE_SEQUENCE_FALLBACK=true ROLLOUT_MAX_PROMPT_LEN=2048 ROLLOUT_MAX_RESPONSE_LEN=16384 EVAL_MAX_PROMPT_LEN=2048 EVAL_MAX_RESPONSE_LEN=16384 SAVE_INTERVAL=5 DEBUG_ROLLOUT_SAVE_INTERVAL=1 bash scripts/run-yulan-cross-tokenizer-opd-train.sh`

这次 full run 很有价值，因为它第一次把跨词表 OPD 放到了：

- 3 卡 train
- 1 卡 teacher
- 3 卡 rollout
- `max_response_len=16384`

这样的高压配置里。

### 1. 这次 full run 暴露出的真实失败点

日志文件：

- `run.log`：`/mnt/ssd/lvzhihao/PostTrain/slime/workdirs/slime_opd_train_workdir_cross_tokenizer_full_7gpu_len16384/logs/run.log`

它不是在 rollout 侧先报错，而是：

- rollout summary 先给出 `status={'ok_recorded': 132}`、`top_errors=[]`
- 随后在训练第一个 `async_train` 中进入 OPD KL 计算时报错

核心异常是：

- `ValueError: alignment_failed: Student token bytes ended before chunk alignment completed.`

也就是说：

- rollout 侧 recorded payload 表面上看是完整的
- 但 train 侧真正做 teacher/student byte-chunk 对齐时，发现两边字节流并没有用同一条 prompt-response 边界

### 2. 根因不是“对齐器不稳”，而是 canonical 文本三元组失配

这次最关键的新结论是：

- 问题首先不在 `align_token_byte_chunks(...)`
- 问题在 `prompt_text / response_text / full_text` 这组三元 canonical 文本本身已经不自洽

具体表现为：

- 一批长截断样本里，`response_text` 已经因为 token 一致性检查失败而回退成 token-derived 文本
- 但 `full_text` 仍然保留了另一套已记录文本
- `prompt_text` 也继续沿用旧的 rendered prompt

于是 teacher 侧会按：

- `prompt_byte_length = len(prompt_text.encode("utf-8"))`

去切：

- `full_text`

从第一个 response byte 开始就切歪。

这在真实失败样本上可以直接观测到：

- `full_text.startswith(prompt_text) == False`
- `full_text != prompt_text + response_text`

而 teacher response bytes 的开头，会混进本应属于 prompt 尾巴的文本片段。

### 3. 这次 full run 的失败是系统性的，不是单个坏样本

对这次产出的：

- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_full_7gpu_len16384/debug_rollouts/rollout_0.pt`

做离线 replay 之后，修复前的统计是：

- 总样本数：`132`
- 失败样本数：`96`
- 失败模式全部一致：`alignment_failed: Student token bytes ended before chunk alignment completed.`

进一步核对还可以看到：

- `132` 个样本中，恰好有 `96` 个满足 `opd_full_text` 不以 `opd_prompt_text` 开头，且不等于 `opd_prompt_text + opd_response_text`
- 这与 replay 失败的 `96` 个样本一一对应

所以这次 full run 的事故不是“长样本偶发不稳定”，而是：

- canonical text boundary drift 在长截断场景下系统性暴露

### 4. 修复点：让 canonical prompt/response/full 始终来自同一条 token 链

这次修复没有去修改 chunk 对齐器，而是改了：

- `slime/rollout/on_policy_distillation.py`
- `_build_canonical_opd_texts(...)`

修复原则是：

- 如果当前 `prompt_text / response_text / full_text` 三元组边界不自洽
- 或者 `full_text` 已经无法继续安全信任
- 那么直接回退到同一条 student token 链上解出来的：
  - `decoded_prompt_text`
  - `decoded_response_text`
  - `decoded_full_text`

这样 teacher byte clipping 与 student recorded alignment 才会真正共享同一条 prompt-response 边界。

### 5. 修复后验证结果

新增回归测试：

- `tests/test_opd_byte_chunk.py` 中增加了“当 `full_text` 回退到 token decode 时，`prompt_text` 也必须同步回退”的用例

定向 pytest 结果：

- `3 passed`

更关键的是，对同一份真实失败产物：

- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_cross_tokenizer_full_7gpu_len16384/debug_rollouts/rollout_0.pt`

重新离线 replay 后，结果从：

- 修复前：`96/132 failed`

变成：

- 修复后：`132/132 ok`

这说明这次修复命中的确实是主因，而不是只掩盖了表面症状。

### 6. 这次追加验证带来的状态更新

到这一轮为止，关于“跨词表 OPD 是否已经足够鲁棒”，更准确的表述应该更新成：

- 在中短 smoke 范围内，主链路已经稳定
- 在 full 规模长响应场景下，主链路曾暴露出 canonical boundary drift
- 这个 drift 已经找到根因，并且对真实失败 rollout 做到了离线 `132/132` 全修复

但我仍然不建议现在就把话说满成“full run 已完全证明十分鲁棒”，因为还差最后一步：

- 用修复后的代码重新发起一次真实 full train

只有这一轮也稳定通过，才算把“长响应 + 多卡 + 无 fallback”这组要求闭环。
