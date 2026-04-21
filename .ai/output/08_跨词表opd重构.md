# 08 跨词表 OPD 重构

## 目标

这份文档用于交付本轮跨词表 OPD 重构的最终结果，重点回答四件事：

- 这次重构到底改了什么架构
- 为什么这次不是“补 patch”，而是换成了更长期正确的链路
- 当前验证做到什么程度，证据在哪里
- 还剩下哪些已知风险或后续建议

本轮工作的核心目标不是继续在旧的“训练后再从 student 文本反推 byte span”的链路上打补丁，而是把跨词表 OPD 的 student 对齐信息前移到 rollout / sample 准备阶段记录下来，再让训练、诊断、回放统一消费这一份记录。

## 背景问题

此前跨词表 OPD 的主要问题有两个：

### 1. 鲁棒性不足

旧实现更依赖：

- student token ids
- response text
- prompt / response 边界的二次推断

然后在训练或诊断阶段重新构建 student response bytes / token byte spans。

这个思路在 same-vocab 或 tokenizer 行为稳定时还能凑合，但在 YuLan <- Nanbeige 这种 cross-vocab 场景下很脆弱，因为：

- prompt 和 response 的边界不一定能靠单纯 decode 稳定恢复
- tokenizer 的 offset / prefix decode 行为不总是和“字符串拼接直觉”一致
- 一旦 canonical full text 与 student reconstruction 的视角不完全一致，就会在 teacher slicing 或 chunk 对齐阶段暴露成 fallback / mismatch

### 2. 复杂度和调试成本过高

旧链路不仅在若干位置存在重复 reconstruction，而且出问题时很难判断：

- 是 sample 本身没带够信息
- 是 canonical text 构建错了
- 还是 chunk 对齐本身出错

导致很多问题表面都长得像“byte-chunk 对齐不鲁棒”，但根因并不在 `align_token_byte_chunks()`。

## 最终采用的架构

本轮最终落地的方案是：

### 1. rollout 侧记录 student 对齐 payload

在 sample 准备阶段直接记录：

- `opd_student_response_bytes`
- `opd_student_token_byte_spans`
- `opd_student_alignment_version`
- `opd_student_alignment_error`

同时保留：

- `opd_prompt_text`
- `opd_response_text`
- `opd_full_text`

这样后续 OPD 消费侧不再把“重新从 response 文本恢复 student bytes”当作主路径，而是优先使用 rollout 已经生成好的 recorded payload。

### 2. full-sequence + prompt-boundary aware 对齐

实践里已经验证，仅靠 `response_text + response_token_ids` 不足以稳定恢复真实 student 响应字节区间。

因此当前 recorded alignment builder 的设计是：

- 基于 canonical full text
- 显式使用 prompt boundary
- 在 full sequence 视角下校验 response byte region

这比“只看 response 片段”的逻辑更重一些，但在 cross-tokenizer 场景下明显更稳。

### 3. 训练、诊断、回放统一优先消费 recorded payload

统一原则是：

- 训练主链路优先使用 recorded payload
- 诊断脚本优先使用 recorded payload
- 仅当 recorded payload 缺失或显式失效时，才进入兼容性 fallback

这样就避免出现“训练和诊断走的不是同一套语义”的问题。

## 本轮主要代码改动

### 已提交的核心重构

本轮已经落地并提交的关键 commit：

- `acb6de8` `backup: save current cross-tokenizer opd work`
- `a096556` `refactor: record rollout-side opd byte alignment`
- `097d9cf` `test: prefer recorded opd alignment in diagnostics`
- `00104a4` `docs: add cross-tokenizer opd maintenance skill`

其中本轮最关键的是：

#### `a096556` `refactor: record rollout-side opd byte alignment`

核心含义：

- 在 rollout / sample 侧直接记录 student response bytes 与 per-token byte spans
- OPD 训练链路默认消费 recorded payload
- 保留兼容 fallback，但不再让 reconstruction 成为主路径

主要涉及文件：

- [`slime/utils/types.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/utils/types.py)
- [`slime/utils/opd_utils.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/utils/opd_utils.py)
- [`slime/rollout/on_policy_distillation.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/rollout/on_policy_distillation.py)
- [`slime/ray/rollout.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/ray/rollout.py)
- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py)
- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py)

#### `097d9cf` `test: prefer recorded opd alignment in diagnostics`

核心含义：

- 让 `replay_debug_rollout_opd.py` / `analyze_opd_alignment.py` 与训练侧口径保持一致
- 诊断时优先使用 recorded payload，而不是沿用旧的 reconstruction 习惯

### 本轮新增 skill

为了后续维护不再重复走弯路，本轮新增了本地 skill：

- [maintain-cross-tokenizer-opd](/mnt/ssd/lvzhihao/PostTrain/slime/.codex/skills/maintain-cross-tokenizer-opd/SKILL.md)

它约束了后续维护时的几个关键原则：

- recorded payload 优先
- 训练与诊断语义一致
- 高风险修改至少跑 128 short + 128 long sampled regression

## 测试与验证

## 单测与回归测试

本轮已验证通过的核心测试包括：

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
```

另外也验证过相关文件可正常编译：

```bash
./.venv/bin/python -m py_compile \
  scripts/analyze_opd_alignment.py \
  scripts/replay_debug_rollout_opd.py \
  slime/utils/opd_utils.py \
  slime/rollout/on_policy_distillation.py \
  slime/ray/rollout.py \
  slime/backends/megatron_utils/loss.py \
  slime/backends/megatron_utils/data.py
```

### 真实 tokenizer sampled regression

本轮使用真实资源做了 sampled integration regression：

- Student model: `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`
- Teacher model: `/mnt/hdd/Nanbeige4.1-3B`
- Short corpus: `/mnt/hdd/lvzhihao/output/OpenMathInstruct-2/correct/segments`
- Long corpus: `/mnt/hdd/lvzhihao/output/OpenThoughts3-1.2M-math-distill-nanbeige4_1_3b/correct/segments`

最终按要求提升到每个语料至少 128 条 sampled rows：

- short 128 / 128 通过
- long 128 / 128 通过

这点很重要，因为它说明当前 recorded-alignment 架构并不是只在手工构造样例上成立，而是在真实 tokenizer + 真实语料条件下也有说服力。

## 最小化真实训练 smoke

为了验证跨词表 OPD 训练链路不是“单测通过但真实训练挂掉”，本轮还做了一次最小化 smoke：

- 工作目录：`/mnt/hdd/lvzhihao/slime_opd_train_workdir_min_smoke`
- 启动脚本：[`scripts/run-yulan-cross-tokenizer-opd-train.sh`](/mnt/ssd/lvzhihao/PostTrain/slime/scripts/run-yulan-cross-tokenizer-opd-train.sh)
- 目标：少量 rollout、小 batch、只跑极少 step，确认 rollout -> teacher -> OPD -> train 整条链路真实可走

### smoke 中暴露并修复的问题

第一次 smoke 并不是直接成功，而是暴露了一个真实问题：

- `log_rollout_data()` 会把新的 recorded alignment payload 当成普通标量列表聚合
- 具体报错是对 `list[list[int]]` / `list[list[list[int]]]` 做 `sum(val) / len(val)`

报错根因不是 OPD 主逻辑坏了，而是训练侧 logging 没跟上新数据结构。

为此做了一个很小但必要的修复：

- 在 [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py) 的 rollout logging skip 列表中加入：
  - `opd_student_response_bytes_list`
  - `opd_student_token_byte_spans_list`
- 新增测试 [`tests/test_rollout_logging.py`](/mnt/ssd/lvzhihao/PostTrain/slime/tests/test_rollout_logging.py)

这个修复当前已经在工作区完成并本地验证通过，但尚未单独提交。

### smoke 最终结果

从 [`workdirs/slime_opd_train_workdir_min_smoke/logs/run.log`](/mnt/ssd/lvzhihao/PostTrain/slime/workdirs/slime_opd_train_workdir_min_smoke/logs/run.log) 看，这次最小化训练已经正常跑通，关键证据是日志尾部出现：

```text
[driver] train(args) done
```

对应离线 WandB latest run 为：

- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_min_smoke/wandb/wandb/offline-run-20260421_005030-uf5dxi44`

从该 run 中可以确认至少完成了 3 个 rollout / 3 个 train step。

## smoke 指标结果

结合 [`05_OPD指标监测.md`](/mnt/ssd/lvzhihao/PostTrain/slime/.ai/output/05_OPD指标监测.md) 的口径，这次 smoke 的关键指标如下。

### rollout 0 / step 0

- `rollout/rollout_log_probs = -0.5005501906077067`
- `rollout/log_probs = -0.5012866655985514`
- `rollout/teacher_log_probs = -1.4604551394780476`
- `rollout/rollout_chunk_log_prob = -0.5005502005418142`
- `rollout/teacher_chunk_log_prob = -1.368876059850057`
- `rollout/opd_reverse_kl = 0.8675893942515055`
- `train/train_rollout_logprob_abs_diff = 0.009359203279018402`
- `train/grad_norm = 24.439029733530877`

### rollout 1 / step 1

- `rollout/rollout_log_probs = -0.4978152811527252`
- `rollout/log_probs = -0.4979561120271683`
- `rollout/teacher_log_probs = -1.1909077564875286`
- `rollout/rollout_chunk_log_prob = -0.4978152811527252`
- `rollout/teacher_chunk_log_prob = -1.1370916763941448`
- `rollout/opd_reverse_kl = 0.6391355792681376`
- `train/train_rollout_logprob_abs_diff = 0.009868377819657326`
- `train/grad_norm = 18.04504776328666`

### rollout 2 / step 2

- `rollout/rollout_log_probs = -0.5200501581033071`
- `rollout/log_probs = -0.520250161488851`
- `rollout/teacher_log_probs = -1.3332574367523193`
- `rollout/rollout_chunk_log_prob = -0.5200501680374146`
- `rollout/teacher_chunk_log_prob = -1.233601689338684`
- `rollout/opd_reverse_kl = 0.7133514881134033`
- `train/train_rollout_logprob_abs_diff = 0.009969414522250494`
- `train/grad_norm = 24.998471632969384`

### 3 步均值

- `rollout/rollout_log_probs ≈ -0.5061385`
- `rollout/log_probs ≈ -0.5064976`
- `rollout/teacher_log_probs ≈ -1.3282068`
- `rollout/rollout_chunk_log_prob ≈ -0.5061385`
- `rollout/teacher_chunk_log_prob ≈ -1.2465231`
- `rollout/opd_reverse_kl ≈ 0.7400255`
- `train/train_rollout_logprob_abs_diff ≈ 0.0097323`
- `train/grad_norm ≈ 22.4942`

## 如何解读这次 smoke 指标

### 1. OPD 口径是对上的

这次 smoke 里，`rollout/opd_reverse_kl` 与：

```text
rollout/log_probs - rollout/teacher_chunk_log_prob
```

是对得上的。

例如 step 2：

```text
-0.520250161488851 - (-1.233601689338684)
= 0.713351527849833
```

与日志里的：

```text
rollout/opd_reverse_kl = 0.7133514881134033
```

基本一致。

这说明当前 cross-vocab byte-chunk OPD 的真实训练口径已经与指标文档一致，没有再出现“看着像对不上，其实比较的不是同一组数”的问题。

### 2. rollout / train student logprob 一致性较好

`train/train_rollout_logprob_abs_diff` 三步都大约在 `0.01` 附近，说明：

- rollout 引擎生成时返回的 student logprob
- train 侧重算得到的 student logprob

两者没有明显漂移。

对于最小化 smoke 来说，这是个很健康的信号。

### 3. teacher chunk 口径稳定参与了训练

从这次 smoke 的 `teacher_log_probs` 与 `teacher_chunk_log_prob` 可以看出：

- teacher 原始 token logprob 与 chunk 对齐后的 teacher 口径是两组不同的数
- 实际进入 OPD penalty 的是 `teacher_chunk_log_prob`

这也再次验证了当前日志解释文档的结论。

## 本轮交付结论

可以把这次工作的结论概括成三条：

### 1. 跨词表 OPD 的长期正确架构已经落地

当前主路径已经从：

- 训练 / 诊断阶段临时 reconstruction student bytes

切换到：

- rollout 侧提前记录 student alignment payload
- 训练、诊断、回放统一优先消费 recorded payload

这比旧路径更稳，也更容易分析问题。

### 2. 风险最高的真实场景回归已经通过

不是只做了单元测试，而是额外完成了：

- 真实 tokenizer sampled regression
- short 128 / 128
- long 128 / 128
- 最小化真实训练 smoke

因此当前可以认为这套重构已经具备了比较扎实的实证基础。

### 3. 最小训练链路已经真实跑通

这说明当前状态不是“对齐函数可以单测通过”，而是：

- rollout 真的产生了样本
- teacher 真的返回了 logprob
- byte-chunk OPD 真的参与了训练
- train step 真的完成了
- driver 也真的正常退出了

## 当前剩余事项

当前还有两个后续事项值得记录：

### 1. 提交 rollout logging 小修复

当前工作区仍有两个未提交改动：

- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py)
- [`tests/test_rollout_logging.py`](/mnt/ssd/lvzhihao/PostTrain/slime/tests/test_rollout_logging.py)

它们修的是 smoke 暴露出的 recorded alignment payload logging 问题。

建议单独提交，例如：

```text
fix: skip recorded opd alignment payloads in rollout logging
```

### 2. 若要做更强实跑验证，可补 5-step 以上 smoke

本次 smoke 的目标 originally 是 4-5 个 step，但实际稳定落盘的是 3 个 step。

就“链路是否走通”这个目标来说已经足够，但如果后续要更强地证明训练稳定性，可以：

- 再跑一个 5-step 左右的 smoke
- 保持小 batch
- 继续记录 debug rollout dump

## 交付摘要

本轮已经完成的交付包括：

- 跨词表 OPD 从 reconstruction-first 重构为 recorded-alignment-first
- 训练侧、诊断侧统一使用 recorded payload 语义
- 真实 tokenizer sampled regression 达到 128 short + 128 long
- 新增本地维护 skill，沉淀后续维护规范
- 最小化 cross-tokenizer OPD 训练真实跑通
- 修复 smoke 暴露出的 rollout logging 兼容性问题

因此当前可以认为：跨词表 OPD 的主链路已经从“勉强可用、容易 fallback、难以定位”提升到了“架构更清晰、真实验证更充分、后续维护路径更明确”的状态。
