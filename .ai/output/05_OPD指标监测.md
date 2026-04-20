# 05 OPD 指标监测

## 目的

这份文档用于统一说明当前 `slime` 中几类 OPD 相关指标的口径，避免在看 WandB 时把不同来源的 `log_prob` 混在一起比较。

重点回答两个问题：

- `train/train_rollout_logprob_abs_diff`
- `train/grad_norm`
- `rollout/opd_reverse_kl`

以及为什么在两组实验中会出现：

1. Qwen same-vocab chunk 路径下，`opd_reverse_kl ~= student_log_prob - teacher_log_prob`
2. YuLan <- Nanbeige cross-vocab 路径下，表面上 `student_log_prob - teacher_log_prob` 与 `opd_reverse_kl` 完全对不上

## 一、几个关键指标到底是什么

### 1. `rollout/rollout_log_probs`

含义：

- rollout 引擎在生成时直接返回的 student token logprob
- 日志中展示的是每个 sample 的 token 平均值，再对 sample 求平均

代码位置：

- [`slime/rollout/sglang_rollout.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/rollout/sglang_rollout.py)
- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py#L482)

说明：

- 这是“rollout 侧的 student logprob”
- 不是 train 阶段重算的 old policy logprob

### 2. `rollout/log_probs`

含义：

- train 侧拿当前 actor 对 rollout 样本重新前向计算出来的 student token logprob
- 日志中同样是 sample mean 再做 batch mean

代码位置：

- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L620)
- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py#L483)

说明：

- 这是“train 侧的 student logprob”
- 在 OPD 训练里，真正参与 `advantages` / `opd_reverse_kl` 计算的 student logprob 是这组，而不是 `rollout/rollout_log_probs`

### 3. `rollout/teacher_log_probs`

含义：

- teacher server 返回的原始 token logprob
- 在 byte-chunk 对齐前，这是一组“teacher 原始口径”的 token logprob

代码位置：

- [`slime/rollout/on_policy_distillation.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/rollout/on_policy_distillation.py#L175)
- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py#L486)

说明：

- 这组数不一定直接参与最终 OPD penalty
- 在 `byte_chunk` 模式下，teacher 还会经过 chunk 对齐

### 4. `rollout/rollout_chunk_log_prob`

含义：

- 按 byte-chunk 对齐后，student 在 chunk 口径下的 logprob

代码位置：

- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py#L51)

说明：

- 它和 `rollout/rollout_log_probs` 可能接近，也可能差很多
- 当跨词表切 chunk 时，这组数比 `rollout/rollout_log_probs` 更接近真正用于 OPD 的口径

### 5. `rollout/teacher_chunk_log_prob`

含义：

- teacher token logprob 在 byte-chunk 对齐后，映射到 student chunk 口径上的结果

代码位置：

- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py#L52)
- [`slime/utils/opd_utils.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/utils/opd_utils.py#L361)

说明：

- 这是 byte-chunk 模式下最接近“真正 teacher 监督信号”的日志指标

### 6. `rollout/opd_reverse_kl`

含义：

- OPD 实际使用的 reverse KL，定义为：

```text
student_log_prob - teacher_log_prob
```

但这里的 student / teacher，必须是“真正进入 OPD 计算的那一组”。

代码位置：

- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L580)
- [`slime/utils/opd_utils.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/utils/opd_utils.py#L251)

更准确地说：

- token 对齐模式：`student_log_probs - teacher_log_probs`
- byte-chunk 模式：`chunk_aligned_student_log_probs - chunk_aligned_teacher_log_probs`

说明：

- 不能简单拿任意一个 `student_log_prob` 和任意一个 `teacher_log_prob` 去验证它

### 7. `train/train_rollout_logprob_abs_diff`

含义：

- train 阶段重算出来的 student logprob 与 rollout 引擎当时返回的 student logprob 的绝对差

公式：

```text
mean(abs(old_log_probs - rollout_log_probs))
```

代码位置：

- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L994)

说明：

- 这个值越小，说明 rollout 侧与 train 侧对 student 的 logprob 口径越一致
- 如果这个值非常大，通常说明当前样本、当前路径或当前实现存在明显不一致

### 8. `train/grad_norm`

含义：

- 训练一步中的梯度范数

代码位置：

- [`slime/backends/megatron_utils/model.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/model.py#L456)

说明：

- 这个值是训练信号强弱的直接表现
- 配置里当前 `clip_grad=1.0`，所以这里看到的大数值更接近“裁剪前的激进程度”，不一定等于最终实际更新量

## 二、WandB 图是怎么画的

这些图不是累加和，而是“按当前 rollout / step 的平均值记录”。

对 token 级指标，代码会先做：

1. 每个 sample 内对 token 求 mean
2. 再对当前 batch 内的 sample 求 mean

代码位置：

- [`slime/backends/megatron_utils/data.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/data.py#L430)
- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L1026)

因此：

- `rollout/opd_reverse_kl` 不是 token 累加和
- `rollout/teacher_log_probs` 不是 token 累加和
- `train/train_rollout_logprob_abs_diff` 不是全局累计量
- `train/grad_norm` 也不是累计量

WandB 上看到的是“每一步的当前值”。

## 三、为什么 Qwen 那组看起来公式成立

Qwen same-vocab chunk 实验中，观察到：

```text
student_log_prob: -3.45175
teacher_log_prob: -3.99693
opd_reverse_kl: 0.54483
```

这里：

```text
-3.45175 - (-3.99693) = 0.54518
```

和 `opd_reverse_kl` 基本一致。

原因通常是：

- same-vocab 时，student / teacher 的 token 切分天然更稳定
- rollout 侧 student logprob、train 侧重算 student logprob、chunk 对齐后的口径彼此更接近
- 因此你拿出来比较的那两个 `log_prob`，碰巧就是与 `opd_reverse_kl` 同一口径的数

## 四、为什么 YuLan 那组表面上完全对不上

YuLan <- Nanbeige cross-vocab 实验某步观测到：

```text
student_log_prob: -0.35198
teacher_log_prob: -1.73904
opd_reverse_kl: -9.89206
```

表面上：

```text
-0.35198 - (-1.73904) = 1.38706
```

显然和 `-9.89206` 完全不一致。

但这不是 `opd_reverse_kl` 公式错了，而是这三个数不是一套口径。

对应真实日志是：

- [`worker-187f...15676.err:86`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_17-16-47_005602_58381/logs/worker-187f6826412de79f2c9366f8ec8c0608cc7971c99e01e8fe127fb72a-01000000-15676.err#L86)

其中同一步实际记录为：

```text
rollout/rollout_log_probs = -0.3519759
rollout/teacher_log_probs = -1.7390448
rollout/log_probs = -11.4797185
rollout/teacher_chunk_log_prob = -1.5876567
rollout/opd_reverse_kl = -9.8920618
```

这时真正成立的是：

```text
rollout/log_probs - rollout/teacher_chunk_log_prob
= -11.4797185 - (-1.5876567)
= -9.8920618
```

也就是说：

- 你拿来比较的 `student_log_prob=-0.35198` 实际上对应 `rollout/rollout_log_probs`
- 你拿来比较的 `teacher_log_prob=-1.73904` 实际上对应 `rollout/teacher_log_probs`
- 但 `opd_reverse_kl` 实际对应的是：
  - student: `rollout/log_probs`
  - teacher: `rollout/teacher_chunk_log_prob`

所以对不上是必然的。

## 五、为什么这轮 YuLan 结果确实异常

虽然上面的“对不上”本身是口径问题，但这轮实验的数值也确实异常，不只是看错列。

### 1. rollout student 与 train student 严重不一致

同一步里：

```text
rollout/rollout_log_probs = -0.3519759
rollout/log_probs = -11.4797185
```

两者相差非常大。

对应训练指标：

```text
train/train_rollout_logprob_abs_diff = 11.1312561
```

见：

- [`worker-187f...15676.err:88`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_17-16-47_005602_58381/logs/worker-187f6826412de79f2c9366f8ec8c0608cc7971c99e01e8fe127fb72a-01000000-15676.err#L88)

这说明 rollout 侧与 train 侧的 student logprob 已经严重失配。

### 2. 真实跨词表 byte-chunk 存在 fallback

日志里出现了：

```text
Falling back to sequence-level OPD penalty after byte_chunk preparation/alignment failure.
reason=student_byte_reconstruction_failed: Tokenizer decode is not prefix-consistent
```

见：

- [`worker-671c...17589.err:22`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_17-16-47_005602_58381/logs/worker-671cc913de7d2088c4b7a7d2c6404b7fd6d64358ee79ad85062a9647-01000000-17589.err#L22)

这说明：

- 当前并不是所有样本都走通了 byte-chunk 主路径
- 有些样本已经回退到了 sequence-level OPD

### 3. response 太长

这轮 rollout 的 response 长度非常夸张：

- `9038`
- `8007`
- `11349`

对应日志：

- [`worker-187f...15676.err:42`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_17-16-47_005602_58381/logs/worker-187f6826412de79f2c9366f8ec8c0608cc7971c99e01e8fe127fb72a-01000000-15676.err#L42)
- [`worker-187f...15676.err:66`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_17-16-47_005602_58381/logs/worker-187f6826412de79f2c9366f8ec8c0608cc7971c99e01e8fe127fb72a-01000000-15676.err#L66)
- [`worker-187f...15676.err:86`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_17-16-47_005602_58381/logs/worker-187f6826412de79f2c9366f8ec8c0608cc7971c99e01e8fe127fb72a-01000000-15676.err#L86)

### 4. 训练信号几乎全由 OPD 项主导

这轮纯 distillation 下：

- `rollout/rewards = 0.0`
- `rollout/returns = 0.0`
- 但 `rollout/advantages` 却在 `9~11`

说明优势几乎完全由 OPD penalty 决定。

代码位置：

- [`slime/rollout/on_policy_distillation.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/rollout/on_policy_distillation.py#L166)
- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L592)

### 5. 因此 `grad_norm` 很大

这轮的：

- `train/grad_norm = 137640.9241`
- `44322.7739`
- `49675.8388`

不是正常小波动，而是“超长 response + logprob 失配 + byte-chunk fallback + 纯 OPD 信号”叠加后的结果。

## 六、实际读图建议

看 OPD 是否正常时，建议按下面顺序看：

1. 先看 `train/train_rollout_logprob_abs_diff`

- 如果接近 `0` 或较小，说明 rollout 与 train 的 student logprob 基本一致
- 如果像这次一样到 `10+`，优先怀疑口径失配或实现路径异常

2. 再看 `rollout/opd_reverse_kl`

- 要结合真正对应的 student / teacher 口径理解
- 在 byte-chunk 模式下，优先和：
  - `rollout/log_probs`
  - `rollout/teacher_chunk_log_prob`
 进行比较

3. 再看 `train/grad_norm`

- 如果前两项已经异常，`grad_norm` 大通常只是后果
- 不要先把 `grad_norm` 当成唯一根因

4. 辅助看 `rollout/response_lengths` 和 `rollout/truncated`

- 超长 response 很容易把跨词表 OPD 的不稳定性放大

## 七、当前结论

目前可以比较明确地认为：

1. `opd_reverse_kl` 的公式本身没有明显算错

- 它在 YuLan 这轮里依然满足“真正参与 OPD 的 student logprob 减去真正参与 OPD 的 teacher chunk logprob”

2. YuLan 这轮异常的关键，不是公式，而是口径分裂和路径退化

- `rollout/rollout_log_probs` 与 `rollout/log_probs` 严重失配
- 真实跨词表样本上发生了 byte-chunk fallback
- response 长度过长

3. 因此后续做 sanity test 时，不能只看单独一个 `student_log_prob`

- 必须先确认它到底是：
  - rollout student
  - train student
  - chunk-aligned student

否则很容易误判成“OPD 公式算错”

## 八、进一步排查：为什么 `rollout/log_probs` 和 `rollout/rollout_log_probs` 差这么大

这一点需要单独拿出来看，因为它已经超出了普通“训推不一致”的量级。

### 1. 先确认它和 teacher 路径无关

`train/train_rollout_logprob_abs_diff` 的定义是：

```text
mean(abs(old_log_probs - rollout_log_probs))
```

对应代码：

- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L994)

这里用到的只有：

- 训练侧重算出来的 student `old_log_probs`
- rollout 引擎返回的 student `rollout_log_probs`

它不依赖：

- `teacher_log_probs`
- `teacher_chunk_log_prob`
- `opd_reverse_kl`
- byte-chunk teacher 对齐

因此：

- 如果 `train/train_rollout_logprob_abs_diff` 非常大
- 那么即使完全不看 teacher
- 也已经能说明 student 的 rollout/train logprob 口径存在独立问题

这是这次排查里最重要的收缩结论。

### 2. 训练侧 student logprob 的生成链路

训练侧 `log_probs` 的计算链路是：

1. 用 rollout 回传的 `tokens`
2. 重新前向 actor
3. 在完整 logits 上按 shifted tokens 提取 response token 的 logprob

关键代码：

- shifted token 构造：
  [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L223)
- response 区间切片：
  [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L284)
- 对 rollout 温度做补偿：
  [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L410)

这说明训练侧已经显式考虑了：

- response 的右移一位预测关系
- rollout temperature

所以“只是忘了除 temperature”并不是当前最优先怀疑项。

### 3. rollout 侧 student logprob 的生成链路

rollout 侧 `rollout_log_probs` 直接来自 SGLang 返回的：

```text
output["meta_info"]["output_token_logprobs"]
```

代码位置：

- [`slime/rollout/sglang_rollout.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/rollout/sglang_rollout.py#L226)

并且会直接累计到：

- `sample.rollout_log_probs`

对应：

- [`slime/rollout/sglang_rollout.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/rollout/sglang_rollout.py#L242)

### 4. 当前最值得怀疑的不是 OPD，而是 student logprob 口径

从代码上看，当前最值得验证的是下面这些分支：

1. SGLang `output_token_logprobs` 的语义，是否与训练侧 fused CE 计算出的 token logprob 完全等价
2. 长 response / truncation 情况下，`response_length`、`tokens[-response_length:]`、`rollout_log_probs` 是否严格一一对应
3. 训练侧 `_extract_per_sample()` 的 response 切片是否在当前模型 / 当前 qkv_format 上有偏移
4. 同一套权重下，rollout engine 与训练侧 actor 是否真的在用同一个 tokenizer / 同一套 token ids 语义

### 5. 当前没有看到的“直接代码铁证”

这轮静态检查没有发现一个可以直接拍板的单点 bug，比如：

- 明确的 `+1/-1` 越界
- 明确长度不相等却没报错
- 明确漏乘或漏除 temperature

所以现阶段更像是：

- student rollout logprob 的来源语义
- 超长样本下的切片口径
- 或者特定模型在 rollout 与 train 两侧的 token/logit 语义

这三者之一出了问题。

## 九、建议补充的测试点

下面这些测试点的目标，不是继续“猜”，而是把 student mismatch 的责任边界精确卡出来。

### A. 纯 student 路径一致性测试

测试目标：

- 不带 teacher
- 不带 OPD
- 只验证 `rollout_log_probs` 与训练侧 `log_probs` 是否一致

建议断言：

```text
abs(train_log_probs - rollout_log_probs) < 很小阈值
```

建议覆盖：

1. `cp_size=1`
2. `qkv_format=thd`
3. `temperature=1.0`
4. `temperature!=1.0`
5. 短 response
6. 长 response
7. 截断 response

应该新增的位置：

- 优先在 `tests/` 中新增 student-only mismatch 测试
- 最接近现有风格的是离线 debug rollout + train-only 两阶段测试

### B. response 切片一致性测试

测试目标：

- 给定一组已知 `tokens`
- 人工构造 logits
- 验证训练侧 `_extract_per_sample()` 拿到的 response log_probs
- 与“最后 `response_length` 个输出 token 的理论 logprob”完全一致

重点覆盖：

1. `response_length=1`
2. `prompt_length>1`
3. 多 sample 拼接
4. 长序列
5. `thd`

关键代码位置：

- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L284)

这个测试能直接排除：

- 训练侧 response 切片 off-by-one
- 拼 batch 后 offset 错位

### C. rollout metadata 对齐测试

测试目标：

- 验证 `len(sample.rollout_log_probs) == sample.response_length`
- 验证 `sample.tokens[-sample.response_length:]` 与 rollout 返回 token 序列完全一致
- 对 `finish_reason=length` 的样本也成立

关键代码位置：

- [`slime/rollout/sglang_rollout.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/rollout/sglang_rollout.py#L226)
- [`slime/utils/types.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/utils/types.py#L156)

这个测试能直接排除：

- 截断样本下 `response_length` 和 `rollout_log_probs` 不一致
- partial rollout 累积过程中 logprob/token 错位

### D. 温度一致性测试

测试目标：

- rollout 端和训练端在相同 `temperature` 下，是否给出一致 token logprob

原因：

- 训练侧已经显式做了 `logits / rollout_temperature`
- 但仍需要一个测试确认 SGLang 返回的 `output_token_logprobs` 口径确实匹配这个定义

关键代码位置：

- [`slime/backends/megatron_utils/loss.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/backends/megatron_utils/loss.py#L410)

建议覆盖：

1. `temperature=1.0`
2. `temperature=0.7`
3. `temperature=1.3`

### E. 同权重离线回放测试

测试目标：

- 先保存 1 个 rollout 的 `.pt`
- 再 train-only 重放
- 记录 `train/train_rollout_logprob_abs_diff`

建议断言：

1. same-vocab smoke：应非常小
2. cross-vocab same student / same rollout engine：student mismatch 不应因为 teacher 路径而变大

关键意义：

- 这个测试最接近真实运行
- 并且能证明 student mismatch 与 teacher 是否正交

### F. teacher 完全关闭的对照测试

测试目标：

- 在同一 student、同一数据、同一 rollout 下
- 关闭 OPD teacher 路径
- 只保留 student rollout/train logprob 对比

期望：

- 如果 mismatch 仍然很大，问题一定在 student 路径
- 如果 mismatch 明显缩小，再去考虑 OPD 路径是否有副作用

这个测试非常关键，因为它能把：

- student logprob 失配
- teacher byte-chunk fallback

彻底拆开。

## 十、推荐的后续监测方式

如果后面继续做真实跨词表 smoke，建议固定同时关注这几列：

- `rollout/log_probs`
- `rollout/rollout_log_probs`
- `rollout/teacher_log_probs`
- `rollout/teacher_chunk_log_prob`
- `rollout/opd_reverse_kl`
- `train/train_rollout_logprob_abs_diff`
- `train/grad_norm`
- `rollout/response_lengths`
- `rollout/truncated`

其中最关键的诊断链路是：

```text
如果 train_rollout_logprob_abs_diff 很大
-> 先看 rollout/log_probs 和 rollout/rollout_log_probs 是否失配
-> 再看是否出现 byte-chunk fallback
-> 再看 response_lengths 是否过长
-> 最后再解释 grad_norm
```
