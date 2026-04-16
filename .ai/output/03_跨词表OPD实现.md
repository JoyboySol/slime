# 跨词表 OPD 实现记录

日期：2026-04-15

## 目标

为 slime 增加一版可运行的跨词表 OPD 能力：

- 内部对齐粒度采用 UTF-8 bytes
- 在公共 bytes 空间上构造最小公共 chunk
- 用 chunk 内 token logprob 求和来替代原先严格逐 token 对齐的 OPD
- 同时覆盖 `sglang teacher` 和 `megatron teacher` 两条链路

## 本次实现内容

### 1. 新增 OPD bytes/chunk 对齐工具

新增文件：

- `slime/utils/opd_utils.py`

主要能力：

- 缓存加载 tokenizer
- 用 tokenizer 对 response 文本重建 token ids
- 构建 token -> UTF-8 bytes span
- 在 student / teacher 两侧 token bytes 上做最小公共 chunk 对齐
- 计算 chunk-level reverse KL，并把结果广播回 student token 位置

其中 chunk-level reverse KL 的定义是：

- 先对每个 chunk 分别求
  - `student_chunk_log_prob = sum(student token logprobs in chunk)`
  - `teacher_chunk_log_prob = sum(teacher token logprobs in chunk)`
- 再计算
  - `reverse_kl = student_chunk_log_prob - teacher_chunk_log_prob`
- 最后按该 chunk 覆盖的 student token 数做平均分摊，再写回对应 student token

这样可以支持：

- 1 对多，如 `"<think>"` vs `"<" + "think" + ">"`
- 多对 1
- 多对多

### 2. OPD 参数扩展

修改：

- `slime/utils/arguments.py`

新增参数：

- `--opd-alignment token|byte_chunk`
  - 默认 `token`
  - `byte_chunk` 用于跨词表 OPD
- `--opd-teacher-hf-checkpoint`
  - 当 `--opd-alignment=byte_chunk` 时必填
  - 用于加载 teacher tokenizer / HF checkpoint

校验逻辑也已补充：

- 开启 `byte_chunk` 对齐时，必须提供 `--opd-teacher-hf-checkpoint`

### 3. `sglang teacher` 模式支持跨词表

修改：

- `slime/rollout/on_policy_distillation.py`

关键变化：

- 旧逻辑是把 student `sample.tokens` 直接作为 `input_ids` 发给 teacher server
- 新增 `byte_chunk` 模式后，改为先用 student tokenizer 把整条 sample 解码为文本，再用 `text` 方式发给 teacher server，让 teacher 按自己的 tokenizer 重新切分

这样 teacher 不再依赖 student token ids，真正支持跨词表打分。

同时在 reward 后处理里：

- `token` 模式仍按 student `response_length` 截 teacher logprobs
- `byte_chunk` 模式改为先用 student 全量 tokens 确定 response 的 byte 起点，再按 teacher tokenizer 对 `full_text` 重新切分，并裁出覆盖 response byte 区间的 teacher token logprobs

### 4. loss 侧支持 chunk-level OPD

修改：

- `slime/backends/megatron_utils/loss.py`

关键变化：

- `token` 模式保持原样：
  - `reverse_kl = student_log_probs - teacher_log_probs`
- `byte_chunk` 模式：
  - 从 student 全量 token ids 恢复 canonical `full_text`
  - 用 student tokenizer 和 teacher tokenizer 在同一条 `full_text` 上重建 token trace
  - 在 UTF-8 bytes 空间做 chunk 对齐
  - 计算 chunk-level reverse KL
  - 再把 chunk reverse KL 按 student chunk 长度平均分摊回 student token 维度，用于后续 advantage 修正

优势更新仍沿用 slime 原有框架：

- `advantages[i] = adv - opd_kl_coef * reverse_kl`

### 5. `megatron teacher` 模式支持跨词表

修改：

- `slime/backends/megatron_utils/actor.py`

关键变化：

- 原先 megatron teacher 直接复用 student rollout_data 里的 token ids 做 teacher forward
- 这在跨词表场景下不成立

现在在 `byte_chunk` 模式下：

- 先用 student tokenizer 从 full sequence 恢复 canonical `full_text`
- 再用 teacher tokenizer 对 `full_text` 重新编码
- 根据 student response 起始 byte，在 teacher token spans 上裁出 teacher response 部分
- 重建一份 teacher 专属 rollout_data
- teacher forward 基于这份 retokenized 数据来计算 `teacher_log_probs`

这样 megatron teacher 也能真正按 teacher 自己的 tokenizer 打分，而不是被迫吃 student token ids。

### 6. rollout 日志兼容

修改：

- `slime/backends/megatron_utils/data.py`
- `slime/backends/megatron_utils/actor.py`

原因：

- `byte_chunk` 模式下 `teacher_log_probs` 长度不再保证等于 student response token 数
- 原有日志和 CP 切片逻辑默认两边长度一致

处理方式：

- 在 `byte_chunk` 模式下，teacher logprobs 不再走原先基于 student response length 的切片逻辑
- rollout 日志中对 `teacher_log_probs` 改为按 sample mean 聚合，避免按 student token 数错误切分

## 测试

新增测试：

- `tests/test_opd_byte_chunk.py`

覆盖点：

1. `many-to-one`
   - teacher 多 token，对齐到 student 单 token chunk

2. `one-to-many`
   - teacher 单 token，对齐到 student 多 token chunk
   - chunk penalty 会在 student chunk 内平均分摊，而不是完整复制

3. `sglang teacher` reward 后处理
   - `byte_chunk` 模式下，teacher response logprobs 按 teacher tokenizer 的 response token 长度截取

4. loss 集成
   - `apply_opd_kl_to_advantages` 在 `byte_chunk` 模式下正确应用 chunk-level reverse KL

已执行验证命令：

```bash
.venv/bin/python -m pytest -q tests/test_opd_byte_chunk.py
.venv/bin/python -m py_compile \
  slime/utils/opd_utils.py \
  slime/rollout/on_policy_distillation.py \
  slime/utils/arguments.py \
  slime/backends/megatron_utils/loss.py \
  slime/backends/megatron_utils/actor.py \
  slime/backends/megatron_utils/data.py \
  tests/test_opd_byte_chunk.py
```

## 当前实现边界

### 已支持

- `token` 模式保持兼容
- `byte_chunk` 模式下：
  - `sglang teacher`
  - `megatron teacher`
  - chunk-level reverse KL
  - UTF-8 bytes 最小公共 chunk 对齐

### 暂未做

- 论文 ALM 中更完整的 debiasing / binarized f-divergence
- 更完整的 e2e CI 用例
- 针对 CP>1 的跨词表 OPD 专项验证
- 专门的可视化 debug dump（如 chunk text / chunk slices 输出）

## 后续建议

建议下一步继续补下面几项：

1. 增加一个真实 tokenizer mismatch 的集成测试
   - student tokenizer 把某段文本切成 1 个 token
   - teacher tokenizer 切成多个 token
   - 跑通一条最小训练/rollout链

2. 给 `byte_chunk` 模式增加更明确的 debug 日志
   - response text
   - chunk text
   - student / teacher token slices
   - chunk reverse KL

3. 如果后续想进一步贴近论文，可以在当前 `opd_utils` 之上继续加：
   - outcome chunk debiasing
   - 更一般的 f-divergence
