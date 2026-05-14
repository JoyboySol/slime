# 跨词表 OPD 实现说明

本文说明 slime 当前 `byte_chunk` 跨词表 OPD 的实现方式。它面向的是 student 与 teacher tokenizer 不一致的蒸馏场景，例如 rollout 使用 student token 生成，但 teacher 需要按自己的词表重新打分。

当前实现的核心原则是：

- OPD 的对齐空间不是 token id，而是同一段 canonical text 的 UTF-8 bytes。
- student 侧优先消费 rollout / sample 阶段记录的 response bytes 与 token byte spans。
- teacher 侧在 SGLang 后端用文本请求重新 tokenize，并把返回的 teacher log-probs 裁到 response byte 区间。
- loss 侧在公共 byte chunk 上聚合 student / teacher log-probs，再把 chunk 级惩罚分摊回 student response token。

## 入口文件

主要实现分布在以下文件：

| 文件 | 作用 |
|------|------|
| `slime/utils/types.py` | 在 `Sample` 上定义 OPD canonical text、student byte alignment payload、generation byte evidence observability 等字段。 |
| `slime/utils/opd_utils.py` | tokenizer byte span 重建、recorded payload 校验、student/teacher byte chunk 对齐、chunk 级 reverse KL 计算。 |
| `slime/rollout/on_policy_distillation.py` | SGLang teacher 请求构造、teacher log-probs 裁剪、student recorded alignment 生成。 |
| `slime/ray/rollout.py` | 将 `Sample` 上的 OPD 字段转成 train data，并按 DP 分发。 |
| `slime/backends/megatron_utils/loss.py` | 在 `compute_advantages_and_returns()` 中应用 OPD KL 到 advantages。 |
| `slime/backends/megatron_utils/data.py` | 训练侧 rollout 日志、byte chunk log-prob 指标、alignment summary。 |
| `scripts/replay_debug_rollout_opd.py` | 回放 debug rollout，复现训练侧 byte chunk OPD 路径。 |
| `scripts/analyze_opd_alignment.py` | 离线分析真实 tokenizer / 数据集的对齐质量。 |

## 跨词表对齐与 SGLang 后端交互

### 1. 为什么不能按 token 对齐

普通 OPD 假设 student 和 teacher 的 response token 一一对应，因此可以直接做：

```text
reverse_kl[t] = student_log_prob[t] - teacher_log_prob[t]
```

跨词表时这个假设不成立。同一段文本可能出现：

- student 一个 token，teacher 多个 token；
- student 多个 token，teacher 一个 token；
- 双方都是多个 token，但切分边界不同；
- teacher tokenizer 对特殊符号、空格、非法 byte token 的渲染方式不同。

因此当前实现把 student / teacher 都投影到同一段 UTF-8 byte 序列，再在 byte 层找公共 chunk。

### 2. canonical OPD text

SGLang teacher 模式的入口是 `slime.rollout.on_policy_distillation.reward_func()`。

当 `--opd-alignment byte_chunk` 时，它不会把 student `sample.tokens` 作为 `input_ids` 发给 teacher，因为这些 token id 属于 student tokenizer。它会先调用 `_build_canonical_opd_texts()` 得到：

- `sample.opd_prompt_text`
- `sample.opd_response_text`
- `sample.opd_full_text`

然后用：

```python
payload["text"] = sample.opd_full_text
```

请求 teacher SGLang server 的 `/generate` 接口，让 teacher 按自己的 tokenizer 对同一段文本重新编码并返回 input token log-probs。

canonical text 的构建规则比较保守：

- 优先使用 sample 中已有的 rendered prompt / response / full text，以保留 chat template 标点、换行和 teacher 真实请求文本。
- response text 必须通过 student tokenizer 的 token-consistency 校验；如果 rendered response 与 `sample.tokens` 的 response token ids 不一致，则退回 token-derived response text。
- `prompt_text`、`response_text`、`full_text` 必须形成同一条自洽边界：
  - `full_text.startswith(prompt_text)`
  - `full_text == prompt_text + response_text`
- 如果 stored rendered text 在长样本或截断样本里发生边界漂移，则整体退回 student token-derived prompt / response / full text，避免 teacher byte clipping 使用错误 prompt 边界。

### 3. recorded student alignment payload

在发送 teacher 请求之前，`reward_func()` 会调用 `_record_student_opd_alignment()`，把 student response 的 byte 证据记录到 `Sample` 上：

- `opd_student_response_bytes`
- `opd_student_token_byte_spans`
- `opd_student_alignment_version`
- `opd_student_alignment_source`
- `opd_student_alignment_complete`
- `opd_student_alignment_validated`
- `opd_student_alignment_status`
- `opd_student_alignment_error`
- `opd_student_alignment_metadata`

当前主路径是 recorded payload first。训练侧和诊断脚本优先消费这份 payload，而不是在 loss 计算时重新猜 student response bytes。

recorded alignment 的构建流程大致是：

1. 从 `sample.tokens[-sample.response_length:]` 拿 student response token ids。
2. 使用 canonical `opd_response_text` 作为 response bytes 的事实文本。
3. 优先通过 token string / tokenizer 语义构造每个 response token 覆盖的 byte span。
4. 必要时退到 full sequence + prompt boundary aware 的构造方式，避免只看 response 片段时丢失上下文。
5. 生成 gap-free、完整覆盖 response bytes 的 spans。
6. 把 bytes 和 spans 记录到 `Sample`，并标记 `ok_recorded` 或失败原因。

这一步的目标是把 student 侧“真实 response bytes 与 token spans”固定下来。后续训练只验证和消费，不再把临时 reconstruction 作为主路径。

### 4. generation byte evidence 的定位

`Sample` 里还保留了：

- `opd_student_token_texts`
- `opd_generation_byte_evidence_attempted`
- `opd_generation_byte_evidence_complete`
- `opd_generation_byte_evidence_validated`
- `opd_generation_byte_evidence_error`
- `opd_generation_byte_evidence_metadata`

这些字段用于观测 generation 侧是否提供了 token text 证据。但当前实现不会把 SGLang 返回的 per-token text 当作唯一权威 byte evidence。

原因是它更像 `decode([token_id])` 的文本片段，在 SentencePiece / LLaMA 风格词表下可能丢失前导空格语义，长响应里也可能不完整。因此它主要用于 observability；真正训练主路径仍然依赖 recorded builder 生成并校验的 `opd_student_response_bytes` 与 `opd_student_token_byte_spans`。

### 5. teacher log-probs 裁剪

SGLang teacher 返回的 `input_token_logprobs` 覆盖整段 `opd_full_text`。`compute_teacher_log_probs_for_sample()` 在 byte chunk 模式下会调用 `_align_teacher_reward_log_probs()`：

1. 用 teacher tokenizer 本地 `encode_text(teacher_tokenizer, full_text)` 得到 teacher token ids。
2. 用 `build_token_byte_spans()` 为 teacher full text token 构造 byte spans。
3. 计算 prompt byte 长度：

```python
prompt_byte_length = len(prompt_text.encode("utf-8"))
```

4. 用 `clip_token_bytes_by_region(..., start_byte=prompt_byte_length)` 裁出 teacher response token bytes 与 response token indices。
5. 从 SGLang 返回的 reward log-probs 里取出 response 区间的 teacher log-probs。

这里 teacher 侧有额外兼容逻辑：如果 SGLang 返回的 token id 序列与本地 teacher tokenizer encode 结果存在一个额外 leading / trailing token，或本地 encode 序列是 reward token ids 的连续子序列，会尝试 trim 后对齐；如果仍无法对齐，则退到基于 reward token ids 自身 byte spans 的裁剪；再失败才使用 sequence mean 生成与 student response length 相同的 teacher log-probs。

注意：这是 teacher reward extraction 阶段的兼容路径。进入训练侧 `compute_byte_chunk_aligned_log_probs()` 后，当前 byte chunk OPD 已经要求严格 byte chunk 对齐；`allow_sequence_fallback` 参数还保留在函数签名里，但失败时实际会抛出 `ValueError`，不会静默降级为 sequence-level OPD。

### 6. rollout 到训练侧的数据传输

`slime/ray/rollout.py` 的 `_convert_samples_to_train_data()` 会把 `Sample` 上的 OPD 字段转成 train data：

- `teacher_log_probs`
- `opd_prompt_texts`
- `opd_response_texts`
- `opd_full_texts`
- `opd_student_response_bytes_list`
- `opd_student_token_byte_spans_list`
- `opd_student_alignment_*_list`
- `opd_generation_byte_evidence_*_list`

随后 `_split_train_data_by_dp()` 按 data parallel partition 分发训练侧需要消费的字段，其中包括 `teacher_log_probs`、canonical text、recorded student alignment bytes / spans，以及 `opd_student_alignment_*_list`。训练侧通过 `get_student_alignment_evidence_from_rollout_data()` 按 sample index 取回 recorded payload。`opd_generation_byte_evidence_*_list` 当前主要用于 rollout 侧观测汇总，不是 loss 计算的必要输入。

## loss 计算方式

### 1. OPD 应用位置

OPD 不直接替换 PPO / GRPO / GSPO 等基础 advantage estimator，而是在 `slime/backends/megatron_utils/loss.py` 的 `compute_advantages_and_returns()` 中调用：

```python
apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs)
```

最终修改每个 response token 的 advantage：

```python
advantages[i] = adv - args.opd_kl_coef * reverse_kl
```

其中 `reverse_kl` 在 token 对齐模式下是逐 token：

```python
reverse_kl = student_log_probs[i] - teacher_log_probs[i]
```

在 `byte_chunk` 模式下则由 `compute_byte_chunk_reverse_kl()` 计算。

### 2. byte chunk 对齐

`compute_byte_chunk_aligned_log_probs()` 的输入包括：

- canonical `full_text` / `prompt_text`
- student full token ids
- response token count
- student response log-probs
- teacher response log-probs
- student tokenizer / teacher tokenizer
- recorded student alignment evidence

函数先取得 student response token bytes：

1. 如果 `student_alignment_evidence` 存在，则检查：
   - `complete` 不能为 `False`
   - `validated` 不能为 `False`
   - `response_bytes` 与 `token_byte_spans` 必须齐全
2. 调用 `validate_recorded_student_response_alignment()` 验证：
   - recorded response bytes 等于 `response_text.encode("utf-8")`
   - span 数等于 response token 数
   - spans 单调、gap-free
   - spans 不超过 response bytes 长度
   - 最后一个 span 覆盖到 response bytes 末尾
3. 如果没有 recorded payload，才走兼容 reconstruction：
   - `build_contextual_suffix_token_bytes()`
   - `build_token_byte_spans()`

teacher 侧则重新 encode `full_text`，构造 full token byte spans，再按 `prompt_byte_length` 裁出 response bytes 与 response token indices，并要求裁出的 teacher response token 数与 `teacher_log_probs.numel()` 一致。

随后 `align_token_byte_chunks(student_response_bytes, teacher_response_bytes)` 使用双指针把两边 token bytes 聚成最小公共 chunk：

```text
student tokens: [b"<think>"]
teacher tokens: [b"<", b"think", b">"]
chunk:          b"<think>"
```

或：

```text
student tokens: [b"<", b"think", b">"]
teacher tokens: [b"<think>"]
chunk:          b"<think>"
```

对齐成功后，每个 chunk 返回：

- student token slice
- teacher token slice
- chunk bytes

### 3. chunk log-prob 聚合与分摊

对每个公共 chunk，loss 侧会先在 chunk 内求和：

```python
student_chunk_log_prob = student_log_probs[student_slice].sum()
teacher_chunk_log_prob = teacher_log_probs[teacher_slice].sum()
```

然后按 chunk 覆盖的 student token 数平均分摊回 student token 位置：

```python
student_chunk_length = student_slice.stop - student_slice.start
student_chunk_log_probs[student_slice] = student_chunk_log_prob / student_chunk_length
teacher_chunk_log_probs[student_slice] = teacher_chunk_log_prob / student_chunk_length
```

最后：

```python
reverse_kl = student_chunk_log_probs - teacher_chunk_log_probs
```

这有两个重要效果：

- 对齐单位是语义上同一段 bytes，而不是 tokenizer 边界。
- 输出仍然是 student response token 维度，可以直接复用现有 advantage / loss pipeline。

举例：

```text
student: ["<", "think", ">"] logp = [-0.1, -0.2, -0.3]
teacher: ["<think>"]         logp = [-0.5]

student chunk sum = -0.6
teacher chunk sum = -0.5
student chunk length = 3

student_chunk_log_probs = [-0.2, -0.2, -0.2]
teacher_chunk_log_probs = [-0.1666667, -0.1666667, -0.1666667]
reverse_kl = [-0.0333333, -0.0333333, -0.0333333]
```

## 边界情况处理

### 1. 空 response

如果 `sample.response_length <= 0`，`_record_student_opd_alignment()` 会记录：

- `opd_student_response_bytes = []`
- `opd_student_token_byte_spans = []`
- `opd_student_alignment_status = "ok_recorded"`
- `opd_student_alignment_complete = True`
- `opd_student_alignment_validated = True`
- metadata 中 `evidence_kind = "empty_response"`

`compute_byte_chunk_aligned_log_probs()` 对 `response_token_count == 0` 返回空 tensor。

### 2. canonical prompt / response / full 边界漂移

长样本、截断样本或 chat template 渲染差异可能导致 stored text 与 `sample.tokens` 不是同一条 token chain。当前处理是：

- response rendered text 不可被 student tokenizer encode 回 response token ids 时，退回 decoded response token text。
- `full_text` 与 `prompt_text + response_text` 不一致时，整体退回 decoded prompt / response / full text。
- teacher byte slicing 永远使用同一套 canonical prompt / response / full 边界。

这个约束避免了典型错误：student alignment 用一种 response 边界，teacher clipping 用另一种 prompt byte length。

### 3. recorded payload 缺失、不完整或无效

训练侧会区分这些情况：

- `student_alignment_evidence_incomplete`
- `student_alignment_evidence_invalid`
- `student_recorded_alignment_incomplete`
- `student_recorded_alignment_failed`
- `student_byte_reconstruction_failed`

当前 byte chunk 主路径失败时会抛错，不会静默跳过 OPD KL。这样做是为了让跨词表 OPD 的数据协议问题尽早暴露，而不是用 sequence mean 把错误掩盖掉。

### 4. tokenizer offset mapping 异常

`build_token_byte_spans()` 的重建顺序是：

1. 优先使用 tokenizer 的 `return_offsets_mapping=True`。
2. 如果 offset mapping 不可用、不单调、数量不匹配、无法重建原文 bytes，则尝试 `convert_ids_to_tokens()` 的 token string fallback。
3. 允许 prefix decode fallback 的调用点还可以退到 prefix decode；strict 调用点则直接失败。

token string fallback 额外处理了几类 tokenizer 语义：

- SentencePiece 的 `▁` 按空格 byte `0x20` 处理。
- 特殊 token 边界上的 standalone `▁` 可以匹配空 bytes。
- GPT-2 byte-level unicode mapping 通过 `bytes_to_unicode()` 反向恢复。
- `<0xAB>` 形式的 hex byte token 可以按 raw byte 匹配。
- 孤立的非 ASCII hex byte token如果在 canonical rendered text 中表现为 Unicode replacement character，则允许匹配 `EF BF BD`。

### 5. 非法 byte token

真实长响应中可能出现类似 `<0x86>` 的孤立 UTF-8 continuation byte。tokenizer 渲染时会把它变成 `�`，即 UTF-8 bytes `EF BF BD`。

当前 token string fallback 的规则是：

- 合法连续 hex byte token run 仍然按 raw bytes 重建。
- 孤立非法 byte token 可以匹配 replacement character bytes。
- 不把所有 `>=0x80` byte token 都无条件映射为 replacement character，避免破坏合法 UTF-8 byte 序列。

这类处理发生在 recorded student alignment 构建阶段。只要最终 token bytes 能完整重建 canonical response bytes，训练侧就能继续严格 byte chunk 对齐。

### 6. teacher token ids 与本地 encode 不一致

SGLang 返回的 `input_token_logprobs` 可能和本地 teacher tokenizer encode 存在小差异。`_align_teacher_reward_log_probs()` 会尝试：

- 去掉一个额外 leading token；
- 去掉一个额外 trailing token；
- 在 reward token ids 中查找本地 encode 序列的连续子序列；
- 如果 token id 无法对齐，但 reward token ids 能重建 byte spans，则直接用 reward token ids 做 response byte clipping；
- 最后才用 teacher sequence mean 生成与 student response length 相同的 log-probs。

不过进入 loss 侧之后，teacher response token 数必须与 `teacher_log_probs` 长度一致，否则会抛出 `teacher_byte_reconstruction_failed`。

### 7. teacher / student bytes 无法完全对齐

`align_token_byte_chunks()` 要求 student response bytes 与 teacher response bytes 完全一致，并能切成公共 chunks。典型失败包括：

- student bytes 先结束；
- teacher bytes 先结束；
- 两边无法形成相等非空 chunk；
- 对齐后任一侧还有剩余 token bytes。

这些会被包装为 `alignment_failed: ...` 并抛错。排查顺序应先看 canonical text 与 recorded payload，而不是直接修改 chunk 对齐器。

### 8. 指标和诊断

rollout 和训练日志会汇总 OPD alignment 状态：

- sample count
- error count / unique error count
- status 分布
- complete / validated 分布
- source 分布
- evidence kind 分布
- generation byte evidence hit / incomplete / invalid rate

诊断脚本应与训练语义一致：优先使用 recorded payload，再进入兼容 reconstruction。

常用命令：

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
./.venv/bin/pytest tests/test_student_logprob_alignment.py -v
./.venv/bin/pytest tests/test_analyze_opd_alignment.py -v
./.venv/bin/pytest tests/test_real_tokenizer_sampled_opd_alignment.py -v
```

回放 debug rollout：

```bash
./.venv/bin/python scripts/replay_debug_rollout_opd.py \
  --rollout-path /path/to/rollout.pt \
  --hf-checkpoint /path/to/student_hf \
  --teacher-hf-checkpoint /path/to/teacher_hf
```

## 默认超参数配置

OPD 参数定义在 `slime/utils/arguments.py`。当前默认值如下：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use-opd` | `False` | 默认不启用 OPD。 |
| `--opd-type` | `None` | 启用 `--use-opd` 后必须显式指定 `sglang` 或 `megatron`。 |
| `--opd-kl-coef` | `1.0` | OPD KL 惩罚系数。 |
| `--opd-alignment` | `token` | 默认仍为同词表 token 对齐；跨词表需要显式设为 `byte_chunk`。 |
| `--opd-teacher-load` | `None` | `--opd-type=megatron` 时必填。 |
| `--opd-teacher-ckpt-step` | `None` | 可选 teacher checkpoint step。 |
| `--opd-teacher-hf-checkpoint` | `None` | `--opd-alignment=byte_chunk` 时必填，用于加载 teacher tokenizer。 |
| `--opd-disable-sequence-fallback` | `False` | 历史兼容开关。当前 byte chunk 训练主路径已经严格失败抛错，不再静默 sequence fallback。 |

跨词表 SGLang teacher 的典型配置见 `scripts/run-yulan-cross-tokenizer-opd-train.sh`：

```bash
--advantage-estimator grpo
--use-opd
--opd-type sglang
--opd-alignment byte_chunk
--opd-kl-coef 1.0
--opd-teacher-hf-checkpoint "${TEACHER_MODEL_PATH}"
--entropy-coef 0.0
--eps-clip 0.2
--custom-rm-path slime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
--rm-url "http://127.0.0.1:${TEACHER_PORT}/generate"
```

teacher SGLang server 的典型启动方式：

```bash
python3 -m sglang.launch_server \
  --model-path "${TEACHER_MODEL_PATH}" \
  --host 127.0.0.1 \
  --port "${TEACHER_PORT}" \
  --tp 1 \
  --mem-fraction-static "${TEACHER_MEM_FRACTION_STATIC}"
```

如果要运行 YuLan student / Nanbeige teacher 的跨词表 OPD，当前脚本还会设置：

```bash
export SLIME_SGLANG_EXTERNAL_MODEL_PACKAGE=sglang_qwen3_next_plugin
```

这属于具体模型后端适配，不是 byte chunk OPD 算法本身的必要条件。
