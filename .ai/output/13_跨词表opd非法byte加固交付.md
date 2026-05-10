# 13 跨词表 OPD 非法 byte token 加固交付

## 背景

本轮问题来自真实训练日志：

- 日志：`workdirs/math_lighteval/logs/run.log`
- debug rollout：`/mnt/hdd/lvzhihao/math_lighteval/debug_rollouts/rollout_58.pt`
- 失败样本：`index=7725`，保存列表里的 `nth=69`

训练在 rollout 58 处失败，核心错误是：

```text
student_alignment_evidence_incomplete:
Tokenizer token strings do not reconstruct the source text.
token_index=8030 token_piece='<0x86>' byte_offset=25632
```

这个样本的 response 很长：

```text
response_length = 20480
bad token index = 8030
tokens after bad token = 12449
```

因此它不是“截断尾部半个 UTF-8 字符”的问题。坏 token 位于 response 中间，不能通过丢弃尾部 token 来合理解决。

## 根因

跨词表 OPD 的 byte-chunk 对齐要求 student 和 teacher 都落在同一份 canonical rendered text 的 UTF-8 byte 空间里。

本次失败点的实际情况是：

```text
student token piece: <0x86>
token raw byte:      86
tokenizer rendered:  �
rendered UTF-8:      EF BF BD
```

`0x86` 是一个不能独立构成合法 UTF-8 字符的 continuation byte。detokenizer 将它渲染成 Unicode replacement character `�`。但是原先 token-piece 主路径只把 `<0x86>` 解释成 raw byte `86`，于是无法匹配 canonical response text 当前位置的 `EF BF BD`。

这说明问题不在 `align_token_byte_chunks(...)` 的双指针对齐器，而在更上游的 token-piece 到 rendered byte span 的解释规则。

## 设计取舍

本轮坚持以下约束：

1. 不过滤样本。
2. 不在训练端跳过 OPD KL。
3. 不启用 sequence fallback。
4. 不把 SGLang `generation token text` 提升为生产级 byte evidence。
5. 不把 `decode([token_id])` 作为普通 token 的通用强证据。
6. 不对 `<0x86>` 写单点特判。

这与此前硬清理后的协议保持一致：

- 主路径仍然是 recorded alignment。
- 训练和 replay 仍然消费结构化 payload。
- generation token text 仍只作为 observability。
- byte-chunk 主路径失败仍显式暴露，不静默降级。

本轮修复的是现有 token-piece 主路径中的解释规则：

> 当 token piece 是孤立的非法 hex byte token 时，允许它匹配 canonical rendered text 中的 UTF-8 replacement character bytes。

也就是：

```text
isolated invalid byte token <0x86> -> EF BF BD
```

同时仍保留原有 raw byte 匹配：

```text
<0xE2><0x9E><0xA1> -> E2 9E A1
```

## 代码改动

### `slime/utils/opd_utils.py`

新增了三个小 helper：

- `_hex_byte_token_value(...)`
- `_is_invalid_standalone_utf8_byte(...)`
- `_replacement_candidate_for_isolated_invalid_byte_token(...)`

主逻辑仍在 `_build_token_byte_spans_via_token_strings(...)` 中：

- 先使用原有 `_token_piece_candidates(...)` 做正常 token-piece byte 匹配。
- 仅当当前 token piece 是孤立非法 hex byte token 时，额外加入 `EF BF BD` candidate。
- 匹配后仍要求最终 token bytes 完整重建 `response_text.encode("utf-8")`。

为了避免破坏尾部截断处理，本轮没有把所有 `>=0x80` 的 byte token 都映射到 replacement char。

特别是这类尾部半截 UTF-8：

```text
<0xE2><0x9F>
```

仍然会保持失败，从而继续触发已有的 truncated suffix trim 逻辑。

### `tests/test_opd_byte_chunk.py`

新增回归测试：

```text
test_build_recorded_student_response_alignment_from_token_ids_maps_invalid_hex_byte_to_replacement_char
```

覆盖最小复现：

```text
response_text = "A�B"
token pieces = ["A", "<0x86>", "B"]
expected spans = [(0, 1), (1, 4), (4, 5)]
```

同时保留并验证已有合法 hex byte token case：

```text
<0xE2><0x9E><0xA1><0xEF><0xB8><0x8F> -> ➡️
```

## 真实样本验证

对原始失败样本重新构造 recorded alignment：

```text
sample_index = 7725
span_count = 20480
bad_token_span = (25632, 25635)
bad_token_bytes_hex = efbfbd
ok = True
```

这说明第 8030 个 response token 现在被严格映射到 canonical response text 中的 `�`，并且整条 response 的 token spans 仍然是 gap-free、完整覆盖的。

进一步把修复后的 payload 注入样本，并走训练入口 helper：

```text
teacher_log_probs = 18894
student_chunk_log_probs = 20480
teacher_chunk_log_probs = 20480
ok = True
```

这说明该样本不只是能重建 recorded payload，也能进入 `prepare_byte_chunk_training_entry(...)` 完成 byte-chunk OPD 前置准备。

## 验证命令

本轮执行过：

```bash
./.venv/bin/pytest tests/test_opd_byte_chunk.py -v
```

结果：

```text
57 passed
```

执行过：

```bash
./.venv/bin/pytest \
  tests/test_cross_tokenizer_opd_training_entry.py \
  tests/test_student_logprob_alignment.py \
  tests/test_analyze_opd_alignment.py \
  -v
```

结果：

```text
10 passed
```

执行过：

```bash
./.venv/bin/python -m py_compile slime/utils/opd_utils.py
```

结果：通过。

## 当前结论

本轮修复没有改变跨词表 OPD 的主架构，也没有引入新的训练端 fallback。

它补齐的是 token-piece 主路径里的一个边界语义：

- 合法 UTF-8 byte token run 仍按 raw bytes 重建。
- 孤立非法 byte token 可以按 tokenizer rendered text 中的 replacement character 重建。
- 尾部未完成 UTF-8 run 仍然保持失败，并继续交给已有 truncated suffix trim 逻辑处理。

因此这次加固与此前 recorded-alignment-first、no-sequence-fallback、generation-text-observability-only 的方向是一致的。
