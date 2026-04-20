# 跨词表 OPD 对齐加固

## 背景

当前 `slime` 的跨词表 OPD 对齐，核心目标是让 student tokenizer 和 teacher tokenizer 在同一条响应文本上，按 UTF-8 字节块建立稳定的 chunk 对齐关系。  
此前实现中，分析侧曾依赖 prefix decode 去逐 token 重建 response bytes，这在 `YuLan-Mini-Nanbeige-Distill` 这类 tokenizer 上会遇到前缀不一致问题，导致少量样本无法完成严格重建。

我们已经确认：

- 问题不是双指针 chunk 对齐器本身
- 问题出在 `token -> bytes` 的重建基准不稳定
- 训练侧和分析侧都必须共用同一套重建逻辑，否则会出现解释不一致

## 目标

1. 让跨词表 OPD 的 chunk 对齐基准不再依赖 prefix decode。
2. 让分析脚本在真实数据上可稳定跑通 512 条样本。
3. 让训练侧继续使用 byte-chunk OPD，但不削弱现有 fallback 保护。
4. 保持 student / teacher 的响应边界与同一份 rendered 文本一致。

## 现状

当前 chunk 对齐链路可以拆成两层：

1. `token -> bytes` 重建
2. `student bytes` 与 `teacher bytes` 的双指针 chunk 对齐

其中第二层已经是稳定的 byte 级双指针逻辑，不需要改。

真正需要加固的是第一层。此前分析侧曾使用：

- `decode(token_ids[:i])` 的 prefix 方式逐步补齐

这会在某些 Unicode / 标点 / 空白边界上失稳，例如：

- `❗`
- `✓`
- `✔`
- `\u2003`

这类字符会导致前缀 decode 的结果在后续 token 到来时改写历史文本，从而破坏“前缀一致性”假设。

## 加固方案

### 1. 统一用 rendered 文本作为基准

分析侧不再用 `decode(full_token_ids)` 作为全文本真值，而是直接使用：

- `rendered_full_text`
- `rendered_prompt_text`

这样 student 和 teacher 都在同一份渲染文本上做字节重建与边界裁切。

### 2. 改成 strict 的整段重建

`build_contextual_suffix_token_bytes(...)` 不再逐 token prefix decode，而是：

1. 先对完整 `full_text` 做一次性 tokenization
2. 通过 `offset_mapping` 还原每个 token 的 byte span
3. 若 `offset_mapping` 不可用，再退到 `convert_ids_to_tokens()` 还原
4. 仅在最后一层保留 prefix decode 作为兼容兜底

这样可以把“前缀不一致”的风险从主路径中移除。

### 3. 允许 prompt / response 边界切进 token

有些样本的 prompt 边界会落在一个 token 内部。  
这类样本不能因为边界切分就失败，正确处理方式是：

- 把 token 按 byte span 裁切
- 只保留 response 区域对应的 suffix bytes

这样既保留 strict 重建，又不牺牲真实数据覆盖率。

### 4. 保留训练侧 fallback

训练侧 byte-chunk OPD 仍然保留 sequence-level fallback：

- student 字节重建失败
- teacher 字节重建失败
- byte chunk 对齐失败

这些情况下仍会回退到 sequence penalty。  
本次加固不删除训练侧回退，只是让分析与训练共享的重建基准更加稳定。

## 影响范围

### 受影响的模块

- `slime/utils/opd_utils.py`
- `scripts/analyze_opd_alignment.py`
- 相关回归测试

### 不改动的部分

- `align_token_byte_chunks(...)` 的双指针逻辑
- 训练侧的 sequence fallback
- OPD 的损失形式和训练目标

## 验证结果

在真实数据 `OpenMathInstruct-2/correct/shards` 上，使用：

- Student model: `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`
- Teacher model: `/mnt/hdd/Nanbeige4.1-3B`

对前 512 条样本进行压测后，最终结果是：

- `ok: 512`
- `sample_error: 0`

这说明新的重建基准已经能够覆盖此前失败的边界样本。

## 后续建议

1. 把这条路径继续固化成回归测试，避免未来重新引入 prefix decode 依赖。
2. 如果后续要进一步提高鲁棒性，可以考虑继续减少对 tokenizer `legacy` 行为的依赖。
3. 若训练侧要做更严格的可解释性验证，可以再单独加一组“边界切进 token”的样本集。

## 结论

跨词表 OPD 的真正加固点，不在 chunk 双指针本身，而在 `token -> bytes` 的重建基准。  
只要把基准统一到 rendered 文本 + strict byte span 重建，就可以在保留训练 fallback 的同时，让分析侧稳定达到 512 条 100% 成功。
