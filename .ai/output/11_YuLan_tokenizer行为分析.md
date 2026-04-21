# 11 YuLan Tokenizer 行为分析

## 目标

在继续修跨词表 OPD 之前，先把 `YuLan-Mini-Nanbeige-Distill` tokenizer 的关键行为摸清楚，尤其是：

- `encode -> decode` 是否稳定
- `convert_ids_to_tokens()` 暴露的 token piece 语义是什么
- SGLang 当前返回的 `output_token_logprobs[*][2]` 与 tokenizer 原始 token piece 的差异是什么
- 哪些差异会直接导致 `generation_byte_evidence` 无效

## 结论摘要

结论非常明确：

1. `YuLan` tokenizer 自身的 `encode -> decode` 在常见文本上基本是稳定的。
2. `YuLan` token piece 高度依赖 `▁` 作为“前导空格 / 词边界”标记。
3. 当前 SGLang 返回的 generation token text 会系统性丢失 `▁`。
4. 一旦 `▁` 被丢失，token-level text 就不再是可逐 token byte 拼接的真实证据。
5. 因此当前 `generation_byte_evidence` 在 YuLan 路径下大量 invalid 是预期现象，不是统计 bug。

一句话总结：

> 对 YuLan 而言，真正关键的是 tokenizer 的原始 token piece surface；当前 SGLang 暴露出来的是去掉边界语义后的 detokenized text，不能直接当 byte evidence。

---

## 一、基础 tokenizer 行为

使用模型：

- `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`

做了一组基础实验，观察 `encode_text()`、`decode_token_ids()` 与 `convert_ids_to_tokens()`。

### 1. 普通英文空格

文本：

```text
'Hello world'
```

token pieces：

```text
['▁Hello', '▁world']
```

说明：

- `▁Hello` 表示 token 本身带有前导空格语义
- `decode` 后文本正常恢复

### 2. 双空格

文本：

```text
'A  B'
```

token pieces：

```text
['▁A', '▁▁', 'B']
```

说明：

- 双空格会被显式编码成独立 token piece `▁▁`
- 空格信息不是“隐含的”，而是 tokenizer surface 的一部分

### 3. 引号前空格

文本：

```text
'"Determine'
```

token pieces：

```text
['▁"', 'Det', 'ermine']
```

说明：

- 即使是引号，前导空格也编码在 `▁"` 中
- 不能把 `▁"` 简单看成 `"`, 这是两个不同的 surface

### 4. LaTeX / 符号

文本：

```text
'\\(a, b, c\\)'
```

token pieces：

```text
['▁\\', '(', 'a', ',', '▁b', ',', '▁c', '\\', ')']
```

说明：

- 标点和 LaTeX 结构周围的空格仍然通过 `▁` 表达
- `b` / `c` 前面的空格语义不会自动推断，必须靠 token piece 保留

### 5. 结论

基础实验说明：

- `convert_ids_to_tokens()` 给出的 piece 才是 YuLan tokenizer 的真实 token surface
- `▁` 是一等公民，不是可忽略装饰
- 只要丢掉 `▁`，就会破坏 token piece 对原始 byte 序列的可恢复性

---

## 二、真实 rollout 样本中的对比

分析文件：

- `/mnt/hdd/lvzhihao/slime_opd_train_workdir_gbe_metrics_train_smoke_v2/debug_rollouts/rollout_0.pt`

对比了三种东西：

1. `sample.tokens[-response_length:]`
2. `tokenizer.convert_ids_to_tokens(response_ids)`
3. `sample.opd_student_token_texts`

### 1. 真实样本的共同现象

对 6 个样本都成立：

- `sample.response == decode(sampled_response_ids)` 为真
- `encode(sample.response) == sampled_response_ids` 为假
- `convert_ids_to_tokens()` 与 `sample.opd_student_token_texts` 从很前面就开始分叉

### 2. 典型对比

真实样本头部大致如下：

`convert_ids_to_tokens()`：

```text
['We', '▁are', '▁asked', ':', '▁"', 'Det', 'ermine', '▁whether', ...]
```

SGLang generation token text：

```text
['We', 'are', 'asked', ':', '"', 'Det', 'ermine', 'whether', ...]
```

从第 1 个 token 就开始差异：

- `▁are` -> `are`
- `▁asked` -> `asked`
- `▁"` -> `"`
- `▁whether` -> `whether`

这不是个别例子，而是普遍现象。

### 3. 统计结果

在这份 `rollout_0.pt` 的 6 个样本里：

- 主要差异模式全部都是 `space_marker_dropped`
- 共计观察到 `1565` 个这类差异

每个样本的差异规模：

- sample 0: `257`
- sample 1: `248`
- sample 2: `227`
- sample 3: `249`
- sample 4: `299`
- sample 5: `285`

这说明：

- 不是零星 edge case
- 而是 generation token text 的系统性语义降级

---

## 三、为什么这会直接导致 generation evidence invalid

当前 generation evidence 的核心假设是：

```python
token_bytes = [token_text.encode("utf-8") for token_text in response_token_texts]
b"".join(token_bytes) == response_text.encode("utf-8")
```

但对 YuLan 来说：

- `▁are` 的 surface 语义不是 `are`
- `▁"` 不是 `"`
- `▁b` 不是 `b`
- `▁▁` 不是空字符串

所以如果 SGLang 返回的是去掉 `▁` 之后的 text：

```text
['We', 'are', 'asked', ':', '"', ...]
```

直接拼接得到的是：

```text
Weareasked:"...
```

而不是：

```text
We are asked: "...
```

因此 `generation_byte_evidence` 必然 invalid。

这正是当前真实日志里看到：

- `source={'recorded_builder': 6}`
- `generation_byte_evidence={hit_rate=1.0000, incomplete_rate=1.0000, invalid_rate=1.0000}`

的直接原因。

---

## 四、这说明当前真正的问题是什么

当前问题不是：

- tokenizer decode 完全坏了
- 训练侧统计错了
- rollout 数据没存下来

当前真正的问题是：

> 我们把 SGLang 返回的“去空格边界后的 detokenized token text”误当成了“可逐 token byte 拼接的原始 token evidence”。

对于 YuLan，这个假设不成立。

---

## 五、对后续实现的直接启发

### 1. 不要再直接用 `sample.opd_student_token_texts` 构造强 byte evidence

这组字段最多只能算弱提示，不能算强证据。

### 2. 真正有价值的是 tokenizer 原始 token piece

后续如果想做 generation-native evidence，优先级应当是：

1. generation 直接返回原始 token piece
2. 或 generation 返回 token ids，由本地 `convert_ids_to_tokens()` 恢复
3. 最差也要拿到不会丢失 `▁` 的 surface

### 3. 对 YuLan 要做 tokenizer-aware 处理

后续修复不应该继续假设：

- token text 直接拼接即可
- 只要 `response == decode(ids)` 就能重建每个 token 的 byte piece

而应该明确处理：

- `▁xxx`
- `▁▁`
- `▁"`
- 首 token 无 `▁`
- 被拆开的子词，如 `Det` + `ermine`
- 纯空格 / 空 surface 位置

---

## 六、当前最推荐的技术方向

### 短期

继续把当前 generation text 路径当兼容路径，只做观测，不做强 byte evidence。

### 中期

基于 `token ids -> convert_ids_to_tokens()` 做一版 YuLan-aware token piece 到 byte piece 的重建实验。

### 长期

推动 generation 侧交付真正的 byte-level evidence 协议，而不是依赖 detokenized token text。

---

## 七、最终结论

如果想真正把跨词表 OPD 调通，必须先承认一件事：

> 对 YuLan，tokenizer 的原始 token piece surface 才是对齐真相；丢掉 `▁` 以后，generation token text 已经不再是可靠的 byte evidence。

因此，后续正确方向不是继续 patch `generation_logprobs_text`，而是：

1. 以 YuLan tokenizer piece 语义为中心重建理解
2. 让 generation 交付更原始、更权威的 token/byte 证据
3. 再决定 OPD 对齐该如何长期收敛到干净链路
