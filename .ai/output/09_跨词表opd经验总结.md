# 09 跨词表 OPD 经验总结

## 目的

这份文档用于沉淀本轮跨词表 OPD 调试、修复、重构和 smoke 验证之后的阶段性经验，重点回答四件事：

- 这次我们到底学到了什么
- 现有方案哪些部分已经足够可靠
- 现有方案哪些部分仍然只是过渡方案
- 如果要走向长期可维护架构，下一步最该做什么

## 一句话结论

当前这版跨词表 OPD 已经从“脆弱且难排障”推进到了“基本可用、可验证、可定位问题”的状态，但它还不是长期最优维护架构。

更具体地说：

- 短中期：这版值得继续用，因为它明显提升了鲁棒性和诊断能力。
- 长期：最推荐的方向仍然是 generation-native byte evidence first，也就是让 rollout / generation 侧直接产出权威的 response bytes 和 token byte spans，训练侧只做消费和校验，不再自己猜。

## 本轮最重要的经验

## 1. 问题根因不在“chunk 对齐算法本身”

最开始看起来，很多问题都像是 `align_token_byte_chunks()` 不够鲁棒。

但实际排下来，真正更常见的根因在更上游：

- canonical text 选择不稳定
- student response bytes 的事实源不唯一
- tokenizer offset / token string / prefix decode 在 cross-tokenizer 下不总一致
- generation 侧给出的 token text 证据并不总是权威且完整

经验是：

如果没有先把“student 端到底以哪份文本 / 字节为准”这件事钉死，后面再怎么修 byte-chunk 对齐都只是表面补丁。

## 2. recorded payload 优先是正确方向

本轮一个已经被验证为正确的方向是：

- 在 sample 准备阶段记录 student response bytes / token byte spans
- 训练、诊断、回放统一优先消费这份 recorded payload
- 不再把“训练阶段临时重建 student bytes”当作主路径

这个改变的价值非常大，因为它把问题从：

- “训练时重新猜一次 student 对齐”

变成了：

- “训练时验证并消费 rollout 阶段已经记录好的对齐证据”

这不仅更稳，也更容易定位问题。

## 3. canonical text 必须先做一致性校验

此前一类真实故障来自：

- rendered response 和 `sample.tokens` 重新 decode / re-encode 后并不一致

如果这时还强行把 rendered response 当 canonical text，后续所有对齐都会建立在错误前提上。

本轮确认的正确策略是：

- 先对 rendered response 做 token-consistency 校验
- 校验失败时退回 token-derived text

这一步不是“小修小补”，而是把 canonical text 的来源从“看起来像”改成了“先验一致”。

## 4. SentencePiece 空格语义是实际高频坑

这轮一个非常实在的收获是：

- `convert_ids_to_tokens()` 里的 SentencePiece 标记 `▁` 必须按空格字节 `0x20` 处理

否则很多明明可以恢复的长样本，会在 token-string fallback 里被错误打成 reconstruction failed。

这个修复虽然小，但收益很明显。此前一批真实 long smoke dump 里的失败样本，在修完这个点之后，已经能被当前路径救回。

## 5. SGLang 的 token text 不是长期权威事实源

这是本轮最关键的架构性认识之一。

虽然我们已经把 `return_text_in_logprobs=True` 接进来了，也能在不少样本里利用 `opd_student_token_texts` 改善对齐，但真实运行证明它并不等于“权威的 generation-native byte evidence”。

原因是它更像：

- `tokenizer.decode([id])`

而不是：

- generation 过程中真实提交给 detokenizer / text stream 的 byte-aligned token piece

这会带来两个问题：

- SentencePiece / LLaMA 风格词表下，单 token text 可能丢掉前导空格语义
- 长响应下，采集到的 token texts 还可能截断或不完整

因此它可以作为有价值的辅助证据，但不适合作为长期唯一事实源。

## 6. 长响应剩余问题已经收敛成“证据完整性问题”

当前剩下的难点，已经不再是最初那种混沌状态，而是逐渐收敛成一类更清晰的问题：

- generation 侧 token text 证据不完整或不可信
- tokenizer reconstruction 作为 fallback 也不总能兜住

例如真实 long smoke 里已经见到这样的复合失败原因：

```text
generation_logprobs_text_failed: Generation token texts do not match canonical response text bytes.; tokenizer_reconstruction_failed: Tokenizer token/offset reconstruction failed.
```

这其实是一个好现象，因为它说明问题已经从“到处都可能错”收敛成了“证据生产协议还不够强”。

## 7. 日志不能只靠 worker logger，自顶向下的观测链路必须统一

本轮另一个重要教训是：

- 单测里 logger 打出来，不代表真实 smoke 的主 `run.log` 就一定看得到

真实运行中，rollout / train / Ray actor / 主进程之间有多层日志边界。

因此长期维护里不能只想着“加几句 `logger.info`”，而要明确：

- 这条日志写在哪个进程
- 这个进程的日志会不会回流到主 `run.log`
- 这个指标是否还需要同步到 WandB 或结构化 metric

换句话说，日志链路本身也需要设计，而不是临时添加。

## 当前方案里已经比较可靠的部分

下面这些结论，基本可以认为已经坐实：

### 1. payload-first 的消费策略是正确的

训练、回放、诊断优先消费 recorded payload，这条路线已经被多轮单测、sampled regression 和 smoke 验证支持。

### 2. token-derived canonical text fallback 是必要的

只信 rendered response 会踩坑，先做一致性校验、失败时退回 token-derived text 是正确策略。

### 3. token-string fallback 需要理解 tokenizer 语义

尤其是 SentencePiece 空格标记，不处理就会误伤很多本来可恢复的样本。

### 4. 128 short + 128 long 的 sampled real-tokenizer regression 是很有价值的基线

这套验证标准明显比“挑几个 case 看看”更有说服力，应该保留。

## 当前方案里仍然不利于长期维护的部分

下面这些问题说明，当前方案还不是最终架构：

### 1. 事实源仍然不够单一

现在系统里仍然在几个事实源之间切换：

- rendered response
- token-derived canonical text
- generation token texts
- tokenizer reconstruction

虽然优先级已经清楚很多，但长期看事实源越多，维护成本越高。

### 2. generation 证据协议不够强

当前拿到的是“文本片段型证据”，不是“权威的 byte span 证据”。

这使得训练侧还需要额外做很多验证和补救。

### 3. fallback 链虽然可读了，但还是一条补救链

现在的逻辑已经比之前干净得多，但本质上还是：

- 先试 generation text
- 不行再试 tokenizer reconstruction
- 还不行再 fallback sequence-level

这条链可以工作，但不应该成为长期主干。

### 4. 日志和指标链路还没完全主进程化

真实 smoke 已经证明，worker logger 并不天然等于主 `run.log`。

如果这部分不统一，后面还是会出现“代码明明记录了，但线上看不到”的维护问题。

## 本轮验证结果对架构判断的影响

本轮 smoke 和测试结果给出的信号是：

### 正向信号

- 当前方案已经比旧方案明显稳
- 首轮 long smoke 可以做到 `ok_recorded=6/6`
- real-tokenizer sampled regression 的 `128 short + 128 long` 能通过
- 失败原因已经能分层表达，而不是全部糊成一个 reconstruction failed

### 负向信号

- 真实 long case 仍然可能出现 generation 证据不完整 / 不可信
- 新增日志并不会自动出现在主 `run.log`
- 说明“数据协议”和“观测协议”都还没有完全到位

所以架构判断没有变：

- 这版是过渡性正确方案
- 不是长期最终方案

## 长期最推荐的目标架构

长期我最推荐的形态是：

### 1. generation 侧直接产出 authoritative byte evidence

最好让 rollout / generation 侧直接记录：

- response bytes
- per-token byte spans
- 证据完整性标记
- 证据协议版本

训练侧不再自己从 text / token ids 反推主证据。

### 2. canonical text 退居到 debug / 展示用途

canonical text 仍然保留，但主要用于：

- 人类阅读
- debug
- teacher 请求构造时的校验辅助

而不是承担 student byte 对齐的核心职责。

### 3. 训练侧只做三件事

- 验证 recorded evidence 是否完整
- 消费 evidence 做 OPD 对齐
- 在 evidence 缺失时走明确、可监控的 fallback

### 4. 指标和日志围绕“证据状态”来设计

长期最应该看的不是“最后 fallback 了没”，而是：

- evidence 来源
- evidence 完整率
- evidence 校验通过率
- fallback 原因分布
- 长响应下的失败占比

## 后续最值得做的事

如果继续推进，我建议优先级如下：

### 1. 让 OPD 摘要日志真正进入主 `run.log`

这件事比继续堆更多 `logger.info` 更重要，因为没有统一观测链路，后续维护成本会一直偏高。

### 2. 设计 generation-native byte evidence 协议

这是当前最关键的长期任务。重点不是继续 patch reconstruction，而是让 generation 直接交付更权威的 byte-level 结果。

### 3. 保留并扩展真实环境回归基线

至少继续保留：

- `test_opd_byte_chunk.py`
- `test_real_tokenizer_sampled_opd_alignment.py`
- 小规模 long smoke

这样后续每次改协议，都能快速知道是在向长期正确架构前进，还是又退回到补丁式修复。

## 最终判断

本轮最大的成功，不是把所有问题都“修没了”，而是把系统从：

- 不清楚哪层在错

推进到了：

- 能明确区分 canonical text 问题
- 能区分 generation evidence 问题
- 能区分 tokenizer reconstruction 问题
- 能用真实语料和真实 smoke 验证这些判断

这意味着跨词表 OPD 现在已经有了继续工程化演进的基础。

但如果目标是长期可维护，那么最终还是要把“student 对齐证据”从推断物，变成 generation 侧直接产出的权威协议对象。
