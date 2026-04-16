# 04 跨词表 OPD 测试

## 目标

在当前 `slime` 代码上，使用真实 student / teacher / 数据集资源，把跨词表 OPD 路径实际跑起来，并确认：

- student 与 teacher tokenizer 不同的情况下，byte-chunk OPD 路径可用
- rollout -> teacher 打分 -> Megatron train -> save_model 整条链路可走通
- 补齐为真实运行暴露出的兼容性问题和测试

本次使用资源如下：

- Student model: `/mnt/hdd/lvzhihao/hf_models/Dist-mathcode10b-s1randg-sch1-CPT-200b-stage3-r640k-GDN2.9b-A7-12_20_21_23_46_48_49-sl32768bs128lr2e5-2e5/merged_10ckpts_iter_61984-hf_to_iter_71525-hf_mean`
- Teacher model: `/mnt/hdd/Nanbeige4.1-3B`
- Data: `/mnt/hdd/huanglisheng/train_data/G-OPD-Training-Data/DeepMath-103K/slime_style_train_data.jsonl`
- GPU 分配：train 使用 `4,5,6`，teacher 使用 `7`

## 本轮实际修改

### 1. rollout logging 修复

真实 smoke 首次进入训练后，`slime/backends/megatron_utils/data.py` 的 `log_rollout_data()` 因尝试对字符串列表求平均而报错：

```text
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

根因是 `opd_full_texts` 这类纯文本字段被误当成数值指标聚合。修复为：

- rollout metric 聚合时跳过 `opd_full_texts`
- 增加 `tests/test_rollout_logging.py`

### 2. qwen3_next bridge 修复

在 student HF checkpoint 转 torch_dist / mcore 过程中，`slime_plugins/mbridge/qwen3_next.py` 暴露了两个真实兼容性问题。

#### 2.1 MLP 参数映射缺失

报错：

```text
NotImplementedError: Unsupported parameter name: decoder.layers.*.mlp.linear_fc1.layer_norm_weight
```

修复：

- 增加 dense / MoE 场景下的 `_MLP_MAPPING`
- 增加 `_weight_name_mapping_mlp()` override
- 补 `tests/test_qwen3_next_bridge_mapping.py`

#### 2.2 QKV packing 假设过死

之前 `_weight_to_mcore_format()` 假设 checkpoint 一定是某种 gated packed QKV 结构，但本次 student 实际是标准 GQA 布局。验证到的 HF 权重形状例如：

```text
model.layers.12.self_attn.q_proj.weight (1920, 1920)
model.layers.12.self_attn.k_proj.weight (384, 1920)
model.layers.12.self_attn.v_proj.weight (384, 1920)
```

修复：

- `_weight_to_mcore_format()` 自适配标准 GQA 与 gated 两种布局
- 现有 `tests/test_qwen3_next_megatron_to_hf.py` 保持通过
- 新增 bridge 侧映射测试 `tests/test_qwen3_next_bridge_mapping.py`

## 验证命令

### 单测

执行：

```bash
source .venv/bin/activate
python3 -m pytest -q \
  tests/test_rollout_logging.py \
  tests/test_opd_byte_chunk.py \
  tests/test_qwen3_next_megatron_to_hf.py \
  tests/test_qwen3_next_bridge_mapping.py
```

结果：

```text
13 passed, 2 warnings in 22.63s
```

### 真实 smoke

执行脚本：

```bash
REF_LOAD=/mnt/ssd/lvzhihao/PostTrain/slime/.tmp/yulan_hybrid_gdn_dense_smoke_rl/yulan_hybrid_gdn_dense_torch_dist \
bash scripts/run-yulan-cross-tokenizer-opd-smoke.sh
```

说明：

- 实际使用脚本：`scripts/run-yulan-cross-tokenizer-opd-smoke.sh`
- 本次成功会话对应 Ray session：
  - `/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-15_13-33-46_294636_31467`

## 真实运行证据

### 1. rollout 已真实发生

来自：

- `.../session_2026-04-15_13-33-46_294636_31467/logs/worker-112e3abbe030fa7fe149cc1ffba5e9c9f36f769994f40b44485c167c-ffffffff-43758.err`

关键日志：

```text
sglang_rollout.py:435 - First rollout sample: [...]
rollout.py:610 - Final collected 6 samples from rollout to train
```

这说明 rollout 侧不仅启动成功，而且已经实际收集到 6 条样本进入训练。

### 2. teacher_log_probs 与 OPD 指标已被真正写入训练侧

来自：

- `.../session_2026-04-15_13-33-46_294636_31467/logs/worker-0b248793d8a589389f7f383f2b9c605ae838750976898fe1827d8e81-01000000-29416.err`

关键日志：

```text
data.py:208 - rollout 0: {
  'rollout/teacher_log_probs': -0.7705047925313314,
  'rollout/opd_reverse_kl': -15.745668411254883,
  'rollout/advantages': 15.745668411254883,
  ...
}
```

这说明：

- teacher 侧打分已接入真实 rollout
- 跨词表 OPD 的 reverse KL 已经完成计算
- 该 KL 已被用来改写 advantages，训练确实走到了 OPD 路径

### 3. Megatron train 已真实跑到 `step 0`

同一日志文件中可见：

```text
model.py:664 - step 0: {... 'train/step': 0}
```

这说明训练侧不是停在数据准备或前向前，而是已经完成了一次真实训练 step。

### 4. 模型已真实保存

同一日志文件中可见：

```text
timer.py:24 - Timer save_model start
timer.py:32 - Timer save_model end (elapsed: 28.0s)
```

另一个对应 `.out` 日志可见：

```text
saving checkpoint at iteration       0 to /mnt/ssd/lvzhihao/PostTrain/slime/.tmp/yulan_cross_tokenizer_opd/actor_ckpt in torch_dist format
successfully saved checkpoint from iteration       0 to /mnt/ssd/lvzhihao/PostTrain/slime/.tmp/yulan_cross_tokenizer_opd/actor_ckpt [ t 1/1, p 1/1 ]
```

### 5. driver 已正常结束

来自：

- `/mnt/ssd/lvzhihao/PostTrain/slime/.tmp/yulan_cross_tokenizer_opd/run.log`

关键日志：

```text
[driver] train(args) done
```

这说明不仅 Ray worker 内部到了 `step 0` 和 `save_model`，外层 smoke wrapper 也完成了正常收尾。

## 结论

这次已经把跨词表 OPD 路径在真实资源上调通，且证据链完整覆盖：

1. student checkpoint 转换成功
2. teacher 服务可用
3. rollout 成功
4. teacher_log_probs 成功回传
5. byte-chunk OPD reverse KL 成功计算
6. Megatron train 跑到 `step 0`
7. `save_model` 成功
8. driver 正常结束

因此当前可以认为：`slime` 的跨词表 OPD smoke 路径已经真实可跑，而不只是单测或静态实现层面的“看起来支持”。

在这条链路跑通之后，又补了一次 OPD chunk penalty 行为调整：

- 原实现是把同一个 byte chunk 的 reverse KL 完整广播给该 chunk 覆盖的每个 student token
- 现实现改为按 chunk 内 student token 数平均分摊
- 这样可以保证 chunk 总惩罚守恒，避免 student tokenizer 更碎时总罚项被放大

## 当前相关测试状态

本轮新老测试合并后的核心覆盖点：

- `tests/test_opd_byte_chunk.py`
  - byte-chunk 对齐与 reverse KL 的核心逻辑
  - 验证 chunk penalty 在 student chunk 内按长度平均分摊
- `tests/test_rollout_logging.py`
  - rollout logging 不再错误聚合文本字段
- `tests/test_qwen3_next_megatron_to_hf.py`
  - qwen3_next QKV 转换兼容标准 GQA 布局
- `tests/test_qwen3_next_bridge_mapping.py`
  - qwen3_next bridge 的 MLP 参数映射与转换路径

## 建议 commit message

如果只看最初 smoke 跑通那一轮修复，可以拆成两个 commit：

```text
fix(opd): skip byte-chunk text fields in rollout metric aggregation
```

```text
fix(qwen3-next): support dense mlp norm and adaptive qkv packing in mbridge
```

如果合并成一个更贴近 smoke 目标的提交，也可以写：

```text
fix(opd): unblock cross-tokenizer smoke run for yulan-mini
```

如果按当前代码状态一起提交，我更推荐：

```text
feat(opd,qwen3-next): add byte-chunk OPD alignment and fix qwen3-next bridge mapping
```

如果把后续的 chunk penalty 平均分摊单独拆出来，可以再补一个：

```text
fix(opd): average byte-chunk KL penalty across student tokens
```

## 相关文件

- 运行脚本：`scripts/run-yulan-cross-tokenizer-opd-smoke.sh`
- 报告：`.ai/output/04_跨词表OPD测试.md`
- 关键修复：
  - `slime/backends/megatron_utils/data.py`
  - `slime_plugins/mbridge/qwen3_next.py`
- 测试：
  - `tests/test_rollout_logging.py`
  - `tests/test_opd_byte_chunk.py`
  - `tests/test_qwen3_next_megatron_to_hf.py`
  - `tests/test_qwen3_next_bridge_mapping.py`
