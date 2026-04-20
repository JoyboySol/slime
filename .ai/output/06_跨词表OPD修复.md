# 06 跨词表 OPD 修复

## 背景

在真实跨词表 OPD 路径中，我们观察到一组明显异常的现象：

- `rollout/rollout_log_probs` 正常，约为 `-0.3`
- `rollout/log_probs` 异常，约为 `-11`
- `train/train_rollout_logprob_abs_diff` 极大，约为 `11+`
- `rollout/opd_reverse_kl` 被同步放大
- `train/grad_norm` 达到 `1e5` 量级

同样的 chunk 路径在 same-vocab 的 Qwen 实验上是正常的，因此问题更像是：

- 真实跨词表 + YuLan student 特定实现
- 而不是 OPD 指标公式本身错误

本次目标是定位根因，并把真实跨词表 smoke 路径恢复到可解释、可继续实验的状态。

## 现象

异常实验使用：

- Student model: `/mnt/hdd/lvzhihao/hf_models/YuLan-Mini-Nanbeige-Distill`
- Teacher model: `/mnt/hdd/Nanbeige4.1-3B`
- Data: `/mnt/hdd/lvzhihao/data/OpenMathInstruct-2/slime_smoke_1k.jsonl`

异常时的关键特征：

1. rollout 侧 student logprob 正常

例如真实日志中：

```text
rollout/rollout_log_probs = -0.335...
```

2. train 侧 student logprob 完全失真

```text
rollout/log_probs = -11.8...
train/train_rollout_logprob_abs_diff = 11.5...
```

3. `grad_norm` 被带到极大值

```text
train/grad_norm = 2e5 左右
```

这说明问题不在 teacher 单侧，而是在 train 侧 student 前向或 student/train 对齐链路。

## 排查过程

### 1. 先排除 rollout logprob 本身错误

对真实 `rollout_0.pt` 离线重算，使用 HF student 按训练侧相同 response window 重新计算 token logprob，结果与 rollout 保存值基本一致：

- `avg_rollout = -0.3354107738`
- `avg_hf_train_slice = -0.3347961307`
- `mean_abs_diff = 0.0066947942`

结论：

- rollout 侧 `rollout_log_probs` 是可信的
- response 切片逻辑本身没有出现 `11+` 量级的错位

### 2. 排除 checkpoint 转换主链错误

将 torch_dist checkpoint 导回 HF 后，与原始 HF student 对同一条真实样本比较，平均 logprob 近乎一致：

- original HF: `-0.335339`
- roundtrip HF: `-0.335346`
- rollout saved: `-0.335411`

结论：

- HF -> torch_dist -> HF 主转换链整体正确
- 问题不在 rollout 保存权重，也不在 bridge 主体输出头映射

### 3. 检查 train 侧 token 级 logprob

在训练 actor 中加入 `SLIME_DEBUG_LOGPROB_DUMP=1` 后，异常版本的 token 级对比显示：

- rollout 前几个 token 的 logprob 接近 `0`
- train 前几个 token 的 logprob 大量落在 `-6 ~ -16`

这说明：

- train 侧不是“差一点”
- 而是 student 前向语义已经显著偏离 HF / rollout

### 4. 缩小到 YuLan/Qwen3Next 线性注意力 wrapper

继续对照：

- [`slime_plugins/models/yulan_mini.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime_plugins/models/yulan_mini.py)
- [`slime_plugins/models/qwen3_next.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime_plugins/models/qwen3_next.py)
- HF checkpoint 中的 `modeling_qwen3_next.py`

发现一个关键差异：

- HF decoder layer 的 `input_layernorm` 使用的是 `LlamaRMSNorm`
- 本地线性注意力 wrapper 错误使用了 `Qwen3NextRMSNorm`

这会导致：

- HF checkpoint 的 `input_layernorm.weight` 被加载到错误语义的 norm 模块中
- 所有 linear-attention layer 的前向被系统性拉偏
- 进而造成 train 侧 `log_probs` 整体失真

## 根因

根因是：

- `slime_plugins/models/yulan_mini.py`
- `slime_plugins/models/qwen3_next.py`

中的线性注意力 wrapper，把 decoder 的 `input_layernorm` 实现错用了 `Qwen3NextRMSNorm`，而真实 HF 模型使用的是 `LlamaRMSNorm`。

这不是一个小数值差异，而是模块语义差异：

- `Qwen3NextRMSNorm`
- `LlamaRMSNorm`

对权重的解释方式不同，因此直接加载 HF 权重后，前向会系统性偏离。

## 修复内容

### 1. 修复 linear-attention wrapper 的 input layernorm

修改文件：

- [`slime_plugins/models/yulan_mini.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime_plugins/models/yulan_mini.py)
- [`slime_plugins/models/qwen3_next.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime_plugins/models/qwen3_next.py)

修复方式：

- 将线性注意力 wrapper 的 `input_layernorm` 从 `Qwen3NextRMSNorm` 改为 `LlamaRMSNorm`
- 使用 `transformers.models.llama.modeling_llama.LlamaRMSNorm`

### 2. 新增回归测试

新增测试文件：

- [`tests/test_linear_attention_input_layernorm.py`](/mnt/ssd/lvzhihao/PostTrain/slime/tests/test_linear_attention_input_layernorm.py)

测试目标：

- 验证 `yulan_mini.Attention` 使用 `LlamaRMSNorm`
- 验证 `qwen3_next.Attention` 使用 `LlamaRMSNorm`

这组测试在修复前失败，修复后通过。

## 其他已确认但非本次主因的修复

在本次定位过程中，还修过两类真实问题，它们不是这次 `11+ mismatch` 的最终根因，但属于必要的链路修补：

1. YuLan bridge 的 gated-qkv packing 修复

- 文件：[`slime_plugins/mbridge/yulan_mini.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime_plugins/mbridge/yulan_mini.py)
- 测试：[`tests/test_yulan_mini_bridge_mapping.py`](/mnt/ssd/lvzhihao/PostTrain/slime/tests/test_yulan_mini_bridge_mapping.py)

2. bridge 模式下 load 路径 fallback 修复

- 文件：[`slime/utils/load_path_utils.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/utils/load_path_utils.py)
- 文件：[`slime/utils/arguments.py`](/mnt/ssd/lvzhihao/PostTrain/slime/slime/utils/arguments.py)
- 测试：[`tests/test_load_path_utils.py`](/mnt/ssd/lvzhihao/PostTrain/slime/tests/test_load_path_utils.py)

## 测试验证

执行：

```bash
source .venv/bin/activate
pytest -q \
  tests/test_linear_attention_input_layernorm.py \
  tests/test_yulan_mini_bridge_mapping.py \
  tests/test_load_path_utils.py \
  tests/test_update_weight_model_name.py \
  tests/test_student_logprob_alignment.py
```

结果：

```text
12 passed
```

## 真实回归结果

真实回归运行对应 Ray session：

- `/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_20-29-06_738754_37752`

关键日志：

- rollout 回流成功：
  [`worker-8ad5...38602.err:37`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_20-29-06_738754_37752/logs/worker-8ad5c62bea6ec5bbe30c44f68cb07c20dd8c33282b4e985b75646e5c-ffffffff-38602.err#L37)
- train 开始消费：
  [`worker-8630...53731.err:26`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_20-29-06_738754_37752/logs/worker-86302185909a01c9a0581ece07507858cc00e2eb8295a1434498e197-01000000-53731.err#L26)
- token 级 logprob dump：
  [`worker-8630...53731.err:40`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_20-29-06_738754_37752/logs/worker-86302185909a01c9a0581ece07507858cc00e2eb8295a1434498e197-01000000-53731.err#L40)
- step 0：
  [`worker-8630...53731.err:45`](/mnt/ssd/cache_tmp/tmp/ray/session_2026-04-19_20-29-06_738754_37752/logs/worker-86302185909a01c9a0581ece07507858cc00e2eb8295a1434498e197-01000000-53731.err#L45)

### 修复后的 step 0 指标

```text
rollout/rollout_log_probs = -0.3341171940
rollout/log_probs = -0.3335898717
train/train_rollout_logprob_abs_diff = 0.0073150005
rollout/opd_reverse_kl = 0.3855798940
train/grad_norm = 18.6043406354
```

### 与修复前对比

修复前典型异常值：

```text
rollout/rollout_log_probs ≈ -0.335
rollout/log_probs ≈ -11.8
train/train_rollout_logprob_abs_diff ≈ 11.5
train/grad_norm ≈ 2e5
```

修复后：

- `rollout/log_probs` 已回到 `-0.33`
- 与 `rollout/rollout_log_probs` 基本一致
- `train/train_rollout_logprob_abs_diff` 从 `11+` 降到 `0.0073`
- `grad_norm` 从 `1e5` 量级降到 `18.6`

这已经不是“略有改善”，而是核心异常被实质性修复。

## token 级证据

修复后 sample 0 的 token 级对比：

```text
train_first8   = [-9.0e-05, -0.00122, -0.5766, ...]
rollout_first8 = [-8.8e-05, -0.00111, -0.5431, ...]
```

以及另一条 sample：

```text
train_first8   = [-2.77e-05, -7.63e-06, -0.0549, ...]
rollout_first8 = [-2.65e-05, -7.39e-06, -0.0601, ...]
```

可以看到：

- train / rollout token 级 logprob 已经同量级
- 差异是正常数值误差与实现细节误差
- 不再存在之前那种“一个接近 0，另一个大面积掉到 -10 左右”的失真

## 结论

本次跨词表 OPD 异常的真正根因，不是：

- OPD 指标公式错误
- rollout logprob 错误
- checkpoint 主转换链错误
- response label 切片错误

真正根因是：

- YuLan/Qwen3Next 线性注意力 wrapper 的 `input_layernorm` 实现选错

修复后，真实跨词表 smoke 的 student rollout/train logprob mismatch 已经恢复正常，`grad_norm` 也回到了可解释范围。

因此当前可以认为：

- 这次 `train_rollout_logprob_abs_diff` 异常已经被实质性修复
- 之前跨词表 OPD 的大部分异常值，是由错误 linear-attention pre-norm 语义传播出来的

## 后续建议

1. 继续盯后续几个训练 step

- 重点看 `grad_norm`
- 重点看 `rollout/opd_reverse_kl`
- 确认修复不是只在 `step 0` 生效

2. 将同样修复纳入后续所有 Qwen3Next / YuLanMini 相关训练路径

- 尤其是带线性注意力 wrapper 的路径

3. 后续若再出现大 mismatch，优先检查：

- 是否还有其他 HF decoder 组件被错误替换为“看起来像但语义不完全同”的模块
- 尤其是 norm、attention wrapper、token mixer 前处理
