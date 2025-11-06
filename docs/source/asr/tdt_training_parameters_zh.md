# NeMo TDT 训练参数指南

本文档提供 NeMo 中 Token-and-Duration Transducer (TDT) 模型的所有可调参数说明。

## 概述

Token-and-Duration Transducer (TDT) 是一种先进的 ASR 模型架构，论文详见 [TDT: Token-and-Duration Transducer for ASR](https://arxiv.org/abs/2304.06795)。TDT 通过同时建模标记（token）和持续时间（duration）扩展了传统的 RNN-T 架构，从而提高了准确性和效率。

TDT 模型的主要优势：

* 推理速度比传统 RNN-T 模型快 2-3 倍
* 在各种 ASR 基准测试上具有更好的准确性
* 支持流式和非流式推理
* 与 FastConformer 和 Conformer 编码器兼容

## 配置文件

NeMo 为 TDT 模型提供了几个示例配置文件：

* `examples/asr/conf/conformer/tdt/conformer_tdt_bpe.yaml` - 标准 Conformer-TDT（BPE 编码）
* `examples/asr/conf/conformer/tdt/conformer_tdt_bpe_stateless.yaml` - 无状态解码器变体
* `examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe.yaml` - 混合 TDT-CTC 模型

## 核心 TDT 参数

### 1. 模型架构参数

#### tdt_durations（持续时间列表）
* **类型**: 整数列表
* **默认值**: `[0, 1, 2, 3, 4]`
* **说明**: 模型可以预测的持续时间值列表。必须包含 0 和 1。最佳最大持续时间 (n) 通常在 4 到 8 之间，取决于数据集。
* **示例值**:
  * 短持续时间: `[0, 1, 2, 3, 4]` - 适用于一般用途
  * 中等持续时间: `[0, 1, 2, 3, 4, 5, 6]` - 适用于较长音频
  * 长持续时间: `[0, 1, 2, 3, 4, 5, 6, 7, 8]` - 适用于非常长的语音
* **推荐**: 从 `[0, 1, 2, 3, 4]` 开始，根据需要增加
* **位置**: `model.model_defaults.tdt_durations`

#### num_tdt_durations（持续时间输出数量）
* **类型**: 整数
* **默认值**: `5`
* **说明**: 持续时间输出的数量。必须等于 `len(tdt_durations)`
* **位置**: `model.model_defaults.num_tdt_durations`

#### num_extra_outputs（额外输出数量）
* **类型**: 整数
* **默认值**: `${model.model_defaults.num_tdt_durations}`
* **说明**: 联合网络除了词汇标记之外的额外输出数量
* **位置**: `model.joint.num_extra_outputs`

### 2. 损失函数参数

#### loss_name（损失函数名称）
* **类型**: 字符串
* **必需值**: `"tdt"`
* **说明**: 指定使用 TDT 损失而不是标准 RNN-T 损失
* **位置**: `model.loss.loss_name`

#### fastemit_lambda（FastEmit 正则化参数）
* **类型**: 浮点数
* **默认值**: `0.001`
* **范围**: `[1e-4, 1e-2]`
* **说明**: FastEmit 正则化参数，用于减少流式场景中的模型延迟。值越大，模型发出标记的速度越快。
* **位置**: `model.loss.tdt_kwargs.fastemit_lambda`
* **参考文献**: [FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization](https://arxiv.org/abs/2010.11148)
* **调参建议**:
  * `0.0` - 无 FastEmit（非流式默认值）
  * `0.0001` - 非常轻微的正则化
  * `0.001` - 流式推理的良好起点
  * `0.01` - 强正则化，可能影响准确性

#### clamp（梯度裁剪阈值）
* **类型**: 浮点数
* **默认值**: `-1.0`
* **说明**: 联合张量的梯度裁剪阈值。设置为正值时，梯度被裁剪到 `[-clamp, clamp]` 范围。设置为 `-1.0` 可禁用。
* **位置**: `model.loss.tdt_kwargs.clamp`
* **调参建议**:
  * `-1.0` - 无裁剪（默认）
  * `1.0` - 轻度裁剪以提高稳定性
  * `5.0` - 中度裁剪
  * `10.0` - 强裁剪

#### sigma（Logit 欠归一化参数）
* **类型**: 浮点数
* **默认值**: `0.05`
* **范围**: `[0.0, 0.1]`
* **说明**: Logit 欠归一化方法的超参数。这种技术有助于稳定 TDT 训练并提高模型性能。
* **位置**: `model.loss.tdt_kwargs.sigma`
* **参考文献**: 参见 [TDT 论文](https://arxiv.org/abs/2304.06795) 第 3.2 节
* **调参建议**:
  * `0.0` - 无欠归一化
  * `0.02` - 轻度欠归一化（混合 TDT-CTC）
  * `0.05` - 推荐默认值
  * `0.1` - 强欠归一化

#### omega（RNN-T 损失权重）
* **类型**: 浮点数
* **默认值**: `0.1`
* **范围**: `[0.0, 0.5]`
* **说明**: 将 TDT 损失与标准 RNN-T 损失结合的权重。最终损失 = (1-omega) * TDT_loss + omega * RNNT_loss。使用小的 omega 值有助于提高训练稳定性。
* **位置**: `model.loss.tdt_kwargs.omega`
* **参考文献**: 参见 [TDT 论文](https://arxiv.org/abs/2304.06795) 第 3.3 节
* **调参建议**:
  * `0.0` - 纯 TDT 损失（可能不太稳定）
  * `0.1` - 推荐默认值
  * `0.2` - 更多 RNN-T 影响
  * `0.3-0.5` - 平衡混合

### 3. 解码参数

#### strategy（解码策略）
* **类型**: 字符串
* **默认值**: `"greedy"`
* **选项**: `"greedy"`, `"greedy_batch"`, `"beam"`
* **说明**: 推理期间使用的解码策略
* **位置**: `model.decoding.strategy`
* **重要提示**: 强烈建议 TDT 模型使用 `"greedy"`。如果 omega 为 0，使用 `"greedy_batch"` 会导致非常差的结果，即使 omega 非零，结果也不准确。

#### model_type（模型类型）
* **类型**: 字符串
* **必需值**: `"tdt"`
* **说明**: 指定这是 TDT 模型，启用 TDT 特定的解码方法
* **位置**: `model.decoding.model_type`

#### durations（解码持续时间列表）
* **类型**: 整数列表
* **默认值**: `${model.model_defaults.tdt_durations}`
* **说明**: 解码的持续时间列表。必须与训练期间使用的持续时间匹配。
* **位置**: `model.decoding.durations`
* **重要提示**: 此值不能为 None，以使用 TDT 特定的解码

### 4. 编码器参数

TDT 模型通常使用 Conformer 或 FastConformer 编码器。主要可调参数：

#### d_model（隐藏维度）
* **类型**: 整数
* **默认值**: `512`（大型模型）
* **选项**: `176`（小型）、`256`（中型）、`512`（大型）、`1024`（超大型）
* **说明**: 编码器的隐藏维度大小
* **位置**: `model.encoder.d_model`

#### n_layers（层数）
* **类型**: 整数
* **默认值**: `17`
* **说明**: Conformer 层的数量
* **位置**: `model.encoder.n_layers`

#### dropout（Dropout 率）
* **类型**: 浮点数
* **默认值**: `0.1`
* **说明**: 编码器层的 Dropout 率
* **位置**: `model.encoder.dropout`

### 5. 解码器（预测网络）参数

#### pred_hidden（预测网络隐藏大小）
* **类型**: 整数
* **默认值**: `640`
* **说明**: 预测网络的隐藏大小
* **位置**: `model.model_defaults.pred_hidden`

#### pred_rnn_layers（RNN 层数）
* **类型**: 整数
* **默认值**: `1`
* **说明**: 预测网络中的 RNN 层数
* **位置**: `model.decoder.prednet.pred_rnn_layers`

### 6. 联合网络参数

#### joint_hidden（联合网络隐藏大小）
* **类型**: 整数
* **默认值**: `640`
* **说明**: 联合网络的隐藏大小
* **位置**: `model.model_defaults.joint_hidden`

#### fused_batch_size（融合批大小）
* **类型**: 整数
* **默认值**: `16`
* **说明**: 融合操作的批大小。较小的值节省内存但减慢训练速度。
* **位置**: `model.joint.fused_batch_size`
* **推荐**: 保持 fused_batch_size 与 train_ds.batch_size 的比率为 1:1 以获得最佳性能

## 训练参数

### 优化器配置

#### lr（学习率）
* **类型**: 浮点数
* **默认值**: `5.0`
* **说明**: 学习率（与 Noam 调度器一起使用）
* **位置**: `model.optim.lr`

#### weight_decay（权重衰减）
* **类型**: 浮点数
* **默认值**: `1e-3`
* **说明**: L2 正则化权重
* **位置**: `model.optim.weight_decay`

### 学习率调度

#### warmup_steps（预热步数）
* **类型**: 整数
* **默认值**: `10000`
* **说明**: 预热步数
* **位置**: `model.optim.sched.warmup_steps`

### 数据增强

#### freq_masks（频率掩码数量）
* **类型**: 整数
* **默认值**: `2`
* **说明**: 频率掩码的数量。设置为 0 可禁用。
* **位置**: `model.spec_augment.freq_masks`

#### time_masks（时间掩码数量）
* **类型**: 整数
* **默认值**: `10`
* **说明**: 时间掩码的数量。设置为 0 可禁用。
* **位置**: `model.spec_augment.time_masks`

### 数据集参数

#### batch_size（批大小）
* **类型**: 整数
* **默认值**: `16`
* **说明**: 训练/验证的批大小
* **位置**: `model.train_ds.batch_size`, `model.validation_ds.batch_size`
* **注意**: 有效批大小 = batch_size × num_gpus × accumulate_grad_batches

#### max_duration（最大音频时长）
* **类型**: 浮点数
* **默认值**: `16.7`
* **说明**: 最大音频时长（秒）
* **位置**: `model.train_ds.max_duration`

### Trainer 配置

#### devices（设备数量）
* **类型**: 整数
* **默认值**: `-1`（使用所有可用 GPU）
* **说明**: 要使用的 GPU 数量
* **位置**: `trainer.devices`

#### max_epochs（最大训练轮数）
* **类型**: 整数
* **默认值**: `500`
* **说明**: 最大训练轮数
* **位置**: `trainer.max_epochs`

#### precision（训练精度）
* **类型**: 整数或字符串
* **默认值**: `32`
* **选项**: `16`, `32`, `"bf16"`
* **说明**: 训练精度
* **位置**: `trainer.precision`
* **注意**: 使用 16 或 bf16 进行混合精度训练

## 快速开始示例

以下是开始训练 TDT 模型的最小示例：

```yaml
# 最小 TDT 训练配置
model:
  model_defaults:
    tdt_durations: [0, 1, 2, 3, 4]  # 从默认持续时间开始
    num_tdt_durations: 5
  
  train_ds:
    manifest_filepath: /path/to/train_manifest.json
    batch_size: 16
  
  validation_ds:
    manifest_filepath: /path/to/val_manifest.json
    batch_size: 16
  
  tokenizer:
    dir: /path/to/tokenizer
    type: bpe
  
  loss:
    loss_name: "tdt"
    tdt_kwargs:
      fastemit_lambda: 0.001  # 启用 FastEmit 用于流式
      clamp: -1.0
      sigma: 0.05  # 推荐默认值
      omega: 0.1   # 推荐默认值
      durations: ${model.model_defaults.tdt_durations}
  
  decoding:
    strategy: "greedy"  # 使用 greedy 以获得最佳准确性
    model_type: "tdt"
    durations: ${model.model_defaults.tdt_durations}
  
  joint:
    num_extra_outputs: ${model.model_defaults.num_tdt_durations}

trainer:
  devices: -1  # 使用所有可用 GPU
  max_epochs: 100
  precision: 16  # 使用混合精度以加快训练速度
```

### 训练命令

```bash
python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=../conf/conformer/tdt \
    --config-name=conformer_tdt_bpe \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
    model.tokenizer.dir=/path/to/tokenizer \
    trainer.devices=4 \
    trainer.max_epochs=100
```

## 超参数调优技巧

### 需要调整的关键参数

调优 TDT 模型时，按重要性顺序关注这些参数：

1. **sigma** (0.02-0.1): 对模型性能影响最大
2. **omega** (0.0-0.3): 影响训练稳定性
3. **tdt_durations**: 根据音频特性尝试 [0,1,2,3,4] 到 [0,1,2,3,4,5,6,7,8]
4. **fastemit_lambda** (1e-4 到 1e-2): 如果针对流式应用
5. **learning rate 和 warmup_steps**: 标准 transformer 调优
6. **batch_size 和 fused_batch_size**: 在内存和速度之间取得平衡

### 调优策略

1. **从默认值开始**: 使用提供的配置文件作为起点
2. **首先调整 sigma**: 尝试 0.02 到 0.1 之间的值，通常 0.05 效果很好
3. **调整持续时间**: 如果处理长音频，尝试扩展到 [0,1,2,3,4,5,6]
4. **调整 omega**: 如果训练不稳定且 omega=0.0，增加到 0.1-0.2
5. **启用 FastEmit**: 如果针对流式，从 fastemit_lambda=0.001 开始
6. **微调学习率**: 根据模型大小和数据集调整

### 常见问题和解决方案

**问题**: 模型训练不稳定，出现 NaN 损失

* **解决方案**: 将 omega 从 0.0 增加到 0.1 或 0.2
* **解决方案**: 启用梯度裁剪：将 clamp 设置为 5.0 或 10.0
* **解决方案**: 降低学习率

**问题**: 模型准确性低于预期

* **解决方案**: 将 sigma 增加到 0.05 或 0.1
* **解决方案**: 使用 greedy 解码而不是 greedy_batch
* **解决方案**: 确保 omega 不要太高（保持 ≤ 0.2）

**问题**: 流式应用的推理速度太慢

* **解决方案**: 启用 FastEmit：将 fastemit_lambda 设置为 0.001-0.01
* **解决方案**: 减少 greedy 解码中的 max_symbols

**问题**: 训练期间内存不足

* **解决方案**: 减少 fused_batch_size
* **解决方案**: 启用梯度累积：增加 accumulate_grad_batches
* **解决方案**: 减少 batch_size
* **解决方案**: 使用混合精度：将 trainer.precision 设置为 16 或 "bf16"

## 参考资料

* [TDT 论文: Token-and-Duration Transducer for ASR](https://arxiv.org/abs/2304.06795)
* [FastEmit 论文](https://arxiv.org/abs/2010.11148)
* [NeMo ASR 文档](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/intro.html)
* [Parakeet-TDT 模型卡片](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
* [示例配置文件](https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/conformer/tdt)

## 代码示例

### Python API 使用

```python
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf

# 加载并自定义 TDT 配置
config = OmegaConf.load('examples/asr/conf/conformer/tdt/conformer_tdt_bpe.yaml')

# 自定义 TDT 参数
config.model.model_defaults.tdt_durations = [0, 1, 2, 3, 4, 5, 6]
config.model.model_defaults.num_tdt_durations = 7
config.model.loss.tdt_kwargs.sigma = 0.05
config.model.loss.tdt_kwargs.omega = 0.1
config.model.loss.tdt_kwargs.fastemit_lambda = 0.001

# 创建模型
model = nemo_asr.models.EncDecRNNTBPEModel(cfg=config.model, trainer=trainer)

# 训练模型
trainer.fit(model)
```

### 微调预训练 TDT 模型

```python
import nemo.collections.asr as nemo_asr

# 加载预训练 TDT 模型
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# 更新微调的训练参数
model.cfg.optim.lr = 0.0001  # 微调时使用较低的学习率
model.cfg.loss.tdt_kwargs.sigma = 0.05

# 设置数据集
model.setup_training_data(train_config)
model.setup_validation_data(val_config)

# 微调
trainer.fit(model)
```

### 使用 TDT 模型进行推理

```python
import nemo.collections.asr as nemo_asr

# 加载模型
model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")

# 设置解码参数
model.cfg.decoding.strategy = "greedy"
model.cfg.decoding.model_type = "tdt"
model.cfg.decoding.durations = [0, 1, 2, 3, 4]
model.change_decoding_strategy(model.cfg.decoding)

# 转录
transcriptions = model.transcribe(["/path/to/audio1.wav", "/path/to/audio2.wav"])
print(transcriptions)
```

## 总结

本文档提供了 NeMo 中 TDT 模型训练的完整参数说明。关键要点：

1. **sigma** 和 **omega** 是影响 TDT 模型性能的最重要参数
2. 使用 **greedy** 解码策略以获得最佳准确性
3. **tdt_durations** 应根据音频长度特性选择
4. 启用 **fastemit_lambda** 可以提高流式推理的速度
5. 从推荐的默认值开始，然后根据具体数据集进行调优

完整的英文文档请参见 [TDT Training Parameters Guide](tdt_training_parameters.rst)。
