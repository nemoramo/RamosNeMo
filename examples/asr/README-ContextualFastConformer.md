# Context-Aware FastConformer with RNN-T / 上下文感知 FastConformer (RNN-T)

This document is available in English and Chinese.
本文档提供英文和中文版本。

Jump to: [English Readme](#english-readme) | [中文说明](#中文说明)

---
<a name="english-readme"></a>
## English Readme

### Overview

This document describes a context-aware Automatic Speech Recognition (ASR) model based on the FastConformer architecture with an RNN-T transducer. This feature allows the ASR model to leverage preceding textual context (e.g., previous utterances or related text) to potentially improve transcription accuracy, especially in scenarios where context can resolve ambiguity or provide domain-specific cues.

The core idea is to feed both the audio input and its preceding textual context into the model. The model then learns to fuse these two sources of information to produce a more accurate transcription.

### Components

The context-aware functionality is implemented through the following new or modified NeMo components:

1.  **`ContextualAudioToBPEDataset`**:
    *   Located in: `nemo/collections/asr/data/contextual_audio_to_text_dataset.py`
    *   This dataset class extends the standard `AudioToBPEDataset`. Its primary role is to read an additional field from the data manifest that contains the previous textual context.
    *   By default, it looks for a field named `"prev_text"` in each manifest entry.

2.  **`SimpleTextContextEncoder`**:
    *   Located in: `nemo/collections/asr/modules/contextual_encoders.py`
    *   This module encodes the tokenized `prev_text` into a sequence of hidden states.
    *   Its architecture consists of a `torch.nn.Embedding` layer followed by a configurable multi-layer Transformer encoder.

3.  **`ContextualFusionModule`**:
    *   Located in: `nemo/collections/asr/modules/contextual_encoders.py`
    *   This module is responsible for fusing the encoded audio features (from the main Conformer encoder) with the encoded text context features (from `SimpleTextContextEncoder`).
    *   It uses a cross-attention mechanism where the audio encodings attend to the text context encodings. The output is a contextualized audio representation.

4.  **`ContextualEncDecRNNTModel`**:
    *   Located in: `nemo/collections/asr/models/contextual_rnnt_models.py`
    *   This is the main ASR model class, inheriting from `EncDecRNNTModel`.
    *   It integrates the `SimpleTextContextEncoder` and `ContextualFusionModule` into the standard RNN-T workflow. The audio features are contextualized by the text features before being passed to the RNN-T joint network.

### Configuration

The context-aware model is configured using a YAML file. An example configuration is provided in:
`examples/asr/conf/fastconformer/contextual_fast-conformer_transducer_bpe.yaml`

Key new or modified sections in this YAML file include:

*   **Model Target:**
    ```yaml
    model:
      _target_: nemo.collections.asr.models.contextual_rnnt_models.ContextualEncDecRNNTModel
    ```

*   **Dataset Target (for `train_ds`, `validation_ds`, `test_ds`):**
    ```yaml
    model:
      train_ds:
        _target_: nemo.collections.asr.data.contextual_audio_to_text_dataset.ContextualAudioToBPEDataset
        # ... other dataset parameters
        prev_text_field: "prev_text" # Specifies the manifest field for context
    ```

*   **Text Context Encoder Configuration:**
    ```yaml
    model:
      text_context_encoder:
        _target_: nemo.collections.asr.modules.contextual_encoders.SimpleTextContextEncoder
        embedding_dim: 256   # Example
        hidden_size: 256     # Example (d_model for its Transformer)
        num_layers: 2        # Example
        num_attention_heads: 4 # Example
        ff_expansion_factor: 4 # Example
        dropout: 0.1         # Example
        # vocab_size is set automatically by the model based on the main ASR tokenizer
    ```

*   **Contextual Fusion Module Configuration:**
    ```yaml
    model:
      contextual_fusion_module:
        _target_: nemo.collections.asr.modules.contextual_encoders.ContextualFusionModule
        num_attention_heads: 8 # Example (relates to audio_d_model)
        dropout: 0.1           # Example
        # audio_d_model and text_context_d_model are set automatically by the main model
    ```

Users need to ensure that `model.tokenizer.dir` (path to the BPE tokenizer model) and dataset manifest paths (`manifest_filepath` in `train_ds`, `validation_ds`, etc.) are correctly filled in their configuration. The main `model.encoder` section should be configured for the desired FastConformer variant (e.g., Large, Medium).

### Dataset Preparation

The dataset manifest files (JSONL format) must include a field for the preceding textual context. By default, this field is named `"prev_text"`.

**Example Manifest Line:**
```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 10.5, "text": "this is the current utterance", "prev_text": "this was the previous utterance providing context"}
{"audio_filepath": "/path/to/audio2.wav", "duration": 8.2, "text": "another current piece of audio", "prev_text": "some relevant preceding text for the second sample"}
```
If `prev_text` is missing or empty for a sample, it will be treated as an empty context.

### Training and Fine-tuning

A dedicated training script is provided: `examples/asr/train_contextual_asr.py`.

**To run training or fine-tuning:**
```bash
python examples/asr/train_contextual_asr.py \
    --config-path=conf/fastconformer \
    --config-name=contextual_fast-conformer_transducer_bpe \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/validation_manifest.json \
    model.tokenizer.dir=/path/to/tokenizer_model_directory \
    exp_manager.exp_dir=/path/to/experiment_output_directory \
    +trainer.devices=1 \
    +trainer.accelerator=gpu
    # Add other overrides as needed
```

**Fine-tuning from a Base Model:**
To fine-tune from a pre-trained non-contextual FastConformer model, specify the path to the `.nemo` checkpoint file in your YAML configuration:
```yaml
model:
  # ... other model settings ...
  fine_tune_from_nemo_model: /path/to/your/base_fastconformer_model.nemo
```
When `fine_tune_from_nemo_model` is set, the script will load weights from this base model. Components matching by name (like the main Conformer encoder, preprocessor, decoder, joint network) will have their weights loaded. The new components (`SimpleTextContextEncoder`, `ContextualFusionModule`) will be initialized randomly (or as per their `init_weights` if defined) as they don't exist in the base model.

### Expected Usage

This context-aware model is intended for scenarios where providing preceding text can help the ASR system. For example:
*   Improving recognition of domain-specific terms or named entities if they appeared in recent context.
*   Resolving ambiguities in speech.
*   Maintaining consistency in long-form transcriptions (e.g., speaker-specific phrasing).

The effectiveness of the context will depend on the quality and relevance of the `prev_text` provided and the model's capacity to learn the fusion.

---
<a name="中文说明"></a>
## 中文说明

### 概述

本文档描述了一个基于 FastConformer 架构和 RNN-T 换能器的上下文感知自动语音识别 (ASR) 模型。此功能允许 ASR 模型利用前文文本上下文（例如，先前的语音片段或相关文本）来提高转录准确性，尤其是在上下文有助于消除歧义或提供特定领域线索的场景中。

其核心思想是将音频输入及其前文文本上下文同时送入模型。然后，模型学习融合这两个信息源，以生成更准确的转录。

### 组件

上下文感知功能通过以下新增或修改的 NeMo 组件实现：

1.  **`ContextualAudioToBPEDataset`**:
    *   位置：`nemo/collections/asr/data/contextual_audio_to_text_dataset.py`
    *   此类继承自标准的 `AudioToBPEDataset`。其主要作用是从数据清单 (manifest) 文件中读取包含前文文本上下文的附加字段。
    *   默认情况下，它在每个清单条目中查找名为 `"prev_text"` 的字段。

2.  **`SimpleTextContextEncoder`**:
    *   位置：`nemo/collections/asr/modules/contextual_encoders.py`
    *   此模块将经过词元化 (tokenized) 的 `prev_text` 编码为一个隐藏状态序列。
    *   其架构包含一个 `torch.nn.Embedding` 层，后跟一个可配置的多层 Transformer 编码器。

3.  **`ContextualFusionModule`**:
    *   位置：`nemo/collections/asr/modules/contextual_encoders.py`
    *   此模块负责将编码后的音频特征（来自主要的 Conformer 编码器）与编码后的文本上下文特征（来自 `SimpleTextContextEncoder`）进行融合。
    *   它采用交叉注意力机制 (cross-attention)，其中音频编码关注文本上下文编码。输出是经过上下文信息调整的音频表征。

4.  **`ContextualEncDecRNNTModel`**:
    *   位置：`nemo/collections/asr/models/contextual_rnnt_models.py`
    *   这是主要的 ASR 模型类，继承自 `EncDecRNNTModel`。
    *   它将 `SimpleTextContextEncoder` 和 `ContextualFusionModule` 集成到标准的 RNN-T 工作流程中。音频特征在传递给 RNN-T 联合网络 (joint network) 之前，会先由文本特征进行上下文调整。

### 配置

上下文感知模型使用 YAML 文件进行配置。示例配置文件位于：
`examples/asr/conf/fastconformer/contextual_fast-conformer_transducer_bpe.yaml`

此 YAML 文件中关键的新增或修改部分包括：

*   **模型目标 (Model Target):**
    ```yaml
    model:
      _target_: nemo.collections.asr.models.contextual_rnnt_models.ContextualEncDecRNNTModel
    ```

*   **数据集目标 (Dataset Target) (针对 `train_ds`, `validation_ds`, `test_ds`):**
    ```yaml
    model:
      train_ds:
        _target_: nemo.collections.asr.data.contextual_audio_to_text_dataset.ContextualAudioToBPEDataset
        # ... 其他数据集参数
        prev_text_field: "prev_text" # 指定清单文件中上下文所对应的字段名
    ```

*   **文本上下文编码器配置 (Text Context Encoder Configuration):**
    ```yaml
    model:
      text_context_encoder:
        _target_: nemo.collections.asr.modules.contextual_encoders.SimpleTextContextEncoder
        embedding_dim: 256   # 示例值
        hidden_size: 256     # 示例值 (其内部 Transformer 的 d_model)
        num_layers: 2        # 示例值
        num_attention_heads: 4 # 示例值
        ff_expansion_factor: 4 # 示例值
        dropout: 0.1         # 示例值
        # vocab_size 由模型在初始化时根据主 ASR 标记器自动设置
    ```

*   **上下文融合模块配置 (Contextual Fusion Module Configuration):**
    ```yaml
    model:
      contextual_fusion_module:
        _target_: nemo.collections.asr.modules.contextual_encoders.ContextualFusionModule
        num_attention_heads: 8 # 示例值 (与 audio_d_model 相关)
        dropout: 0.1           # 示例值
        # audio_d_model 和 text_context_d_model 由主模型在初始化时自动设置
    ```

用户需要确保 `model.tokenizer.dir`（BPE 标记器模型目录的路径）和数据集清单路径（`train_ds`, `validation_ds` 中的 `manifest_filepath` 等）在其配置中已正确填写。主 `model.encoder` 部分应配置为所需的 FastConformer 型号（例如，Large 或 Medium）。

### 数据集准备

数据集清单文件（JSONL 格式）必须包含一个用于前文文本上下文的字段。默认情况下，此字段名为 `"prev_text"`。

**清单行示例 (Example Manifest Line):**
```json
{"audio_filepath": "/path/to/audio1.wav", "duration": 10.5, "text": "这是当前的语音内容", "prev_text": "这是提供上下文的先前语音内容"}
{"audio_filepath": "/path/to/audio2.wav", "duration": 8.2, "text": "另一段当前的音频", "prev_text": "为第二个样本提供的一些相关前文文本"}
```
如果某个样本的 `prev_text` 缺失或为空，则将其视为空上下文。

### 训练与微调

提供了一个专用的训练脚本：`examples/asr/train_contextual_asr.py`。

**运行训练或微调：**
```bash
python examples/asr/train_contextual_asr.py \
    --config-path=conf/fastconformer \
    --config-name=contextual_fast-conformer_transducer_bpe \
    model.train_ds.manifest_filepath=/path/to/train_manifest.json \
    model.validation_ds.manifest_filepath=/path/to/validation_manifest.json \
    model.tokenizer.dir=/path/to/tokenizer_model_directory \
    exp_manager.exp_dir=/path/to/experiment_output_directory \
    +trainer.devices=1 \
    +trainer.accelerator=gpu
    # 根据需要添加其他覆盖参数
```

**从基础模型进行微调：**
要从预训练的非上下文 FastConformer 模型进行微调，请在 YAML 配置中指定 `.nemo` 检查点文件的路径：
```yaml
model:
  # ... 其他模型设置 ...
  fine_tune_from_nemo_model: /path/to/your/base_fastconformer_model.nemo
```
当设置了 `fine_tune_from_nemo_model` 时，脚本将从此基础模型加载权重。名称匹配的组件（如主要的 Conformer 编码器、预处理器、解码器、联合网络）的权重将被加载。新的组件（`SimpleTextContextEncoder`, `ContextualFusionModule`）将随机初始化（或根据其 `init_weights`（如果已定义）），因为它们在基础模型中不存在。

### 预期用途

此上下文感知模型适用于提供前文文本有助于 ASR 系统的场景。例如：
*   如果领域特定术语或命名实体在最近的上下文中出现过，则提高其识别率。
*   解决语音中的歧义。
*   在长篇幅转录中保持一致性（例如，特定说话人的用语习惯）。

上下文的有效性将取决于所提供的 `prev_text` 的质量和相关性，以及模型学习融合信息的能力。
