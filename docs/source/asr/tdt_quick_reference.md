# TDT Parameter Quick Reference

Quick reference guide for the most important TDT training parameters in NeMo.

For detailed documentation, see [TDT Training Parameters Guide](tdt_training_parameters.rst)

## Critical Parameters (Start Here!)

### 1. sigma (Under-normalization)
```yaml
model.loss.tdt_kwargs.sigma: 0.05
```
- **Range**: 0.02 - 0.1
- **Impact**: ⭐⭐⭐⭐⭐ (Highest)
- **Default**: 0.05
- **Purpose**: Stabilizes training and improves accuracy

### 2. omega (RNN-T Loss Weight)
```yaml
model.loss.tdt_kwargs.omega: 0.1
```
- **Range**: 0.0 - 0.3
- **Impact**: ⭐⭐⭐⭐
- **Default**: 0.1
- **Purpose**: Balances TDT and RNN-T losses for stability

### 3. tdt_durations (Duration List)
```yaml
model.model_defaults.tdt_durations: [0, 1, 2, 3, 4]
```
- **Options**: [0,1,2,3,4] to [0,1,2,3,4,5,6,7,8]
- **Impact**: ⭐⭐⭐⭐
- **Default**: [0, 1, 2, 3, 4]
- **Purpose**: Defines duration modeling capacity

## Essential Configuration Template

```yaml
model:
  model_defaults:
    tdt_durations: [0, 1, 2, 3, 4]
    num_tdt_durations: 5
  
  loss:
    loss_name: "tdt"
    tdt_kwargs:
      sigma: 0.05          # Most important: Under-normalization
      omega: 0.1           # RNN-T loss weight for stability
      fastemit_lambda: 0.001  # For streaming (0.0 for offline)
      clamp: -1.0          # Gradient clipping (-1.0 = disabled)
      durations: ${model.model_defaults.tdt_durations}
  
  decoding:
    strategy: "greedy"     # MUST use greedy for best results
    model_type: "tdt"
    durations: ${model.model_defaults.tdt_durations}
  
  joint:
    num_extra_outputs: ${model.model_defaults.num_tdt_durations}
```

## Quick Tuning Guide

### If training is unstable (NaN losses)
1. Increase `omega` from 0.0 → 0.1 → 0.2
2. Enable gradient clipping: `clamp: 5.0`
3. Reduce learning rate

### If accuracy is lower than expected
1. Increase `sigma` from 0.05 → 0.08 → 0.1
2. Use `strategy: "greedy"` (not greedy_batch)
3. Ensure `omega ≤ 0.2`

### If inference is too slow for streaming
1. Enable FastEmit: `fastemit_lambda: 0.001`
2. Reduce `max_symbols` in decoding

### If you need longer audio support
1. Extend durations: `[0, 1, 2, 3, 4, 5, 6]`
2. Update `num_tdt_durations: 7`

## Parameter Priority for Tuning

1. **sigma** → Most impactful
2. **omega** → Affects stability
3. **tdt_durations** → Based on audio length
4. **fastemit_lambda** → Only if streaming needed
5. Standard params (lr, batch_size, etc.)

## Common Presets

### Preset 1: General Purpose (Recommended)
```yaml
sigma: 0.05
omega: 0.1
tdt_durations: [0, 1, 2, 3, 4]
fastemit_lambda: 0.0
```

### Preset 2: Streaming ASR
```yaml
sigma: 0.05
omega: 0.1
tdt_durations: [0, 1, 2, 3, 4]
fastemit_lambda: 0.001
```

### Preset 3: Long Audio
```yaml
sigma: 0.05
omega: 0.1
tdt_durations: [0, 1, 2, 3, 4, 5, 6]
fastemit_lambda: 0.0
```

### Preset 4: Hybrid TDT-CTC
```yaml
sigma: 0.02  # Lower for hybrid models
omega: 0.1
tdt_durations: [0, 1, 2, 3, 4]
aux_ctc.ctc_loss_weight: 0.3
```

## Model Size Templates

### Small (14M params)
```yaml
encoder:
  d_model: 176
  n_layers: 16
model_defaults:
  pred_hidden: 320
  joint_hidden: 320
```

### Medium (32M params)
```yaml
encoder:
  d_model: 256
  n_layers: 16
model_defaults:
  pred_hidden: 640
  joint_hidden: 640
```

### Large (120M params) - Recommended
```yaml
encoder:
  d_model: 512
  n_layers: 17
model_defaults:
  pred_hidden: 640
  joint_hidden: 640
```

## Training Command

```bash
python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
    --config-path=../conf/conformer/tdt \
    --config-name=conformer_tdt_bpe \
    model.train_ds.manifest_filepath=/path/to/train.json \
    model.validation_ds.manifest_filepath=/path/to/val.json \
    model.tokenizer.dir=/path/to/tokenizer \
    model.loss.tdt_kwargs.sigma=0.05 \
    model.loss.tdt_kwargs.omega=0.1 \
    trainer.devices=4 \
    trainer.max_epochs=100
```

## Reference

- Full documentation: [tdt_training_parameters.rst](tdt_training_parameters.rst)
- Chinese version: [tdt_training_parameters_zh.md](tdt_training_parameters_zh.md)
- TDT Paper: https://arxiv.org/abs/2304.06795
- Example configs: `examples/asr/conf/conformer/tdt/`
