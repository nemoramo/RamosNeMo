.. _tdt-training-parameters:

TDT Training Parameters
========================

This guide provides a comprehensive overview of all tunable parameters for training Token-and-Duration Transducer (TDT) models in NeMo.

Overview
--------

Token-and-Duration Transducer (TDT) is an advanced ASR model architecture described in the paper `TDT: Token-and-Duration Transducer for ASR <https://arxiv.org/abs/2304.06795>`__. TDT extends the traditional RNN-T architecture by modeling both token emissions and their durations, leading to improved accuracy and efficiency.

Key advantages of TDT models:

* Faster inference speed compared to conventional RNN-T models
* Better accuracy on various ASR benchmarks
* Support for streaming and non-streaming inference
* Compatible with FastConformer and Conformer encoders

Configuration Files
-------------------

NeMo provides several example configuration files for TDT models:

* ``examples/asr/conf/conformer/tdt/conformer_tdt_bpe.yaml`` - Standard Conformer-TDT with BPE encoding
* ``examples/asr/conf/conformer/tdt/conformer_tdt_bpe_stateless.yaml`` - Stateless decoder variant
* ``examples/asr/conf/fastconformer/hybrid_transducer_ctc/fastconformer_hybrid_tdt_ctc_bpe.yaml`` - Hybrid TDT-CTC model

Core TDT Parameters
-------------------

Model Architecture Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These parameters define the basic TDT model structure:

**tdt_durations**
  * Type: List of integers
  * Default: ``[0, 1, 2, 3, 4]``
  * Description: List of duration values that the model can predict. Must include 0 and 1. The optimal maximum duration (n) typically ranges from 4 to 8 depending on your dataset.
  * Example values:
    
    * Short durations: ``[0, 1, 2, 3, 4]`` - Good for general use
    * Medium durations: ``[0, 1, 2, 3, 4, 5, 6]`` - Better for longer audio
    * Long durations: ``[0, 1, 2, 3, 4, 5, 6, 7, 8]`` - For very long utterances
  
  * Recommendation: Start with ``[0, 1, 2, 3, 4]`` and increase if needed

**num_tdt_durations**
  * Type: Integer
  * Default: ``5``
  * Description: Number of duration outputs. Must equal ``len(tdt_durations)``
  * Location: ``model.model_defaults.num_tdt_durations``
  * Example: If ``tdt_durations: [0, 1, 2, 3, 4]``, then ``num_tdt_durations: 5``

**num_extra_outputs**
  * Type: Integer
  * Default: ``${model.model_defaults.num_tdt_durations}``
  * Description: Number of additional outputs from the joint network besides the vocabulary tokens
  * Location: ``model.joint.num_extra_outputs``
  * Note: This is automatically set to match ``num_tdt_durations``

Loss Function Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

These parameters control the TDT loss computation and training behavior:

**loss_name**
  * Type: String
  * Required: ``"tdt"``
  * Description: Specifies that TDT loss should be used instead of standard RNN-T loss
  * Location: ``model.loss.loss_name``

**fastemit_lambda**
  * Type: Float
  * Default: ``0.001``
  * Range: ``[1e-4, 1e-2]``
  * Description: FastEmit regularization parameter for reducing model latency in streaming scenarios. Higher values encourage the model to emit tokens faster.
  * Location: ``model.loss.tdt_kwargs.fastemit_lambda``
  * Reference: `FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization <https://arxiv.org/abs/2010.11148>`__
  * Tuning recommendations:
    
    * ``0.0`` - No FastEmit (default for non-streaming)
    * ``0.0001`` - Very mild regularization
    * ``0.001`` - Good starting point for streaming
    * ``0.01`` - Strong regularization, may affect accuracy
  
**clamp**
  * Type: Float
  * Default: ``-1.0``
  * Description: Gradient clamping threshold for the joint tensor. When set to a positive value, gradients are clamped to the range ``[-clamp, clamp]``. Set to ``-1.0`` to disable.
  * Location: ``model.loss.tdt_kwargs.clamp``
  * Tuning recommendations:
    
    * ``-1.0`` - No clamping (default)
    * ``1.0`` - Light clamping for stability
    * ``5.0`` - Moderate clamping
    * ``10.0`` - Strong clamping
  
**sigma**
  * Type: Float
  * Default: ``0.05``
  * Range: ``[0.0, 0.1]``
  * Description: Hyper-parameter for logit under-normalization method. This technique helps stabilize TDT training and improve model performance.
  * Location: ``model.loss.tdt_kwargs.sigma``
  * Reference: See Section 3.2 in `TDT paper <https://arxiv.org/abs/2304.06795>`__
  * Tuning recommendations:
    
    * ``0.0`` - No under-normalization
    * ``0.02`` - Light under-normalization (hybrid TDT-CTC)
    * ``0.05`` - Recommended default
    * ``0.1`` - Strong under-normalization
  
**omega**
  * Type: Float
  * Default: ``0.1``
  * Range: ``[0.0, 0.5]``
  * Description: Weight for combining TDT loss with standard RNN-T loss. Final loss = (1-omega) * TDT_loss + omega * RNNT_loss. Using a small omega value helps with training stability.
  * Location: ``model.loss.tdt_kwargs.omega``
  * Reference: See Section 3.3 in `TDT paper <https://arxiv.org/abs/2304.06795>`__
  * Tuning recommendations:
    
    * ``0.0`` - Pure TDT loss (may be less stable)
    * ``0.1`` - Recommended default
    * ``0.2`` - More RNN-T influence
    * ``0.3-0.5`` - Balanced mix
  
**durations**
  * Type: List of integers
  * Default: ``${model.model_defaults.tdt_durations}``
  * Description: Reference to the duration list defined in model_defaults
  * Location: ``model.loss.tdt_kwargs.durations``
  * Note: This should always reference the same durations used in the model architecture

Decoding Parameters
~~~~~~~~~~~~~~~~~~~

These parameters control how the TDT model performs inference:

**strategy**
  * Type: String
  * Default: ``"greedy"``
  * Options: ``"greedy"``, ``"greedy_batch"``, ``"beam"``
  * Description: Decoding strategy to use during inference
  * Location: ``model.decoding.strategy``
  * Important: Using ``"greedy"`` is highly recommended for TDT models. Using ``"greedy_batch"`` will give very poor results if omega is 0, and inaccurate results even with non-zero omega.
  * Recommendations:
    
    * ``"greedy"`` - Recommended for best accuracy
    * ``"greedy_batch"`` - Only use with non-zero omega and if batch processing is critical
    * ``"beam"`` - Experimental beam search for TDT

**model_type**
  * Type: String
  * Required: ``"tdt"``
  * Description: Specifies that this is a TDT model, enabling TDT-specific decoding methods
  * Location: ``model.decoding.model_type``

**durations**
  * Type: List of integers
  * Default: ``${model.model_defaults.tdt_durations}``
  * Description: Duration list for decoding. Must match the durations used during training.
  * Location: ``model.decoding.durations``
  * Important: This must not be None to use TDT-specific decoding

**max_symbols**
  * Type: Integer
  * Default: ``10``
  * Description: Maximum number of symbols to emit per frame during greedy decoding
  * Location: ``model.decoding.greedy.max_symbols``

Encoder Parameters
~~~~~~~~~~~~~~~~~

TDT models typically use Conformer or FastConformer encoders. Key tunable parameters:

**d_model**
  * Type: Integer
  * Default: ``512`` (Large model)
  * Options: ``176`` (Small), ``256`` (Medium), ``512`` (Large), ``1024`` (XLarge)
  * Description: Hidden dimension size of the encoder
  * Location: ``model.encoder.d_model``

**n_layers**
  * Type: Integer
  * Default: ``17``
  * Description: Number of Conformer layers
  * Location: ``model.encoder.n_layers``

**n_heads**
  * Type: Integer
  * Default: ``8``
  * Description: Number of attention heads
  * Location: ``model.encoder.n_heads``

**conv_kernel_size**
  * Type: Integer
  * Default: ``31``
  * Description: Kernel size for the convolution module
  * Location: ``model.encoder.conv_kernel_size``

**dropout**
  * Type: Float
  * Default: ``0.1``
  * Description: Dropout rate for encoder layers
  * Location: ``model.encoder.dropout``

Decoder (Prediction Network) Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**pred_hidden**
  * Type: Integer
  * Default: ``640``
  * Description: Hidden size of the prediction network
  * Location: ``model.model_defaults.pred_hidden``

**pred_rnn_layers**
  * Type: Integer
  * Default: ``1``
  * Options: ``1`` (Small/Medium/Large), ``2`` (XLarge)
  * Description: Number of RNN layers in the prediction network
  * Location: ``model.decoder.prednet.pred_rnn_layers``

**dropout**
  * Type: Float
  * Default: ``0.2``
  * Description: Dropout rate for the prediction network
  * Location: ``model.decoder.prednet.dropout``

Joint Network Parameters
~~~~~~~~~~~~~~~~~~~~~~~

**joint_hidden**
  * Type: Integer
  * Default: ``640``
  * Description: Hidden size of the joint network
  * Location: ``model.model_defaults.joint_hidden``

**activation**
  * Type: String
  * Default: ``"relu"``
  * Options: ``"relu"``, ``"tanh"``, ``"silu"``
  * Description: Activation function for the joint network
  * Location: ``model.joint.jointnet.activation``

**dropout**
  * Type: Float
  * Default: ``0.2``
  * Description: Dropout rate for the joint network
  * Location: ``model.joint.jointnet.dropout``

**fuse_loss_wer**
  * Type: Boolean
  * Default: ``true``
  * Description: Whether to fuse prediction net, joint net, loss, and WER calculation to run on sub-batches
  * Location: ``model.joint.fuse_loss_wer``

**fused_batch_size**
  * Type: Integer
  * Default: ``16``
  * Description: Batch size for fused operations. Smaller values preserve memory but slow training.
  * Location: ``model.joint.fused_batch_size``
  * Recommendation: Keep ratio of fused_batch_size to train_ds.batch_size at 1:1 for best performance

Training Parameters
------------------

Optimizer Configuration
~~~~~~~~~~~~~~~~~~~~~~

**name**
  * Type: String
  * Default: ``"adamw"``
  * Description: Optimizer type
  * Location: ``model.optim.name``

**lr**
  * Type: Float
  * Default: ``5.0``
  * Description: Learning rate (used with Noam scheduler)
  * Location: ``model.optim.lr``

**betas**
  * Type: List of floats
  * Default: ``[0.9, 0.98]``
  * Description: Adam optimizer beta parameters
  * Location: ``model.optim.betas``

**weight_decay**
  * Type: Float
  * Default: ``1e-3``
  * Options: ``0.0`` (Small), ``1e-3`` (Medium/Large/XLarge)
  * Description: L2 regularization weight
  * Location: ``model.optim.weight_decay``

Learning Rate Schedule
~~~~~~~~~~~~~~~~~~~~~

**name**
  * Type: String
  * Default: ``"NoamAnnealing"``
  * Description: Learning rate scheduler type
  * Location: ``model.optim.sched.name``

**d_model**
  * Type: Integer
  * Default: ``${model.encoder.d_model}``
  * Description: Model dimension for Noam scheduler
  * Location: ``model.optim.sched.d_model``

**warmup_steps**
  * Type: Integer
  * Default: ``10000``
  * Description: Number of warmup steps
  * Location: ``model.optim.sched.warmup_steps``

**min_lr**
  * Type: Float
  * Default: ``1e-6``
  * Description: Minimum learning rate
  * Location: ``model.optim.sched.min_lr``

Data Augmentation
~~~~~~~~~~~~~~~~

Spectrogram Augmentation:

**freq_masks**
  * Type: Integer
  * Default: ``2``
  * Description: Number of frequency masks. Set to 0 to disable.
  * Location: ``model.spec_augment.freq_masks``

**time_masks**
  * Type: Integer
  * Default: ``10``
  * Description: Number of time masks. Set to 0 to disable.
  * Location: ``model.spec_augment.time_masks``

**freq_width**
  * Type: Integer
  * Default: ``27``
  * Description: Maximum width of frequency masks
  * Location: ``model.spec_augment.freq_width``

**time_width**
  * Type: Float
  * Default: ``0.05``
  * Description: Maximum width of time masks as fraction of sequence length
  * Location: ``model.spec_augment.time_width``

Dataset Parameters
~~~~~~~~~~~~~~~~~

**batch_size**
  * Type: Integer
  * Default: ``16``
  * Description: Batch size for training/validation
  * Location: ``model.train_ds.batch_size``, ``model.validation_ds.batch_size``
  * Note: Effective batch size is batch_size * num_gpus * accumulate_grad_batches

**num_workers**
  * Type: Integer
  * Default: ``8``
  * Description: Number of data loading workers
  * Location: ``model.train_ds.num_workers``

**max_duration**
  * Type: Float
  * Default: ``16.7``
  * Description: Maximum audio duration in seconds
  * Location: ``model.train_ds.max_duration``

**bucketing_strategy**
  * Type: String
  * Default: ``"synced_randomized"``
  * Description: Strategy for bucketing samples by length
  * Location: ``model.train_ds.bucketing_strategy``

Trainer Configuration
~~~~~~~~~~~~~~~~~~~~

**devices**
  * Type: Integer
  * Default: ``-1`` (use all available GPUs)
  * Description: Number of GPUs to use
  * Location: ``trainer.devices``

**max_epochs**
  * Type: Integer
  * Default: ``500``
  * Description: Maximum number of training epochs
  * Location: ``trainer.max_epochs``

**precision**
  * Type: Integer or String
  * Default: ``32``
  * Options: ``16``, ``32``, ``"bf16"``
  * Description: Training precision
  * Location: ``trainer.precision``
  * Note: Use 16 or bf16 for mixed precision training

**accumulate_grad_batches**
  * Type: Integer
  * Default: ``1``
  * Description: Number of batches to accumulate gradients over
  * Location: ``trainer.accumulate_grad_batches``
  * Use case: Increase effective batch size without increasing memory

**gradient_clip_val**
  * Type: Float
  * Default: ``0.0``
  * Description: Gradient clipping value. 0.0 means no clipping.
  * Location: ``trainer.gradient_clip_val``

Model Size Recommendations
-------------------------

Here are recommended configurations for different TDT model sizes:

Small (14M parameters)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    model:
      encoder:
        d_model: 176
        n_heads: 4
        n_layers: 16
        conv_kernel_size: 31
      
      model_defaults:
        pred_hidden: 320
        joint_hidden: 320
        tdt_durations: [0, 1, 2, 3, 4]
        num_tdt_durations: 5
      
      decoder:
        prednet:
          pred_rnn_layers: 1
      
      optim:
        weight_decay: 0.0

Medium (32M parameters)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    model:
      encoder:
        d_model: 256
        n_heads: 4
        n_layers: 16
        conv_kernel_size: 31
      
      model_defaults:
        pred_hidden: 640
        joint_hidden: 640
        tdt_durations: [0, 1, 2, 3, 4]
        num_tdt_durations: 5
      
      decoder:
        prednet:
          pred_rnn_layers: 1
      
      optim:
        weight_decay: 1e-3

Large (120M parameters)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    model:
      encoder:
        d_model: 512
        n_heads: 8
        n_layers: 17
        conv_kernel_size: 31
      
      model_defaults:
        pred_hidden: 640
        joint_hidden: 640
        tdt_durations: [0, 1, 2, 3, 4]
        num_tdt_durations: 5
      
      decoder:
        prednet:
          pred_rnn_layers: 1
      
      optim:
        weight_decay: 1e-3

XLarge (644M parameters)
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    model:
      encoder:
        d_model: 1024
        n_heads: 8
        n_layers: 24
        conv_kernel_size: 5
      
      model_defaults:
        pred_hidden: 640
        joint_hidden: 640
        tdt_durations: [0, 1, 2, 3, 4]
        num_tdt_durations: 5
      
      decoder:
        prednet:
          pred_rnn_layers: 2
      
      optim:
        weight_decay: 1e-3

Quick Start Example
------------------

Here's a minimal example to start training a TDT model:

.. code-block:: yaml

    # Minimal TDT training config
    model:
      model_defaults:
        tdt_durations: [0, 1, 2, 3, 4]  # Start with default durations
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
          fastemit_lambda: 0.001  # Enable FastEmit for streaming
          clamp: -1.0
          sigma: 0.05  # Recommended default
          omega: 0.1   # Recommended default
          durations: ${model.model_defaults.tdt_durations}
      
      decoding:
        strategy: "greedy"  # Use greedy for best accuracy
        model_type: "tdt"
        durations: ${model.model_defaults.tdt_durations}
      
      joint:
        num_extra_outputs: ${model.model_defaults.num_tdt_durations}
    
    trainer:
      devices: -1  # Use all available GPUs
      max_epochs: 100
      precision: 16  # Use mixed precision for faster training

Training Command
~~~~~~~~~~~~~~~

.. code-block:: bash

    python examples/asr/asr_transducer/speech_to_text_rnnt_bpe.py \
        --config-path=../conf/conformer/tdt \
        --config-name=conformer_tdt_bpe \
        model.train_ds.manifest_filepath=/path/to/train_manifest.json \
        model.validation_ds.manifest_filepath=/path/to/val_manifest.json \
        model.tokenizer.dir=/path/to/tokenizer \
        trainer.devices=4 \
        trainer.max_epochs=100

Hyperparameter Tuning Tips
--------------------------

Key Parameters to Tune
~~~~~~~~~~~~~~~~~~~~~

When tuning TDT models, focus on these parameters in order of importance:

1. **sigma** (0.02-0.1): Most impactful for model performance
2. **omega** (0.0-0.3): Affects training stability
3. **tdt_durations**: Try [0,1,2,3,4] to [0,1,2,3,4,5,6,7,8] based on audio characteristics
4. **fastemit_lambda** (1e-4 to 1e-2): If targeting streaming applications
5. **learning rate and warmup_steps**: Standard tuning for transformers
6. **batch_size and fused_batch_size**: Balance between memory and speed

Tuning Strategy
~~~~~~~~~~~~~~

1. **Start with defaults**: Use the provided config files as starting points
2. **Tune sigma first**: Try values between 0.02 and 0.1, typically 0.05 works well
3. **Adjust durations**: If dealing with long audio, try extending to [0,1,2,3,4,5,6]
4. **Tune omega**: If training is unstable with omega=0.0, increase to 0.1-0.2
5. **Enable FastEmit**: If targeting streaming, start with fastemit_lambda=0.001
6. **Fine-tune learning rate**: Adjust based on model size and dataset

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: Model training is unstable with NaN losses

* **Solution**: Increase omega from 0.0 to 0.1 or 0.2
* **Solution**: Enable gradient clipping: set clamp to 5.0 or 10.0
* **Solution**: Reduce learning rate

**Issue**: Model accuracy is lower than expected

* **Solution**: Increase sigma to 0.05 or 0.1
* **Solution**: Use greedy decoding instead of greedy_batch
* **Solution**: Ensure omega is not too high (keep it ≤ 0.2)

**Issue**: Inference is too slow for streaming applications

* **Solution**: Enable FastEmit: set fastemit_lambda to 0.001-0.01
* **Solution**: Reduce max_symbols in greedy decoding

**Issue**: Out of memory during training

* **Solution**: Reduce fused_batch_size
* **Solution**: Enable gradient accumulation: increase accumulate_grad_batches
* **Solution**: Reduce batch_size
* **Solution**: Use mixed precision: set trainer.precision to 16 or "bf16"

Advanced Features
----------------

Stateless Decoder
~~~~~~~~~~~~~~~~

For very fast inference, consider using the stateless decoder variant:

.. code-block:: yaml

    model:
      decoder:
        _target_: nemo.collections.asr.modules.StatelessTransducerDecoder
        context_size: 2  # Use 2 words as context
        normalization_mode: layer

Hybrid TDT-CTC Models
~~~~~~~~~~~~~~~~~~~~

Combine TDT with CTC for improved accuracy:

.. code-block:: yaml

    model:
      aux_ctc:
        ctc_loss_weight: 0.3  # 30% CTC, 70% TDT
        decoder:
          _target_: nemo.collections.asr.modules.ConvASRDecoder
      
      loss:
        loss_name: "tdt"
        tdt_kwargs:
          sigma: 0.02  # Use lower sigma for hybrid models

References
----------

* `TDT Paper: Token-and-Duration Transducer for ASR <https://arxiv.org/abs/2304.06795>`__
* `FastEmit Paper <https://arxiv.org/abs/2010.11148>`__
* `NeMo ASR Documentation <https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/asr/intro.html>`__
* `Parakeet-TDT Model Cards <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2>`__
* `Example Config Files <https://github.com/NVIDIA/NeMo/tree/main/examples/asr/conf/conformer/tdt>`__

Code Examples
------------

Python API Usage
~~~~~~~~~~~~~~~

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    from omegaconf import OmegaConf
    
    # Load and customize TDT config
    config = OmegaConf.load('examples/asr/conf/conformer/tdt/conformer_tdt_bpe.yaml')
    
    # Customize TDT parameters
    config.model.model_defaults.tdt_durations = [0, 1, 2, 3, 4, 5, 6]
    config.model.model_defaults.num_tdt_durations = 7
    config.model.loss.tdt_kwargs.sigma = 0.05
    config.model.loss.tdt_kwargs.omega = 0.1
    config.model.loss.tdt_kwargs.fastemit_lambda = 0.001
    
    # Create model
    model = nemo_asr.models.EncDecRNNTBPEModel(cfg=config.model, trainer=trainer)
    
    # Train model
    trainer.fit(model)

Fine-tuning Pretrained TDT Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    
    # Load pretrained TDT model
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    
    # Update training parameters for fine-tuning
    model.cfg.optim.lr = 0.0001  # Lower learning rate for fine-tuning
    model.cfg.loss.tdt_kwargs.sigma = 0.05
    
    # Setup datasets
    model.setup_training_data(train_config)
    model.setup_validation_data(val_config)
    
    # Fine-tune
    trainer.fit(model)

Inference with TDT Model
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import nemo.collections.asr as nemo_asr
    
    # Load model
    model = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2")
    
    # Set decoding parameters
    model.cfg.decoding.strategy = "greedy"
    model.cfg.decoding.model_type = "tdt"
    model.cfg.decoding.durations = [0, 1, 2, 3, 4]
    model.change_decoding_strategy(model.cfg.decoding)
    
    # Transcribe
    transcriptions = model.transcribe(["/path/to/audio1.wav", "/path/to/audio2.wav"])
    print(transcriptions)
