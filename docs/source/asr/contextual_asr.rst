.. _context_aware_asr_fastconformer:

Context-Aware ASR with FastConformer
====================================

Introduction
------------

Context-aware Automatic Speech Recognition (ASR) refers to the capability of an ASR model to utilize information from preceding utterances or related text as contextual input to improve the accuracy and relevance of its current transcription. This is particularly beneficial for:

*   **Resolving Ambiguity**: Words or phrases that sound similar can be disambiguated based on the surrounding conversation.
*   **Improving Accuracy for Domain-Specific Terms**: Acronyms, jargon, or named entities (e.g., product names, person names) that have appeared in recent context are more likely to be transcribed correctly.
*   **Enhancing Coherence**: The transcription of the current utterance can be made more coherent with the preceding dialogue.

This document describes how to use the context-aware ASR feature with FastConformer models in NeMo. The implementation currently focuses on using the immediate previous utterance as text context.

How it Works (Briefly)
----------------------

The context-aware ASR functionality in NeMo's FastConformer models involves modifications primarily to the encoder architecture:

1.  **Modified Encoder Layers**: The ``ConformerEncoder`` and its constituent ``ConformerLayer`` modules have been enhanced.
2.  **Text Context Processing**:
    *   A dedicated embedding layer (``torch.nn.Embedding``) is used to convert the tokenized input text context into dense vector representations.
    *   A linear projection layer then maps these embeddings to the dimensionality expected by the attention mechanism in the Conformer layers.
3.  **Cross-Attention Fusion**:
    *   Each ``ConformerLayer`` now includes a cross-attention module.
    *   This module allows the acoustic features (derived from the current audio input) to attend to the processed text context embeddings from the previous utterance.
    *   The output of this cross-attention mechanism is then fused with the audio representation, allowing the model to incorporate contextual cues during transcription.

Manifest File Format
--------------------

To train or fine-tune a context-aware ASR model, your manifest files need to include the contextual information. A new field, ``text_context`` (string), should be added to your JSON manifest entries.

**Example:**

.. code-block:: json

    {"audio_filepath": "path/to/audio.wav", "duration": 10.0, "text": "this is the transcript", "text_context": "this is the previous utterance context"}
    {"audio_filepath": "path/to/another.wav", "duration": 5.0, "text": "another transcript"}
    {"audio_filepath": "path/to/audio3.wav", "duration": 7.0, "text": "a third example", "text_context": ""}

**Notes**:

*   The ``text_context`` field should contain the transcript of the utterance immediately preceding the current audio sample.
*   If a ``text_context`` field is omitted for an entry, or if its value is an empty string, it will be treated as an empty context (no contextual information will be used for that specific sample).
*   The dataset classes (``AudioToBPEDataset``, ``TarredAudioToBPEDataset``) will tokenize this string using the model's tokenizer.

Model Configuration
-------------------

To enable and configure the context-aware capabilities in your FastConformer model, you need to update the ``model.encoder`` section of your YAML configuration file.

The following new parameters are available:

*   ``text_vocab_size`` (int):
    *   Specifies the size of the vocabulary used for the input ``text_context``. This should match the vocabulary size of the tokenizer being used for the context.
*   ``text_d_model`` (int):
    *   Defines the hidden dimensionality of the internal text embedding layer before it's projected to ``d_model`` (the Conformer's main hidden size) for the cross-attention mechanism.
*   ``cross_attention_model`` (str, default: ``'abs_pos'``):
    *   Determines the type of cross-attention mechanism to be used within the ``ConformerLayer``.
    *   Currently, only ``'abs_pos'`` (absolute positional encoding based attention) is supported for the cross-attention between audio and text context.

**Example Snippet (YAML):**

.. code-block:: yaml

    model:
      # ... other model configurations ...
      encoder:
        _target_: nemo.collections.asr.modules.ConformerEncoder
        # ... other conformer encoder parameters (d_model, n_layers, etc.) ...
        text_vocab_size: 128          # Example: Size of your BPE tokenizer vocab for context
        text_d_model: 256           # Example: Internal dimension for text embeddings
        cross_attention_model: 'abs_pos'  # Explicitly set or rely on default

The main ASR model class for instantiation, such as ``nemo.collections.asr.models.EncDecCTCModelBPE``, remains the same. The new encoder parameters will be passed down when the model is instantiated.

Tokenizer for Text Context
--------------------------

The current implementation uses the **same tokenizer** for both:
1.  Tokenizing the target ``text`` field (the transcript to be predicted).
2.  Tokenizing the input ``text_context`` field.

This tokenizer is specified under the ``model.tokenizer`` section in your YAML configuration file (e.g., pointing to a SentencePiece BPE model). Ensure that this tokenizer is appropriate for both the target language and the language of the context you are providing.

Example Usage
-------------

NeMo provides an example script and configuration to demonstrate fine-tuning a FastConformer model with context-aware capabilities:

*   **Fine-tuning Script**: ``<NeMo_ROOT>/examples/asr/speech_to_text_context_aware_finetune.py``
*   **Configuration File**: ``<NeMo_ROOT>/examples/asr/conf/fastconformer/fast-conformer_ctc_bpe_context_aware.yaml``

Users are encouraged to adapt this example script and configuration file for their specific datasets and fine-tuning requirements. Remember to update placeholders in the YAML (like manifest paths and tokenizer paths) to point to your actual data and tokenizer model.

Fine-tuning
-----------

To effectively leverage this feature, a model typically needs to be **fine-tuned** on a dataset that includes the ``text_context`` field in its manifest entries. The standard FastConformer models available from NGC or through ``from_pretrained()`` are generally **not** pre-trained with this contextual capability by default.

During fine-tuning, the model will learn to utilize the provided ``text_context`` through the newly added text encoder and cross-attention mechanisms in the Conformer layers. This allows the model to adapt its predictions based on the contextual information, leading to potential improvements in transcription accuracy and robustness, especially for contextually dependent speech.
