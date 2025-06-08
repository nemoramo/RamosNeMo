import torch
import logging
from nemo.collections.asr.data.audio_to_text_dataset import AudioToBPEDataset, AudioTextDatasetSample
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.asr.parts.preprocessing.segment import ChannelSelectorType
from typing import Optional, List, Dict, Any, Tuple, Union

logger = logging.getLogger(__name__)

class ContextualAudioToBPEDataset(AudioToBPEDataset):
    """
    Dataset that loads audio and transcriptions, and also previous textual context.
    Modified from nemo.collections.asr.data.audio_to_text_dataset.AudioToBPEDataset.
    """

    def __init__(
        self,
        manifest_filepath: str,
        tokenizer: TokenizerSpec,
        sample_rate: float,
        prev_text_field: str = "prev_text",
        **kwargs
    ):
        """
        Args:
            manifest_filepath: Path to manifest file.
            tokenizer: TokenizerSpec instance.
            sample_rate: Sample rate of the audio.
            prev_text_field: The field name in the manifest for the previous text context.
            **kwargs: Additional keyword arguments passed to AudioToBPEDataset.
        """
        self.prev_text_field = prev_text_field
        super().__init__(manifest_filepath=manifest_filepath, tokenizer=tokenizer, sample_rate=sample_rate, **kwargs)

        # Post-process self.data to include prev_text
        # self.data is a list of AudioTextDatasetSample objects
        # self.manifest_processor.collection is a list of dicts (original manifest entries)
        if len(self.data) != len(self.manifest_processor.collection):
            logger.warning(
                "Mismatch between length of self.data and self.manifest_processor.collection. "
                "Previous text context might not be added correctly."
            )
        else:
            for i in range(len(self.data)):
                sample_in_data = self.data[i]
                original_manifest_entry = self.manifest_processor.collection[i]
                # Dynamically add prev_text to the AudioTextDatasetSample object
                setattr(sample_in_data, 'prev_text', original_manifest_entry.get(self.prev_text_field, ""))

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get (audio_features, audio_features_len, transcript_tokens, transcript_tokens_len)
        # Note: super().__getitem__ processes the audio sample from self.data[index]
        # which includes loading audio, feature extraction etc.
        features, features_len, transcript, transcript_len = super().__getitem__(index)

        sample_info = self.data[index]
        prev_text_str = getattr(sample_info, 'prev_text', "")

        tokenized_prev_text_ids = self.tokenizer.text_to_ids(prev_text_str)
        if tokenized_prev_text_ids is None:
            tokenized_prev_text_ids = []

        prev_text_tokens_tensor = torch.tensor(tokenized_prev_text_ids, dtype=torch.long)
        prev_text_tokens_len_tensor = torch.tensor(len(tokenized_prev_text_ids), dtype=torch.long)

        return (
            features,
            features_len,
            transcript,
            transcript_len,
            prev_text_tokens_tensor,
            prev_text_tokens_len_tensor,
        )

    def collate_fn(self, batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
        # Separate the components from the batch
        features_list, features_lengths_list, transcripts_list, transcript_lengths_list = [], [], [], []
        prev_texts_tokens_list, prev_text_tokens_lengths_list = [], []

        for sample in batch:
            features_list.append(sample[0])
            features_lengths_list.append(sample[1])
            transcripts_list.append(sample[2])
            transcript_lengths_list.append(sample[3])
            prev_texts_tokens_list.append(sample[4])
            # prev_text_tokens_lengths_list.append(sample[5]) # This is already a tensor

        audio_transcript_sub_batch = []
        for sample in batch:
            audio_transcript_sub_batch.append((sample[0], sample[1], sample[2], sample[3]))

        processed_audio_signal, processed_audio_signal_length, \
            processed_transcript_tokens, processed_transcript_tokens_length = \
                super().collate_fn(audio_transcript_sub_batch)

        pad_id_to_use = self.pad_id
        if pad_id_to_use is None:
            logger.error("self.pad_id is None in collate_fn. This might cause issues with padding prev_text.")
            pad_id_to_use = 0

        max_prev_text_len = 0
        if prev_texts_tokens_list:
            valid_prev_texts_tokens_list = [t for t in prev_texts_tokens_list if isinstance(t, torch.Tensor) and t.ndim > 0]
            if valid_prev_texts_tokens_list:
                max_prev_text_len = max(tokens_tensor.size(0) for tokens_tensor in valid_prev_texts_tokens_list)
            elif any(isinstance(t, torch.Tensor) and t.ndim == 0 and t.numel() == 0 for t in prev_texts_tokens_list):
                pass
            elif prev_texts_tokens_list:
                 pass

        batched_prev_text_tokens = torch.full(
            (len(batch), max_prev_text_len), fill_value=pad_id_to_use, dtype=torch.long
        )

        current_prev_text_lengths = []
        for i, tokens_tensor in enumerate(prev_texts_tokens_list):
            if isinstance(tokens_tensor, torch.Tensor):
                current_len = tokens_tensor.size(0) if tokens_tensor.ndim > 0 else 0
                if current_len > 0 : # Ensure we only copy if there's something to copy
                    batched_prev_text_tokens[i, :current_len] = tokens_tensor
                current_prev_text_lengths.append(current_len)
            else:
                current_prev_text_lengths.append(0)

        batched_prev_text_tokens_length = torch.tensor(current_prev_text_lengths, dtype=torch.long)

        return (
            processed_audio_signal,
            processed_audio_signal_length,
            processed_transcript_tokens,
            processed_transcript_tokens_length,
            batched_prev_text_tokens,
            batched_prev_text_tokens_length,
        )
