import torch
from omegaconf import DictConfig, OmegaConf, open_dict # Added OmegaConf for mutable copy
from lightning.pytorch import Trainer
from typing import Optional, Dict, Tuple, List, Any

from nemo.collections.asr.models.rnnt_models import EncDecRNNTModel
# Assuming SimpleTextContextEncoder and ContextualFusionModule are in the specified path
# and that nemo.collections.asr.modules.__init__.py makes them available.
from nemo.collections.asr.modules.contextual_encoders import SimpleTextContextEncoder, ContextualFusionModule
from nemo.core.neural_types import NeuralType, LengthsType, SpectrogramType, AcousticEncodedRepresentation, TokenIndex, LogprobsType
from nemo.utils import logging # logging was used but not imported

class ContextualEncDecRNNTModel(EncDecRNNTModel):
    """
    Encoder-decoder RNNT-based model with an additional text context encoder
    and a fusion module to make the ASR context-aware.
    """

    def __init__(self, cfg: DictConfig, trainer: Optional[Trainer] = None):
        super().__init__(cfg=cfg, trainer=trainer)

        text_encoder_cfg_in = cfg.get('text_context_encoder', None)
        if text_encoder_cfg_in is None:
            raise ValueError("Missing 'text_context_encoder' config for ContextualEncDecRNNTModel")

        # Create a mutable copy for modification using OmegaConf.to_container and then DictConfig
        text_encoder_cfg_dict = OmegaConf.to_container(text_encoder_cfg_in, resolve=True)
        text_encoder_cfg = DictConfig(text_encoder_cfg_dict) # Convert back to DictConfig if methods expect it
                                                           # or operate on dict directly

        if hasattr(self, 'tokenizer') and hasattr(self.tokenizer, 'vocab_size'):
            text_encoder_vocab_size = self.tokenizer.vocab_size
        else:
            logging.warning("Attempting to derive vocab_size from self.cfg.labels for text_context_encoder. "
                            "Ensure this is appropriate for your tokenizer setup. "
                            "self.cfg.labels includes the blank token.")
            # Typically, for a text encoder, vocab should not include ASR's blank.
            # If self.tokenizer is not available, this part is tricky.
            # For char models, self.cfg.labels might be the only source.
            # A common pattern: len(self.joint.vocabulary) is vocab_size + 1 (for blank)
            # So, len(self.joint.vocabulary) - 1 might be text_encoder_vocab_size
            # Or, if cfg.labels is just the char set without blank, then len(cfg.labels) is fine.
            # Given it's BPE, self.tokenizer.vocab_size is the primary source.
            # This fallback needs to be robust if a char-based non-tokenizer model is used.
            text_encoder_vocab_size = len(self.cfg.labels) # Placeholder if no tokenizer.vocab_size

        # Use open_dict for DictConfig if it's still a DictConfig, otherwise direct dict modification
        if isinstance(text_encoder_cfg, DictConfig):
            with OmegaConf.open_dict(text_encoder_cfg):
                 text_encoder_cfg.vocab_size = text_encoder_vocab_size
            self.text_context_encoder = SimpleTextContextEncoder(**text_encoder_cfg)
        else: # if it was converted to a plain dict
             text_encoder_cfg['vocab_size'] = text_encoder_vocab_size
             self.text_context_encoder = SimpleTextContextEncoder(**text_encoder_cfg)


        fusion_cfg_in = cfg.get('contextual_fusion_module', None)
        if fusion_cfg_in is None:
            raise ValueError("Missing 'contextual_fusion_module' config for ContextualEncDecRNNTModel")

        fusion_cfg_dict = OmegaConf.to_container(fusion_cfg_in, resolve=True)
        # No need to convert back to DictConfig if module takes dict kwargs

        fusion_cfg_dict['audio_d_model'] = self.encoder.d_model
        fusion_cfg_dict['text_context_d_model'] = self.text_context_encoder.d_model
        self.contextual_fusion_module = ContextualFusionModule(**fusion_cfg_dict)


    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        base_input_types = super().input_types
        if base_input_types is None:
            base_input_types = {}

        context_input_types = {
            "prev_text_tokens": NeuralType(('B', 'T_prev_text'), TokenIndex()),
            "prev_text_tokens_length": NeuralType(('B'), LengthsType()),
        }
        updated_input_types = {**base_input_types, **context_input_types}
        return updated_input_types

    def forward(
        self,
        input_signal: Optional[torch.Tensor] = None,
        input_signal_length: Optional[torch.Tensor] = None,
        processed_signal: Optional[torch.Tensor] = None,
        processed_signal_length: Optional[torch.Tensor] = None,
        prev_text_tokens: Optional[torch.Tensor] = None,
        prev_text_tokens_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        audio_encoded, audio_encoded_len = super().forward(
            input_signal=input_signal,
            input_signal_length=input_signal_length,
            processed_signal=processed_signal,
            processed_signal_length=processed_signal_length
        )

        if prev_text_tokens is None or prev_text_tokens_length is None or torch.all(prev_text_tokens_length == 0):
            if self.training:
                # This should ideally not happen if data loader is correct
                logging.error("prev_text_tokens or prev_text_tokens_length is None/empty during training!")
                # Depending on strictness, could raise error or try to proceed (might fail in fusion)
                # For now, let it proceed, fusion module might handle zero-length tensors if designed for it.
                # Or, more safely, return audio_encoded directly if context is unusable.
                return audio_encoded, audio_encoded_len
            else:
                logging.debug("No previous text context provided or context is empty, bypassing fusion.")
            return audio_encoded, audio_encoded_len

        text_context_encoded, _ = self.text_context_encoder(
            context_tokens=prev_text_tokens,
            context_tokens_length=prev_text_tokens_length
        )

        contextualized_audio_encoded, _ = self.contextual_fusion_module( # length remains audio_encoded_len
            audio_encoded=audio_encoded,
            audio_encoded_length=audio_encoded_len,
            text_context_encoded=text_context_encoded,
            text_context_encoded_length=prev_text_tokens_length
        )

        return contextualized_audio_encoded, audio_encoded_len # Return original audio_encoded_len

    def _prepare_forward_inputs_from_batch(self, batch: Any) -> Dict[str, Optional[torch.Tensor]]:
        """Helper to extract all necessary inputs for the forward method from a batch."""
        # This filters for None to ensure only present signals are passed to super().forward()
        # and also extracts prev_text parts.
        kwargs = {}
        if isinstance(batch, dict):
            kwargs["input_signal"] = batch.get('audio_signal')
            kwargs["input_signal_length"] = batch.get('audio_signal_length')
            kwargs["processed_signal"] = batch.get('processed_signal')
            kwargs["processed_signal_length"] = batch.get('processed_signal_length')
            kwargs["prev_text_tokens"] = batch.get('prev_text_tokens')
            kwargs["prev_text_tokens_length"] = batch.get('prev_text_tokens_length')
        else:
            kwargs["input_signal"] = batch[0]
            kwargs["input_signal_length"] = batch[1]
            # Processed signal is not typically in tuple batch, parent forward handles it
            kwargs["prev_text_tokens"] = batch[4]
            kwargs["prev_text_tokens_length"] = batch[5]

        return {k: v for k, v in kwargs.items() if v is not None}

    def _get_transcripts_from_batch(self, batch: Any) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if isinstance(batch, dict):
            return batch.get('transcript'), batch.get('transcript_length')
        return batch[2], batch[3]

    def training_step(self, batch: Any, batch_idx: int):
        forward_inputs = self._prepare_forward_inputs_from_batch(batch)
        transcript, transcript_len = self._get_transcripts_from_batch(batch)

        encoded, encoded_len = self.forward(**forward_inputs)

        decoder_output, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)

        # Simplified for non-fused loss, full logic in parent EncDecRNNTModel.training_step
        joint_output = self.joint(encoder_outputs=encoded, decoder_outputs=decoder_output)
        loss_value = self.loss(
            log_probs=joint_output, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
        )
        loss_value = self.add_auxiliary_losses(loss_value)

        tensorboard_logs = {
            'train_loss': loss_value,
            'learning_rate': self._optimizer.param_groups[0]['lr'],
            'global_step': torch.tensor(self.trainer.global_step, dtype=torch.float32),
        }

        if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
             if hasattr(self, 'wer') and self.wer is not None and not self.joint.fuse_loss_wer:
                self.wer.update(
                    predictions=encoded, predictions_lengths=encoded_len,
                    targets=transcript, targets_lengths=transcript_len
                )
                # Batch WER logging is complex; parent model might have specific way.
                # For now, just update. Epoch end WER is more standard.
                # _, scores, words = self.wer.compute() # This might reset or be inaccurate for batch.
                # self.wer.reset()
                # tensorboard_logs['training_batch_wer'] = scores.float() / words

        self.log_dict(tensorboard_logs, sync_dist=True) # Added sync_dist
        if hasattr(self, '_optim_normalize_joint_txu') and self._optim_normalize_joint_txu:
             self._optim_normalize_txu = [encoded_len.max(), transcript_len.max()]
        return {'loss': loss_value}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        forward_inputs = self._prepare_forward_inputs_from_batch(batch)
        transcript, transcript_len = self._get_transcripts_from_batch(batch)

        encoded, encoded_len = self.forward(**forward_inputs)

        # Simplified loss calculation for validation from parent's validation_pass
        val_loss = None
        if self.compute_eval_loss and not self.joint.fuse_loss_wer:
            decoder_output, target_length, _ = self.decoder(targets=transcript, target_length=transcript_len)
            joint_output = self.joint(encoder_outputs=encoded, decoder_outputs=decoder_output)
            val_loss = self.loss(
                log_probs=joint_output, targets=transcript, input_lengths=encoded_len, target_lengths=target_length
            )

        if hasattr(self, 'wer') and self.wer is not None and not self.joint.fuse_loss_wer:
            self.wer.update(
                predictions=encoded, predictions_lengths=encoded_len,
                targets=transcript, targets_lengths=transcript_len
            )

        metrics = {}
        if val_loss is not None:
            metrics[f'val_loss_dl_{dataloader_idx}'] = val_loss # Log per dataloader

        # Store metrics for epoch end calculation (as done in parent)
        # Parent uses self.validation_step_outputs.append(metrics_dict_from_validation_pass)
        # where metrics_dict_from_validation_pass contains 'val_loss', 'val_wer_num', 'val_wer_denom'
        # For simplicity here, we are just passing what we computed.
        # The actual wer_num/denom for aggregation would come from self.wer at epoch_end.
        # This part needs to align with how multi_validation_epoch_end expects data.
        # Let's ensure it has at least the loss if computed.
        # For WER, parent's multi_validation_epoch_end computes it from self.wer's accumulated state.

        # To align with parent's `multi_validation_epoch_end` which sums 'val_wer_num' and 'val_wer_denom'
        # from step outputs, we should try to provide them if possible, even if placeholders for now.
        # However, these are not computed directly in parent's val_step, but in validation_pass.
        # The parent's validation_step calls self.validation_pass which returns these.
        # Our structure is a bit different. We'll rely on self.wer being updated.
        # The multi_validation_epoch_end in parent does:
        #   wer_num = torch.stack([x['val_wer_num'] for x in outputs]).sum()
        #   wer_denom = torch.stack([x['val_wer_denom'] for x in outputs]).sum()
        # This implies `validation_step` needs to return these keys if they are to be aggregated this way.
        # This is a known simplification point.

        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            if not hasattr(self, 'validation_step_outputs') or self.validation_step_outputs is None : self.validation_step_outputs = [[] for _ in range(len(self.trainer.val_dataloaders))]
            self.validation_step_outputs[dataloader_idx].append(metrics)
        else:
            if not hasattr(self, 'validation_step_outputs') or self.validation_step_outputs is None : self.validation_step_outputs = []
            self.validation_step_outputs.append(metrics)

        # Log actual computed metrics, not the list for accumulation
        self.log_dict({k:v for k,v in metrics.items() if isinstance(v, torch.Tensor)}, sync_dist=True)
        return metrics # Return what was logged for PTL internal use if any

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> List[Tuple[Any, Any]]:
        forward_inputs = self._prepare_forward_inputs_from_batch(batch)

        sample_ids = []
        if isinstance(batch, dict):
            sample_ids = batch.get('sample_id', []) # Ensure it's a list
        elif hasattr(batch, 'id'):
            sample_ids = [batch.id] # Ensure list for zip
        else: # Fallback if no explicit sample_id
            # Get batch size from one of the length tensors if possible
            bs = 1
            if forward_inputs.get("input_signal_length") is not None:
                bs = forward_inputs["input_signal_length"].size(0)
            elif forward_inputs.get("processed_signal_length") is not None:
                bs = forward_inputs["processed_signal_length"].size(0)
            elif forward_inputs.get("prev_text_tokens_length") is not None:
                 bs = forward_inputs["prev_text_tokens_length"].size(0)
            sample_ids = [f"dl{dataloader_idx}_b{batch_idx}_s{i}" for i in range(bs)]

        encoded, encoded_len = self.forward(**forward_inputs)

        hypotheses = self.decoding.rnnt_decoder_predictions_tensor(
            encoder_output=encoded, encoded_lengths=encoded_len, return_hypotheses=True
        )

        if isinstance(sample_ids, torch.Tensor): # Should be list of strings or numbers
            sample_ids = sample_ids.cpu().detach().numpy().tolist()

        if len(sample_ids) != len(hypotheses):
            logging.warning(f"Mismatch in length of sample_ids ({len(sample_ids)}) and hypotheses ({len(hypotheses)}) in predict_step. Truncating to shorter.")
            min_len = min(len(sample_ids), len(hypotheses))
            sample_ids = sample_ids[:min_len]
            hypotheses = hypotheses[:min_len]

        return list(zip(sample_ids, hypotheses))
