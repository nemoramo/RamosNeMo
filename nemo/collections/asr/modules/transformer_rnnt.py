# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

from nemo.collections.asr.modules import rnnt_abstract
from nemo.collections.asr.modules.transformer.transformer_encoders import TransformerEncoder
from nemo.collections.asr.modules.transformer.transformer_modules import FixedPositionalEncoding
from nemo.collections.asr.parts.utils import adapter_utils, rnnt_utils
from nemo.collections.common.parts import rnn
from nemo.core.classes import adapter_mixins, typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.neural_types import LabelsType, LengthsType, NeuralType, EmbeddedTextType

__all__ = ["TransformerRNNTDecoder"]


class _TransformerRNNTEmbedding(nn.Module):
    """Token + positional embedding that allows custom padding / blank id."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        padding_idx: Optional[int],
        max_sequence_length: int = 4096,
        embedding_dropout: float = 0.0,
        learn_positional_encodings: bool = False,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=padding_idx)
        if learn_positional_encodings:
            self.position_embedding = nn.Embedding(max_sequence_length, hidden_size)
        else:
            self.position_embedding = FixedPositionalEncoding(hidden_size, max_sequence_length)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(embedding_dropout)

    def forward(self, tokens: torch.Tensor, start_pos: Union[int, torch.Tensor] = 0) -> torch.Tensor:
        seq_length = tokens.size(1)
        if isinstance(start_pos, torch.Tensor):
            position_ids = torch.arange(seq_length, device=tokens.device).unsqueeze(0)
            position_ids = position_ids + start_pos.unsqueeze(1)
        else:
            position_ids = torch.arange(start_pos, start_pos + seq_length, device=tokens.device).unsqueeze(0)
            position_ids = position_ids.expand(tokens.size(0), -1)

        embeddings = self.token_embedding(tokens) + self.position_embedding(position_ids)
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerRNNTDecoder(rnnt_abstract.AbstractRNNTDecoder, Exportable, adapter_mixins.AdapterModuleMixin):
    """
    Causal Transformer-based RNNT/TDT prediction network.

    Uses a Transformer encoder stack with future-masked self-attention and per-layer KV caching.

    Args:
        prednet: Dict-like object with keys:
            pred_hidden: model dimension of the decoder.
            num_layers: number of Transformer layers.
            inner_size: feed-forward hidden size.
            num_attention_heads: number of attention heads.
            attn_score_dropout / attn_layer_dropout / ffn_dropout: dropout probabilities.
            embedding_dropout: embedding dropout probability.
            max_sequence_length: maximum positional embedding length.
            learn_positional_encodings: whether to learn positional encodings.
        vocab_size: Vocabulary size excluding blank token.
        blank_idx: Index of the blank token (0 or vocab_size). Defaults to vocab_size.
        blank_as_pad: When True, reserves an extra embedding slot so blank is also padding id.
        pre_ln: Whether to use pre-layer-norm Transformer blocks.
        pre_ln_final_layer_norm: Whether to append a final LayerNorm when pre_ln is enabled.
    """

    @property
    def input_types(self):
        return {
            "targets": NeuralType(('B', 'T'), LabelsType()),
            "target_length": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType(('B', 'T'), LabelsType(), optional=True)],
        }

    @property
    def output_types(self):
        return {
            "outputs": NeuralType(('B', 'D', 'T'), EmbeddedTextType()),
            "prednet_lengths": NeuralType(tuple('B'), LengthsType()),
            "states": [NeuralType(('B', 'T'), LabelsType(), optional=True)],
        }

    def input_example(self, max_batch=1, max_dim=1):
        targets = torch.full((max_batch, max_dim), fill_value=self.blank_idx, dtype=torch.int32).to(
            next(self.parameters()).device
        )
        target_length = torch.randint(0, max_dim, size=(max_batch,), dtype=torch.int32).to(targets.device)
        states = None
        return (targets, target_length, states)

    def __init__(
        self,
        prednet: Dict[str, Any],
        vocab_size: int,
        blank_idx: Optional[int] = None,
        blank_as_pad: bool = True,
        pre_ln: bool = False,
        pre_ln_final_layer_norm: bool = True,
    ):
        self.pred_hidden = prednet['pred_hidden']
        self.num_layers = prednet['num_layers']
        self.blank_idx = vocab_size if blank_idx is None else blank_idx
        self.blank_as_pad = blank_as_pad

        super().__init__(vocab_size=vocab_size, blank_idx=self.blank_idx, blank_as_pad=blank_as_pad)

        inner_size = prednet.get('inner_size', 4 * self.pred_hidden)
        num_attention_heads = prednet.get('num_attention_heads', 8)
        attn_score_dropout = prednet.get('attn_score_dropout', 0.0)
        attn_layer_dropout = prednet.get('attn_layer_dropout', 0.0)
        ffn_dropout = prednet.get('ffn_dropout', 0.0)
        embedding_dropout = prednet.get('embedding_dropout', 0.0)
        max_sequence_length = prednet.get('max_sequence_length', 4096)
        learn_positional_encodings = prednet.get('learn_positional_encodings', False)

        vocab_for_embedding = vocab_size + 1 if self.blank_idx == vocab_size else vocab_size
        padding_idx = self.blank_idx if blank_as_pad else None

        self.embedding = _TransformerRNNTEmbedding(
            vocab_size=vocab_for_embedding,
            hidden_size=self.pred_hidden,
            padding_idx=padding_idx,
            max_sequence_length=max_sequence_length,
            embedding_dropout=embedding_dropout,
            learn_positional_encodings=learn_positional_encodings,
        )

        self.transformer = TransformerEncoder(
            num_layers=self.num_layers,
            hidden_size=self.pred_hidden,
            inner_size=inner_size,
            mask_future=True,
            num_attention_heads=num_attention_heads,
            attn_score_dropout=attn_score_dropout,
            attn_layer_dropout=attn_layer_dropout,
            ffn_dropout=ffn_dropout,
            hidden_act=prednet.get('hidden_act', 'relu'),
            pre_ln=pre_ln,
            pre_ln_final_layer_norm=pre_ln_final_layer_norm,
        )

        self._rnnt_export = False

    def _unwrap_state(
        self, state: Optional[Union[Tuple[Sequence[torch.Tensor], torch.Tensor], Sequence[torch.Tensor]]]
    ) -> Tuple[Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        if state is None:
            return None, None

        if isinstance(state, (list, tuple)) and len(state) == 2 and isinstance(state[0], (list, tuple)):
            mems = [m for m in state[0]]
            lengths = state[1]
        else:
            mems = [m for m in state] if isinstance(state, (list, tuple)) else None
            lengths = None
        return mems, lengths

    def _pack_state(
        self, mems: Optional[List[torch.Tensor]], lengths: Optional[torch.Tensor]
    ) -> Optional[Tuple[List[torch.Tensor], Optional[torch.Tensor]]]:
        if mems is None:
            return None
        return mems, lengths

    def _prepare_tokens(
        self,
        y: Optional[torch.Tensor],
        state_lengths: Optional[torch.Tensor],
        add_sos: bool,
        batch_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if y is None:
            if batch_size is not None:
                B = batch_size
            elif state_lengths is not None:
                B = state_lengths.shape[0]
            else:
                B = 1
            y = torch.full((B, 1), fill_value=self.blank_idx, device=self._device, dtype=torch.long)
        else:
            y = rnn.label_collate(y)
            if y.device != self._device:
                y = y.to(self._device)

        B = y.size(0)
        prev_lengths = (
            state_lengths.to(self._device) if state_lengths is not None else torch.zeros(B, device=self._device, dtype=torch.long)
        )

        if add_sos:
            sos = torch.full((B, 1), fill_value=self.blank_idx, device=self._device, dtype=torch.long)
            y = torch.cat([sos, y], dim=1)

        return y, prev_lengths

    @property
    def _device(self):
        return next(self.parameters()).device

    @typecheck()
    def forward(self, targets, target_length, states=None):
        if self._rnnt_export:
            add_sos = False
        else:
            add_sos = True

        g, states = self.predict(targets, state=states, add_sos=add_sos, target_length=target_length)
        g = g.transpose(1, 2)
        return g, target_length, states

    def predict(
        self,
        y: Optional[torch.Tensor] = None,
        state: Optional[Union[Tuple[Sequence[torch.Tensor], torch.Tensor], Sequence[torch.Tensor]]] = None,
        add_sos: bool = True,
        batch_size: Optional[int] = None,
        target_length: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[List[torch.Tensor], torch.Tensor]]]:
        mems, prev_lengths = self._unwrap_state(state)
        y, prev_lengths = self._prepare_tokens(y, prev_lengths, add_sos=add_sos, batch_size=batch_size)

        if target_length is not None:
            eff_lengths = target_length.to(self._device) + (1 if add_sos else 0)
            decoder_mask = (
                torch.arange(y.size(1), device=self._device, dtype=torch.long).unsqueeze(0)
                < eff_lengths.unsqueeze(1)
            ).long()
        else:
            decoder_mask = torch.ones_like(y, dtype=torch.long)

        embeddings = self.embedding(y, start_pos=prev_lengths)
        mems_out = self.transformer(
            encoder_states=embeddings,
            encoder_mask=decoder_mask,
            encoder_mems_list=mems,
            return_mems=True,
        )

        new_lengths = prev_lengths + y.size(1)
        final_hidden = mems_out[-1]
        # Extract only newly generated portion (exclude cached history).
        new_outputs = final_hidden[:, -y.size(1) :, :]

        return new_outputs, self._pack_state(mems_out, new_lengths)

    def initialize_state(self, y: torch.Tensor) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
        batch = y.size(0)
        zero_lengths = torch.zeros(batch, device=y.device, dtype=torch.long)
        return [], zero_lengths

    def batch_initialize_states(
        self, decoder_states: List[Optional[Tuple[Sequence[torch.Tensor], torch.Tensor]]]
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
        if decoder_states is None or len(decoder_states) == 0 or all(s is None for s in decoder_states):
            return None

        mems_per_sample = []
        lengths_per_sample = []
        for state in decoder_states:
            mems, lengths = self._unwrap_state(state)
            if mems is None or lengths is None:
                return None
            mems_per_sample.append([m for m in mems])
            lengths_per_sample.append(lengths)

        lengths_tensor = torch.stack(lengths_per_sample, dim=0).to(self._device)
        num_layers = len(mems_per_sample[0])
        stacked_mems: List[torch.Tensor] = []
        for layer in range(num_layers):
            layer_mems = [mems_per_sample[b][layer] for b in range(len(mems_per_sample))]
            max_len = max(m.size(1) for m in layer_mems)
            padded = []
            for mem in layer_mems:
                if mem.size(1) < max_len:
                    pad_len = max_len - mem.size(1)
                    pad = torch.zeros(mem.size(0), pad_len, mem.size(2), device=mem.device, dtype=mem.dtype)
                    mem = torch.cat([mem, pad], dim=1)
                padded.append(mem)
            stacked = torch.stack(padded, dim=0).to(self._device)  # [B, T, H]
            stacked_mems.append(stacked)

        return stacked_mems, lengths_tensor

    def batch_select_state(
        self, batch_states: Optional[Tuple[Sequence[torch.Tensor], torch.Tensor]], idx: int
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
        if batch_states is None:
            return None
        mems, lengths = self._unwrap_state(batch_states)
        if mems is None or lengths is None:
            return None
        selected_mems = [m[idx : idx + 1] for m in mems]
        selected_lengths = lengths[idx : idx + 1]
        return self._pack_state(selected_mems, selected_lengths)

    def batch_concat_states(
        self, batch_states: List[Optional[Tuple[Sequence[torch.Tensor], torch.Tensor]]]
    ) -> Optional[Tuple[List[torch.Tensor], torch.Tensor]]:
        return self.batch_initialize_states(batch_states)

    @classmethod
    def batch_replace_states_mask(
        cls,
        src_states: Tuple[Sequence[torch.Tensor], torch.Tensor],
        dst_states: Tuple[Sequence[torch.Tensor], torch.Tensor],
        mask: torch.Tensor,
        other_src_states: Optional[Tuple[Sequence[torch.Tensor], torch.Tensor]] = None,
    ):
        src_mems, src_lengths = src_states
        dst_mems, dst_lengths = dst_states
        other_mems, other_lengths = other_src_states if other_src_states is not None else dst_states
        mask_bool = mask.to(dtype=torch.bool)
        mask_exp = mask_bool.unsqueeze(-1).unsqueeze(-1)
        for i in range(len(dst_mems)):
            torch.where(mask_exp, src_mems[i].to(dst_mems[i].dtype), other_mems[i].to(dst_mems[i].dtype), out=dst_mems[i])
        torch.where(mask_bool, src_lengths.to(dst_lengths.dtype), other_lengths.to(dst_lengths.dtype), out=dst_lengths)

    @classmethod
    def batch_replace_states_all(
        cls, src_states: Tuple[Sequence[torch.Tensor], torch.Tensor], dst_states: Tuple[Sequence[torch.Tensor], torch.Tensor],
    ):
        src_mems, src_lengths = src_states
        dst_mems, dst_lengths = dst_states
        for i in range(len(dst_mems)):
            dst_mems[i].copy_(src_mems[i])
        dst_lengths.copy_(src_lengths)

    @classmethod
    def clone_state(
        cls, state: Tuple[Sequence[torch.Tensor], torch.Tensor]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        mems, lengths = state
        return [m.clone() for m in mems], lengths.clone()

    @classmethod
    def batch_aggregate_states(
        cls,
        src_states: Tuple[Sequence[torch.Tensor], torch.Tensor],
        batch_size: int,
        beam_size: int,
        indices: torch.Tensor,
        dst_states: Optional[Tuple[Sequence[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
        src_mems, src_lengths = src_states
        flat_indices = indices.view(-1)

        if dst_states is None:
            new_mems: List[torch.Tensor] = []
        else:
            new_mems = list(dst_states[0])

        for i, mem in enumerate(src_mems):
            gathered = mem.index_select(0, flat_indices)
            if dst_states is None:
                new_mems.append(gathered)
            else:
                new_mems[i].copy_(gathered)

        new_lengths = src_lengths.index_select(0, flat_indices)
        if dst_states is not None:
            dst_states[1].copy_(new_lengths)
            return dst_states
        return new_mems, new_lengths

    @classmethod
    def batch_split_states(
        cls, batch_states: Tuple[Sequence[torch.Tensor], torch.Tensor]
    ) -> List[Tuple[List[torch.Tensor], torch.Tensor]]:
        mems, lengths = batch_states
        return [( [m[i : i + 1] for m in mems], lengths[i : i + 1]) for i in range(lengths.shape[0])]

    @classmethod
    def batch_unsplit_states(
        cls, batch_states: List[Tuple[Sequence[torch.Tensor], torch.Tensor]], device=None, dtype=None
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        mems_per_layer: List[List[torch.Tensor]] = []
        lengths = []
        for mems, l in batch_states:
            lengths.append(l.squeeze(0))
            for i, m in enumerate(mems):
                if len(mems_per_layer) <= i:
                    mems_per_layer.append([])
                mems_per_layer[i].append(m)
        stacked_mems: List[torch.Tensor] = []
        for layer_mems in mems_per_layer:
            stacked_mems.append(torch.cat(layer_mems, dim=0).to(device=device, dtype=dtype))
        stacked_lengths = torch.stack(lengths, dim=0).to(device=device, dtype=dtype if dtype else None)
        return stacked_mems, stacked_lengths

    def mask_select_states(
        self, states: Tuple[Sequence[torch.Tensor], torch.Tensor], mask: torch.Tensor
    ) -> Tuple[Sequence[torch.Tensor], torch.Tensor]:
        mems, lengths = states
        return [m[mask] for m in mems], lengths[mask]

    def batch_copy_states(
        self,
        old_states: List[torch.Tensor],
        new_states: List[torch.Tensor],
        ids: List[int],
        value: Optional[float] = None,
    ) -> List[torch.Tensor]:
        for i in range(len(old_states)):
            if value is None:
                old_states[i][ids] = new_states[i][ids]
            else:
                old_states[i][ids] *= 0.0
                old_states[i][ids] += value
        return old_states

    def score_hypothesis(
        self, hypothesis: rnnt_utils.Hypothesis, cache: Dict[Tuple[int], Any]
    ) -> Tuple[torch.Tensor, Optional[Tuple[List[torch.Tensor], torch.Tensor]], torch.Tensor]:
        """
        Lightweight per-hypothesis scoring used by beam search implementations.

        Args:
            hypothesis: Current RNNT hypothesis being extended.
            cache: Memoized decoder outputs keyed by prefix token tuples to avoid recomputation.

        Returns:
            Tuple of decoder output for the last token, cached decoder state, and the last label id.
        """
        device = self._device
        mems = None
        if hypothesis.dec_state is not None:
            mems, lengths = self._unwrap_state(hypothesis.dec_state)
            if mems is not None and len(mems) > 0:
                device = mems[0].device
        else:
            lengths = None

        last_token = hypothesis.y_sequence[-1]
        target = torch.full((1, 1), fill_value=last_token, device=device, dtype=torch.long)
        lm_token = target[:, -1]
        sequence = tuple(hypothesis.y_sequence)

        if sequence in cache:
            y, new_state = cache[sequence]
        else:
            if len(hypothesis.y_sequence) > 0 and last_token == self.blank_idx:
                # Blank does not reset the prediction network; reuse cached transformer state.
                new_state = hypothesis.dec_state
                if new_state is None:
                    # Initial step with no cached state: seed predictor once.
                    y, new_state = self.predict(None, state=None, add_sos=True, batch_size=1)
                else:
                    if mems is None or len(mems) == 0:
                        mems, lengths = self._unwrap_state(new_state)
                    last_hidden = mems[-1]
                    if lengths is not None:
                        last_pos = (lengths - 1).clamp(min=0)
                        y = last_hidden.gather(
                            dim=1, index=last_pos.view(-1, 1, 1).expand(-1, 1, last_hidden.size(-1))
                        )
                    else:
                        y = last_hidden[:, -1:, :]
            else:
                y, new_state = self.predict(target, state=hypothesis.dec_state, add_sos=False, batch_size=1)

            y = y[:, -1:, :]
            cache[sequence] = (y, new_state)

        return y, new_state, lm_token

    # Adapter method overrides
    def add_adapter(self, name: str, cfg: DictConfig):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        super().add_adapter(name=name, cfg=cfg)

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        return adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.pred_hidden)
