# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pytest
import torch

from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder


class TestStochasticDepth:
    """Testing stochastic depth functionality."""

    def test_stochastic_depth_model_creation(self):
        """Testing basic model creation and the drop probs are correctly assigned."""
        n_layers = 4
        model = ConformerEncoder(feat_in=10, n_layers=n_layers, d_model=4, feat_out=8)

        # checking that by default SD is disabled
        assert model.layer_drop_probs == [0.0] * n_layers

        # linear mode
        for drop_prob in [0.3, 0.5, 0.9]:
            for start_layer in [1, 3]:
                model = ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_drop_prob=drop_prob,
                    stochastic_depth_start_layer=start_layer,
                )
                L = n_layers - start_layer
                assert model.layer_drop_probs == [0.0] * start_layer + [drop_prob * l / L for l in range(1, L + 1)]

        # uniform mode
        for drop_prob in [0.3, 0.5, 0.9]:
            model = ConformerEncoder(
                feat_in=10,
                n_layers=n_layers,
                d_model=4,
                feat_out=8,
                stochastic_depth_drop_prob=drop_prob,
                stochastic_depth_mode="uniform",
                stochastic_depth_start_layer=start_layer,
            )
            L = n_layers - start_layer
            assert model.layer_drop_probs == [0.0] * start_layer + [drop_prob] * L

        # checking for errors
        for drop_prob in [-1.0, 1.0]:
            with pytest.raises(ValueError, match="stochastic_depth_drop_prob has to be in"):
                ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_drop_prob=drop_prob,
                    stochastic_depth_mode="uniform",
                )

        with pytest.raises(ValueError, match="stochastic_depth_mode has to be one of"):
            ConformerEncoder(feat_in=10, n_layers=n_layers, d_model=4, feat_out=8, stochastic_depth_mode="weird")

        for start_layer in [-1, 0, 5]:
            with pytest.raises(ValueError, match="stochastic_depth_start_layer has to be in"):
                ConformerEncoder(
                    feat_in=10,
                    n_layers=n_layers,
                    d_model=4,
                    feat_out=8,
                    stochastic_depth_start_layer=start_layer,
                )

    @pytest.mark.pleasefixme
    def test_stochastic_depth_forward(self):
        """Testing that forward works and we get randomness during training, but not during eval."""
        random_input = torch.rand((1, 2, 2))
        random_length = torch.tensor([2], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=2,
            n_layers=3,
            d_model=4,
            feat_out=4,
            stochastic_depth_drop_prob=0.8,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        model.train()
        outputs = [None] * 5
        for i in range(5):
            outputs[i] = model(audio_signal=random_input, length=random_length)[0]
        # checking that not all outputs are the same
        num_diff = 0
        for i in range(1, 5):
            if not torch.allclose(outputs[i], outputs[0]):
                num_diff += 1
        assert num_diff > 0

        model.eval()
        outputs = [None] * 5
        for i in range(5):
            outputs[i] = model(audio_signal=random_input, length=random_length)[0]
        # checking that not all outputs are the same
        num_diff = 0
        for i in range(1, 5):
            if not torch.allclose(outputs[i], outputs[0]):
                num_diff += 1
        assert num_diff == 0


class TestBypassPreEncode:
    """Testing bypass pre-encode functionality."""

    def test_bypass_pre_encode_forward(self):
        """Testing that forward works with "bypass pre-encode" mode."""
        # For pre-encoded embeddings, the shape is (batch_size, n_frames, emb_dim)
        batch_size = 2
        n_frames, emb_dim, feat_out = 17, 16, 8
        random_input = torch.rand((batch_size, n_frames, emb_dim))
        random_length = torch.tensor([n_frames], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=10,
            n_layers=3,
            d_model=emb_dim,
            feat_out=feat_out,
            stochastic_depth_drop_prob=0.0,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        model.train()
        fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=random_input, length=random_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

    def test_error_shape_invalid_bypass_pre_encode_forward(self):
        """
        Testing that error messages are correctly triggered regarding "bypass pre-encode" mode.
        Both correct samples and wrongs samples are tested.

        (1) bypass_pre_encode = False (default):
            `audio_signal` must be a tensor containing audio features.
            Shape: (batch, self._feat_in, n_frames)
        (2) bypass_pre_encode = True:
            `audio_signal` must be a tensor containing pre-encoded embeddings.
            Shape: (batch, n_frame, self.d_model)
        """
        batch_size = 2
        n_frames, emb_dim, feat_in, feat_out = 17, 16, 10, 8

        pre_encode_input = torch.rand((batch_size, n_frames, emb_dim))
        feat_input = torch.rand((batch_size, feat_in, n_frames))
        input_length = torch.tensor([n_frames], dtype=torch.int64)

        model = ConformerEncoder(
            feat_in=feat_in,
            n_layers=3,
            d_model=emb_dim,
            feat_out=feat_out,
            stochastic_depth_drop_prob=0.0,
            dropout=0.0,
            dropout_pre_encoder=0.0,
            dropout_emb=0.0,
            conv_norm_type="layer_norm",
            conv_kernel_size=3,
        )
        sub_sampled_n_frames = np.ceil(n_frames / model.subsampling_factor)

        # Test with bypass_pre_encode = True, should be pre_encode_input but given feat_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=feat_input, length=input_length, bypass_pre_encode=True)

        # Test with bypass_pre_encode = True, given the correct input pre_encode_input.
        model.train()
        fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        model.eval()
        fwd_outputs = model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=True)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, n_frames)

        # Test with bypass_pre_encode = False, should be feat_input but given pre_encode_input.
        model.train()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        model.eval()
        with pytest.raises(ValueError):
            model(audio_signal=pre_encode_input, length=input_length, bypass_pre_encode=False)

        # Test with bypass_pre_encode = False, given the correct input feat_input.
        model.train()
        fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)


class TestConformerEncoderTextContext:
    @pytest.mark.unit
    def test_conformer_encoder_with_text_context(self):
        batch_size = 4
        feat_in = 80
        audio_seq_len = 100
        d_model = 64
        text_vocab_size = 10
        text_d_model_internal = 32 # Internal embedding dim for text before projection
        text_seq_len = 20
        subsampling_factor = 4 # Default for ConformerEncoder usually implies ConvSubsampling with factor 4

        encoder = ConformerEncoder(
            feat_in=feat_in,
            n_layers=2,
            d_model=d_model,
            ff_expansion_factor=1, # Smaller for faster test
            n_heads=2,
            conv_kernel_size=3,
            subsampling_factor=subsampling_factor, # Make it explicit
            text_vocab_size=text_vocab_size,
            text_d_model=text_d_model_internal,
            cross_attention_model='abs_pos',
        )
        encoder.eval()

        audio_signal = torch.randn(batch_size, feat_in, audio_seq_len)
        # Lengths should be <= audio_seq_len
        lengths = torch.randint(audio_seq_len // 2, audio_seq_len + 1, (batch_size,), dtype=torch.long)
        lengths[0] = audio_seq_len # Ensure at least one max length sample for consistency

        text_context = torch.randint(0, text_vocab_size, (batch_size, text_seq_len))

        # Forward with text context
        encoded_audio_ctx, encoded_len_ctx = encoder(
            audio_signal=audio_signal.clone(), length=lengths.clone(), text_context=text_context.clone()
        )

        expected_encoded_len = torch.ceil(lengths.float() / subsampling_factor).long()
        # Output shape: (B, D_model, T_encoded)
        assert encoded_audio_ctx.shape[0] == batch_size
        assert encoded_audio_ctx.shape[1] == d_model
        # Max length of encoded audio, considering padding from other samples in batch
        assert encoded_audio_ctx.shape[2] == expected_encoded_len.max().item()
        torch.testing.assert_close(encoded_len_ctx, expected_encoded_len)


        # Forward without text context
        encoded_audio_no_ctx, encoded_len_no_ctx = encoder(
            audio_signal=audio_signal.clone(), length=lengths.clone(), text_context=None
        )
        assert encoded_audio_no_ctx.shape == encoded_audio_ctx.shape
        torch.testing.assert_close(encoded_len_no_ctx, encoded_len_ctx)

        # Check that outputs are different
        assert not torch.allclose(encoded_audio_ctx, encoded_audio_no_ctx, atol=1e-6)

    @pytest.mark.unit
    def test_conformer_encoder_no_text_encoder_with_context(self):
        # Test behavior when text_context is provided but text embedding layers are not configured
        batch_size = 2
        feat_in = 80
        audio_seq_len = 50
        d_model = 32
        text_seq_len = 10
        subsampling_factor = 4

        encoder = ConformerEncoder(
            feat_in=feat_in, n_layers=1, d_model=d_model, subsampling_factor=subsampling_factor,
            text_vocab_size=None, text_d_model=None # Text encoder parts not configured
        )
        encoder.eval()

        audio_signal = torch.randn(batch_size, feat_in, audio_seq_len)
        lengths = torch.tensor([audio_seq_len, audio_seq_len - 10], dtype=torch.long)
        text_context = torch.randint(0, 10, (batch_size, text_seq_len)) # Dummy context

        # Expect a warning to be logged (if logging.warning_once was used)
        # For now, just check graceful execution and expected output shape
        encoded_audio, encoded_len = encoder(
            audio_signal=audio_signal, length=lengths, text_context=text_context
        )

        expected_encoded_len = torch.ceil(lengths.float() / subsampling_factor).long()
        assert encoded_audio.shape[0] == batch_size
        assert encoded_audio.shape[1] == d_model
        assert encoded_audio.shape[2] == expected_encoded_len.max().item()
        torch.testing.assert_close(encoded_len, expected_encoded_len)

    @pytest.mark.unit
    def test_conformer_encoder_text_embedding_passthrough(self):
        from unittest.mock import patch

        batch_size = 2
        feat_in = 80
        audio_seq_len = 50
        d_model = 64
        text_vocab_size = 10
        text_d_model_internal = 32
        text_seq_len = 10

        encoder = ConformerEncoder(
            feat_in=feat_in,
            n_layers=1, # Test with one layer for simplicity
            d_model=d_model,
            text_vocab_size=text_vocab_size,
            text_d_model=text_d_model_internal,
            cross_attention_model='abs_pos',
        )
        encoder.eval()

        audio_signal = torch.randn(batch_size, feat_in, audio_seq_len)
        lengths = torch.tensor([audio_seq_len, audio_seq_len - 5], dtype=torch.long)
        text_context = torch.randint(0, text_vocab_size, (batch_size, text_seq_len))

        # Patch the forward method of the ConformerLayer(s)
        # We target the actual layer instance within the encoder's ModuleList
        assert len(encoder.layers) > 0, "Encoder should have at least one ConformerLayer"

        # Path to the forward method of the first ConformerLayer
        # The string should be 'module_instance_in_test.method_to_patch'
        # Here, encoder.layers[0] is the instance.
        with patch.object(encoder.layers[0], 'forward', wraps=encoder.layers[0].forward) as mock_layer_forward:
            encoder(audio_signal=audio_signal.clone(), length=lengths.clone(), text_context=text_context.clone())

            mock_layer_forward.assert_called()
            # call_args is a tuple (args, kwargs) or CallArgs (args, kwargs) namedtuple
            # We are interested in kwargs
            called_kwargs = mock_layer_forward.call_args.kwargs
            assert 'text_context_embedding' in called_kwargs
            assert called_kwargs['text_context_embedding'] is not None
            assert called_kwargs['text_context_embedding'].shape == (batch_size, text_seq_len, d_model)

        # Test without text_context
        with patch.object(encoder.layers[0], 'forward', wraps=encoder.layers[0].forward) as mock_layer_forward_no_ctx:
            encoder(audio_signal=audio_signal.clone(), length=lengths.clone(), text_context=None)

            mock_layer_forward_no_ctx.assert_called()
            called_kwargs_no_ctx = mock_layer_forward_no_ctx.call_args.kwargs
            assert 'text_context_embedding' in called_kwargs_no_ctx
            assert called_kwargs_no_ctx['text_context_embedding'] is None

    @pytest.mark.unit
    def test_conformer_encoder_text_param_validation(self):
        # Should raise ValueError if only one of text_vocab_size or text_d_model is provided
        with pytest.raises(ValueError, match="Both text_vocab_size and text_d_model must be provided if one is."):
            ConformerEncoder(feat_in=80, n_layers=1, d_model=64, text_vocab_size=10, text_d_model=None)

        with pytest.raises(ValueError, match="Both text_vocab_size and text_d_model must be provided if one is."):
            ConformerEncoder(feat_in=80, n_layers=1, d_model=64, text_vocab_size=None, text_d_model=32)

        # Should work fine if both are provided or both are None
        try:
            ConformerEncoder(feat_in=80, n_layers=1, d_model=64, text_vocab_size=10, text_d_model=32)
            ConformerEncoder(feat_in=80, n_layers=1, d_model=64, text_vocab_size=None, text_d_model=None)
        except ValueError:
            pytest.fail("ConformerEncoder instantiation failed unexpectedly for valid text_param configurations.")

        model.eval()
        fwd_outputs = model(audio_signal=feat_input, length=input_length, bypass_pre_encode=False)[0]
        assert fwd_outputs.shape == (batch_size, feat_out, sub_sampled_n_frames)
