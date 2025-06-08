import torch
import torch.nn as nn
from nemo.core.classes import NeuralModule
from nemo.core.neural_types import NeuralType, LengthsType, EncodedRepresentation, TokenIndex

class SimpleTextContextEncoder(NeuralModule):
    """
    A simple encoder for textual context, using an embedding layer
    followed by a Transformer encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int, # This will be d_model for the Transformer
        num_layers: int = 1,
        num_attention_heads: int = 4,
        ff_expansion_factor: int = 4, # d_ffn = hidden_size * ff_expansion_factor
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # Assuming 0 is pad_id

        # Transformer Encoder Layer
        # PyTorch's TransformerEncoderLayer expects d_model (hidden_size) as input
        # If embedding_dim is different from hidden_size, a projection might be needed.
        # For simplicity, let's assume embedding_dim == hidden_size for now,
        # or add a linear layer if they must differ.
        # Let's make embedding_dim the primary dimension and use it as d_model for the transformer.
        # So, hidden_size will be equivalent to embedding_dim in this context.

        if embedding_dim != hidden_size:
            # This case is common. Embedding dim can be different from transformer d_model.
            # Let's rename hidden_size to d_model for clarity in transformer context.
            # The input to TransformerEncoderLayer should be (seq_len, batch, d_model)
            # Embedding output is (batch, seq_len, embedding_dim)
            # So, we'll make d_model = embedding_dim for simplicity here.
            # If a different d_model is desired for the transformer itself,
            # a projection layer would be needed after embedding or hidden_size should be d_model.
            # Let's stick to hidden_size being the d_model of the transformer.
            # Add a projection if embedding_dim != hidden_size
             self.input_projection = nn.Linear(embedding_dim, hidden_size)
        else:
            self.input_projection = None

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=hidden_size * ff_expansion_factor,
            dropout=dropout,
            activation='relu', # Common activation
            batch_first=True  # Important: NeMo generally uses batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_encoder_layer,
            num_layers=num_layers
        )
        self._d_model = hidden_size


    @property
    def input_types(self):
        return {
            "context_tokens": NeuralType(('B', 'T_context'), TokenIndex()),
            "context_tokens_length": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context_encoded": NeuralType(('B', 'T_context', 'D_context_encoded'), EncodedRepresentation()),
            "context_encoded_length": NeuralType(('B'), LengthsType()),
        }

    @property
    def d_model(self):
        return self._d_model

    def forward(self, context_tokens: torch.Tensor, context_tokens_length: torch.Tensor):
        # context_tokens: (batch_size, seq_len)
        embedded_context = self.embedding(context_tokens)  # (batch_size, seq_len, embedding_dim)

        if self.input_projection is not None:
            embedded_context = self.input_projection(embedded_context) # (batch_size, seq_len, hidden_size)

        # Create src_key_padding_mask for TransformerEncoder
        # Mask should be True for padding positions, False for non-padding
        # context_tokens_length gives the unpadded length for each sequence in batch
        max_len = context_tokens.size(1)
        indices = torch.arange(max_len, device=context_tokens_length.device).unsqueeze(0) # (1, max_len)
        src_key_padding_mask = indices >= context_tokens_length.unsqueeze(1) # (batch_size, max_len)

        # TransformerEncoder expects input of shape (seq_len, batch_size, hidden_size) if batch_first=False (default)
        # or (batch_size, seq_len, hidden_size) if batch_first=True. We set batch_first=True.
        encoded_context = self.transformer_encoder(
            src=embedded_context,
            src_key_padding_mask=src_key_padding_mask
        ) # (batch_size, seq_len, hidden_size)

        return encoded_context, context_tokens_length # Lengths remain the same after transformer if padding is handled


class ContextualFusionModule(NeuralModule):
    """
    A module to fuse audio encodings with text context encodings using cross-attention.
    Audio encodings attend to text context encodings.
    """

    def __init__(
        self,
        audio_d_model: int,         # Dimension of audio encoder output
        text_context_d_model: int,  # Dimension of text context encoder output
        num_attention_heads: int = 4,
        dropout: float = 0.1,
        activation: str = 'relu' # activation for the optional feedforward layer
    ):
        super().__init__()

        self.audio_d_model = audio_d_model
        self.text_context_d_model = text_context_d_model

        # MultiheadAttention expects query, key, value to have the same embed_dim (d_model)
        # If audio_d_model and text_context_d_model are different,
        # we need projections for key and value (from text_context) to audio_d_model,
        # or project query (audio) to text_context_d_model.
        # Let's project text K, V to audio_d_model.
        # The output of attention will then be audio_d_model.

        if audio_d_model != text_context_d_model:
            self.key_proj = nn.Linear(text_context_d_model, audio_d_model)
            self.value_proj = nn.Linear(text_context_d_model, audio_d_model)
        else:
            self.key_proj = None
            self.value_proj = None

        self.attention = nn.MultiheadAttention(
            embed_dim=audio_d_model,  # This is the dimension of queries, and keys/values after projection
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True  # NeMo standard
        )

        self.norm1 = nn.LayerNorm(audio_d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Optional: Add a PositionwiseFeedForward layer like in Transformer blocks
        # For simplicity, we'll make this optional or a simple version first.
        # Let's include a simple one for now.
        self.ffn = nn.Sequential(
            nn.Linear(audio_d_model, audio_d_model * 4), # Expansion factor of 4
            # nn.GELU() if activation == 'gelu' else nn.ReLU(), # Common activations
            nn.ReLU(), # Sticking to ReLU as specified in TransformerEncoderLayer earlier
            nn.Dropout(dropout),
            nn.Linear(audio_d_model * 4, audio_d_model)
        )
        self.norm2 = nn.LayerNorm(audio_d_model)
        self.dropout2 = nn.Dropout(dropout)


    @property
    def input_types(self):
        return {
            "audio_encoded": NeuralType(('B', 'T_audio', 'D_audio_encoded'), EncodedRepresentation()),
            "audio_encoded_length": NeuralType(('B'), LengthsType()),
            "text_context_encoded": NeuralType(('B', 'T_context', 'D_context_encoded'), EncodedRepresentation()),
            "text_context_encoded_length": NeuralType(('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "contextualized_audio_encoded": NeuralType(('B', 'T_audio', 'D_audio_encoded'), EncodedRepresentation()),
            "contextualized_audio_encoded_length": NeuralType(('B'), LengthsType()),
        }

    def forward(
        self,
        audio_encoded: torch.Tensor,          # (B, T_audio, D_audio)
        audio_encoded_length: torch.Tensor,   # (B) - not directly used by MHA if src_key_padding_mask is not for query
        text_context_encoded: torch.Tensor,   # (B, T_context, D_text_context)
        text_context_encoded_length: torch.Tensor # (B)
    ):
        query = audio_encoded
        key = text_context_encoded
        value = text_context_encoded

        if self.key_proj is not None:
            key = self.key_proj(key)
        if self.value_proj is not None:
            value = self.value_proj(value)

        # Create key_padding_mask for the text_context (keys and values)
        # True for padded positions, False for actual tokens
        max_text_len = text_context_encoded.size(1)
        text_indices = torch.arange(max_text_len, device=text_context_encoded_length.device).unsqueeze(0) # (1, T_context)
        # text_context_encoded_length is (B), needs to be (B, 1) for broadcasting
        text_key_padding_mask = text_indices >= text_context_encoded_length.unsqueeze(1) # (B, T_context)

        # attn_output: (B, T_audio, D_audio)
        # attn_weights: (B, T_audio, T_context) - not used here but good to know
        attn_output, _ = self.attention(
            query=query,
            key=key,
            value=value,
            key_padding_mask=text_key_padding_mask
        )

        # First residual connection block (Attention)
        contextualized_audio = query + self.dropout1(attn_output)
        contextualized_audio = self.norm1(contextualized_audio)

        # Second residual connection block (FFN)
        ffn_output = self.ffn(contextualized_audio)
        contextualized_audio = contextualized_audio + self.dropout2(ffn_output)
        contextualized_audio = self.norm2(contextualized_audio)

        return contextualized_audio, audio_encoded_length # Length of audio sequence remains unchanged
