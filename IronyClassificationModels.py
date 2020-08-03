import torch
import torch.nn as nn

import math

import time


"""class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout_p: float = 0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_multi_head_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_p)
        self.dropout_0 = nn.Dropout(p=dropout_p)
        self.layer_norm_0 = nn.LayerNorm(normalized_shape=d_model)
        self.fc = nn.Linear(in_features=d_model, out_features=d_model)
        self.dropout_1 = nn.Dropout(p=dropout_p)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model)

        #self.activation_function = nn._get_activation_fn(activation)
        self.activation_fn = nn.GELU()

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = nn.GELU()
        super(TransformerEncoderLayer, self).__setstate__()

    def forward(self, x: torch.Tensor):
        x_2 = self.self_multi_head_attn(query=x, key=x, value=x)[0]
        x = x + self.dropout_0(x_2)
        x = self.layer_norm_0(x)

        x_2 = self.fc(x)
        x = x + self.dropout_1(x_2)
        x = self.layer_norm_1(x)

        return x"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_timescale: int = 1.0e4, max_seq_len: int = 5000, start_index: int = 0):
        """
        Position Encoder for transformer network. Adds position encodings to word embeddings.

        Args:
            d_model (int): dim of word embedding vector.
            max_timescale (int): choose depending on d_model (increase d_model -> decrease max_timescale).
            max_seq_len (int): maximum sequence length.
            start_index (int): start position index.
        """

        super(PositionalEncoding, self).__init__()

        position_encoding = torch.empty(max_seq_len, d_model)

        position = torch.arange(start_index, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                             (- math.log(max_timescale) / d_model))     # OpenAI's position encoding based on BERT
        # using position encoding as in the paper, not as in the code
        position_encoding[:, 0::2] = torch.sin(div_term * position)     # for every even embedding vector index
        position_encoding[:, 1::2] = torch.cos(div_term * position)     # for every odd embedding vector index
        position_encoding = position_encoding.unsqueeze(1)

        self.register_buffer('position_encoding', position_encoding)        # position encoding is not trainable

    def forward(self, x):
        x = x + self.positional_encoding[:x.shape[0]]

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_tokens, d_model, n_heads, n_hid, n_layers, dropout_p=0.5):
        super(TransformerEncoder, self).__init__()
        self.src_mask = None
        self.word_embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        # TODO: audio spectrogram embedding
        self.positional_encoder = PositionalEncoding(d_model, dropout_p)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=n_hid, dropout=dropout_p, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)


x = PositionalEncoding(
    d_model=512
)
