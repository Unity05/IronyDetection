import torch
import torch.nn as nn

import math
import warnings
from typing import Optional, Any
import torch.multiprocessing as mp

import time


"""
ASR model parts
"""


class LayerNorm(nn.Module):
    def __init__(self, n_features):
        super(LayerNorm, self).__init__()

        self.layer_norm = nn.LayerNorm(n_features)

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.layer_norm(x)

        return x.transpose(2, 3)


class ResidualCNN(nn.Module):
    """
    Residual Networks: https://arxiv.org/pdf/1512.03385.pdf

    Structure:
    input -> LayerNorm -> ActivationFunction -> Dropout -> Conv2d -> ... -> output + input -> output
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p, n_features):
        super(ResidualCNN, self).__init__()

        self.conv_block = nn.Sequential(
            LayerNorm(n_features=n_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=int(kernel_size / 2)),
            LayerNorm(n_features=n_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=int(kernel_size / 2))
        )

    def forward(self, x):
        identity = x
        x = self.conv_block(x)
        x += identity

        return x


class BidirectionalGRU(nn.Module):
    """
    Bidirectional GRU block.

    Structure:
    input -> LayerNorm -> ActivationFunction -> BidirectionalGRU -> Dropout -> output
    """

    def __init__(self, input_size, hidden_size, dropout_p, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.preprocessing_block = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_size),
            nn.ReLU()
        )
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=batch_first,
                          bidirectional=True)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):
        x = self.preprocessing_block(x)
        x, _ = self.gru(x)
        x = self.dropout(x)

        return x


class SpeechModel(nn.Module):
    """
    Pretty similar to 'Deep Speech 2': https://arxiv.org/pdf/1512.02595.pdf

    Structure:
    spectrogram -> InitialConv2d -> ResConv2d - Layers -> Linear - (Transition - / Connection -) Layer -> BiGRU - Layers -> Classifier
    """

    def __init__(self, n_res_cnn_layers, n_bi_gru_layers, bi_gru_dim, n_classes, n_features, dropout_p, device, dataset,
                 d_audio_embedding):
        super(SpeechModel, self).__init__()

        self.dataset = dataset

        self.device = device

        # InitialConv2d
        self.init_conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        n_features = int(n_features / 2)        # feature dimension decreased by first Conv2d - Layer

        # General ResidualConv2d - Layers
        self.residual_cnn_general = nn.Sequential(
            *[ResidualCNN(in_channels=32, out_channels=32, kernel_size=3, stride=1, dropout_p=dropout_p,
                          n_features=n_features) for i in range(int(n_res_cnn_layers / 2))]
        )

        # Specific ResidualConv2d - Layers
        self.residual_cnn_specific = nn.Sequential(
            *[ResidualCNN(in_channels=32, out_channels=32, kernel_size=3, stride=1, dropout_p=dropout_p,
                          n_features=n_features) for i in range((n_res_cnn_layers - int(n_res_cnn_layers / 2)))]
        )

        # Linear - (Transition - / Connection -) Layer
        self.linear_layer_connection = nn.Linear(in_features=n_features * 32, out_features=bi_gru_dim)

        # BidirectionalGRU - Layers
        self.bi_gru_nn = nn.Sequential(
            *[BidirectionalGRU(input_size=bi_gru_dim if i == 0 else bi_gru_dim * 2, hidden_size=bi_gru_dim,
                               dropout_p=dropout_p, batch_first=(i == 0)) for i in range(n_bi_gru_layers)]
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=bi_gru_dim * 2 if n_bi_gru_layers > 1 else bi_gru_dim, out_features=bi_gru_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features=bi_gru_dim, out_features=n_classes)
        )

        # Decoder
        self.decoder = FinalDecoder(input_size=1, hidden_size=1024, output_size=9897, device=device)


        # Audio Embedding model for irony classificaton model
        self.audio_embedding = AudioEmbedding(n_features=n_features, dropout_p=dropout_p,
                                              d_audio_embedding=d_audio_embedding)

    def do_audio_embedding(self, x):
        x = self.audio_embedding(x)

        return x

    def forward(self, x, this_model_train):

        x = self.init_conv2d(x)
        x = self.residual_cnn_general(x)

        # Start parallel audio processing for following irony classification
        #self.audio_embedding(x, mp.Queue())
        #print('Hi.')
        #audio_embedding_input = x.data
        #print('is_leaf: ', audio_embedding_input.is_leaf)
        #print('audio_embedding_input: ', audio_embedding_input)
        #print('x.data: ', x.data)
        if not self.training and not this_model_train:
            mp.set_start_method('spawn')

            audio_embedding_return = mp.Queue()
            audio_embedding_process = mp.Process(
                target=self.audio_embedding,
                args=(x.clone(), audio_embedding_return)
            )
            #print('Okay.')
            audio_embedding_process.start()
            #self.do_audio_embedding(x=x.copy())
            #print('Finished.')
            #exit(-1)
        else:
            conv_output_for_audio_embedding = x.clone()

        x = self.residual_cnn_specific(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).transpose(1, 2)     # reshape for linear layer
        x = self.linear_layer_connection(x)
        x = self.bi_gru_nn(x)

        x = self.classifier(x)

        if this_model_train is True:        # for seperatly training the ASR model
            return x

        x = nn.Softmax(dim=2)(x) * 1.0e3        # Why? Comment out?

        if not self.training:
            # Adjust audio embeddings with ASR - classifier outputs
            audio_embedding_process.join()
            adjusted_audio_embedding_return = mp.Queue()
            adjusted_audio_embedding_process = mp.Process(
                target=self.audio_embedding.audio_embedding_adjustments,
                args=(audio_embedding_return.get(), x, adjusted_audio_embedding_return)
            )
            adjusted_audio_embedding_process.start()
        else:
            audio_embedding_return = self.audio_embedding(conv_output_for_audio_embedding)
            adjusted_audio_embedding = self.audio_embedding.audio_embedding_adjustments(
                audio_embeddings=audio_embedding_return, asr_classifications=x)

        output, original_word_lens = self.decoder(x=x)

        if not self.training:
            # Get adjusted audio embeddings
            adjusted_audio_embedding = adjusted_audio_embedding_return.get()

        if self.training:
            return output, adjusted_audio_embedding
        else:
            return output, adjusted_audio_embedding, original_word_lens


class FinalDecoder(nn.Module):
    """
    Matches letter wise output from SpeechModel with the corresponding word index from vocabulary.

    Structure:
    input_words (letter wise) -> feed forward -> word index
    """

    def __init__(self, input_size, hidden_size, output_size, device):
        super(FinalDecoder, self).__init__()

        self.device = device

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=False)
        self.fc_0 = nn.Linear(in_features=hidden_size, out_features=(hidden_size * 2))
        self.fc_1 = nn.Linear(in_features=(hidden_size * 2), out_features=int(output_size / 2))
        self.fc_2 = nn.Linear(in_features=int(output_size / 2), out_features=output_size)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.softmax = nn.Softmax(dim=2)

    def data_preprocessing(self, x):
        # split letter classifier output in word tensors
        word_tensors = []
        original_word_lens = []
        start_index = 0
        previous_batch_i = 0

        word_info_tuples = torch.cat(((x.argmax(dim=2) == 1).nonzero(), torch.Tensor(
            [[batch_i.item(), (x.shape[1] - 1)] for batch_i in torch.unique((x.argmax(dim=2) == 1)[:, 0])]).long().to(
            self.device)))

        for word_index in word_info_tuples:
            batch_i, end_char_i = word_index
            if end_char_i == start_index:
                continue
            if batch_i != previous_batch_i:
                start_index = 0
                previous_batch_i = batch_i

            word_tensor = torch.argmax(x[batch_i.item()][start_index: (end_char_i.item() + 1)], dim=1, keepdim=True)
            adjusted_words = []
            prev_i = 1
            for char_i in word_tensor[:-1]:
                char_i = char_i.item()
                if char_i is not prev_i:
                    if char_i is 28:
                        prev_i = 28
                    else:
                        adjusted_words.append(char_i)
                        prev_i = char_i
            word_tensor = torch.Tensor(adjusted_words).unsqueeze(1)
            word_tensors.append(word_tensor)
            original_word_lens.append(word_tensor.shape[0])
            start_index = end_char_i.item() + 1

        if len(word_tensors) == 0:
            word_tensors = [y for y in x]

        word_tensors = torch.nn.utils.rnn.pad_sequence(sequences=word_tensors, padding_value=0.0,
                                                       batch_first=False).to(self.device)  # (char, n_words, n_classes)

        return word_tensors, original_word_lens

    def forward(self, x):
        if not self.training:
            x, original_word_lens = self.data_preprocessing(x=x)        # dtype int to float
        output, hidden = self.gru(x)
        output = self.elu(self.fc_0(output))
        output = self.elu(self.fc_1(output))
        output = self.fc_2(output)
        output = self.log_softmax(output)

        if self.training:
            return output
        else:
            return output, original_word_lens


"""
Irony classifier model parts
"""


class AudioEmbedding(nn.Module):
    def __init__(self, n_features, dropout_p, d_audio_embedding):
        super(AudioEmbedding, self).__init__()

        self.conv_block = nn.Sequential(
            LayerNorm(n_features=n_features),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1,
                      padding=int(3 / 2))
        )

        self.fc_0 = nn.Linear(in_features=(32 * n_features), out_features=d_audio_embedding)
        self.fc_1 = nn.Linear(in_features=d_audio_embedding, out_features=d_audio_embedding)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor, audio_embedding_return=None):
        #print(self.device)
        #print('is_cuda: ', next(self.parameters()).is_cuda)
        #print(x.shape)
        x = self.conv_block(x)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).transpose(1, 2)     # reshape for linear layer

        x = self.dropout(self.activation_fn(self.fc_0(x)))
        x = self.dropout(self.activation_fn(self.fc_1(x)))

        # return_value.put(x)
        if audio_embedding_return is None:
            return x
        else:
            audio_embedding_return.put(x)

    def audio_embedding_adjustments(self, audio_embeddings: torch.Tensor, asr_classifications: torch.Tensor):
        # TODO

        return audio_embeddings


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


class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 12, d_feedforward: int = 2048, dropout_p: float = 0.1, activation: str = 'ReLU'):
        super(CustomTransformerDecoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_p)
        self.dropout_0 = nn.Dropout(p=dropout_p)

        self.layer_norm_0 = nn.LayerNorm(normalized_shape=d_model)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout_p)
        self.dropout_1 = nn.Dropout(p=dropout_p)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model)

        self.fc_0 = nn.Linear(in_features=d_model, out_features=d_feedforward)
        self.dropout_2 = nn.Dropout(p=dropout_p)

        self.fc_1 = nn.Linear(in_features=d_feedforward, out_features=d_model)
        self.dropout_3 = nn.Dropout(p=dropout_p)

        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_model)

        self.activation_fn = getattr(nn, activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            warnings.warn(message='"state" does not contain "activation". "nn.ReLU()" is used as default.')
            state['activation'] = nn.ReLU()
        super(CustomTransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt2 = self.self_attn(query=tgt, key=tgt, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout_0(tgt2)
        tgt = self.layer_norm_0(tgt)

        tgt2 = self.multihead_attn(query=tgt, key=memory, value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + memory + self.dropout_1(tgt2)
        tgt = self.layer_norm_1(tgt)

        tgt2 = self.dropout_2(self.activation_fn(self.fc_0(tgt)))
        tgt2 = self.fc_1(tgt2)
        tgt = tgt + self.dropout_3(tgt2)
        tgt = self.layer_norm_2(tgt)

        return tgt


class Transformer(nn.Module):
    def __init__(self, n_tokens, d_model, n_heads, n_hid, n_layers, dropout_p=0.5):
        super(Transformer, self).__init__()

        self.word_embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        # TODO: audio spectrogram embedding
        self.positional_encoder = PositionalEncoding(d_model, dropout_p)

        # encoder definition
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=n_hid,
                                                   dropout=dropout_p, activation='gelu')
        transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

        # decoder definition
        decoder_layer = CustomTransformerDecoderLayer(d_model=d_model, n_heads=n_heads, d_feedforward=n_hid,
                                                      dropout_p=dropout_p, activation='GELU')
        transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=6)

        self.transformer = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=6, num_decoder_layers=6,
                                          dim_feedforward=n_hid, dropout=dropout_p, activation='GELU',
                                          custom_encoder=transformer_encoder, custom_decoder=transformer_decoder)

    def generate_decoder_tgt_mask(self, T: int) -> torch.Tensor:
        """
        Generates a triangular target mask for transformer decoder input.

        Args:
            T (int): target sequence length.

        Returns:
            (torch.Tensor) triangular target mask of shape (T, T).
        """

        mask = (torch.triu(torch.ones((T, T))) == 0).transpose(1, 0)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))

        return mask

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if tgt_mask is None or tgt_mask.shape[0] is not tgt.shape[0]:
            tgt_mask = self.generate_decoder_tgt_mask(T=tgt.shape[0]).to(self.device)

        output = self.transformer(src=src, tgt=tgt, tgt_mask=tgt_mask)

        return output


class IronyClassifier(nn.Module):
    def __init__(self, batch_size, n_tokens, d_model, d_context, n_heads, n_hid, n_layers, dropout_p=0.5):
        super(IronyClassifier, self).__init__()

        self.batch_size = batch_size
        self.d_model = d_model
        self.d_context = d_context

        self.word_embedding = nn.Embedding(num_embeddings=n_tokens, embedding_dim=d_model)
        self.positional_encoder = PositionalEncoding(d_model, dropout_p)

        # encoder definition
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=n_hid,
                                                   dropout=dropout_p, activation='gelu')
        transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

    def generate_src_mask(self, utterance_lens: tuple) -> torch.Tensor:
        max_len = max(utterance_lens)
        # src_mask = torch.Tensor(([0] * utterance_lens[0]) + ([float('-inf')] * (1 - utterance_lens[0])))

        src_mask = []

        for current_len in utterance_lens:
            src_mask.append(([0] * current_len) + ([float('-inf')] * (max_len - current_len)))

        src_mask = torch.Tensor(src_mask)

        return src_mask

    def generate_context(self) -> torch.Tensor:
        context_tensor = torch.zeros((self.d_context, self.batch_size))

        return context_tensor

    def forward(self, src: torch.Tensor, utterance_lens: tuple, context_tensor: torch.Tensor):
        # get src mask
        src_mask = self.generate_src_mask(utterance_lens=utterance_lens)
        src = self.word_embedding(src) * math.sqrt(self.d_model)
        src = self.positional_encoder(src)
        out = self.transformer_encoder(src, src_mask)
