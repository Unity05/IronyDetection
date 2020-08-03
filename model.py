import torch
import torch.nn as nn

import time


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
        #x, identities = x
        x = self.preprocessing_block(x)
        #print('X_SHAPE: ', x.shape)
        x, _ = self.gru(x)
        #print(x.shape)
        x = self.dropout(x)

        #identities.append(h)

        return x


"""class AttentionDecoder(nn.Module):
    """
#Nice introduction: https://www.youtube.com/watch?v=FMXUkEbjf9k .
"""
    def __init__(self, hidden_size, output_size, n_classes):
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size

        # ==== model ====
        self.attention_weight = nn.Linear(in_features=(output_size + hidden_size), out_features=1)      # computes 'e<t, t'>'
        self.gru = nn.GRU(input_size=(hidden_size + n_classes), hidden_size=output_size, num_layers=1)
        self.classifier = nn.Linear(in_features=output_size, out_features=n_classes)

    def forward(self, hidden, encoder_outputs, input):
        # ==== compute attention weights ====
        # One layer to compute 'e<t, t'>' with given previous hiddens and encoder_output at time t.
        weights = []
        for i in range(encoder_outputs.shape[1]):
            #print(hidden.shape)
            weights.append(self.attention_weight(torch.cat((hidden[0], encoder_outputs.transpose(0, 1)[i]), dim=1)))
        # Normalize attention weights, so that:     sum[t∈T](exp(e<t, t'>)) = 1     (-> Softmax)
        weights = nn.Softmax(dim=1)(torch.cat(weights, dim=1))
        #print('WEIGHTS_SHAPE: ', weights.shape)

        # ==== apply attention weights ====
        weighted_encoder_output = torch.bmm(weights.unsqueeze(1), encoder_outputs.view(1, -1, self.hidden_size))
        #print('WEIGHTED_ENCODER_OUTPUT_SHAPE: ', weighted_encoder_output.shape, ' | (SHOULD BE 2 * BI_GRU_DIM)')

        # ==== prepare RNN - Input ====
        rnn_input = torch.cat((weighted_encoder_output, input), dim=2)
        #print('RNN_INPUT_SHAPE: ', rnn_input.shape)

        # ==== forepropagation through RNN ====
        output, hidden = self.gru(rnn_input, hidden)
        #print(f'OUTPUT_SHAPE: {output.shape} | HIDDEN_SHAPE: {hidden.shape}')

        # ==== classifier ====
        output = self.classifier(output)
        #print(f'OUTPUT_SHAPE: {output.shape}')

        return output, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(batch_size, 1, self.output_size)"""


class SpeechModel(nn.Module):
    """
    Pretty similar to 'Deep Speech 2': https://arxiv.org/pdf/1512.02595.pdf

    Structure:
    spectrogram -> InitialConv2d -> ResConv2d - Layers -> Linear - (Transition - / Connection -) Layer -> BiGRU - Layers -> Classifier
    """
    def __init__(self, n_res_cnn_layers, n_bi_gru_layers, bi_gru_dim, n_classes, n_features, dropout_p, device, dataset):
        super(SpeechModel, self).__init__()

        self.dataset = dataset

        self.device = device
        #self.identity_weights = identity_weights

        # InitialConv2d
        self.init_conv2d = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2, padding=1)
        n_features = int(n_features / 2)        # feature dimension decreased by first Conv2d - Layer

        # ResidualConv2d - Layers
        self.residual_cnn = nn.Sequential(
            *[ResidualCNN(in_channels=32, out_channels=32, kernel_size=3, stride=1, dropout_p=dropout_p,
                          n_features=n_features) for i in range(n_res_cnn_layers)]
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
        self.decoder = FinalDecoder(input_size=1, hidden_size=1024, output_size=9897)
        #self.decoder = FinalDecoder(input_size=(29 + 1024), hidden_size=1, output_size=9896)

        # Attention
        #self.attention = AttentionDecoder(hidden_size=(bi_gru_dim * 2), output_size=256, n_classes=n_classes)

    """def forward(self, x):
        #print('Hi.')
        x = self.init_conv2d(x)
        x = self.residual_cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).transpose(1, 2)     # reshape for linear layer
        x = self.linear_layer_connection(x)
        x = self.bi_gru_nn(x)
        #identity = sum([a*b for a, b in zip(self.identity_weights, identities)])
        #x += identity
        #print(x.shape)"""

    """     # ==== Attention ====
        y = torch.zeros(1, 1, 29).to(self.device)
        hidden = self.attention.init_hidden().to(self.device)
        output = torch.zeros(1, 1, 29).to(self.device)
        print('Attention.')
        for i in range(x.shape[1]):
            #print('ENCODER_OUTPUTS_SHAPE: ', x.shape)
            output, hidden = self.attention(hidden, x, output)
            y = torch.cat((y, output), dim=0)
            #if torch.argmax(output) == 29:      # '<EOS>'
            #    break"""

    """     #print('Hi.')

        x = self.classifier(x)
        #print(x.shape)

        return x"""

    def forward(self, x):
        #print('Hi.')
        x = self.init_conv2d(x)
        x = self.residual_cnn(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]).transpose(1, 2)     # reshape for linear layer
        x = self.linear_layer_connection(x)
        x = self.bi_gru_nn(x)
        #identity = sum([a*b for a, b in zip(self.identity_weights, identities)])
        #x += identity
        #print('SHAPE_0: ', x.shape)

        #print('Hi.')

        x = self.classifier(x)
        print('X_SHAPE: ', x.shape)
        #print(x.shape)
        x = nn.Softmax(dim=2)(x) * 1.0e3
        #x = nn.Sigmoid()(x) * 1.0e10
        #print('SHAPE_1: ', x.shape)
        #print('X_ARGMAX: ', (x.argmax(dim=2) == 1).nonzero())
        #print(x[0][0:2])

        # split letter classifier output in word tensors
        word_tensors = []
        original_word_lens = []
        start_index = 0
        previous_batch_i = 0

        word_info_tuples = torch.cat(((x.argmax(dim=2) == 1).nonzero(), torch.Tensor([[batch_i.item(), (x.shape[1] - 1)] for batch_i in torch.unique((x.argmax(dim=2) == 1)[:, 0])]).long().to(self.device)))

        for word_index in word_info_tuples:
            batch_i, end_char_i = word_index
            if end_char_i == start_index:
                continue
            if batch_i != previous_batch_i:
                start_index = 0
                previous_batch_i = batch_i

            word_tensor = torch.argmax(x[batch_i.item()][start_index: (end_char_i.item() + 1)], dim=1, keepdim=True)
            #print('(0) ', word_tensor)
            adjusted_words = []
            prev_i = 1
            for char_i in word_tensor[:-1]:
                char_i = char_i.item()
                #print(char_i)
                if char_i is not prev_i:
                    if char_i is 28:
                        prev_i = 28
                    else:
                        adjusted_words.append(char_i)
                        prev_i = char_i
            word_tensor = torch.Tensor(adjusted_words).unsqueeze(1)
            #print('(0) ', torch.argmax(word_tensor, dim=1))
            #word_tensor = torch.argmax(word_tensor[torch.argmax(word_tensor, dim=1) != 28], dim=1, keepdim=True)
            #print('(1) ', word_tensor)
            word_tensors.append(word_tensor)
            original_word_lens.append(word_tensor.shape[0])
            start_index = end_char_i.item() + 1
        #print('N_WORDS: ', len(word_tensors))
        if len(word_tensors) == 0:
            word_tensors = [y for y in x]

        #print('(0) WORD_TENSOR_SHAPE: ', word_tensors[0].shape)
        #print('(0) WORD_TENSORS: ', word_tensors)
        #print('(0): ', torch.argmax(word_tensors[0], dim=1))
        #print('(0): ', self.dataset.indices_to_text(torch.argmax(word_tensors[0], dim=1).tolist(), policy=self.dataset.index_char_policy))
        word_tensors = torch.nn.utils.rnn.pad_sequence(sequences=word_tensors, padding_value=0.0, batch_first=False)      # (char, n_words, n_classes)
        #print('ORIGINAL_WORD_LENS: ', original_word_lens)
        #print('WORD_TENSORS_SHAPE: ', word_tensors.shape)
        #print('(1) WORD_TENSORS_SHAPE: ', word_tensors.shape)
        #print('(1) WORD_TENSORS: ', word_tensors)
        #('(1): ', torch.argmax(word_tensors.transpose(1, 0)[0], dim=1))
        #print('(1): ', self.dataset.indices_to_text(torch.argmax(word_tensors.transpose(1, 0)[0], dim=1).tolist(), policy=self.dataset.index_char_policy))

        #hidden = self.decoder.init_hidden(n_words=word_tensors.shape[1]).to(self.device)

        #word_tensors = torch.nn.utils.rnn.pack_padded_sequence(input=word_tensors, lengths=original_word_lens,
         #                                                      enforce_sorted=False, batch_first=False)
        #print('Hi: ', torch.nn.utils.rnn.pad_packed_sequence(sequence=word_tensors, batch_first=False)[0].shape)
        #word_tensors = torch.argmax(word_tensors, dim=2, keepdim=True).float()
        #print(word_tensors)
        #print(hidden.shape)
        #print('WORD_TENSOR_SHAPE: ', word_tensors.shape)
        """for char_i in word_tensors:
            #print('CHAR: ', char_i)
            output, hidden = self.decoder(x=char_i, hidden=hidden.squeeze(0))
            print('OUTPUT: ', torch.argmax(output[0], dim=1))
            print('HIDDEN: ', hidden[0])"""
            #print(hidden)
        #print(len(word_tensors))
        #print(word_tensors.shape)
        #output, hidden = self.decoder(x=word_tensors, hidden=hidden, original_word_lens=original_word_lens)
        output, hidden = self.decoder(x=word_tensors.float().to(self.device))       # dtype int to float
        #output, unpadded_output_lens= self.decoder(x=word_tensors)
        #print('DURATION: ', time.process_time() - a)

        return output, hidden, original_word_lens, word_tensors
        #return output, unpadded_output_lens


class FinalDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FinalDecoder, self).__init__()

        #self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=False)
        #nn.LSTM()
        #self.gru = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc_0 = nn.Linear(in_features=hidden_size, out_features=(hidden_size * 2))
        self.fc_1 = nn.Linear(in_features=(hidden_size * 2), out_features=int(output_size / 2))
        self.fc_2 = nn.Linear(in_features=int(output_size / 2), out_features=output_size)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, hidden=None, original_word_lens=None):
        #print('input: ', len(x[0]))
        #print('input: ', x)
        #x = x[:5]
        #print('x_shape: ', x.shape)
        #print(x)
        #x = torch.cat((torch.randn((x.shape[0] // 1), x.shape[1], x.shape[2]).to(torch.device('cuda')),
        #              torch.zeros((x.shape[0] // 10), x.shape[1], x.shape[2]).to(torch.device('cuda'))))
        #print(x.shape)
        #print(x)
        #print('hidden_shape: ', hidden.shape)
        #combined = torch.cat((x, hidden), dim=1).unsqueeze(0).transpose(1, 0)
        #print('combined: ', combined)
        #print('combined_shape: ', combined.shape)

        #output, hidden = self.gru(input=combined)
        output, hidden = self.gru(x)
        #output = nn.ELU(output)
        #print(output)
        #print(hidden.shape)
        #output = output.transpose(1, 0)
        #print(output)
        #print('OUTPUT_SHAPE: ', output.shape)
        #print('(0) LAST_HIDDEN_SHAPE: ', hidden.shape)
        #print('ORIGINAL_WORDS: ', original_word_lens)
        #for i in range(output.shape[0]):
         #   print(i)
          #  x = output[i][original_word_lens[i]]

        #output, unpadded_output_lens = torch.nn.utils.rnn.pad_packed_sequence(sequence=output, batch_first=True)

        #print(unpadded_output_lens)
        #print(output.shape)
        #hidden = torch.stack([output[i][(original_word_lens[i] - 1)] for i in range(output.shape[0])]).unsqueeze(0)
        #y = torch.stack([x[i][a[i]] for i in b])
        #print('(1) LAST_HIDDEN_SHAPE: ', hidden.shape)
        #output = hidden
        #output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence=output, batch_first=False)
        #print(output.shape)

        #output = nn.ReLU(output)
        #hidden = nn.ReLU(hidden)
        #print('output_0: ', output)
        #output = self.fc_0(output)
        #output = self.relu(output)
        #print(hidden)
        output = self.elu(self.fc_0(output))
        #output = self.relu(output)
        #output = self.elu(output)
        output = self.elu(self.fc_1(output))
        output = self.fc_2(output)
        #print(output)
        #print('OUTPUT_SHAPE: ', output.shape)
        #output = self.tanh(output)
        #print('output_1: ', output)
        output = self.log_softmax(output)
        #output = self.softmax(output)
        #print('OUTPUT_SHAPE: ', output.shape)

        return output, hidden
        #return output, unpadded_output_lens

    def init_hidden(self, n_words):
        return torch.zeros(1, n_words, self.hidden_size)
