# import torchaudio
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import json
import os
import re
import random as rnd

# import matplotlib.pyplot as plt


class Dataset:
    def __init__(self, root, url, mode, n_features, download=False):
        self.root = root

        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)

        with open(os.path.join(root, 'char_index_policy.json')) as policy_json_file:
            self.char_index_policy = json.load(policy_json_file)
        self.index_char_policy = {v: k for k, v in self.char_index_policy.items()}

        with open(os.path.join(root, 'english_word_dict_10k.json'), 'r') as word_dict_file:
            self.index_word_policy = json.load(word_dict_file)
        self.word_index_policy = {v: k for k, v in self.index_word_policy.items()}

        if mode == 'train':
            self.audio_transforms = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_features),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
                torchaudio.transforms.TimeMasking(time_mask_param=30)
            )
        else:
            self.audio_transforms = torchaudio.transforms.MelSpectrogram()

        #self.word_distribution = self.create_word_distribution()

    def create_word_distribution(self, saved=True):
        if saved:
            word_distribution = np.load('data/word_distribution.npy')
        else:
            word_distribution = np.zeros(((len(self.index_word_policy) + 1), ))
            for i in range(len(self.dataset)):
                _, _, utterance, _, _, _ = self.dataset.__getitem__(i)
                utterance = self.text_to_indices(text=re.split(' ', utterance.lower()), policy=self.word_index_policy)
                for word_index in utterance:
                    word_distribution[word_index] += 1

            word_distribution = 1 / word_distribution
            word_distribution[word_distribution == float('inf')] = 0
            np.save('data/word_distribution.npy', word_distribution)

        return torch.from_numpy(word_distribution).float()

    def text_to_indices(self, text, policy):
        indices = []
        for char in text:
            try:
                indices.append(int(policy[char]))
            except KeyError:
                indices.append(int(policy['NAME']))

        return indices

    def indices_to_text(self, indices, policy, decoder=False):
        #print('INDICES: ', indices)
        text = []
        for index in indices:
            if decoder:
                index = index[0]
            #print(index)
            try:
                text.append(policy[int(index)])
            except Exception:
                #print('Hi. :-)')
                pass

        return ''.join(text)

    @staticmethod
    def pad_batch(spectrograms, targets):
        spectrograms = nn.utils.rnn.pad_sequence(sequences=spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        targets = nn.utils.rnn.pad_sequence(sequences=targets, batch_first=True)

        return spectrograms, targets

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        waveform, _, utterance, _, _, _ = data
        spectrogram = self.audio_transforms(waveform).squeeze(0).transpose(0, 1)
        target = torch.Tensor(self.text_to_indices(text=utterance.lower(), policy=self.char_index_policy))
        input_len = int(spectrogram.shape[0] / 2)
        target_len = len(target)

        # decoder targets
        word_wise_target = self.text_to_indices(text=re.split(' ', utterance.lower()), policy=self.word_index_policy)

        return spectrogram, target, input_len, target_len, word_wise_target

    def __len__(self):
        return self.dataset.__len__()


class DatasetASRDecoder:
    def __init__(self, root, url, download=False):
        self.root = root

        self.dataset = torchaudio.datasets.LIBRISPEECH(root=root, url=url, download=download)

        with open(os.path.join(root, 'char_index_policy.json')) as policy_json_file:
            self.char_index_policy = json.load(policy_json_file)
        self.index_char_policy = {v: k for k, v in self.char_index_policy.items()}

        with open(os.path.join(root, 'english_word_dict_10k.json'), 'r') as word_dict_file:
            self.index_word_policy = json.load(word_dict_file)
        self.word_index_policy = {v: k for k, v in self.index_word_policy.items()}

    def create_word_distribution(self, saved=True):
        if saved:
            word_distribution = np.load('data/word_distribution.npy')
        else:
            word_distribution = np.zeros(((len(self.index_word_policy) + 1), ))
            for i in range(len(self.dataset)):
                _, _, utterance, _, _, _ = self.dataset.__getitem__(i)
                utterance = self.text_to_indices(text=re.split(' ', utterance.lower()), policy=self.word_index_policy)
                for word_index in utterance:
                    word_distribution[word_index] += 1

            word_distribution = 1 / word_distribution
            word_distribution[word_distribution == float('inf')] = 0
            np.save('data/word_distribution.npy', word_distribution)

        return torch.from_numpy(word_distribution).float()

    def text_to_word_indices(self, text):
        indices = []
        for char in re.split(pattern=' ', string=text):
            try:
                #print(char)
                #print(int(self.word_index_policy[char]))
                indices.append(int(self.word_index_policy[char]))
            except KeyError:
                indices.append(int(self.word_index_policy['NAME']))

        return indices

    def text_to_noised_char_indices(self, text, p):
        indices = []
        word_lens = []
        for word in re.split(pattern=' ', string=text):
            word_len = len(word)
            same_word_indices = []
            p_word = p * ((np.exp(word_len / 7) * 0.25) - 0.25)     # exponentially higher probability of noise for longer words
            for char in word:
                if rnd.random() < p_word:
                    if rnd.random() < p_word:
                        word_len -= 1
                        continue
                    char_index = [rnd.randint(2, (len(self.index_char_policy) - 1))]
                    if rnd.random() < p:
                        char_index.append(int(self.char_index_policy[char]))
                        word_len += 1
                    p_word *= 0.5
                else:
                    char_index = [int(self.char_index_policy[char])]
                same_word_indices += char_index
            word_lens.append(word_len)
            indices.append(torch.Tensor(same_word_indices))

        return indices, word_lens

    def indices_to_text(self, indices, policy):
        text = []
        for index in indices:
            try:
                text.append(policy[int(index.item())])
            except Exception:
                # print('Hi. :-)')
                #print(index)
                text.append(policy[str(int(index.item()))])
                pass

        return ''.join(text)

    """def indices_to_text(self, indices, policy):
        print('INDICES: ', indices)
        text = []
        for index in indices:
            #print(index)
            try:
                text.append(policy[index])
            except Exception:
                pass

        return ''.join(text)"""

    """@staticmethod
    def pad_batch(spectrograms, targets):
        spectrograms = nn.utils.rnn.pad_sequence(sequences=spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
        targets = nn.utils.rnn.pad_sequence(sequences=targets, batch_first=True)

        return spectrograms, targets"""

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        _, _, utterance, _, _, _ = data
        target = torch.Tensor(self.text_to_word_indices(text=utterance.lower()))
        input, word_lens = self.text_to_noised_char_indices(text=utterance.lower(), p=0.1)     # p=0.1 ?
        #print(len(word_lens))
        #print(target)
        #input = torch.nn.utils.rnn.pack_padded_sequence(torch.nn.utils.rnn.pad_sequence(
         #   sequences=input, batch_first=True), lengths=word_lens, batch_first=True, enforce_sorted=False)

        return input, target, word_lens

    def __len__(self):
        return self.dataset.__len__()


class IronyClassificationDataset:
    def __init__(self, mode, top_k=1.0e5, root='data/irony_data'):
        self.mode = mode
        self.root = root

        self.df = pd.read_csv(os.path.join(root, f'train-balanced-sarcasm-{mode}.csv'))

        # self.non_sarcastic_audio_files = os.listdir(os.path.join(root, 'Audio/non_sarcastic'))
        # self.sarcastic_audio_files = os.listdir(os.path.join(root, 'Audio/sarcastic'))

        # TODO: self.vocabulary_dict = ...
        with open(os.path.join(root, 'vocabulary.json'), 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict.items()}
        self.vocabulary_dict['ukw'] = '10001'

        """if mode == 'train':
            self.start_index = 0
            # self.length = int(3 * (len(self.df) / 5))
        elif mode == 'valid':
            self.start_index = int(3 * (len(self.df) / 5))
            # self.length = int(1 * (len(self.df) / 5))
            print('Hi.')
        elif mode == 'test':
            self.start_index = int(4 * (len(self.df) / 5))
            # self.length = int(1 * (len(self.df) / 5))
        # print(self.length)"""

        """if mode == 'train':
            self.audio_transforms = nn.Sequential(
                torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=n_features),
                torchaudio.transforms.FrequencyMasking(freq_mask_param=10),
                torchaudio.transforms.TimeMasking(time_mask_param=30)
            )
        else:
            self.audio_transforms = torchaudio.transforms.MelSpectrogram()"""

    def text_to_indices(self, utterance):
        indices = [10005]       # '<cls>' token
        for word in utterance.split():
            # print(word)
            # TODO
            try:
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word

        return indices

    def __getitem__(self, index):
        try:
            # print('Start index: ', self.start_index)
            # print('Length: ', self.length)
            # print('Index: ', index)
            # index += self.start_index       # to isolate data for train, validation and test purpose

            # row = self.df[[index]]
            row = self.df.loc[index]
            # print('row: ', row['label'])

            """if row['label'] == 1:
                audio_file_path = os.path.join(self.root, f'Audio/sarcastic/{os.listdir(os.path.join(self.root, "Audio/sarcastic"))[row["file_index"]]}')
            else:
                audio_file_path = os.path.join(self.root, f'Audio/non_sarcastic/{os.listdir(os.path.join(self.root, "Audio/non_sarcastic"))[row["file_index"]]}')
            waveform, sample_rate = torchaudio.load(audio_file_path)
            spectrogram = self.audio_transforms(waveform).squeeze(0).transpose(0, 1)"""

            utterance = row['comment']
            utterance = torch.Tensor(self.text_to_indices(utterance=utterance.lower()))
            utterance_len = utterance.shape[0]

            parent_utterance = row['parent_comment']
            parent_utterance = torch.Tensor(self.text_to_indices(utterance=parent_utterance.lower()))
            parent_utterance_len = parent_utterance.shape[0]

            # target = row['target']
            # target = torch.Tensor(target)

            target = row['label']
            # target = torch.Tensor(target)

            # return spectrogram, utterance, target

            return utterance, utterance_len, parent_utterance, parent_utterance_len, target

        except AttributeError:      # if utterance or parent utterance is 'nan'
            return self.__getitem__((index - 1))

    def __len__(self):
        # print('Length: ', self.length)
        return len(self.df)


#x = DatasetASRDecoder(root='data', url='train-clean-100')
#x.__getitem__(0)
#print(x.dataset.__len__())
#print(x.dataset.__getitem__(0))
