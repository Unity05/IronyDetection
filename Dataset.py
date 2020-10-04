import torchaudio
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import nlpaug.augmenter.word as naw
from nltk import bigrams

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
    def __init__(self, mode, top_k=1.0e5, root='data/irony_data', phase=0):
        self.mode = mode
        self.root = root

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(root, f'train-balanced-sarcasm-{mode}-{phase}-adjusted.csv'))
            # self.df = pd.read_csv(os.path.join(root, f'train-balanced-sarcasm-{mode}-{phase}.csv'))
            # self.df = pd.read_csv(os.path.join(root, 'train-balanced-sarcasm-train-very-small.csv'))
        else:
            self.df = pd.read_csv(os.path.join(root, f'train-balanced-sarcasm-{mode}.csv'))

        """print(mode)
        print('a: ', len(self.df.loc[self.df['label'] == 1]))
        print('b: ', len(self.df.loc[self.df['label'] == 0]))"""

        # self.non_sarcastic_audio_files = os.listdir(os.path.join(root, 'Audio/non_sarcastic'))
        # self.sarcastic_audio_files = os.listdir(os.path.join(root, 'Audio/sarcastic'))

        # TODO: self.vocabulary_dict = ...
        with open(os.path.join(root, 'glove_adjusted_vocabulary.json'), 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict['100003'] = 'sep'
        self.vocabulary_dict['100001'] = 'cls'
        self.vocabulary_dict['100002'] = 'pad'
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict.items()}
        self.vocabulary_dict['ukw'] = '100000'

        self.aug = naw.SynonymAug()

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

    def text_to_indices(self, utterance, first):
        #  if first:
        if True:
            indices = [100001]       # '<cls>' token
        else:
            indices = [100003]
        pass
        # print(len(utterance.split()))
        for word in utterance.split():
            # print(word)
            # TODO
            try:
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                # print('ukw: ', word)
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word
        """if not first:
            indices.append(100003)"""

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
            # print(utterance + ' \n LOLOLOLOLOLOLO')
            utterance = utterance.lower()
            """if rnd.random() < 0.95 and self.mode == 'train':
            # if self.mode == 'train':
                utterance = self.aug.augment(data=utterance)        # text augmentation (SynonymAug)"""
            utterance = torch.Tensor(self.text_to_indices(utterance=utterance, first=False))
            utterance_len = utterance.shape[0]

            parent_utterance = row['parent_comment']
            # print(parent_utterance + ' \n LULULULULULU')
            parent_utterance = parent_utterance.lower()
            """if rnd.random() < 0.75 and self.mode == 'train':
                parent_utterance = self.aug.augment(data=parent_utterance)"""
            parent_utterance = torch.Tensor(self.text_to_indices(utterance=parent_utterance, first=True))
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


class SARC_2_0_IronyClassificationDataset:
    def __init__(self, mode, top_k=1.0e5, root='data/irony_data'):
        self.mode = mode
        self.root = root

        # ==== Load Training Data ====

        with open(os.path.join(root, 'SARC_2.0/adjusted-comments.json'), 'r') as comments_json_file:
            self.comments_json = json.load(comments_json_file)

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(root, 'SARC_2.0/train-unbalanced-adjusted.csv'), names=['data'], encoding='utf-8')
        else:
            self.df = pd.read_csv(os.path.join(root, 'SARC_2.0/test-balanced.csv'), names=['data'], encoding='utf-8')

        # ==== Load Vocabulary Data ====

        #  with open(os.path.join(root, 'SARC_2.0/glove_adjusted_vocabulary_original_adjusted.json'), 'r') as vocabulary_file:
        #  with open(os.path.join(root, 'SARC_2.0/vocabulary_fast_text_adjusted.json'), 'r') as vocabulary_file:
        with open(os.path.join(root, 'SARC_2.0/glove_adjusted_vocabulary.json'), 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict_indices = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict_indices['100000'] = 'ukw'
        self.vocabulary_dict_indices['100003'] = 'sep'
        self.vocabulary_dict_indices['100001'] = 'cls'
        self.vocabulary_dict_indices['100002'] = 'pad'
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict_indices.items()}

        self.n_current_samples = 0.1
        self.n_sarcastic = 0

    def text_to_indices(self, utterance, first):
        #  print(utterance)
        if first:
            indices = [100001]       # '<cls>' token
        else:
            indices = []
        if not first:
            indices.append(100003)
        num_unknown_words = 1.0e-9
        for word in utterance.split():
            # print(word)
            # TODO
            try:
                #  indices.append(int(self.vocabulary_dict[rnd.choice(list(self.vocabulary_dict.keys()))]))
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                num_unknown_words += 1
                # print('Hi.')
                # print(word)
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word
        if not first:
            indices.append(100003)

        try:
            unkown_words_frequency = (num_unknown_words / len(utterance.split()))
        except ZeroDivisionError:
            unkown_words_frequency = 1.0

        return indices, unkown_words_frequency

    def indices_to_text(self, indices):
        assert type(indices) == torch.Tensor, f"'indices' (type: {type(indices)}) should be of type 'torch.Tensor'."
        indices = indices.tolist()
        utterance = []
        for index in indices:
            utterance.append(self.vocabulary_dict_indices[str(int(index))])

        return ' '.join(utterance)

    def __getitem__(self, index):
        data = self.df.iloc[index].item()
        data = data.split('|')

        post_ids = data[0].split(' ')
        # post_id = post_ids[0]       # TODO: Is this always the best choice?
        post_id = post_ids[-1]

        response_ids = data[1].split(' ')
        labels = data[2].split(' ')

        """for p_id in post_ids:
            print(self.comments_json[p_id])
        print('-' * 24)
        for r_id in response_ids:
            print(self.comments_json[r_id])
        print('-' * 24)
        print(labels)
        # for r_id in response_ids"""

        # Shuffle lists, so that 'index()' does not always choose the same element / comment id.
        # print(post_ids)
        # print(response_ids)
        # print(labels)
        combined_ids = list(zip((post_ids * len(labels)), response_ids, labels))
        # print(combined_ids)
        rnd.shuffle(combined_ids)
        # print(combined_ids)
        # print(*combined_ids)
        post_ids, response_ids, labels = zip(*combined_ids)
        # print(post_ids)
        # print(response_ids)
        # print(labels)

        class_ratio = (self.n_sarcastic / self.n_current_samples)
        try:
            if class_ratio < 0.4:
                response_index = labels.index('1')
            elif class_ratio > 0.6:
                response_index = labels.index('0')
        except ValueError:
            response_index = rnd.randint(0, (len(response_ids) - 1))

        # post_id = post_ids[0]       # TODO: Is this always the best choice?
        response_id = response_ids[response_index]
        label = int(labels[response_index])

        # print(label)

        """print(self.comments_json[post_id])
        print(self.comments_json[response_id])
        print(label)"""

        #  print('parent_utterance: ', self.comments_json[post_id])
        # parent_utterance = torch.Tensor(self.text_to_indices(utterance=self.comments_json[post_id][0].lower()))
        parent_utterance, parent_unknown_words_ratio = self.text_to_indices(utterance=self.comments_json[post_id][0].lower(), first=True)
        parent_utterance = torch.Tensor(parent_utterance)
        parent_utterance_len = parent_utterance.shape[0]

        # print(parent_unknown_words_ratio)

        # utterance = torch.Tensor(self.text_to_indices(utterance=self.comments_json[response_id][0].lower()))
        utterance, unknown_words_ratio = self.text_to_indices(utterance=self.comments_json[response_id][0].lower(), first=False)
        utterance = torch.Tensor(utterance)
        #  print(self.indices_to_text(indices=utterance))
        #  print('utterance: ', self.comments_json[response_id][0])
        utterance_len = utterance.shape[0]

        if parent_unknown_words_ratio > 0.2 or unknown_words_ratio > 0.2:
            self.__getitem__(index=(index - 1))

        # print(unknown_words_ratio)

        # class equilibrium
        self.n_current_samples += 1
        self.n_sarcastic += label

        return utterance, utterance_len, parent_utterance, parent_utterance_len, label, class_ratio

    def __len__(self):
        # return len(self.comments_json)
        return len(self.df)


class SARC_2_0_Dataset:
    def __init__(self, mode, top_k=1.0e5, root='data/irony_data'):
        self.mode = mode

        # ==== Load Training Data ====

        with open(os.path.join(root, 'SARC_2.0/adjusted-comments.json'), 'r') as comments_json_file:
            self.comments_json = json.load(comments_json_file)

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(root, 'SARC_2.0/train-unbalanced-adjusted.csv'), encoding='utf-8')
        else:
            #  self.df = pd.read_csv(os.path.join(root, 'SARC_2.0/test-balanced-adjusted.csv'), names=['data'], encoding='utf-8')
            self.df = pd.read_csv(os.path.join(root, 'SARC_2.0/test-balanced-adjusted.csv'), encoding='utf-8')

        # ==== Load Vocabulary Data ====

        with open(os.path.join(root, 'SARC_2.0/glove_adjusted_vocabulary.json'), 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict_indices = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict_indices['100000'] = 'ukw'
        self.vocabulary_dict_indices['100003'] = 'sep'
        self.vocabulary_dict_indices['100001'] = 'cls'
        self.vocabulary_dict_indices['100002'] = 'pad'
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict_indices.items()}

    def text_to_indices(self, utterance, first):
        #  print(utterance)
        if first:
            indices = [100001]       # '<cls>' token
        else:
            indices = []
        if not first:
            indices.append(100003)
        #  num_unknown_words = 1.0e-9
        for word in utterance.split():
            # print(word)
            # TODO
            try:
                #  indices.append(int(self.vocabulary_dict[rnd.choice(list(self.vocabulary_dict.keys()))]))
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                #  num_unknown_words += 1
                #  print('Hi.')
                #  print(word)
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word
        if not first:
            indices.append(100003)
        #  print(indices)

        """try:
            unkown_words_frequency = (num_unknown_words / len(utterance.split()))
        except ZeroDivisionError:
            unkown_words_frequency = 1.0"""

        #  return indices, unkown_words_frequency
        return indices

    def __getitem__(self, index):
        #  print('Hi.')
        #  data = self.df.loc[[index]]['data']
        #  print(self.df)
        #  data = self.df.iloc[index].item()
        data = self.df.loc[index]
        #  print(data)
        #  print(data['data'])
        #  print(data.columns)
        #  print(data['parent_ids'])
        #  print(data['response_ids'])
        #  print(data['labels'])

        #  print(data)

        parent_id = data['parent_ids']
        response_id = data['response_ids']
        label = int(data['labels'])
        #  print(label)

        parent_utterance = self.comments_json[parent_id][0].lower()
        #  print(parent_utterance)
        parent_utterance = self.text_to_indices(utterance=parent_utterance, first=True)
        parent_utterance = torch.Tensor(parent_utterance)
        parent_utterance_len = parent_utterance.shape[0]
        response_utterance = self.comments_json[response_id][0].lower()
        #  print(response_utterance)
        response_utterance = self.text_to_indices(utterance=response_utterance, first=False)
        response_utterance = torch.Tensor(response_utterance)
        response_utterance_len = response_utterance.shape[0]

        return response_utterance, response_utterance_len, parent_utterance, parent_utterance_len, label

    def __len__(self):
        return len(self.df)


class SARC_2_0_Dataset_Bigram:
    def __init__(self, mode, top_k=1.0e5, root='data/irony_data'):
        self.mode = mode

        # ==== Load Training Data ====

        with open(os.path.join(root, 'SARC_2.0/adjusted-comments.json'), 'r') as comments_json_file:
            self.comments_json = json.load(comments_json_file)

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(root, 'SARC_2.0/train-balanced-adjusted.csv'), encoding='utf-8')
        else:
            self.df = pd.read_csv(os.path.join(root, 'SARC_2.0/test-balanced.csv'), names=['data'], encoding='utf-8')

        # ==== Load Vocabulary Data ====

        with open(os.path.join(root, 'SARC_2.0/glove_adjusted_vocabulary.json'), 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict_indices = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict_indices['400000'] = 'ukw'
        self.vocabulary_dict_indices['400003'] = 'sep'
        self.vocabulary_dict_indices['400001'] = 'cls'
        self.vocabulary_dict_indices['400002'] = 'pad'
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict_indices.items()}

    def text_to_indices(self, utterance, first):
        # ==== bigrams ====
        bigram_list = list(bigrams(utterance.split()))

        #  print(utterance)
        if first:
            indices = [400001]       # '<cls>' token
        else:
            indices = []
        if not first:
            indices.append(400003)

        for bigram in bigram_list:
            bigram = list(bigram)
            bigram_string = ' '.join(bigram)
            try:
                indices.append(int(self.vocabulary_dict[bigram_string]))
            except KeyError:        # bigram not in vocabulary --> try to get unigram token from vocabulary
                for unigram in bigram:
                    try:
                        indices.append(int(self.vocabulary_dict[unigram]))
                    except KeyError:        # unknown word --> append 'ukw' token
                        indices.append(int(self.vocabulary_dict['ukw']))

        """#  num_unknown_words = 1.0e-9
        for word in utterance.split():
            # print(word)
            # TODO
            try:
                #  indices.append(int(self.vocabulary_dict[rnd.choice(list(self.vocabulary_dict.keys()))]))
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                #  num_unknown_words += 1
                #  print('Hi.')
                #  print(word)
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word"""
        if not first:
            indices.append(400003)

        """try:
            unkown_words_frequency = (num_unknown_words / len(utterance.split()))
        except ZeroDivisionError:
            unkown_words_frequency = 1.0"""

        #  return indices, unkown_words_frequency
        return indices

    def indices_to_text(self, indices):
        assert type(indices) == torch.Tensor, f"'indices' (type: {type(indices)}) should be of type 'torch.Tensor'."
        indices = indices.tolist()
        utterance = []
        for index in indices:
            utterance.append(self.vocabulary_dict_indices[str(int(index))])

        return ' '.join(utterance)

    def __getitem__(self, index):
        #  data = self.df.loc[[index]]['data']
        #  print(self.df)
        #  data = self.df.iloc[index].item()
        data = self.df.loc[index]
        #  print(data)
        #  print(data['data'])
        #  print(data.columns)
        #  print(data['parent_ids'])
        #  print(data['response_ids'])
        #  print(data['labels'])

        parent_id = data['parent_ids']
        response_id = data['response_ids']
        label = int(data['labels'])

        parent_utterance = self.comments_json[parent_id][0].lower()
        parent_utterance = self.text_to_indices(utterance=parent_utterance, first=True)
        parent_utterance = torch.Tensor(parent_utterance)
        parent_utterance_len = parent_utterance.shape[0]
        response_utterance = self.comments_json[response_id][0].lower()
        response_utterance = self.text_to_indices(utterance=response_utterance, first=False)
        response_utterance = torch.Tensor(response_utterance)
        response_utterance_len = response_utterance.shape[0]

        return response_utterance, response_utterance_len, parent_utterance, parent_utterance_len, label

    def __len__(self):
        return len(self.df)


class SarcasmHeadlinesDataset:
    def __init__(self, mode, top_k=1.0e5, root='data/irony_data'):
        self.mode = mode
        self.root = root

        # ==== Load Training Data ====

        if mode == 'train':
            self.sarcasm_headlines_df = pd.read_csv(os.path.join(root, 'sarcastic_headlines/Sarcasm_Headlines_Dataset_Train.csv'))
        else:
            self.sarcasm_headlines_df = pd.read_csv(os.path.join(root, 'sarcastic_headlines/Sarcasm_Headlines_Dataset_Valid.csv'))

        with open(os.path.join(root, 'glove_adjusted_vocabulary.json'), 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict.items()}
        self.vocabulary_dict['ukw'] = '100000'

    def text_to_indices(self, utterance):
        indices = [100001]       # '<cls>' token
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
            row = self.sarcasm_headlines_df.loc[index]

            parent_utterance = row['headline']
            # print(parent_utterance)
            parent_utterance = parent_utterance.lower()
            parent_utterance = torch.Tensor(self.text_to_indices(utterance=parent_utterance))
            parent_utterance_len = parent_utterance.shape[0]

            target = row['is_sarcastic']

            utterance = None
            utterance_len = None
        except AttributeError:
            return self.__getitem__((index - 1))    # if utterance or parent utterance is 'nan'

        return utterance, utterance_len, parent_utterance, parent_utterance_len, target

    def __len__(self):
        return len(self.sarcasm_headlines_df)

#x = DatasetASRDecoder(root='data', url='train-clean-100')
#x.__getitem__(0)
#print(x.dataset.__len__())
#print(x.dataset.__getitem__(0))

"""# string = 'Hi, my name is fish and I do not like cars.'

string = 'i can not wait until @potus start a twitter war against morning joe'

# aug = nlpaug.augmenter.word.synonym.SynonymAug()

aug = naw.SynonymAug()

augmented_data = aug.augment(data=string)

print(augmented_data)"""

"""#  x = SARC_2_0_IronyClassificationDataset(mode='train', top_k=1.0e5, root='data/irony_data')
x = SARC_2_0_Dataset(mode='valid', top_k=1.0e5, root='data/irony_data')

# print(x.__getitem__(0))

for i in range(10):
    y = rnd.randint(0, 10000)
    #  _, _, _, _, z, _ = x.__getitem__(y)
    _ = x.__getitem__(y)
    #  print(z)
#  _, _, _, _, z, _ = x.__getitem__((x.__len__() - 1))
#  print(z)"""

"""import time


x = SARC_2_0_Dataset_Bigram(mode='train', top_k=4.0e5, root='data/irony_data')

a = time.time()
y, _, _, _, _ = x.__getitem__(0)
print(time.time() - a)

print(y)

z = x.indices_to_text(indices=y)

print(z)"""

"""x = Dataset(root='data', url='train-clean-100', mode='train', n_features=128, download=True)

print(x.__getitem__(0))"""
