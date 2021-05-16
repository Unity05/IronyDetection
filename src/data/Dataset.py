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
        text = []
        for index in indices:
            if decoder:
                index = index[0]
            try:
                text.append(policy[int(index)])
            except Exception:
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
                text.append(policy[str(int(index.item()))])
                pass

        return ''.join(text)

    def __getitem__(self, index):
        data = self.dataset.__getitem__(index)
        _, _, utterance, _, _, _ = data
        target = torch.Tensor(self.text_to_word_indices(text=utterance.lower()))
        input, word_lens = self.text_to_noised_char_indices(text=utterance.lower(), p=0.1)     # p=0.1 ?

        return input, target, word_lens

    def __len__(self):
        return self.dataset.__len__()


class IronyClassificationDataset:
    def __init__(self, mode, top_k=1.0e5, root='data/irony_data', phase=0):
        self.mode = mode
        self.root = root

        if mode == 'train':
            self.df = pd.read_csv(os.path.join(root, f'train-balanced-sarcasm-{mode}-{phase}-adjusted.csv'))
        else:
            self.df = pd.read_csv(os.path.join(root, f'train-balanced-sarcasm-{mode}.csv'))

        with open(os.path.join(root, 'glove_adjusted_vocabulary.json'), 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict['100003'] = 'sep'
        self.vocabulary_dict['100001'] = 'cls'
        self.vocabulary_dict['100002'] = 'pad'
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict.items()}
        self.vocabulary_dict['ukw'] = '100000'

        self.aug = naw.SynonymAug()

    def text_to_indices(self, utterance, first):
        if True:
            indices = [100001]       # '<cls>' token
        else:
            indices = [100003]
        pass
        for word in utterance.split():
            try:
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word

        return indices

    def __getitem__(self, index):
        try:
            row = self.df.loc[index]

            utterance = row['comment']
            utterance = utterance.lower()
            utterance = torch.Tensor(self.text_to_indices(utterance=utterance, first=False))
            utterance_len = utterance.shape[0]

            parent_utterance = row['parent_comment']
            parent_utterance = parent_utterance.lower()
            parent_utterance = torch.Tensor(self.text_to_indices(utterance=parent_utterance, first=True))
            parent_utterance_len = parent_utterance.shape[0]

            target = row['label']

            return utterance, utterance_len, parent_utterance, parent_utterance_len, target

        except AttributeError:      # if utterance or parent utterance is 'nan'
            return self.__getitem__((index - 1))

    def __len__(self):
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
        if first:
            indices = [100001]       # '<cls>' token
        else:
            indices = []
        if not first:
            indices.append(100003)
        num_unknown_words = 1.0e-9
        for word in utterance.split():
            try:
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                num_unknown_words += 1
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
        post_id = post_ids[-1]

        response_ids = data[1].split(' ')
        labels = data[2].split(' ')

        combined_ids = list(zip((post_ids * len(labels)), response_ids, labels))
        rnd.shuffle(combined_ids)
        post_ids, response_ids, labels = zip(*combined_ids)

        class_ratio = (self.n_sarcastic / self.n_current_samples)
        try:
            if class_ratio < 0.4:
                response_index = labels.index('1')
            elif class_ratio > 0.6:
                response_index = labels.index('0')
        except ValueError:
            response_index = rnd.randint(0, (len(response_ids) - 1))

        response_id = response_ids[response_index]
        label = int(labels[response_index])

        parent_utterance, parent_unknown_words_ratio = self.text_to_indices(utterance=self.comments_json[post_id][0].lower(), first=True)
        parent_utterance = torch.Tensor(parent_utterance)
        parent_utterance_len = parent_utterance.shape[0]

        utterance, unknown_words_ratio = self.text_to_indices(utterance=self.comments_json[response_id][0].lower(), first=False)
        utterance = torch.Tensor(utterance)
        utterance_len = utterance.shape[0]

        if parent_unknown_words_ratio > 0.2 or unknown_words_ratio > 0.2:
            self.__getitem__(index=(index - 1))


        # class equilibrium
        self.n_current_samples += 1
        self.n_sarcastic += label

        return utterance, utterance_len, parent_utterance, parent_utterance_len, label, class_ratio

    def __len__(self):
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
        if first:
            indices = [100001]       # '<cls>' token
        else:
            indices = []
        if not first:
            indices.append(100003)
        for word in utterance.split():
            try:
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word
        if not first:
            indices.append(100003)

        return indices

    def __getitem__(self, index):
        data = self.df.loc[index]

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

        if not first:
            indices.append(400003)

        return indices

    def indices_to_text(self, indices):
        assert type(indices) == torch.Tensor, f"'indices' (type: {type(indices)}) should be of type 'torch.Tensor'."
        indices = indices.tolist()
        utterance = []
        for index in indices:
            utterance.append(self.vocabulary_dict_indices[str(int(index))])

        return ' '.join(utterance)

    def __getitem__(self, index):
        data = self.df.loc[index]

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
            try:
                indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                indices.append(int(self.vocabulary_dict['ukw']))        # unknown word

        return indices

    def __getitem__(self, index):
        try:
            row = self.sarcasm_headlines_df.loc[index]

            parent_utterance = row['headline']
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
