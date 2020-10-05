from nltk.corpus import wordnet
from nltk import pos_tag, help, data, load_parser
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
import time
import numpy as np
import torch
import os
import json

from model import IronyClassifier


class SentenceManipulator:
    def __init__(self, irony_regressor_file_path: str, vocabulary_file_path: str, attn_layer: int, attn_head: int, top_k=1.0e5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.last_utterance_indices = None
        self.last_word_embedding = None
        self.attn_layer = attn_layer
        self.attn_head = attn_head
        self.PRIMARY_AUXILIARY_VERBS = ['be', 'have', 'do']
        self.lemmatizer = WordNetLemmatizer()
        self.character_replacement_dict = {'.': '', '…': '', ',': '', '(': '', ')': '', '-': ' ', ';': '', ':': '',
                                           '?': ' ?', '!': ' !', '=': '', '*': '', '~': ' ', '%': '', '"': '', '$': '',
                                           '^': ' ', '#': '', '<': ' ', '>': ' ', '_': ' ', '{': ' ', '}': ' ',
                                           '/': ' ', '\\': ' ', '|': ''}
        with open('data/irony_data/abbr_replacers_inference', 'r') as file:
            self.abbreviation_policy = json.load(file)

        irony_regressor = IronyClassifier(
            batch_size=1,
            n_tokens=1.0e5,
            d_model=500,
            d_context=500,
            n_heads=10,
            n_hid=1024,
            n_layers=12,
            dropout_p=0.5
        )
        irony_regressor.load_state_dict(state_dict=torch.load(irony_regressor_file_path)['model_state_dict'])
        irony_regressor.to(self.device)
        irony_regressor.eval()
        self.irony_regressor = irony_regressor
        #  self.irony_regressor = torch.load(f=irony_regressor_file_path, map_location=self.device)
        #  self.irony_regressor.eval()

        with open(vocabulary_file_path, 'r') as vocabulary_file:
            vocabulary_dict = json.load(vocabulary_file)
        self.vocabulary_dict_indices = dict(list(vocabulary_dict.items())[:int(top_k)])
        self.vocabulary_dict_indices['100000'] = 'ukw'
        self.vocabulary_dict_indices['100003'] = 'sep'
        self.vocabulary_dict_indices['100001'] = 'cls'
        self.vocabulary_dict_indices['100002'] = 'pad'
        self.vocabulary_dict = {v: k for k, v in self.vocabulary_dict_indices.items()}

    def text_to_indices(self, utterance: str) -> list:
        """# ==== Create Last Utterance Indices ====

        if self.last_utterance_indices == None:
            last_utterance_indices = [100001]
        else:
            last_utterance_indices = [100001] + self.last_utterance_indices"""

        # ==== Create Utterance Indices ====

        utterance_indices = [100003]
        for word in utterance.split():
            try:
                #  print(word)
                #  print(int(self.vocabulary_dict[word]))
                utterance_indices.append(int(self.vocabulary_dict[word]))
            except KeyError:
                utterance_indices.append(int(self.vocabulary_dict['ukw']))
        self.last_utterance_indices = utterance_indices[1:]
        utterance_indices.append(100003)

        #  return last_utterance_indices, utterance_indices
        return utterance_indices

    synonyms = []
    antonyms = []

    # print(wordnet.antsets('end'))
    @staticmethod
    def get_first_antonym(word):
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                # synonyms.append(l.name())
                if l.antonyms():
                    # antonyms.append(l.antonyms()[0].name())
                    return l.antonyms()[0].name()

    def get_pos_tag(self, word: list, main_word_index: int):
        """
        Parser for the type of the given word for further processing.

        Args:
            word (list): the list containing the 'main word' and its 'context' to return the 'main word''s corresponding tag / type
            main_word_index (int): position of the 'main word' in the word list

        Returns:
            (str) the corresponding tag / type for the given word.
            (Possible tags / types:
                'ADJ': adjective,
                'VER': verb
                'AUX': auxiliary verb (primary auxiliary verb and modal auxiliary verb)
        """

        #  word_pos_tag = pos_tag([word])[0][1]
        #  word_pos_tag = pos_tag(word)[0][1]
        word_pos_tag = pos_tag(word)[main_word_index][1]
        #  print(word_pos_tag)
        #  print(pos_tag([word]))
        #  print(word)
        #  print(pos_tag(word))
        # ==== Check If Word Is Adjective ====
        if word_pos_tag.startswith('J'):
            return 'ADJ'

        # ==== Check If Word Is An Auxiliary Verb (Primary Auxiliary Verbs And Modal Auxiliary Verbs) OR A Normal Verb ====
        if word_pos_tag.startswith('V'):
            if self.lemmatizer.lemmatize(word[main_word_index], 'v') in self.PRIMARY_AUXILIARY_VERBS:
                return 'AUX'
            else:
                return 'VER'
        elif word_pos_tag == 'MD':
            return 'AUX'
        elif word_pos_tag.startswith('NN'):
            return 'NN'

        """# a = time.process_time()
        # lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize('is', 'v')
        a = time.time()
        x = lemmatizer.lemmatize('is', 'v')
        # b = time.time()
        # b = time.process_time()
        print(time.time() - a)
        print(x)"""

    def get_manipulated_sentence(self, sentence: str, original_utterance: str, distribution: np.array):
        """
        TODO
        """

        # ==== Get Most Likely Word ====
        #  print(distribution)
        word_index = np.argmax(distribution)
        #  print(word_index)
        most_likely_word = sentence.split()[word_index]
        #  print(most_likely_word)
        if len(distribution) == 1:      # Utterance consists only of one word.
            main_word_index = 0
            word_list = [most_likely_word]
        elif word_index == 0:
            additional_context_word = sentence.split()[(word_index + 1)]
            main_word_index = 0
            word_list = [most_likely_word, additional_context_word]
        else:
            additional_context_word = sentence.split()[(word_index - 1)]
            main_word_index = 1
            word_list = [additional_context_word, most_likely_word]
        #  word_tag = self.get_pos_tag(word=([additional_context_word, most_likely_word]), main_word_index=main_word_index)
        word_tag = self.get_pos_tag(word=word_list, main_word_index=main_word_index)
        #  print(word_tag)

        # ==== Replace Word ====
        replacement = []

        if word_tag is 'ADJ' or word_tag is 'NN':
            replacement.append(SentenceManipulator.get_first_antonym(word=most_likely_word))
        elif word_tag is 'AUX':
            replacement += [most_likely_word, 'not']
        elif word_tag is 'VER':
            replacement += ['do', 'not', most_likely_word]
        #  print(replacement)

        if replacement == None or replacement == [None]:       # Nothing to change found. :C
            replacement = [most_likely_word]

        # ==== New Sentence ====

        #  new_sentence = sentence.split()
        new_sentence = original_utterance.split()
        #  print(new_sentence)
        new_sentence[word_index] = ' '.join(replacement)
        #  print(new_sentence)
        new_sentence = ' '.join(new_sentence)
        #  print(new_sentence)

        return new_sentence

    def get_wordnet_pos(self, treebank_tag):

        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def processing(self, utterance: str) -> (str, bool):
        """
        TODO
        """

        #  utterance = '...'       # TODO
        original_utterance = utterance
        #  print('Utterance: ', utterance)
        if utterance.replace(' ', '') is not '':
            try:
                #  print(utterance)
                utterance = ''.join([self.abbreviation_policy.get(e, e) for e in utterance.lower().replace(" '", "").replace("' ", "").replace("'s", " is").replace("'d", " would").replace("'re", " are")])
                original_utterance = utterance
                #  print(utterance)
                utterance = ' '.join(self.lemmatizer.lemmatize(y[0], self.get_wordnet_pos(y[1])) for y in pos_tag(utterance.split(' ')))
                #  print(utterance)
            except IndexError:
                return original_utterance, False

        utterance_indices = self.text_to_indices(utterance=utterance)
        utterance_indices = torch.Tensor(utterance_indices).to(self.device)
        utterance_len = utterance_indices.shape[0]

        # ==== Forward ====

        #  print('Last: ', self.last_word_embedding)

        if utterance_len < 3:
            return utterance, False
        elif self.last_word_embedding is None:
            output, word_embedding, attn_weights = self.irony_regressor(src=utterance_indices,
                                                                        utterance_lens=utterance_len,
                                                                        first=True)
            self.last_word_embedding = word_embedding
            return original_utterance, False     # Assumption: First utterance is never sarcastic / ironic.
        else:
            #  print(self.last_word_embedding.shape)
            #  print(self.last_word_embedding)
            #  print(utterance_indices)
            output, word_embedding, attn_weights = self.irony_regressor(src=utterance_indices,
                                                                        utterance_lens=utterance_len,
                                                                        first=False,
                                                                        last_word_embedding=self.last_word_embedding.unsqueeze(1),
                                                                        last_utterance_lens=self.last_word_embedding.shape[0])
        self.last_word_embedding = word_embedding

        is_ironic = (output.item() > 0.0)

        #  print(is_ironic)
        #  print(output)

        if not is_ironic:
            return original_utterance, False

        attn_vector = attn_weights[self.attn_layer][0][self.attn_head][0][3:-1]
        new_sentence = self.get_manipulated_sentence(sentence=utterance, original_utterance=original_utterance, distribution=attn_vector.cpu().detach().numpy())

        return new_sentence, True
