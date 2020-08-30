# https://escholarship.org/content/qt85h6g1ps/qt85h6g1ps.pdf

from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk import FreqDist, pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import pandas as pd
import sklearn

import re
import collections

import os
import json

import time


def update_data(root='data'):
    df = pd.read_csv(os.path.join(root, 'irony_data/train-balanced-sarcasm-adjusted.csv'))
    with open(os.path.join(root, 'irony_data/Audio/sarcastic/sarcasm_indices.txt')) as file:
        sarcastic_indices = file.read().splitlines()
    with open(os.path.join(root, 'irony_data/Audio/non_sarcastic/non_sarcasm_indices.txt')) as file:
        non_sarcastic_indices = list(map(int, file.read().splitlines()))

    sarcastic_df = df.loc[df['label'] == 1][:len(sarcastic_indices)]
    non_sarcastic_df = df.loc[df['label'] == 0][:len(non_sarcastic_indices)][list(map(bool, non_sarcastic_indices))]        # spoken non - sarcastic data

    # sarcastic targets
    keep_list = []
    targets = []
    for comment, sarcastic_index in zip(sarcastic_df['comment'], sarcastic_indices):
        comment = comment.split()

        if sarcastic_index is '-':
            keep_list.append(False)
            continue
        else:
            keep_list.append(True)

        range_delimiter = ', '
        index_delimiter = ':'

        if sarcastic_index is index_delimiter:
            targets.append(([1] * len(comment)))
            continue

        sarcastic_indices_list = re.split(pattern=f'[{index_delimiter}{range_delimiter}]', string=sarcastic_index)

        if '' in sarcastic_indices_list:
            sarcastic_indices_list = list(filter(('').__ne__, sarcastic_indices_list))

        target = []
        sarcastic_index_i = 0
        n_sarcastic_indices_indices = len(sarcastic_indices_list)
        digit = 0
        for i in range(len(comment)):
            if int(sarcastic_indices_list[sarcastic_index_i]) == i:
                digit = (digit + 1) % 2
                if sarcastic_index_i < (n_sarcastic_indices_indices - 1):
                    sarcastic_index_i += 1
                if digit == 0:          # inconsistent indexing for digit change in training data: *real*[start, (end - 1)] *wanted*[start, end]
                    target.append(1)
                    continue

            target.append(digit)

        targets.append(target)

    sarcastic_df = sarcastic_df[keep_list]
    sarcastic_df['targets'] = targets
    sarcastic_df['file_index'] = [*range(len(targets))]

    # non - sarcastic targets
    targets = []
    for comment in non_sarcastic_df['comment']:
        comment = comment.split()
        targets.append(([0] * len(comment)))

    non_sarcastic_df['targets'] = targets
    non_sarcastic_df['file_index'] = [*range(len(targets))]

    final_df = pd.concat([sarcastic_df, non_sarcastic_df])

    final_df.to_csv(os.path.join(root, 'irony_data/train-balanced-sarcasm-final.csv'), index=False, encoding='utf-8')


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def lemmatize_list(x, abbreviation_policy):
    try:
        y = ' '.join([abbreviation_policy.get(e, e) for e in x])
    except TypeError:
        y = ''

    return y


class CreateVocabulary:
    def __init__(self):
        # self.vocab = collections.Counter(dict())
        self.vocab = []
        self.i = 0

    def add(self, x):
        # a = time.process_time()
        # frequency_dict = FreqDist(samples=x.split())
        # self.vocab += collections.Counter(frequency_dict)
        # print('Duration: ', time.process_time() - a)
        # print(x)
        self.vocab += x.split()
        # print('i: ', self.i)
        self.i += 1


def generate_vocabulary(root='data', max_len=0):
    if not os.path.isfile(os.path.join(root, 'irony_data/train-balanced-sarcasm-adjusted.csv')):
        lemmatizer = WordNetLemmatizer()

        df = pd.read_csv(os.path.join(root, 'irony_data/train-balanced-sarcasm.csv'))

        with open(os.path.join(root, 'irony_data/abbr_replacers'), 'r') as file:
            abbreviation_policy = json.load(file)

        relevant_column_names = ['comment', 'parent_comment']
        character_replacement_dict = {'.': '', '…': '', ',': '', '(': '', ')': '', '-': ' ', ';': '', ':': '',
                                      '?': ' ?', '!': ' !', '=': '', '*': '', '~': ' ', '%': '', '"': '', '$': '',
                                      '^': ' ', '#': '', '<': ' ', '>': ' ', '_': ' ', '{': ' ', '}': ' ', '/': ' ',
                                      '\\': ' ', '|': ''}
        for relevant_column_name in relevant_column_names:
            print(df[relevant_column_name].str)
            df[relevant_column_name] = df[relevant_column_name].str.lower().str.translate(
                str.maketrans(character_replacement_dict)).str.replace(" '", "").replace("' ", "").str.split().apply(
                lambda x: lemmatize_list(x=x, abbreviation_policy=abbreviation_policy))
            #df[relevant_column_name] = df[relevant_column_name].str.split().apply(
             #   lambda x: lemmatize_list(x=x, abbreviation_policy=abbreviation_policy))
            # df[relevant_column_name] = df[relevant_column_name].str.replace('[...…,()--;:?!=*~%"$^#<>_{}/]', ' ').replace("'", " ").replace("\\", " ").str.lower().str.split().apply(
              #   lambda x: lemmatize_list(x=x, abbreviation_policy=abbreviation_policy))     # 'comment' column contains 'nan' elements; we can ignore that since the first one appears to be at index >50,000
            df[relevant_column_name] = df[relevant_column_name].apply(lambda x: ' '.join([lemmatizer.lemmatize(
                y[0], get_wordnet_pos(y[1])) for y in pos_tag(x.split())]))

        print('Okay.')

        #df = df.loc[df['label'] == 0]        # just for audio recording

        df.to_csv(os.path.join(root, 'irony_data/train-balanced-sarcasm-adjusted.csv'), index=False, encoding='utf-8')
    else:
        df = pd.read_csv(os.path.join(root, 'irony_data/train-balanced-sarcasm-adjusted.csv'))

    print(df['comment'])
    # print(df[~(df['comment'].str.len() > max_len)])
    # print(df[df['comment'].apply(lambda x: print(type(len(str(x).split()))))])

    # df = df[~(df['comment'].str.len() <= max_len)]
    # df = df[~(df['parent_comment'].str.len() <= max_len)]
    # print(df[df['comment'].apply(lambda x: print(type(len(str(x).split()) <= max_len)))])
    df = df[df['comment'].apply(lambda x: len(str(x).split()) <= max_len)]
    df = df[df['parent_comment'].apply(lambda x: len(str(x).split()) <= max_len)]
    print('len_df: ', len(df))
    df.to_csv('data/irony_data/train-balanced-sarcasm-adjusted-length.csv')

    # print(df.parent_comment.tolist())

    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    vectorizer.fit_transform(df.parent_comment.astype('U').tolist())
    vocabulary_dict = vectorizer.vocabulary_
    sorted_vocabulary_dict_items = sorted(vocabulary_dict.items(), reverse=True)

    # df['frequencies_comments'] = df.comment.astype('U').apply(lambda x: FreqDist(samples=x.split()))
    vocabulary_creator = CreateVocabulary()
    df.comment.astype('U').apply(lambda x: vocabulary_creator.add(x=x))
    # print('First half finished.')
    # df['frequencies_parent_comments'] = df.parent_comment.astype('U').apply(lambda x: FreqDist(samples=x.split()))
    df.parent_comment.astype('U').apply(lambda x: vocabulary_creator.add(x=x))
    # print('Second half finished.')

    vocab = vocabulary_creator.vocab
    vocab_words_frequencies = FreqDist(samples=vocab)

    # print(df['frequencies'])

    """vocab = collections.Counter(dict())

    i = 0
    for frequency_name in ['frequencies_comments', 'frequencies_parent_comments']:
        for frequency_dict in df[frequency_name]:
            print(vocab)
            vocab += collections.Counter(frequency_dict)
            print(f'{((i / 2.0e6) * 100)}% done.')
            i += 1"""

    # print(frequency_dict)

    #for top_i_item in sorted_vocabulary_dict_items:
     #   print(top_i_item)

    # print(vectorizer.vocabulary_)

    # print(dict(vocab_words_frequencies))
    k = -1
    # print(f'{k} most frequent words: {list(vocab_words_frequencies)[:k]}')

    vocabulary = {}
    for i, word in enumerate(list(vocab_words_frequencies)[:k]):
        vocabulary[i] = word

    with open(os.path.join(root, 'irony_data/vocabulary.json'), 'w') as vocabulary_file:
        json.dump(vocabulary, vocabulary_file)


def split_data(root='data/irony_data', p_train=0.8, p_valid=0.1, p_test=0.1):
    df = pd.read_csv(os.path.join(root, 'train-balanced-sarcasm-adjusted-length.csv'))
    df_len = len(df)
    train_len = int(df_len * p_train)
    valid_len = int(df_len * p_valid)
    test_len = int(df_len * p_test)

    df_train_one = df.loc[df['label'] == 1].head(int(train_len / 2))
    df_train_one_len_median = df_train_one['comment'].apply(lambda x: len(str(x).split())).median()
    df_train_one_0 = df_train_one[df_train_one['comment'].apply(lambda x: len(str(x).split()) <= df_train_one_len_median)]
    df_train_one_1 = df_train_one[df_train_one['comment'].apply(lambda x: len(str(x).split()) > df_train_one_len_median)]

    df_train_zero = df.loc[df['label'] == 0].head(int(train_len / 2))
    df_train_zero_len_median = df_train_zero['comment'].apply(lambda x: len(str(x).split())).median()
    df_train_zero_0 = df_train_zero[df_train_zero['comment'].apply(lambda x: len(str(x).split()) <= df_train_zero_len_median)]
    df_train_zero_1 = df_train_zero[df_train_zero['comment'].apply(lambda x: len(str(x).split()) > df_train_zero_len_median)]

    # df_train = df.loc[:train_len]
    df_train = pd.concat([df_train_one, df_train_zero], ignore_index=True)
    df_train_0 = pd.concat([df_train_one_0, df_train_zero_0], ignore_index=True)
    df_train_1 = pd.concat([df_train_one_1, df_train_zero_1], ignore_index=True)
    print('len_df_train: ', len(df_train))
    print('len_df_train_0: ', len(df_train_0), ' | len_df_train_one_0: ', len(df_train_one_0), ' | len_df_train_zero_0: ', len(df_train_zero_0))
    print('len_df_train_1: ', len(df_train_1), ' | len_df_train_one_1: ', len(df_train_one_1), ' | len_df_train_zero_1: ', len(df_train_zero_1))

    df_valid = df.loc[train_len:(train_len + valid_len)]
    df_test = df.loc[(train_len + valid_len):(train_len + valid_len + test_len)]
    # print(len(df))

    df_train.to_csv(os.path.join(root, 'train-balanced-sarcasm-train-2.csv'), index=False, encoding='utf-8')
    df_train_0.to_csv(os.path.join(root, 'train-balanced-sarcasm-train-0.csv'), index=False, encoding='utf-8')
    df_train_1.to_csv(os.path.join(root, 'train-balanced-sarcasm-train-1.csv'), index=False, encoding='utf-8')
    df_valid.to_csv(os.path.join(root, 'train-balanced-sarcasm-valid.csv'), index=False, encoding='utf-8')
    df_test.to_csv(os.path.join(root, 'train-balanced-sarcasm-test.csv'), index=False, encoding='utf-8')


# print(', ( )'.lower().replace('[...…,()]', 'a'), ' | ', ', ( )'.replace('[...…,()]', 'a'))
# generate_vocabulary(root='data', max_len=64)
# update_data(root='data')
split_data()
