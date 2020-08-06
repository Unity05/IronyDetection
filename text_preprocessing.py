# https://escholarship.org/content/qt85h6g1ps/qt85h6g1ps.pdf

from nltk.stem import PorterStemmer, WordNetLemmatizer, SnowballStemmer
from nltk import FreqDist
from nltk.tokenize import word_tokenize
import pandas as pd
import sklearn

import collections

import os
import json


def generate_vocabulary(root='data'):

    if not os.path.isfile(os.path.join(root, 'irony_data/train-balanced-sarcasm-adjusted.csv')):
        lemmatizer = WordNetLemmatizer()

        df = pd.read_csv(os.path.join(root, 'irony_data/1309_36545_compressed_train-balanced-sarcasm.csv/train-balanced-sarcasm.csv'))

        with open(os.path.join(root, 'irony_data/abbr_replacers'), 'r') as file:
            abbreviation_policy = json.load(file)

        df['parent_comment'] = df.parent_comment.str.replace('[...…,()--;:?!]', ' ').str.lower().str.split().apply(
            lambda x: ' '.join([abbreviation_policy.get(e, e) for e in x]))
        df['parent_comment'] = df.parent_comment.apply(lambda x: ' '.join([lemmatizer.lemmatize(y) for y in x.split()]))

        df.to_csv(os.path.join(root, 'irony_data/train-balanced-sarcasm-adjusted.csv'), index=False, encoding='utf-8')
    else:
        df = pd.read_csv(os.path.join(root, 'irony_data/train-balanced-sarcasm-adjusted.csv'))

    print(df['parent_comment'])

    # print(df.parent_comment.tolist())

    vectorizer = sklearn.feature_extraction.text.CountVectorizer()
    vectorizer.fit_transform(df.parent_comment.astype('U').tolist())
    vocabulary_dict = vectorizer.vocabulary_
    sorted_vocabulary_dict_items = sorted(vocabulary_dict.items(), reverse=True)

    df['frequencies'] = df.parent_comment.astype('U').apply(lambda x: FreqDist(samples=x.split()))

    print(df['frequencies'])

    vocab = collections.Counter(dict())

    for frequency_dict in df['frequencies']:
        print(vocab)
        vocab += collections.Counter(frequency_dict)

    print(frequency_dict)

    #for top_i_item in sorted_vocabulary_dict_items:
     #   print(top_i_item)

    # print(vectorizer.vocabulary_)


# print(', ( )'.lower().replace('[...…,()]', 'a'), ' | ', ', ( )'.replace('[...…,()]', 'a'))
generate_vocabulary(root='data')
