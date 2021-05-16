import pandas as pd
import json
import os
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist, bigrams
from nltk.corpus import wordnet


class CreateVocabulary:
    def __init__(self):
        self.vocab = []
        self.i = 0

    def add(self, x):
        self.vocab += x.split()
        self.i += 1


class CreateBigramVocabulary:
    def __init__(self):
        self.vocab = []
        self.i = 0

    def add(self, x):
        self.vocab += list(bigrams(x.split()))
        self.i += 1


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def generate_vocabulary(root='data'):
    if not os.path.isfile(os.path.join(root, 'irony_data/SARC_2.0/comments-original-adjusted.json')):
        lemmatizer = WordNetLemmatizer()

        with open(os.path.join(root, 'irony_data/abbr_replacers'), 'r') as file:
            abbreviation_policy = json.load(file)

        with open(os.path.join(root, 'irony_data/SARC_2.0/comments.json'), 'r') as comments_json_file:
            comments_json = json.load(comments_json_file)

        character_replacement_dict = {'.': '', '…': '', ',': '', '(': '', ')': '', '-': ' ', ';': '', ':': '',
                                      '?': ' ?', '!': ' !', '=': '', '*': '', '~': ' ', '%': '', '"': '', '$': '',
                                      '^': ' ', '#': '', '<': ' ', '>': ' ', '_': ' ', '{': ' ', '}': ' ', '/': ' ',
                                      '\\': ' ', '|': ''}

        print(len(comments_json))
        for i, comment_json_key in enumerate(comments_json.keys()):
            adjusted_comment_text_list = [abbreviation_policy.get(e, e) for e in
                                          comments_json[comment_json_key]['text'].lower().translate(
                                              str.maketrans(character_replacement_dict)).
                                              replace(" '", "").replace("' ", "").replace("'s", " is").
                                              replace("'d", " would").replace("'re", " are")]
            comments_json[comment_json_key] = [''.join(adjusted_comment_text_list)]

            if i % 10000 == 0:
                print((i / len(comments_json)) * 100)

        with open(os.path.join(root, 'irony_data/SARC_2.0/comments-original-adjusted.json'), 'w') as adjusted_comments_json_file:
            json.dump(comments_json, adjusted_comments_json_file)
    else:
        with open(os.path.join(root, 'irony_data/SARC_2.0/comments-original-adjusted.json'), 'r') as comments_json_file:
            comments_json = json.load(comments_json_file)

    vocabulary_creator = CreateVocabulary()
    for i, comment_json_key in enumerate(comments_json.keys()):
        vocabulary_creator.add(x=comments_json[comment_json_key][0])

        if i % 100000 == 0:
            print((i / len(comments_json)) * 100)

    vocab = vocabulary_creator.vocab
    vocab_words_frequencies = FreqDist(samples=vocab)

    k = -1

    vocabulary = {}
    for i, word in enumerate(list(vocab_words_frequencies)[:k]):
        vocabulary[i] = word

    with open(os.path.join(root, 'irony_data/SARC_2.0/vocabulary-original-adjusted.json'), 'w') as vocabulary_file:
        json.dump(vocabulary, vocabulary_file)


def generate_bigram_vocabulary(root='data'):
    with open(os.path.join(root, 'irony_data/SARC_2.0/adjusted-comments.json'), 'r') as comments_json_file:
        comments_json = json.load(comments_json_file)

    bigram_vocabulary_creator = CreateBigramVocabulary()
    for i, comment_json_key in enumerate(comments_json.keys()):
        bigram_vocabulary_creator.add(x=comments_json[comment_json_key][0])

        if i % 100000 == 0:
            print((i / len(comments_json)) * 100)

    bigram_vocab = bigram_vocabulary_creator.vocab
    vocab_bigram_frequencies = FreqDist(samples=bigram_vocab)

    k = 300000

    vocabulary = {}
    for i, word in enumerate(list(vocab_bigram_frequencies)[:k]):
        vocabulary[i] = ' '.join(word)

    with open('../../../data/irony_data/SARC_2.0/glove_adjusted_vocabulary.json', 'r') as vocabulary_file:
        vocabulary_dict = json.load(vocabulary_file)

    for i, word in enumerate(list(vocabulary_dict.values())[:100000]):
        vocabulary[(i + k)] = word

    with open(os.path.join(root, 'irony_data/SARC_2.0/vocabulary-original-adjusted-bigrams.json'), 'w') as vocabulary_file:
        json.dump(vocabulary, vocabulary_file)


def get_label_ratio(root='data'):
    df = pd.read_csv(os.path.join(root, 'irony_data/SARC_2.0/test-balanced-adjusted.csv'))
    ratio = (
        len(df.loc[df['labels'] == 0]) / len(df.loc[df['labels'] == 1])
    )
    print(df)
    print(ratio)
    print(len(df.loc[df['labels'] == 0]))
    print(len(df.loc[df['labels'] == 1]))

#  generate_vocabulary(root='data')
#  generate_bigram_vocabulary(root='data')
get_label_ratio(root='data')
