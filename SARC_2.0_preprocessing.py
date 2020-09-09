import pandas as pd
import json
import os
import re
from nltk.stem import WordNetLemmatizer
from nltk import FreqDist, pos_tag
from nltk.corpus import wordnet
import sklearn


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
    if not os.path.isfile(os.path.join(root, 'irony_data/SARC_2.0/adjusted-comments.json')):
        lemmatizer = WordNetLemmatizer()

        # df = pd.read_csv(os.path.join(root, 'irony_data/SARC_2.0/train-balanced.csv', names=['data'], encoding='utf-8'))

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
                                              replace(" '", "").replace("' ", "").split()]
            comments_json[comment_json_key] = [' '.join(lemmatizer.lemmatize(y[0], get_wordnet_pos(y[1]))
                                                        for y in pos_tag(adjusted_comment_text_list))]

            if i % 10000 == 0:
                print((i / len(comments_json)) * 100)

        with open(os.path.join(root, 'irony_data/SARC_2.0/adjusted-comments.json'), 'w') as adjusted_comments_json_file:
            json.dump(comments_json, adjusted_comments_json_file)
    else:
        with open(os.path.join(root, 'irony_data/SARC_2.0/adjusted-comments.json'), 'r') as comments_json_file:
            comments_json = json.load(comments_json_file)

    vocabulary_creator = CreateVocabulary()
    for i, comment_json_key in enumerate(comments_json.keys()):
        # print(comments_json[comment_json_key][0].split())
        vocabulary_creator.add(x=comments_json[comment_json_key][0])

        if i % 100000 == 0:
            print((i / len(comments_json)) * 100)

    vocab = vocabulary_creator.vocab
    vocab_words_frequencies = FreqDist(samples=vocab)

    k = -1

    vocabulary = {}
    for i, word in enumerate(list(vocab_words_frequencies)[:k]):
        vocabulary[i] = word

    with open(os.path.join(root, 'irony_data/SARC_2.0/vocabulary.json'), 'w') as vocabulary_file:
        json.dump(vocabulary, vocabulary_file)


"""root = 'data/irony_data/SARC_2.0'

print(os.listdir(root))

with open(os.path.join(root, 'comments.json'), 'r') as comments_json_file:
    comments_json = json.load(comments_json_file)

train_df = pd.read_csv(os.path.join(root, 'train-balanced.csv'), names=['data'], encoding='utf-8')

test_df = pd.read_csv(os.path.join(root, 'test-balanced.csv'), names=['data'], encoding='utf-8')

print(train_df.iloc[0])

print(train_df)

x = train_df.iloc[128538].item()

print(x)

y = re.split('|', x)

post_ids = re.split(' ', y[0])
response_ids = re.split(' ', y[1])
labels = re.split(' ', y[2])

y = x.split('|')

post_ids = y[0].split(' ')
response_ids = y[1].split(' ')
labels = y[2].split(' ')

print(post_ids)
print(response_ids)
print(labels)

print(comments_json[post_ids[0]])
print(comments_json[post_ids[1]])
print(comments_json[response_ids[0]])
print(comments_json[response_ids[1]])"""

generate_vocabulary(root='data')
