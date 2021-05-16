import json
import bcolz
import numpy as np
import random as rnd


vectors = bcolz.carray(np.zeros(1), rootdir='data/irony_data/SARC_2.0/6B.300_original.dat', mode='w')


with open('../../../data/irony_data/SARC_2.0/vocabulary-original-adjusted.json', 'r') as word_dict_file:
    word_dict = json.load(word_dict_file)

word_list = [word for _, word in word_dict.items()]
top_k = 100000
words_changed = 0

with open('../../../data/irony_data/glove/glove.6B.300d.txt', 'rb') as glove_file:
    glove_word_dict = {line.decode().split()[0]: np.array(line.decode().split()[1:]) for line in glove_file}
    glove_word_dict_list = list(glove_word_dict)
    i = 0
    for _, word in word_dict.items():
        if i % 10000 == 0:
            print('i: ', i, ' | word: ', word)
        try:
            vectors.append(glove_word_dict[word].astype(np.float))
        except KeyError:        # glove has no data about the current word - choose a random other one
            x = 0
            while x is 0:
                random_word = rnd.choice(glove_word_dict_list)
                if random_word not in word_list:
                    x = 1
            vectors.append(np.array(glove_word_dict[random_word].astype(np.float)))
            word_dict[str(i)] = random_word
            words_changed += 1
        i += 1
        if i == top_k:
            break

vectors = bcolz.carray(vectors[1:].reshape((top_k, 300)),
                       rootdir='data/irony_data/SARC_2.0/6B.300_original_adjusted.dat', mode='w')
vectors.flush()

with open('../../../data/irony_data/SARC_2.0/glove_adjusted_vocabulary_original_adjusted.json', 'w') as glove_adjusted_vocabulary_file:
    json.dump(word_dict, glove_adjusted_vocabulary_file)

print('words_changed: ', words_changed)
