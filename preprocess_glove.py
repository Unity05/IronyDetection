import json
import bcolz
import numpy as np
import random as rnd
import torch


"""# vectors = bcolz.carray(np.zeros(1), rootdir='data/irony_data/glove/6B.300.dat', mode='w')
vectors = bcolz.carray(np.zeros(1), rootdir='data/irony_data/SARC_2.0/6B.300_original.dat', mode='w')


with open('data/irony_data/SARC_2.0/vocabulary-original-adjusted.json', 'r') as word_dict_file:
    word_dict = json.load(word_dict_file)

word_list = [word for _, word in word_dict.items()]
top_k = 100000
words_changed = 0
print(word_list)

with open('data/irony_data/glove/glove.6B.300d.txt', 'rb') as glove_file:
    glove_word_dict = {line.decode().split()[0]: np.array(line.decode().split()[1:]) for line in glove_file}
    glove_word_dict_list = list(glove_word_dict)
    print(glove_word_dict_list)
    # print(glove_word_dict)
    i = 0
    for _, word in word_dict.items():
        if i % 10000 == 0:
            print('i: ', i, ' | word: ', word)
        # print(word)
        try:
            # vectors.append(np.array(glove_file[glove_word_dict_list.index(word)][1:]).astype(np.float()))
            vectors.append(glove_word_dict[word].astype(np.float))
        except KeyError:        # glove has no data about the current word - choose a random other one
            x = 0
            while x is 0:
                random_word = rnd.choice(glove_word_dict_list)
                if random_word not in word_list:
                    x = 1
            # vectors.append(np.array(glove_file[glove_word_dict_list.index(random_word)][1:]).astype(np.float()))
            vectors.append(np.array(glove_word_dict[random_word].astype(np.float)))
            word_dict[str(i)] = random_word
            words_changed += 1
        i += 1
        if i == top_k:
            break

# vectors = bcolz.carray(vectors[1:].reshape((top_k, 300)), rootdir='data/irony_data/glove/6B.300.dat', mode='w')
vectors = bcolz.carray(vectors[1:].reshape((top_k, 300)), rootdir='data/irony_data/SARC_2.0/6B.300_original_adjusted.dat', mode='w')
vectors.flush()

with open('data/irony_data/SARC_2.0/glove_adjusted_vocabulary_original_adjusted.json', 'w') as glove_adjusted_vocabulary_file:
    json.dump(word_dict, glove_adjusted_vocabulary_file)

print('words_changed: ', words_changed)"""

"""vectors = bcolz.open('data/irony_data/glove/6B.200.dat')
weights_matrix = np.zeros((100000, 200))
for i in range(100000):
    weights_matrix[i] = vectors[i]

torch.save(torch.from_numpy(weights_matrix), 'data/irony_data/glove/6B.200.dat.pth')"""

vectors = bcolz.carray(np.zeros(1), rootdir='data/irony_data/FastText/300d-1M.dat', mode='w')

with open('data/irony_data/SARC_2.0/vocabulary.json', 'r') as sarc_2_0_comments_dict_file:
    sarc_2_0_comments_dict = json.load(sarc_2_0_comments_dict_file)

# word_list = [word for word, _ in sarc_2_0_comments_dict.items()]
word_list = [word for _, word in sarc_2_0_comments_dict.items()]
top_k = 100000
words_changed = 0
print(word_list)
word_dict = {}

with open('data/irony_data/FastText/300d-1M-dict.json', 'r') as fast_text_dict_file:
    fast_text_dict = json.load(fast_text_dict_file)
    i = 0
    # for _, word in word_list:
    for word in word_list:
        if i % 10000 == 0:
            print('i: ', i, ' | word: ', word)
        try:
            vectors.append(np.array(fast_text_dict[word], dtype=np.float))
            # vectors.append(np.array(fast_text_dict[word]).astype(np.float()))
            word_dict[str(i)] = word
        except KeyError:        # FastText has no data about the current word - choose a random other one
            x = 0
            while x is 0:
                random_word = rnd.choice(list(fast_text_dict.keys()))
                if random_word not in word_list:
                    x = 1
            vectors.append(np.array(fast_text_dict[random_word], dtype=np.float))
            word_dict[str(i)] = random_word
            words_changed += 1
        i += 1
        if i == top_k:
            break

vectors = bcolz.carray(vectors[1:].reshape((top_k, 300)), rootdir='data/irony_data/FastText/300d-1M-SARC_2_0_Adjusted.dat', mode='w')
vectors.flush()

with open('data/irony_data/SARC_2.0/vocabulary_fast_text_adjusted.json', 'w') as vocabulary_fast_text_adjusted_dict_file:
    json.dump(word_dict, vocabulary_fast_text_adjusted_dict_file)

print('words changed: ', words_changed)

"""vectors = bcolz.open('data/irony_data/glove/6B.200.dat')"""
