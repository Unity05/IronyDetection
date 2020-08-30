import json
import bcolz
import numpy as np
import random as rnd
import torch


vectors = bcolz.carray(np.zeros(1), rootdir='data/irony_data/glove/6B.200.dat', mode='w')


with open('data/irony_data/vocabulary.json', 'r') as word_dict_file:
    word_dict = json.load(word_dict_file)

word_list = [word for _, word in word_dict.items()]
top_k = 100000
words_changed = 0
print(word_list)

with open('data/irony_data/glove/glove.6B.200d.txt', 'rb') as glove_file:
    glove_word_dict = {line.decode().split()[0]: np.array(line.decode().split()[1:]) for line in glove_file}
    glove_word_dict_list = list(glove_word_dict)
    print(glove_word_dict_list)
    # print(glove_word_dict)
    i = 0
    for _, word in word_dict.items():
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

vectors = bcolz.carray(vectors[1:].reshape((top_k, 200)), rootdir='data/irony_data/glove/6B.200.dat', mode='w')
vectors.flush()

with open('data/irony_data/glove_adjusted_vocabulary.json', 'w') as glove_adjusted_vocabulary_file:
    json.dump(word_dict, glove_adjusted_vocabulary_file)

print('words_changed: ', words_changed)

"""vectors = bcolz.open('data/irony_data/glove/6B.200.dat')
weights_matrix = np.zeros((100000, 200))
for i in range(100000):
    weights_matrix[i] = vectors[i]

torch.save(torch.from_numpy(weights_matrix), 'data/irony_data/glove/6B.200.dat.pth')"""

