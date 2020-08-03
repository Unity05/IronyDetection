import numpy as np
import json


with open('data/char_index_policy.json') as policy_json_file:
    char_index_policy = json.load(policy_json_file)


def char_to_index(char):
    return char_index_policy[char]


pi = np.zeros((29, ), dtype=float)

a = np.zeros((29, 29), dtype=float)

p_q = np.zeros((29, ), dtype=float)

#path = 'data/hmm_data/english3.txt'
path = 'data/hmm_data/english_word_list.txt'
with open(path, 'r') as english_word_list_file:
    english_word_list = english_word_list_file.read().splitlines()

for word in english_word_list:
    last_letter_index = None
    for letter in word:
        letter_index = char_to_index(char=letter)
        if last_letter_index is None or last_letter_index == 1:
            pi[letter_index] += 1
            a[1][letter_index] += 1
        else:
            a[last_letter_index][letter_index] += 1

        p_q[letter_index] += 1

        last_letter_index = letter_index
    a[last_letter_index][1] += 1

p_q[1] = len(english_word_list)
p_q_sum = np.sum(p_q)
for i, v in enumerate(p_q):
    p_q[i] = (v / p_q_sum)

pi_sum = np.sum(pi)
for i, v in enumerate(pi):
    pi[i] = (v / pi_sum)

for row_i, row in enumerate(a[1:-2], start=1):
    row_sum = np.sum(row)
    for i, v in enumerate(row):
        a[row_i][i] = (v / row_sum)
a = np.transpose(a)
a[1] = [0] * 29

np.save('data/hmm_data/initial_probabilities.npy', pi)
np.save('data/hmm_data/transition_probabilities.npy', a)
np.save('data/hmm_data/state_probabilities.npy', p_q)
