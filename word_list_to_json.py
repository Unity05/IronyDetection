import json


path = 'data/english_word_list_10k.txt'
target_path = 'data/english_word_dict_10k.json'

dict = {}

with open(path, 'r') as txt_file:
    for i, word in enumerate(txt_file):
        dict[(i + 1)] = word[:-1]
    dict[(i + 2)] = 'NAME'

with open(target_path, 'w') as target_file:
    json.dump(dict, target_file)
