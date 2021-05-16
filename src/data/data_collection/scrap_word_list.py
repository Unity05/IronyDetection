#### USING: http://www.mieliestronk.com/corncob_lowercase.txt
#### USING: https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt

import urllib.request
import re


english_word_list = list(filter(None, re.split('\n|\r', urllib.request.urlopen(
    'https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-no-swears.txt').read().decode('utf-8'))))

with open('data/english_word_list_10k.txt', 'w') as english_word_list_file:
    for word in english_word_list:
        english_word_list_file.write((word + '\n'))
