import numpy as np

# From word to number.
word_dict = {}

# From number to word, numbers need to minus one to convert to list indices.
word_array = []

# Word frequency
word_freq = []

with open('../data/sentiment3/rt-polarity.pos.prep.txt') as f:
    lines = f.readlines()

pos_marker = len(lines)
with open('../data/sentiment3/rt-polarity.neg.prep.txt') as f:
    lines.extend(f.readlines())

line_max = 0
line_numbers = []

# Key is 1-based, 0 is reserved for sentence end.
key = 1
for i in range(0, len(lines)):
    words = lines[i].split(' ')
    for j in range(0, len(words) - 1):
        if len(words) - 1 > line_max:
            line_max = len(words) - 1
        if not word_dict.has_key(words[j]):
            word_dict[words[j]] = key
            word_array.append(words[j])
            word_freq.append(1)
            key += 1
        else:
            k = word_dict[words[j]]
            word_freq[k - 1] += 1

input_ = np.zeros((len(lines), line_max), int)
target_ = np.zeros((len(lines), 1), int)
count = 0

for i in range(0, len(lines)):
    if i < pos_marker:
        target_[count, 0] = 1
    else:
        target_[count, 0] = 0
    words = lines[i].split(' ')

    # Last word is \n.
    for j in range(0, len(words) - 1):
        input_[count, j] = word_dict[words[j]]
    count += 1

data = np.array((word_dict, word_array, word_freq, input_, target_), object)
np.save('../data/sentiment3/train.npy', data)
