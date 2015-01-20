import numpy as np

# From word to number.
word_dict = {}

# From number to word, numbers need to minus one to convert to list indices.
word_array = []

# Word frequency
word_freq = []

with open('../data/sentiment/train2.txt') as f:
    lines = f.readlines()

line_max = 0
sentence_dict = {}
line_numbers = []

# Key is 1-based, 0 is reserved for sentence end.
key = 1
for i in range(0, len(lines)):
    # Remove duplicate records.
    if not sentence_dict.has_key(lines[i]):
        sentence_dict[lines[i]] = 1
        line_numbers.append(i)
        words = lines[i].split(' ')
        for j in range(1, len(words) - 1):
            if len(words) - 2 > line_max:
                line_max = len(words) - 2
            if not word_dict.has_key(words[j]):
                word_dict[words[j]] = key
                word_array.append(words[j])
                word_freq.append(1)
                key += 1
            else:
                k = word_dict[words[j]]
                word_freq[k - 1] += 1

input_ = np.zeros((len(line_numbers), line_max), int)
target_ = np.zeros((len(line_numbers), 1), int)
count = 0
for i in line_numbers:
    if lines[i][0] == 'p':
        target_[count, 0] = 1
    else:
        target_[count, 0] = 0
    words = lines[i].split(' ')

    # First word is target, last word is \n.
    for j in range(1, len(words) - 1):
        input_[count, j - 1] = word_dict[words[j]]
    count += 1

# # Output vocab file for converting word2vec
# with open('../data/sentiment3/vocabs.txt', 'w+') as vocab_f:
#     for word in word_array_unk:
#         vocab_f.write(word + '\n')

data = np.array((word_dict, word_array, word_freq, input_, target_), object)
np.save('../data/sentiment/train.npy', data)
