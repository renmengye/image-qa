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

# Sort frequency
sorted_key = sorted(range(len(word_freq)), key=lambda k: word_freq[k], reverse=True)

with open('../data/sentiment3/word_freq.txt', 'w+') as f:
    for k in sorted_key:
        f.write('%s, %d\n' % (word_array[k], word_freq[k]))

# Replace low frequency words as unk_
key = 1
word_dict_unk = {}
word_array_unk = []
lowest_freq = 1
unknown = 'unk_'
count = 0
for k in sorted_key:
    count += 1
    if word_freq[k] < lowest_freq:
        break
word_dict_unk[unknown] = count
for i in range(0, len(lines)):
    words = lines[i].split(' ')
    for j in range(0, len(words) - 1):
        word = words[j]
        if not word_dict_unk.has_key(word):
            if word_freq[word_dict[word] - 1] >= lowest_freq:
                word_dict_unk[word] = key
                key += 1
                word_array_unk.append(word)
word_array_unk.append(unknown)

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
        word = words[j]
        if word_dict_unk.has_key(word):
            input_[count, j] = word_dict_unk[word]
        else:
            input_[count, j] = word_dict_unk[unknown]
    count += 1

data = np.array((word_dict_unk, word_array_unk, word_freq, input_, target_), object)

# Output vocab file for converting word2vec
with open('../data/sentiment3/vocabs-1.txt', 'w+') as vocab_f:
    for word in word_array_unk:
        vocab_f.write(word + '\n')

np.save('../data/sentiment3/train-1.npy', data)
