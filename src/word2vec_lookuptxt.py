import sys
import numpy as np
import io
import struct
from StringIO import StringIO
#with open('D:\\Projects\\word2vec_win\\Debug\\vocab_index.txt', 'r') as f:

# Usage: word2vec_read vocab-file output-file vec-bin-file vec-bin-index-file
vocabFilename = sys.argv[1]
vecOutputFilename = sys.argv[2]
word2vecTxtFilename = sys.argv[3]

#word2vecTxtFilename = '../../../data/mscoco/train/vec.txt'
wordDict = {'UNK': 0}
i = 1
count = 0
array = []
with open(word2vecTxtFilename) as f:
    for line in f:
        count += 1
        if count <= 2:
            continue
        word = line[:line.index(' ')]
        wordDict[word] = i
        i += 1
        array.append(
            np.loadtxt(
                StringIO(line[line.index(' ') + 1:-2]), dtype='float32'))
        array[-1] = array[-1].reshape(1, array[-1].shape[0])

array.insert(0, np.zeros((1, array[0].shape[-1]), dtype='float32'))
array = np.concatenate(array)

wordIdx = []
with open(vocabFilename) as f:
    for line in f:
        word = line[:-1]
        if wordDict.has_key(word):
            wordIdx.append(wordDict[word])
        else:
            wordIdx.append(0)

wordarray = array[wordIdx]
np.save(vecOutputFilename, wordarray)