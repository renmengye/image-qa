import sys
import numpy as np
import io
import struct
#with open('D:\\Projects\\word2vec_win\\Debug\\vocab_index.txt', 'r') as f:

# Usage: word2vec_read vocab-file output-file vec-bin-file vec-bin-index-file
if len(sys.argv) > 2:
    vocabFilename = sys.argv[1]
    vecOutputFilename = sys.argv[2]
else:
    vocabFilename = '../data/sentiment3/vocabs.txt'
    vecOutputFilename = '../data/sentiment3/vocabs-vec.npy'
if len(sys.argv) > 4:
    word2vecBinFilename = sys.argv[3]
    word2vecIdxFilename = sys.argv[4]
else:
    word2vecBinFilename = 'C:\\Users\\renme_000\\Downloads\\GoogleNews-vectors-negative300.bin\\GoogleNews-vectors-negative300.bin'
    word2vecIdxFilename = 'C:\\Users\\renme_000\\Downloads\\GoogleNews-vectors-negative300.bin\\vocab_index.txt'
word2vecIdxFile = open(word2vecIdxFilename, 'r')
word2vecBinFile = open(word2vecBinFilename, 'rb')
vocabFile = open(vocabFilename, 'r')

numWords = 3000000
numDim = 300

wordLocationDict = {}

allVocabs = vocabFile.readlines()
allVocabsDict = {}
i = 0
for word in allVocabs:
    allVocabsDict[word[0:-1]] = i
    i += 1

vecArray = np.zeros((len(allVocabs), numDim), float)

for line in word2vecIdxFile.readlines():
    parts = line.split(',')
    word = parts[0].lower()
    if not wordLocationDict.has_key(word) and allVocabsDict.has_key(word):
        location = long(parts[-1])
        wordLocationDict[word] = location
        word2vecBinFile.seek(location, io.SEEK_SET)
        i = allVocabsDict[word]
        for j in range(0, numDim):
            vecArray[i, j] = struct.unpack('f', word2vecBinFile.read(4))[0]

np.save(vecOutputFilename, vecArray)
word2vecBinFile.close()
word2vecIdxFile.close()
vocabFile.close()
