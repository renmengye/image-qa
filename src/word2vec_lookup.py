import sys
import numpy as np
import io
import struct

word2vecBinFilename = '/ais/gobi3/u/mren/data/word2vec/GoogleNews-vectors-negative300.bin'
word2vecIdxFilename = '/ais/gobi3/u/mren/data/word2vec/vocab_index.txt'
numDim = 300

def lookup(words, binFilename=word2vecBinFilename, idxFilename=word2vecIdxFilename, numDim=300):
    vocabsDict = {}
    wordLocationDict = {}
    idxFile = open(word2vecIdxFilename, 'r')
    binFile = open(word2vecBinFilename, 'rb')
    i = 0
    for word in words:
        vocabsDict[word[0:-1]] = i
        i += 1

    vecArray = np.zeros((len(words), numDim), float)

    for line in idxFile.readlines():
        parts = line.split(',')
        word = parts[0].lower()
        if not wordLocationDict.has_key(word) and vocabsDict.has_key(word):
            location = long(parts[-1])
            wordLocationDict[word] = location
            binFile.seek(location, io.SEEK_SET)
            i = vocabsDict[word]
            for j in range(0, numDim):
                vecArray[i, j] = struct.unpack('f', binFile.read(4))[0]

    binFile.close()
    idxFile.close()
    return vecArray

if __name__ == '__main__':
    """
    Usage:
    python word2vec_lookup.py
        -w[ord] {input word list}
        -o[utput] {output npy file}
        [-b[in] {word2vec binary file}]
        [-i[ndex] {word2vec index file}]
        [-d[im] {number of word vector dimension}]
    """
    vocabFilename = None
    vecOutputFilename = None
    for i, flag in enumerate(sys.argv):
        if flag == '-w' or flag == '-word':
            vocabFilename = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            vecOutputFilename = sys.argv[i + 1]
        elif flag == '-b' or flag == '-bin':
            word2vecBinFilename = sys.argv[i + 1]
        elif flag == '-i' or flag == '-index':
            word2vecIdxFilename = sys.argv[i + 1]
        elif flag == '-d' or flag == '-dim':
            numDim = int(sys.argv[i + 1])

    if vocabFilename is None:
        raise Exception('No input word list provided.')
    if vecOutputFilename is None:
        raise Exception('No output npy file provided.')

    with open(vocabFilename) as f:
        words = f.readlines()
    vecArray = lookup(words,
                      binFilename=word2vecBinFilename,
                      idxFilename=word2vecIdxFilename,
                      numDim=numDim)
    np.save(vecOutputFilename, vecArray)
