import sys
import numpy as np
from StringIO import StringIO

def lookup(words, word2vecTxtFilename):
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
    for word in words:
        if wordDict.has_key(word):
            wordIdx.append(wordDict[word])
        else:
            wordIdx.append(0)

    wordarray = array[wordIdx]
    return wordarray

if __name__ == '__main__':
    """
    Usage:
    python word2vec_lookuptxt.py \
        -w[ords] {vocab file name}
        -o[utput] {output file name}
        [-t[xt] {word2vec text file}]
    """
    vocabFilename = None
    vecOutputFilename = None
    word2vecTxtFilename = '/ais/gobi3/u/$USER/data/mscoco/word2vec300.txt'
    for i, flag in enumerate(sys.argv):
        if flag == '-w' or flag == '-word':
            vocabFilename = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            vecOutputFilename = sys.argv[i + 1]
        elif flag == '-t' or flag == '-txt':
            word2vecTxtFilename = sys.argv[i + 1]

    if vocabFilename is None:
        raise Exception('No input word list provided.')
    if vecOutputFilename is None:
        raise Exception('No output npy file provided.')

    words = []
    with open(vocabFilename) as f:
        for line in f:
            words.append(line[:-1])

    wordarray = lookup(words, word2vecTxtFilename)
    np.save(vecOutputFilename, wordarray)