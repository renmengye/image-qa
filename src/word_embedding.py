import numpy as np
import tsne
import sys

def getWordEmbedding(weights, initSeed, initRange, pcaDim=0, transpose=True):
    np.random.seed(initSeed)
    for i in range(weights.shape[0]):
        if weights[i, 0] == 0.0:
            weights[i, :] = np.random.rand(weights.shape[1]) * initRange - initRange / 2.0
    if pcaDim > 0:
        weights = tsne.pca(weights, pcaDim)
    if transpose:
        return weights.transpose()
    else:
        return weights

if __name__ == '__main__':
    if len(sys.argv) > 4:
        vocabFile = sys.argv[1]
        outputFile = sys.argv[2]
        pcaDim = int(sys.argv[3])
        transpose = sys.argv[4] == 'transpose'
    else:
        vocabFile = '../data/sentiment3/vocabs-vec-1.npy'
        pcaDim = 0
        transpose = True
        outputFile = '../data/sentiment3/word-embed-%d.npy' % pcaDim

    weights = np.load(vocabFile)
    weights = getWordEmbedding(
        weights=weights,
        initSeed=1,
        initRange=0.42,
        pcaDim=pcaDim,
        transpose=transpose)
    np.save(outputFile, weights)