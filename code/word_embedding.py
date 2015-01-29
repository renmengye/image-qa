import numpy as np
import tsne

vocabFile = '../data/sentiment3/vocabs-vec-1.npy'
pcaDim = 0
outputFile = '../data/sentiment3/word-embed-%d.npy' % pcaDim

def getWordEmbedding(initSeed, initRange, pcaDim=0):
    np.random.seed(initSeed)
    weights = np.load(vocabFile)
    for i in range(weights.shape[0]):
        if weights[i, 0] == 0.0:
            weights[i, :] = np.random.rand(weights.shape[1]) * initRange - initRange / 2.0
    if pcaDim > 0:
        weights = tsne.pca(weights, pcaDim)
    return weights.transpose()

if __name__ == '__main__':
    weights = getWordEmbedding(
        initSeed=1,
        initRange=0.42,
        pcaDim=pcaDim)
    np.save(outputFile, weights)