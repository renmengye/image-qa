import numpy as np
import tsne
import sys

def getWordEmbedding(weights, initSeed=1, initRange=0.42, pcaDim=0):
    np.random.seed(initSeed)
    for i in range(weights.shape[0]):
        if weights[i, 0] == 0.0:
            weights[i, :] = np.random.rand(weights.shape[1]) * initRange - initRange / 2.0
    if pcaDim > 0:
        weights = tsne.pca(weights, pcaDim)
    return weights

if __name__ == '__main__':
    """
    Usage:
    python word_embedding.py
        -v[ec] {numpy vec array}
        -o[utput] {output file}
        [-d[im] {PCA dimension}]
    """
    pcaDim = 0
    vocabFile = None
    outputFile = None

    for i, flag in enumerate(sys.argv):
        if flag == '-v' or flag == '-vec':
            vocabFile = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFile = sys.argv[i + 1]

    weights = np.load(vocabFile)
    weights = getWordEmbedding(
        weights=weights,
        pcaDim=pcaDim)
    np.save(outputFile, weights)