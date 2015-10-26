import numpy as np
import sys
import os
import h5py

def replaceEmpty(data, vocabSize):
    data2 = np.copy(data)
    for i in range(data2.shape[0]):
        for j in range(data2.shape[1]):
            if data2[i, j] == 0:
                data2[i, j] = vocabSize
    return data2

if __name__ == '__main__':
    dataFolder = '../data/cocoqa'
    for i, flag in enumerate(sys.argv):
        if flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
    trainFile = 'train.npy'
    validFile = 'valid.npy'
    testFile = 'test.npy'
    answerVocabFile = 'answer_vocabs.txt'

    trainData = np.load(os.path.join(dataFolder, trainFile))
    validData = np.load(os.path.join(dataFolder, validFile))
    testData = np.load(os.path.join(dataFolder, testFile))
    outputFile = h5py.File(os.path.join(dataFolder, 'all_id.h5'), 'w')
    outputFile['trainData'] = trainData[0].reshape(trainData[0].shape[0], trainData[0].shape[1])
    outputFile['trainLabel'] = trainData[1].reshape(trainData[1].size)
    outputFile['validData'] = validData[0].reshape(validData[0].shape[0], validData[0].shape[1])
    outputFile['validLabel'] = validData[1].reshape(validData[1].size)
    outputFile['testData'] = testData[0].reshape(testData[0].shape[0], testData[0].shape[1])
    outputFile['testLabel'] = testData[1].reshape(testData[1].size)
    outputFile.close()
    outputFile = h5py.File(os.path.join(dataFolder, 'all_id_unk.h5'), 'w')
    with open(os.path.join(dataFolder, answerVocabFile)) as f:
        vocabs = f.readline()
        vocabSize = len(vocabs)
    outputFile['trainData'] = \
        replaceEmpty(trainData[0].reshape(
        trainData[0].shape[0], trainData[0].shape[1]), vocabSize)
    outputFile['trainLabel'] = trainData[1].reshape(trainData[1].size) + 1
    outputFile['validData'] = \
        replaceEmpty(validData[0].reshape(
        validData[0].shape[0], validData[0].shape[1]), vocabSize)
    outputFile['validLabel'] = validData[1].reshape(validData[1].size) + 1
    outputFile['testData'] = \
        replaceEmpty(testData[0].reshape(
        testData[0].shape[0], testData[0].shape[1]), vocabSize)
    outputFile['testLabel'] = testData[1].reshape(testData[1].size) + 1
    outputFile.close()
