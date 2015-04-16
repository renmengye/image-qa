import nn
import os
import sys
import numpy as np

def calcPrecision(Y, T):
    # Calculate precision
    correctAt1 = 0
    correctAt5 = 0
    correctAt10 = 0
    for n in range(0, Y.shape[0]):
        sortIdx = np.argsort(Y[n], axis=0)
        sortIdx = sortIdx[::-1]
        for i in range(0, 10):
            if sortIdx[i] == T[n, 0]:
                if i == 0:
                    correctAt1 += 1
                if i <= 4:
                    correctAt5 += 1
                correctAt10 += 1
    print 'rate @ 1: %.4f' % (correctAt1 / float(Y.shape[0]))
    print 'rate @ 5: %.4f' % (correctAt5 / float(Y.shape[0]))
    print 'rate @ 10: %.4f' % (correctAt10 / float(Y.shape[0]))

if __name__ == '__main__':
    """
    Usage: test.py id -data dataFolder
    """
    taskId = sys.argv[1]
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]

    resultFolder = '../results/%s' % taskId
    trainOutFile = os.path.join('../results/%s' % taskId, '%s.train.o.npy' % taskId)
    testOutFile = os.path.join('../results/%s' % taskId, '%s.test.o.npy' % taskId)
    testAnswerFile = os.path.join('../results/%s' % taskId, '%s.test.o.txt' % taskId)
    testTruthFile = os.path.join('../results/%s' % taskId, '%s.test.t.txt' % taskId)
    modelFile = '../results/%s/%s.model.yml' % (taskId, taskId)
    model = nn.load(modelFile)
    model.loadWeights(
        np.load('../results/%s/%s.w.npy' % (taskId, taskId)))

    trainDataFile = os.path.join(dataFolder, 'train.npy')
    testDataFile = os.path.join(dataFolder, 'test.npy')
    trainData = np.load(trainDataFile)
    testData = np.load(testDataFile)

    X = trainData[0]
    Y = nn.test(model, X)
    T = trainData[1]
    TX = testData[0]
    TY = nn.test(model, TX)
    TT = testData[1]
    vocabDict = np.load(dictFile)
    answerArray = vocabDict[3]
    with open(testTruthFile, 'w+') as f:
        for n in range(0, TT.shape[0]):
            f.write(answerArray[TT[n, 0]] + '\n')
    with open(testAnswerFile, 'w+') as f:
        for n in range(0, TY.shape[0]):
            f.write(answerArray[np.argmax(TY[n, :])] + '\n')
    calcPrecision(Y, T)
    calcPrecision(TY, TT)
    np.save(trainOutFile, Y)
    np.save(testOutFile, TY)