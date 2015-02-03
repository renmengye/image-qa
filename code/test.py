from trainer import *

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
    taskId = sys.argv[1]
    dataFile = sys.argv[2]
    dictFile = sys.argv[3]
    trainOutFile = os.path.join('../results/%s' % taskId, '%s.train.o.npy' % taskId)
    testOutFile = os.path.join('../results/%s' % taskId, '%s.test.o.npy' % taskId)
    testAnswerFile = os.path.join('../results/%s' % taskId, '%s.test.o.txt' % taskId)
    testTruthFile = os.path.join('../results/%s' % taskId, '%s.test.t.txt' % taskId)
    configFile = '../results/%s/%s.yaml' % (taskId, taskId)

    trainer = Trainer.initFromConfig(
        name='test',
        configFilename=configFile,
        outputFolder='../results')
    trainer.loadWeights('../results/%s/%s.w.npy' % (taskId, taskId))
    data = np.load(dataFile)
    Y = trainer.test(data[0],data[1])
    T = data[1]
    TY = trainer.test(data[2],data[3])
    TT = data[3]

    vocabDict = np.load(dictFile)
    answerArray = vocabDict[3]
    with open(testTruthFile, 'w+') as f:
        for n in range(0, T.shape[0]):
            f.write(answerArray[T[n, 0]] + '\n')
    with open(testAnswerFile, 'w+') as f:
        for n in range(0, Y.shape[0]):
            f.write(answerArray[np.argmax(Y[n, :])] + '\n')

    calcPrecision(Y, T)
    calcPrecision(TY, TT)
    np.save(trainOutFile, Y)
    np.save(testOutFile, TY)
