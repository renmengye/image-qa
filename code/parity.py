from lstm import *

def getData(trainSize, testSize, length):
    trainInput = np.zeros((trainSize, length, 1), float)
    trainTarget = np.zeros((trainSize, length, 1), float)
    testInput = np.zeros((testSize, length, 1), float)
    testTarget = np.zeros((testSize, length, 1), float)
    for i in range(0, trainSize):
        for j in range(0, length):
            trainInput[i, j, :] = np.round(np.random.rand(1))
            trainTarget[i, j, :] = np.mod(np.sum(trainInput[i, :, :]), 2)
    for i in range(0, testSize):
        for j in range(0, length):
            testInput[i, j, :] = np.round(np.random.rand(1))
            testTarget[i, j, :] = np.mod(np.sum(testInput[i, :, :]), 2)
    return trainInput, trainTarget, testInput, testTarget

if __name__ == '__main__':
    lstm = LSTM(
        inputDim=1,
        memoryDim=1,
        initRange=0.01,
        initSeed=2)

    trainOpt = {
        'learningRate': 0.8,
        'numEpoch': 1200,
        'momentum': 0.9,
        'batchSize': 1,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'needValid': True,
        'name': 'parity_train',
        'plotFigs': True,
        'combineFnDeriv': simpleSumDeriv,
        'calcError': True,
        'decisionFn': simpleSumDecision,
        'stoppingE': 0.015,
        'stoppingR': 1.0
    }

    np.random.seed(2)
    trainSize = 6
    testSize = 1000
    length = 8
    trainInput, trainTarget, testInput, testTarget = getData(trainSize, testSize, length)
    lstm.train(trainInput, trainTarget, trainOpt)
    rate, correct, total = lstm.testRate(testInput, testTarget, simpleSumDecision)

    print 'TR: %.4f' % rate
    lstm.save('parity.npy')
    raw_input('Press Enter to continue.')
    pass