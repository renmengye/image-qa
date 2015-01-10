from lstm import *

def getData(size, length):
    input_ = np.zeros((size, length, 2), float)
    target_ = np.zeros((size, length, 1), float)
    for i in range(0, size):
        carry = 0
        for j in range(0, length):
            if j != length - 1:
                input_[i, j, 0] = np.round(np.random.rand(1))
                input_[i, j, 1] = np.round(np.random.rand(1))
            s = np.sum(input_[i, j, :]) + carry
            if s >= 2:
                carry = 1
                target_[i, j, 0] = np.mod(s, 2)
            else:
                carry = 0
                target_[i, j, 0] = s
        # if i < 10:
        #     print input_[i, :, 0]
        #     print input_[i, :, 1]
        #     print target_[i, :, 0]
        #     print
    return input_, target_

if __name__ == '__main__':
    lstm = LSTM(
        inputDim=2,
        memoryDim=3,
        initRange=0.01,
        initSeed=2)

    trainOpt = {
        'learningRate': 0.8,
        'numEpoch': 2000,
        'momentum': 0.9,
        'batchSize': 1,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'needValid': True,
        'name': 'binadd_train',
        'plotFigs': True,
        'combineFnDeriv': simpleSumDeriv,
        'calcError': True,
        'decisionFn': simpleSumDecision,
        'stoppingE': 0.002,
        'stoppingR': 1.0
    }

    np.random.seed(2)
    trainSize = 40
    testSize = 1000
    length = 8
    trainInput, trainTarget = getData(trainSize, length)
    testInput, testTarget = getData(testSize, length)
    lstm.train(trainInput, trainTarget, trainOpt)
    rate, correct, total = lstm.testRate(testInput, testTarget, simpleSumDecision, printEx=True)

    print 'TR: %.4f' % rate
    lstm.save('binadd.npy')
    raw_input('Press Enter to continue.')
    pass