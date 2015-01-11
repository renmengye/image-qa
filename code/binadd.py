from lstm import *
from simplesum import *
from pipeline import *
from util_func import *

def getData(size, length, seed=2):
    np.random.seed(seed)
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
    return input_, target_

if __name__ == '__main__':
    pipeline = Pipeline(
        name='binadd',
        costFn=meanSqErr,
        decisionFn=hardLimit)

    pipeline.addStage(LSTM(
        inputDim=2,
        memoryDim=3,
        initRange=0.01,
        initSeed=2))

    pipeline.addStage(SimpleSum())

    trainOpt = {
        'learningRate': 0.8,
        'numEpoch': 2000,
        'heldOutRatio': 0.5,
        'momentum': 0.9,
        'batchSize': 1,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'needValid': True,
        'plotFigs': True,
        'calcError': True,
        'stopE': 0.006
    }

    trainInput, trainTarget = getData(
        size=40,
        length=8,
        seed=2)

    testInput, testTarget = getData(
        size=1000,
        length=8,
        seed=2)

    pipeline.train(trainInput, trainTarget, trainOpt)
    pipeline.testRate(testInput, testTarget)
    pipeline.save()
    pass