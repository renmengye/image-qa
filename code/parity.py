from lstm import *
from simplesum import *
from pipeline import *
from util_func import *

def getData(size, length, seed=2):
    np.random.seed(seed)
    input_ = np.zeros((size, length, 1), float)
    target_ = np.zeros((size, length, 1), float)
    for i in range(0, size):
        for j in range(0, length):
            input_[i, j, :] = np.round(np.random.rand(1))
            target_[i, j, :] = np.mod(np.sum(input_[i, :, :]), 2)
    return input_, target_

if __name__ == '__main__':
    pipeline = Pipeline(
        name='parity',
        costFn=meanSqErr,
        decisionFn=hardLimit)

    pipeline.addStage(LSTM(
        inputDim=1,
        memoryDim=1,
        initRange=0.01,
        initSeed=2))

    pipeline.addStage(SimpleSum())

    trainOpt = {
        'learningRate': 0.8,
        'numEpoch': 1200,
        'needValid': True,
        'heldOutRatio': 0.5,
        'momentum': 0.9,
        'batchSize': 1,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'plotFigs': True,
        'calcError': True,
        'stopE': 0.015
    }

    trainInput, trainTarget = getData(
        size=6,
        length=8,
        seed=2)

    testInput, testTarget = getData(
        size=1000,
        length=8,
        seed=3)

    pipeline.train(trainInput, trainTarget, trainOpt)
    pipeline.testRate(testInput, testTarget)
    pipeline.save()
    pass