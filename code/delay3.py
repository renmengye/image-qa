from lstm import *
from simplesum import *
from softmax import *
from sigmoid import *
from time_unfold import *
from time_fold import *
from pipeline import *
from util_func import *

def getData(size, length, seed=2):
    np.random.seed(seed)
    input_ = np.zeros((size, length, 1), float)
    target_ = np.zeros((size, length, 1), float)
    for i in range(0, size):
        for j in range(0, length):
            input_[i, j, :] = np.round(np.random.rand(1))
            if j >= 3:
                target_[i, j, :] = input_[i, j - 3, :]
    return input_, target_

if __name__ == '__main__':
    # pipeline = Pipeline(
    #     name='delay3',
    #     costFn=meanSqErr,
    #     decisionFn=hardLimit)
    # pipeline.addStage(LSTM(
    #     inputDim=1,
    #     memoryDim=5,
    #     initRange=0.01,
    #     initSeed=2))
    # pipeline.addStage(SimpleSum())
    # trainOpt = {
    #     'learningRate': 0.1,
    #     'numEpoch': 2000,
    #     'heldOutRatio': 0.5,
    #     'momentum': 0.9,
    #     'batchSize': 1,
    #     'learningRateDecay': 1.0,
    #     'momentumEnd': 0.9,
    #     'needValid': True,
    #     'plotFigs': True,
    #     'calcError': True,
    #     'stopE': 0.005
    # }

    # pipeline = Pipeline(
    #     name='delay3',
    #     costFn=crossEntIdx,
    #     decisionFn=argmax)
    pipeline = Pipeline(
        name='delay3',
        costFn=crossEntOne,
        decisionFn=hardLimit)
    pipeline.addStage(LSTM(
        inputDim=1,
        memoryDim=5,
        initRange=0.01,
        initSeed=2),
        learningRate=0.8,
        weightClip=0.1)
    pipeline.addStage(TimeUnfold())
    # pipeline.addStage(Softmax(
    #     inputDim=3,
    #     outputDim=2,
    #     initRange=0.01,
    #     initSeed=3))
    pipeline.addStage(Sigmoid(
        inputDim=5,
        outputDim=1,
        initRange=0.01,
        initSeed=3),
        learningRate=0.1)
    pipeline.addStage(TimeFold(
        timespan=8))
    trainOpt = {
        'learningRate': 0.1,
        'numEpoch': 2000,
        'heldOutRatio': 0.2,
        'momentum': 0.9,
        'batchSize': 5,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'dropout': False,
        'shuffle': False,
        'needValid': True,
        'writeRecord': False,
        'plotFigs': True,
        'everyEpoch': False,
        'calcError': True,
        'stopE': 0.01,
        'progress': False
    }

    trainInput, trainTarget = getData(
        size=80,
        length=8,
        seed=2)
    testInput, testTarget = getData(
        size=1000,
        length=8,
        seed=3)
    pipeline.train(trainInput, trainTarget, trainOpt)
    pipeline.test(testInput, testTarget)
    pipeline.save()
    pass