from lstm import *
from simplesum import *
from softmax import *
from sigmoid import *
from time_unfold import *
from time_fold import *
from time_select import *
from dropout import *
from linear_map import *
from linear_dict import *
from pipeline import *
from util_func import *
import sys

def getTrainData():
    data = np.load('../data/sentiment3/train-info-1.npy')
    input_ = data[3]
    target_ = data[4]
    return input_, target_

if __name__ == '__main__':
    trainInput, trainTarget = getTrainData()          # 8555 records

    np.random.seed(1)
    subset = np.arange(0, 8555)                       # permute
    if len(sys.argv) < 3:
        subset = np.random.permutation(subset)
    trainInput = trainInput[subset]
    trainInput = trainInput.reshape(trainInput.shape[0], trainInput.shape[1], 1)
    trainTarget = trainTarget[subset]
    timespan = trainInput.shape[1]

    trainOpt = {
        'numEpoch': 2000,
        'heldOutRatio': 0.1,
        'momentum': 0.9,
        'batchSize': 25,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'shuffle': True,
        'needValid': True,
        'writeRecord': True,
        'saveModel': True,
        'plotFigs': True,
        'everyEpoch': True,
        'calcError': True,
        'stopCost': 0.01,
        'progress': True,
        'displayDw': 4
    }

    pipeline = Pipeline(
        name='sentiment3',
        trainOpt=trainOpt,
        costFn=crossEntOne,
        decisionFn=hardLimit,
        outputFolder='../results')
    pipeline.addStage(TimeUnfold())
    pipeline.addStage(LinearDict(
        inputDim=np.max(trainInput)+1,
        outputDim=40,
        initRange=0.1,
        initSeed=2),
        learningRate=0.8,
        outputdEdX=False)
    pipeline.addStage(TimeFold(
        timespan=timespan))
    pipeline.addStage(Dropout(
        dropoutRate=0.2))
    pipeline.addStage(LSTM(
        inputDim=40,
        outputDim=20,
        initRange=0.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=True),
        learningRate=0.8,
        gradientClip=0.1)
    pipeline.addStage(Dropout(
        dropoutRate=0.5))
    pipeline.addStage(LSTM(
        inputDim=20,
        outputDim=10,
        initRange=0.1,
        initSeed=4,
        cutOffZeroEnd=True,
        multiErr=False),
        learningRate=0.8,
        gradientClip=0.1)
    pipeline.addStage(Sigmoid(
        inputDim=10,
        outputDim=1,
        initRange=0.1,
        initSeed=5),
        learningRate=0.01)

    pipeline.train(trainInput, trainTarget)