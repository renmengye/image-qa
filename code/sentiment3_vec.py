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
from trainer import *
from util_func import *
import sys
import tsne

trainDataFilename = '../data/sentiment3/train-info-1.npy'
vocabVecFilename = '../data/sentiment3/vocabs-vec-1.npy'

def getTrainData():
    data = np.load(trainDataFilename)
    input_ = data[3]
    target_ = data[4]
    return input_, target_

def getWordEmbedding(initSeed, initRange, pcaDim=0):
    np.random.seed(initSeed)
    weights = np.load(vocabVecFilename)
    for i in range(weights.shape[0]):
        if weights[i, 0] == 0.0:
            weights[i, :] = np.random.rand(weights.shape[1]) * initRange - initRange / 2.0
    if pcaDim > 0:
        weights = tsne.pca(weights, pcaDim)
    return weights.transpose()

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
        'dropout': True,
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

    pipeline = Trainer(
        name='sentiment3-vec',
        trainOpt=trainOpt,
        costFn=crossEntOne,
        decisionFn=hardLimit,
        outputFolder='../results')
    pipeline.addStage(TimeUnfold())
    pipeline.addStage(LinearDict(
        inputDim=np.max(trainInput)+1,
        outputDim=100,
        needInit=False,
        initWeights=getWordEmbedding(
            initSeed=2,
            initRange=0.42,
            pcaDim=100)),       # std ~= 0.12. U~[0.21, 0.21].
        learningRate=0.0)
    pipeline.addStage(TimeFold(
        timespan=timespan))
    pipeline.addStage(Dropout(
        dropoutRate=0.2,
        initSeed=3))
    pipeline.addStage(LSTM(
        inputDim=100,
        outputDim=50,
        initRange=0.1,
        initSeed=4,
        cutOffZeroEnd=True,
        multiErr=True),
        learningRate=0.8,
        gradientClip=0.1,
        weightClip=10.0,
        weightRegConst=5e-5,
        annealConst=0.01,
        outputdEdX=False)
    pipeline.addStage(Dropout(
        dropoutRate=0.5,
        initSeed=5))
    pipeline.addStage(LSTM(
        inputDim=50,
        outputDim=50,
        initRange=0.1,
        initSeed=6,
        cutOffZeroEnd=True,
        multiErr=False),
        learningRate=0.8,
        gradientClip=0.1,
        weightClip=20.0,
        weightRegConst=5e-5,
        annealConst=0.01)
    pipeline.addStage(Sigmoid(
        inputDim=50,
        outputDim=1,
        initRange=0.1,
        initSeed=7),
        learningRate=0.01,
        gradientClip=0.1,
        weightClip=3.0,
        weightRegConst=5e-5,
        annealConst=0.01)

    pipeline.train(trainInput, trainTarget)
