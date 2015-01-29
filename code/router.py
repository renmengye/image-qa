__author__ = 'renme_000'
from util_func import *
from lstm import *
from linear_dict import *
from time_unfold import *
from time_fold import *
from dropout import *
from sigmoid import *
from softmax import *

def routeFn(name):
    if name == 'crossEntOne':
        return crossEntOne
    elif name == 'crossEntInx':
        return crossEntIdx
    elif name == 'hardLimit':
        return hardLimit
    elif name == 'argmax':
        return argmax
    else:
        raise 'Function ' + name + ' not found.'
    pass

def routeStage(stageDict):
    if stageDict['type'] == 'lstm':
        return LSTM(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=np.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True,
            cutOffZeroEnd=stageDict['cutOffZeroEnd'],
            multiErr=stageDict['multiErr']
        )
    elif stageDict['type'] == 'linearDict':
        return LinearDict(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=np.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True
        )
        pass
    elif stageDict['type'] == 'timeUnfold':
        return TimeUnfold()
    elif stageDict['type'] == 'timeFold':
        return TimeFold(
            timespan=stageDict['timespan']
        )
    elif stageDict['type'] == 'dropout':
        return Dropout(
            dropoutRate=stageDict['dropoutRate'],
            initSeed=stageDict['initSeed']
        )
    elif stageDict['type'] == 'sigmoid':
        return Sigmoid(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=np.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True
        )
    elif stageDict['type'] == 'softmax':
        return Softmax(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=np.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True
        )
    else:
        raise 'Stage type ' + stageDict['type'] + ' not found.'