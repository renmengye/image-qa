from util_func import *
from lstm import *
from linear_dict import *
from time_unfold import *
from time_fold import *
from dropout import *
from sigmoid import *
from softmax import *
import numpy

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
    stage = None
    if stageDict['type'] == 'lstm':
        stage = LSTM(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=numpy.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True,
            cutOffZeroEnd=stageDict['cutOffZeroEnd'],
            multiErr=stageDict['multiErr'],
            learningRate=stageDict['learningRate']
            if stageDict.has_key('learningRate') else 0.0,
            learningRateAnnealConst=stageDict['learningRateAnnealConst']
            if stageDict.has_key('learningRatennealConst') else 0.0,
            momentum=stageDict['momentum']
            if stageDict.has_key('momentum') else 0.0,
            deltaMomentum=stageDict['deltaMomentum']
            if stageDict.has_key('deltaMomentum') else 0.0,
            gradientClip=stageDict['gradientClip']
            if stageDict.has_key('gradientClip') else 0.0,
            weightClip=stageDict['weightClip']
            if stageDict.has_key('weightClip') else 0.0,
            weightRegConst=stageDict['weightRegConst']
            if stageDict.has_key('weightRegConst') else 0.0,
            outputdEdX=stageDict['outputdEdX']
            if stageDict.has_key('outputdEdX') else True
        )
    elif stageDict['type'] == 'linearDict':
        stage = LinearDict(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=numpy.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True,
            learningRate=stageDict['learningRate']
            if stageDict.has_key('learningRate') else 0.0,
            learningRateAnnealConst=stageDict['learningRateAnnealConst']
            if stageDict.has_key('learningRateAnnealConst') else 0.0,
            momentum=stageDict['momentum']
            if stageDict.has_key('momentum') else 0.0,
            deltaMomentum=stageDict['deltaMomentum']
            if stageDict.has_key('deltaMomentum') else 0.0,
            gradientClip=stageDict['gradientClip']
            if stageDict.has_key('gradientClip') else 0.0,
            weightClip=stageDict['weightClip']
            if stageDict.has_key('weightClip') else 0.0,
            weightRegConst=stageDict['weightRegConst']
            if stageDict.has_key('weightRegConst') else 0.0
        )
    elif stageDict['type'] == 'timeUnfold':
        stage = TimeUnfold()
    elif stageDict['type'] == 'timeFold':
        stage = TimeFold(
            timespan=stageDict['timespan']
        )
    elif stageDict['type'] == 'dropout':
        stage = Dropout(
            dropoutRate=stageDict['dropoutRate'],
            initSeed=stageDict['initSeed']
        )
    elif stageDict['type'] == 'sigmoid':
        stage = Sigmoid(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=numpy.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True,
            learningRate=stageDict['learningRate']
            if stageDict.has_key('learningRate') else 0.0,
            learningRateAnnealConst=stageDict['learningRateAnnealConst']
            if stageDict.has_key('learningRateAnnealConst') else 0.0,
            momentum=stageDict['momentum']
            if stageDict.has_key('momentum') else 0.0,
            deltaMomentum=stageDict['deltaMomentum']
            if stageDict.has_key('deltaMomentum') else 0.0,
            gradientClip=stageDict['gradientClip']
            if stageDict.has_key('gradientClip') else 0.0,
            weightClip=stageDict['weightClip']
            if stageDict.has_key('weightClip') else 0.0,
            weightRegConst=stageDict['weightRegConst']
            if stageDict.has_key('weightRegConst') else 0.0,
            outputdEdX=stageDict['outputdEdX']
            if stageDict.has_key('outputdEdX') else True
        )
    elif stageDict['type'] == 'softmax':
        stage = Softmax(
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=stageDict['initSeed']
            if stageDict.has_key('initSeed') else 0,
            initRange=stageDict['initRange']
            if stageDict.has_key('initRange') else 1.0,
            initWeights=numpy.load(stageDict['initWeights'])
            if stageDict.has_key('initWeights') else 0,
            needInit=False
            if stageDict.has_key('initWeights') else True,
            learningRate=stageDict['learningRate']
            if stageDict.has_key('learningRate') else 0.0,
            learningRateAnnealConst=stageDict['learningRateAnnealConst']
            if stageDict.has_key('learningRateAnnealConst') else 0.0,
            momentum=stageDict['momentum']
            if stageDict.has_key('momentum') else 0.0,
            deltaMomentum=stageDict['deltaMomentum']
            if stageDict.has_key('deltaMomentum') else 0.0,
            gradientClip=stageDict['gradientClip']
            if stageDict.has_key('gradientClip') else 0.0,
            weightClip=stageDict['weightClip']
            if stageDict.has_key('weightClip') else 0.0,
            weightRegConst=stageDict['weightRegConst']
            if stageDict.has_key('weightRegConst') else 0.0,
            outputdEdX=stageDict['outputdEdX']
            if stageDict.has_key('outputdEdX') else True
        )
    else:
        raise 'Stage type ' + stageDict['type'] + ' not found.'

    return stage