from util_func import *
from lstm import *
from linear_dict import *
from linear_map import *
from time_unfold import *
from time_fold import *
from dropout import *
from sigmoid import *
from softmax import *
from sequential import *
from parallel import *

def routeFn(name):
    if name == 'crossEntOne':
        return crossEntOne
    elif name == 'crossEntIdx':
        return crossEntIdx
    elif name == 'hardLimit':
        return hardLimit
    elif name == 'argmax':
        return argmax
    else:
        raise 'Function ' + name + ' not found.'
    pass

stageLib = {}

def getStage(name):
    return stageLib[name]

def routeStage(stageDict):
    stage = None
    if stageDict['type'] == 'lstm':
        stage = LSTM(
            name=stageDict['name'],
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
            name=stageDict['name'],
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
    elif stageDict['type'] == 'linearMap':
        stage = LinearMap(
            name=stageDict['name'],
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
    elif stageDict['type'] == 'timeUnfold':
        stage = TimeUnfold(
            name=stageDict['name'])
    elif stageDict['type'] == 'timeFold':
        stage = TimeFold(
            name=stageDict['name'],
            timespan=stageDict['timespan']
        )
    elif stageDict['type'] == 'dropout':
        stage = Dropout(
            name=stageDict['name'],
            dropoutRate=stageDict['dropoutRate'],
            initSeed=stageDict['initSeed']
        )
    elif stageDict['type'] == 'sigmoid':
        stage = Sigmoid(
            name=stageDict['name'],
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
            name=stageDict['name'],
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
    elif stageDict['type'] == 'sequential':
        stages = stageDict['stages']
        realStages = []
        for i in range(len(stages)):
            realStages.append(stageLib[stages[i]])
        stage = Sequential(
            name=stageDict['name'],
            stages=realStages,
            outputdEdX=stageDict['outputdEdX']
            if stageDict.has_key('outputdEdX') else True
        )
    elif stageDict['type'] == 'parallel':
        stages = stageDict['stages']
        realStages = []
        for i in range(len(stages)):
            realStages.append(stageLib[stages[i]])
        stage = Parallel(
            name=stageDict['name'],
            stages=realStages,
            axis=stageDict['axis'],
            splits=stageDict['splits'],
            outputdEdX=stageDict['outputdEdX']
            if stageDict.has_key('outputdEdX') else True
        )
    else:
        raise Exception('Stage type ' + stageDict['type'] + ' not found.')

    stageLib[stageDict['name']] = stage
    return stage
