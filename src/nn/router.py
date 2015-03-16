from lstm_old import *
from lut import *
from reshape import *
from time_sum import *
from inner_prod import *
from dropout import *
from sequential import *
from const_weights import *
from cos_sim import *
from lstm import *

import pickle

def routeFn(name):
    if name == 'crossEntOne':
        return crossEntOne
    elif name == 'crossEntIdx':
        return crossEntIdx
    elif name == 'rankingLoss':
        return rankingLoss
    elif name == 'hardLimit':
        return hardLimit
    elif name == 'argmax':
        return argmax
    elif name == 'sigmoid':
        return SigmoidActiveFn
    elif name == 'softmax':
        return SoftmaxActiveFn
    elif name == 'tanh':
        return TanhActiveFn
    elif name == 'identity':
        return IdentityActiveFn
    elif name == 'mse':
        return meanSqErr
    else:
        raise Exception('Function ' + name + ' not found.')
    pass

stageLib = {}

def getStage(name):
    return stageLib[name]

def routeStage(stageDict):
    stage = None

    initSeed=stageDict['initSeed']\
    if stageDict.has_key('initSeed') else 0
    initRange=stageDict['initRange']\
    if stageDict.has_key('initRange') else 1.0
    
    if stageDict.has_key('initWeights'):
        if stageDict.has_key('sparse') and stageDict['sparse']:
            with open(stageDict['initWeights'], 'rb') as f:
                initWeights = pickle.load(f)
        else:
            initWeights = np.load(stageDict['initWeights'])
    else:
        initWeights = 0
    
    needInit=False\
    if stageDict.has_key('initWeights') else True    
    biasInitConst=stageDict['biasInitConst']\
    if stageDict.has_key('biasInitConst') else -1.0
    learningRate=stageDict['learningRate']\
    if stageDict.has_key('learningRate') else 0.0
    learningRateAnnealConst=stageDict['learningRateAnnealConst']\
    if stageDict.has_key('learningRatennealConst') else 0.0
    momentum=stageDict['momentum']\
    if stageDict.has_key('momentum') else 0.0
    deltaMomentum=stageDict['deltaMomentum']\
    if stageDict.has_key('deltaMomentum') else 0.0
    gradientClip=stageDict['gradientClip']\
    if stageDict.has_key('gradientClip') else 0.0
    weightClip=stageDict['weightClip']\
    if stageDict.has_key('weightClip') else 0.0
    weightRegConst=stageDict['weightRegConst']\
    if stageDict.has_key('weightRegConst') else 0.0
    outputdEdX=stageDict['outputdEdX']\
    if stageDict.has_key('outputdEdX') else True
    defaultValue=(np.zeros(stageDict['outputDim']) + stageDict['defaultValue'])\
    if stageDict.has_key('defaultValue') else None
    if stageDict.has_key('inputs'):
        inputList = stageDict['inputs'].split(',')
        for i in range(len(inputList)):
            inputList[i] = inputList[i].strip()
    else:
        inputList = None

    if stageDict['type'] == 'lstm_old':
        stage = LSTM_Old(
            name=stageDict['name'],
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=initSeed,
            initRange=initRange,
            initWeights=initWeights,
            needInit=needInit,
            cutOffZeroEnd=stageDict['cutOffZeroEnd'],
            multiErr=stageDict['multiErr'],
            learningRate=learningRate,
            learningRateAnnealConst=learningRateAnnealConst,
            momentum=momentum,
            deltaMomentum=deltaMomentum,
            gradientClip=gradientClip,
            weightClip=weightClip,
            weightRegConst=weightRegConst,
            outputdEdX=outputdEdX
        )
    elif stageDict['type'] == 'lstm':
        stage = LSTM(
            name=stageDict['name'],
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            timespan=stageDict['timespan'],
            defaultValue=defaultValue,
            initSeed=initSeed,
            initRange=initRange,
            initWeights=initWeights,
            needInit=needInit,
            multiOutput=stageDict['multiErr'] if stageDict.has_key('multiErr') else stageDict['multiOutput'],
            learningRate=learningRate,
            learningRateAnnealConst=learningRateAnnealConst,
            momentum=momentum,
            deltaMomentum=deltaMomentum,
            gradientClip=gradientClip,
            weightClip=weightClip,
            weightRegConst=weightRegConst,
            outputdEdX=outputdEdX
        )
    elif stageDict['type'] == 'lut':
        stage = LUT(
            name=stageDict['name'],
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            inputNames=inputList,
            initSeed=initSeed,
            initRange=initRange,
            initWeights=initWeights,
            sparse=stageDict['sparse'] if stageDict.has_key('sparse') else False,
            needInit=needInit,
            learningRate=learningRate,
            learningRateAnnealConst=learningRateAnnealConst,
            momentum=momentum,
            deltaMomentum=deltaMomentum,
            gradientClip=gradientClip,
            weightClip=weightClip,
            weightRegConst=weightRegConst
        )
    elif stageDict['type'] == 'map':
        stage = Map(
            name=stageDict['name'],
            outputDim=stageDict['outputDim'],
            inputNames=inputList,
            activeFn=routeFn(stageDict['activeFn']),
            initSeed=initSeed,
            initRange=initRange,
            initWeights=initWeights,
            needInit=needInit,
            learningRate=learningRate,
            learningRateAnnealConst=learningRateAnnealConst,
            momentum=momentum,
            deltaMomentum=deltaMomentum,
            gradientClip=gradientClip,
            weightClip=weightClip,
            weightRegConst=weightRegConst,
            outputdEdX=outputdEdX
        )
    elif stageDict['type'] == 'timeUnfold':
        stage = TimeUnfold(
            name=stageDict['name'])
    elif stageDict['type'] == 'timeSum':
        stage = TimeSum(
            name=stageDict['name'])
    elif stageDict['type'] == 'innerProd':
        stage = InnerProduct(
            name=stageDict['name'],
            inputNames=inputList,
            outputDim=stageDict['outputDim'],
            learningRate=learningRate,
            learningRateAnnealConst=learningRateAnnealConst,
            momentum=momentum,
            deltaMomentum=deltaMomentum)
    elif stageDict['type'] == 'timeFold':
        stage = TimeFold(
            name=stageDict['name'],
            timespan=stageDict['timespan']
        )
    elif stageDict['type'] == 'reshape':
        stage = Reshape(
            name=stageDict['name'],
            reshapeFn=stageDict['reshapeFn']
        )
    elif stageDict['type'] == 'dropout':
        stage = Dropout(
            name=stageDict['name'],
            dropoutRate=stageDict['dropoutRate'],
            initSeed=stageDict['initSeed']
        )
    elif stageDict['type'] == 'sequential':
        stages = stageDict['stages']
        realStages = []
        for i in range(len(stages)):
            realStages.append(stageLib[stages[i]])
        stage = Sequential(
            name=stageDict['name'],
            stages=realStages,
            outputdEdX=outputdEdX
        )
    elif stageDict['type'] == 'constWeights':
        stage = ConstWeights(
            name=stageDict['name'],
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            initSeed=initSeed,
            initRange=initRange,
            initWeights=initWeights,
            needInit=needInit,
            learningRate=learningRate,
            learningRateAnnealConst=learningRateAnnealConst,
            momentum=momentum,
            deltaMomentum=deltaMomentum,
            gradientClip=gradientClip,
            weightClip=weightClip,
            weightRegConst=weightRegConst
        )
    elif stageDict['type'] == 'cosSimilarity':
        stage = CosSimilarity(
            name=stageDict['name'],
            inputNames=inputList,
            outputDim=stageDict['outputDim'],
            bankDim=stageDict['bankDim']
        )
    elif stageDict['type'] == 'componentProd':
        stage = ElementProduct(
            name=stageDict['name'],
            inputNames=inputList,
            outputDim=stageDict['outputDim']
        )
    elif stageDict['type'] == 'selector':
        stage = Selector_Recurrent(
            name=stageDict['name'],
            inputNames=inputList,
            axis=stageDict['axis'] if stageDict.has_key('axis') else -1,
            start=stageDict['start'],
            end=stageDict['end']
        )
    elif stageDict['type'] == 'reshape':
        stage = Reshape_Recurrent(
            name=stageDict['name'],
            inputNames=inputList,
            reshapeFn=stageDict['reshapeFn']
        )
    elif stageDict['type'] == 'sum':
        stage = Sum_Recurrent(
            name=stageDict['name'],
            inputNames=inputList,
            numComponents=stageDict['numComponents'],
            outputDim=stageDict['outputDim'],
            defaultValue=defaultValue
        )
    elif stageDict['type'] == 'elemProd':
        stage = ComponentProduct_Recurrent(
            name=stageDict['name'],
            inputNames=inputList,
            outputDim=stageDict['outputDim'],
            defaultValue=defaultValue
        )
    elif stageDict['type'] == 'active':
        stage = Active_Recurrent(
            name=stageDict['name'],
            inputNames=inputList,
            outputDim=stageDict['outputDim'],
            activeFn=routeFn(stageDict['activeFn']),
            defaultValue=defaultValue
        )
    elif stageDict['type'] == 'sumProd':
        stage = SumProduct_Recurrent(
            name=stageDict['name'],
            inputNames=inputList,
            sumAxis=stageDict['sumAxis'],
            outputDim=stageDict['outputDim']
        )
    elif stageDict['type'] == 'recurrent':
        stages = stageDict['stages']
        realStages = []
        for i in range(len(stages)):
            realStages.append(stageLib[stages[i]])
        stage = Recurrent(
            name=stageDict['name'],
            inputDim=stageDict['inputDim'],
            outputDim=stageDict['outputDim'],
            inputType=stageDict['inputType'] if stageDict.has_key('inputType') else 'float',
            timespan=stageDict['timespan'],
            stages=realStages,
            multiOutput=stageDict['multiOutput'],
            outputStageName=stageDict['outputStageName'],
            outputdEdX=outputdEdX
        )
    else:
        raise Exception('Stage type ' + stageDict['type'] + ' not found.')

    stageLib[stageDict['name']] = stage
    return stage
