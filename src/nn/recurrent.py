import copy
import time
from stage import *
from active_func import *

class RecurrentSubstage(Stage):
    def __init__(self, name,
                 inputNames,
                 outputDim,
                 defaultValue=None,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True):
        Stage.__init__(self,
                 name=name,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        if defaultValue is None:
            self.defaultValue = np.zeros((outputDim))
        else:
            self.defaultValue = defaultValue
        self.inputs = None
        self.inputNames = inputNames # Before binding with actual objects
        self.inputDim = None
        self.outputDim = outputDim
        #self.tmpError = []
        self.dEdY = 0.0
        self.dEdX = 0.0
        self.splX = None
        self.splXsize = None
        self.N = 0
        self.X = 0
        pass

    def addInput(self, stage):
        if self.inputs is None:
            self.inputs = [stage]
        else:
            self.inputs.append(stage)

    #@profile
    def getInput(self):
        # fetch input from each input stage
        # concatenate input into one vector
        if len(self.inputs) > 1:
            self.splX = []
            for stage in self.inputs:
                X = stage.Y
                self.splX.append(X)
            return np.concatenate(self.splX, axis=-1)
        else:
            return self.inputs[0].Y

    #@profile
    def sendError(self, dEdX):
        # iterate over input list and send dEdX
        if len(self.inputs) > 1:
            s = 0
            for stage in self.inputs:
                s2 = s + stage.Y.shape[-1]
                stage.dEdY += dEdX[:, s : s2]
                s = s2
        else:
            self.inputs[0].dEdY += dEdX

    def clearError(self):
        self.dEdY = 0.0

    def getOutputError(self):
        return self.dEdY
    
    #@profile
    def graphForward(self):
        X = self.getInput()
        self.forward(X)

    #@profile
    def graphBackward(self):
        dEdX = self.backward(self.dEdY)
        if self.outputdEdX:
            self.sendError(dEdX)

    def forward(self, X):
        """Subclasses need to implement this"""
        pass

    def backward(self, dEdY):
        """Subclasses need to implement this"""
        pass

    def copy(self):
        return copy.copy(self)

class Active_Recurrent(RecurrentSubstage):
    def __init__(self,
                 activeFn,
                 inputNames,
                 outputDim,
                 defaultValue=None,
                 inputDim=None,
                 outputdEdX=True,
                 name=None):
        RecurrentSubstage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 defaultValue=defaultValue,
                 outputdEdX=outputdEdX)
        self.activeFn = activeFn
    def forward(self, X):
        self.Y = self.activeFn.forward(X)
        return self.Y
    def backward(self, dEdY):
        self.dEdW = 0
        return self.activeFn.backward(dEdY, self.Y, 0)

class Map_Recurrent(RecurrentSubstage):
    def __init__(self,
                 outputDim,
                 activeFn,
                 inputNames,
                 initRange=1.0,
                 biasInitConst=-1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
                 defaultValue=None,
                 name=None):
        RecurrentSubstage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 defaultValue=defaultValue,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        self.activeFn = activeFn
        self.random = np.random.RandomState(initSeed)
        if not needInit:
            self.W = initWeights
        else:
            self.W = None
        self.initRange = initRange
        self.biasInitConst = biasInitConst
        self.X = 0
        self.Y = 0
        pass

    def initWeights(self):
        if self.biasInitConst >= 0.0:
            self.W = np.concatenate((self.random.uniform(
                -self.initRange/2.0, self.initRange/2.0, (self.outputDim, self.inputDim)),
                np.ones((self.outputDim, 1)) * self.biasInitConst), axis=-1)
        else:
            self.W = self.random.uniform(
                -self.initRange/2.0, self.initRange/2.0, (self.outputDim, self.inputDim + 1))
    
    #@profile
    def forward(self, X):
        if self.inputDim is None: self.inputDim = X.shape[-1]
        if self.W is None: self.initWeights()
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        Z = np.inner(self.X, self.W)
        self.Y = self.activeFn.forward(Z)
        return self.Y

    def backward(self, dEdY):
        dEdZ = self.activeFn.backward(dEdY, self.Y, 0)
        self.dEdW = np.dot(dEdZ.transpose(), self.X)
        dEdX = np.dot(dEdZ, self.W[:, :-1])
        #self.dEdX = dEdX
        return dEdX if self.outputdEdX else None


class Selector_Recurrent(RecurrentSubstage):
    def __init__(self, 
                 name, 
                 inputNames,
                 start, 
                 end, 
                 axis=-1):
        RecurrentSubstage.__init__(
                 self,
                 name=name, 
                 inputNames=inputNames,
                 outputDim=end-start)
        self.start = start
        self.end = end
        self.axis = axis

    def forward(self, X):
        self.X = X
        if self.axis == 0:
            self.Y = X[self.start:self.end]
        else:
            self.Y = X[:, self.start:self.end]
        # print self.name
        # print self.X.shape
        # print self.Y.shape
        return self.Y

    def backward(self, dEdY):
        dEdX = np.zeros(self.X.shape)
        if self.axis == 0:
            dEdX[self.start:self.end] = dEdY
        else:
            dEdX[:, self.start:self.end] = dEdY
        return dEdX

class Input_Recurrent(RecurrentSubstage):
    def __init__(self, name, outputDim):
        RecurrentSubstage.__init__(self, name=name, inputNames=[], outputDim=outputDim)

    def setValue(self, value):
        self.Y = value

    def getInput(self):
        return self.X

    def sendError(self, dEdX):
        pass

    def forward(self, X):
        return X

    def backward(self, dEdY):
        return dEdY

class Output_Recurrent(RecurrentSubstage):
    def __init__(self, name, outputDim=0):
        RecurrentSubstage.__init__(self, name=name, inputNames=[], outputDim=outputDim)
    def graphForward(self):
        self.Y = self.getInput()
        self.dEdX = np.zeros(self.Y.shape)
    def graphBackward(self):
        pass

class Constant_Recurrent(RecurrentSubstage):
    """Stage emitting constant value for all samples."""
    def __init__(self, name, outputDim, value):
        RecurrentSubstage.__init__(self, name=name, inputNames=[], outputDim=outputDim)
        self.value = np.reshape(value, (1, value.size))
    def forward(self, X):
        return np.tile(self.value, (X.shape[0], 1))
    def graphForward(self):
        self.Y = self.forward(self.X)

class ComponentProduct_Recurrent(RecurrentSubstage):
    """Stage multiplying first half of the input with second half"""
    def __init__(self, name, inputNames, outputDim,
                 defaultValue=None):
        RecurrentSubstage.__init__(
            self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim,
            defaultValue=defaultValue)
    def forward(self, X):
        self.X = X
        self.Y = X[:,:X.shape[1]/2] * X[:,X.shape[1]/2:]
        return self.Y
    def backward(self, dEdY):
        self.dEdW = 0.0
        return np.concatenate((self.X[:,self.X.shape[1]/2:] * dEdY, self.X[:,:self.X.shape[1]/2] * dEdY), axis=-1)

class Sum_Recurrent(RecurrentSubstage):
    """Stage summing first hald of the input with second half."""
    def __init__(self, name, inputNames, numComponents, outputDim,
                 defaultValue=None):
        RecurrentSubstage.__init__(
            self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim,
            defaultValue=defaultValue)
        self.numComponents = numComponents
    def forward(self, X):
        self.Y = np.sum(X.reshape(X.shape[0], self.numComponents, X.shape[1]/self.numComponents), axis=1)
        return self.Y
    def backward(self, dEdY):
        self.dEdW = 0.0
        return np.tile(dEdY, 2)

class LUT_Recurrent(RecurrentSubstage):
    def __init__(self,
                 inputNames,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 sparse=False,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 name=None):
        RecurrentSubstage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 learningRate=learningRate,
                 outputDim=outputDim,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=False)
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.initRange = initRange
        self.random = np.random.RandomState(initSeed)
        self.needInit = needInit

        # Zeroth dimension of the weight matrix is reserved
        # for empty word at the end of a sentence.
        if needInit:
            self.W = None
        else:
            if sparse:
                initWeights = np.array(initWeights.todense())
                self.W = np.concatenate(
                    (np.zeros((1, outputDim)), initWeights), axis=0)
            else:
                self.W = np.concatenate(
                    (np.zeros((1, outputDim)), initWeights), axis=0)
        self.X = 0
        self.Y = 0
        self.sparse = sparse
        pass

    def initWeights(self):
        self.W = np.concatenate(
            (np.zeros((1, self.outputDim)),
             self.random.uniform(
            -self.initRange/2.0, self.initRange/2.0,
            (self.inputDim, self.outputDim))), axis=0)
    
    #@profile
    def forward(self, X):
        if self.W is None: self.initWeights()
        X = X.reshape(X.size)
        Y = np.zeros((X.shape[0], self.outputDim))
        for n in range(0, X.shape[0]):
             Y[n] = self.W[X[n]]
        self.X = X
        self.Y = Y
        return Y

    def backward(self, dEdY):
        X = self.X
        if self.learningRate > 0.0:
            self.dEdW = np.zeros(self.W.shape)
            for n in range(0, X.shape[0]):
                self.dEdW[X[n]] += dEdY[n]
        return None

    def loadWeights(self, W):
        if self.learningRate == 0.0:
            return
        else:
            Stage.loadWeights(self, W)

    def getWeights(self):
        if self.learningRate == 0.0:
            return 0
        else:
            return W

class Reshape_Recurrent(RecurrentSubstage):
    def __init__(self, name, inputNames, reshapeFn):
        # Please don't put recurrent connection here.
        RecurrentSubstage.__init__(self, name=name, inputNames=inputNames, outputDim=0)
        self.reshapeFn = eval('lambda x: ' + reshapeFn)
        self.Xshape = 0

    def forward(self, X):
        self.Xshape = X.shape
        # print X.shape
        # print self.reshapeFn(X.shape)
        self.Y = np.reshape(X, self.reshapeFn(X.shape))
        return self.Y

    def backward(self, dEdY):
        # print self.Xshape
        # print dEdY.shape
        return np.reshape(dEdY, self.Xshape)

class SumProduct_Recurrent(RecurrentSubstage):
    def __init__(self, name, inputNames, sumAxis, outputDim):
        RecurrentSubstage.__init__(self, name=name, inputNames=inputNames, outputDim=outputDim)
        self.sumAxis = sumAxis

    def getInput(self):
        # Assume that the input size is always 2
        # Rewrite get input logic into two separate arrays
        return [self.inputs[0].Y, self.inputs[1].Y]

    def sendError(self, dEdX):
        self.inputs[0].dEdY += dEdX[0]
        self.inputs[1].dEdY += dEdX[1]
        #self.inputs[0].tmpError.append(dEdX[0])
        #self.inputs[1].tmpError.append(dEdX[1])

    def forward(self, X):
        self.X = X
        self.Y = np.sum(X[0] * X[1], axis=self.sumAxis)
        return self.Y

    def backward(self, dEdY):
        # Need to generalize, but now, let's assume it's the attention model.
        dEdX = []
        dEdY = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
        dEdX.append(np.sum(dEdY * self.X[1], axis=2))
        dEdX.append(dEdY * self.X[0])
        #self.dEdX = dEdX
        return dEdX

class Dropout_Recurrent(RecurrentSubstage):
    def __init__(self, 
                 name, 
                 inputNames,
                 outputDim, 
                 dropoutRate, 
                 initSeed, 
                 debug=False):
        RecurrentSubstage.__init__(self, 
            name=name, 
            inputNames=inputNames,
            outputDim=outputDim)
        self.dropout = True
        self.dropoutVec = 0
        self.dropoutRate = dropoutRate
        self.debug = debug
        self.random = np.random.RandomState(initSeed)
        self.seed = initSeed

    def forward(self, X):
        if self.dropoutRate > 0.0 and self.dropout:
            if self.debug:
                self.random = np.random.RandomState(self.seed)
            self.dropoutVec = (self.random.uniform(0, 1, (X.shape[-1])) >
                               self.dropoutRate)
            Y = X * self.dropoutVec
        else:
            Y = X * (1 - self.dropoutRate)
        self.X = X
        self.Y = Y
        return Y

    def backward(self, dEdY):
        dEdX = None
        if self.outputdEdX:
            if self.dropout:
                dEdX = dEdY * self.dropoutVec
            else:
                dEdX = dEdY / (1 - self.dropoutRate)
        return dEdX

class Recurrent(Stage):
    """
    Recurrent stage.
    Propagate through time.
    """
    def __init__(self,
                 stages,
                 timespan,
                 outputStageName,
                 inputDim,
                 outputDim,
                 inputType='float',
                 multiOutput=True,
                 name=None,
                 outputdEdX=True):
        Stage.__init__(self, name=name, outputdEdX=outputdEdX)
        self.stages = []
        self.stageDict = {}
        self.constStages = []
        self.timespan = timespan
        self.multiOutput = multiOutput
        self.inputDim = inputDim
        self.inputType = inputType
        self.outputDim = outputDim
        self.outputStageName = outputStageName
        self.Xend = 0
        self.XendAll = 0
        self.X = 0
        for t in range(timespan):
            self.stages.append([])
            inputStage = Input_Recurrent(name='input', outputDim=self.inputDim)
            self.stages[t].append(inputStage)
            self.stageDict[('input-%d' % t)] = inputStage

        for stage in stages:
            self.register(stage)

        self.link()
        self.dEdW = []
        for stage in self.stages[0]:
            self.dEdW.append(0.0)
        self.testRun()

    def register(self, stage):
        """
        Register a substage
        :param stage: new recurrent substage
        :return:
        """
        for t in range(self.timespan):
            if t == 0:
                stageNew = stage
            else:
                stageNew = stage.copy()
            stageNew.used = False
            self.stages[t].append(stageNew)
            self.stageDict[('%s-%d' % (stage.name, t))] = stageNew

    def link(self):
        """
        Link substages with their input strings
        :return:
        """
        for t in range(self.timespan):
            outputStage = Output_Recurrent(name='output')
            self.stages[t].append(outputStage)
            self.stageDict[('output-%d' % t)] = outputStage
            outputStage.used = True
            outputStageInput = self.stageDict[('%s-%d' % (self.outputStageName, t))]
            outputStageInput.used = True
            outputStage.addInput(outputStageInput)
            for stage in self.stages[t]:
                for inputStageStr in stage.inputNames:
                    if '(' in inputStageStr:
                        stageName = inputStageStr[:inputStageStr.index('(')]
                        stageTimeStr = \
                            inputStageStr[inputStageStr.index('(') + 1 : inputStageStr.index(')')]
                        if stageTimeStr[0] == '$':
                            stageTime = int(stageTimeStr[1:]) - t
                        else:
                            stageTime = int(stageTimeStr)
                    else:
                        stageName = inputStageStr
                        stageTime = 0
                    if stageTime > 0:
                        raise Exception('Recurrent model definition is non-causal.')
                    # stageNameTime = '%s-%d' % (stageName, stageTime)
                    if t + stageTime < 0:
                        stageInput = Constant_Recurrent(
                            name=('%s-%s-%d'%('const', stageName, t)),
                            outputDim=self.stageDict[stageName + '-0'].outputDim,
                            value=self.stageDict[('%s-%d') % (stageName, 0)].defaultValue)
                        self.constStages.append(stageInput)
                    else:
                        stageInput = self.stageDict[('%s-%d' % (stageName, t + stageTime))]
                        stageInput.used = True
                    stage.addInput(stageInput)

    def testRun(self):
        """Test run through the recurrent net to initialize all the weights."""
        if self.inputType == 'float':
            X = np.random.rand(2, self.timespan, self.inputDim)
        elif self.inputType == 'int':
            X = np.round(np.random.rand(2, self.timespan, self.inputDim) * 5).astype(int)
        
        self.forward(X)
        for t in range(1, self.timespan):
            for s in range(1, len(self.stages[0]) - 1):
                self.stages[t][s].W = self.stages[0][s].W
        for t in range(self.timespan):
            for s in range(len(self.stages[0])):
                self.stages[t][s].X = 0
                self.stages[t][s].Y = 0

    #@profile
    def forward(self, X, dropout=True):
        # print 'recurrent'
        # print X.shape
        N = X.shape[0]
        self.Xend = np.zeros(N, dtype=int) + X.shape[1]
        reachedEnd = np.sum(X, axis=-1) == 0.0
        if self.multiOutput:
            Y = np.zeros((N, self.timespan, self.outputDim))
        else:
            Y = np.zeros((N, self.outputDim))
        for n in range(N):
            for t in range(X.shape[1]):
                if reachedEnd[n, t]:
                    self.Xend[n] = t
                    break
        self.XendAll = np.max(self.Xend)
        for s in self.constStages:
            s.Y = np.tile(s.value, (X.shape[0], 1))
        for t in range(self.XendAll):
            self.stages[t][0].Y = X[:, t, :]
            for s in range(1, len(self.stages[t])):
                if self.stages[t][s].used:
                    if hasattr(self.stages[t][s], 'dropout'):
                        self.stages[t][s].dropout = dropout
                    self.stages[t][s].graphForward()
        if self.multiOutput:
            for n in range(N):
                if self.Xend[n] > 0:
                    for t in range(self.Xend[n]):
                        Y[n, t, :] = self.stages[t][-1].Y[n]
        else:
            for n in range(N):
                if self.Xend[n] > 0:
                    Y[n, :] = self.stages[self.Xend[n] - 1][-1].Y[n]
        self.Y = Y
        self.X = X
        return Y

    #@profile
    def backward(self, dEdY):
        N = self.X.shape[0]
        dEdX = np.zeros(self.X.shape)
        if self.outputdEdX:
            dEdX = np.zeros(self.X.shape)

        if self.multiOutput:
            for t in range(self.XendAll):
                self.stages[t][-1].sendError(dEdY[:, t, :])
        else:
            for n in range(N):
                if self.Xend[n] > 0:
                    err = np.zeros(dEdY.shape)
                    err[n] = dEdY[n]
                    self.stages[self.Xend[n] - 1][-1].sendError(err)

        for t in reversed(range(self.XendAll)):
            for s in reversed(range(0, len(self.stages[0]) - 1)):
                if self.stages[t][s].used:
                    self.stages[t][s].graphBackward()

        # Collect input error
        if self.outputdEdX:
            for t in range(self.XendAll):
                dEdX[:, t, :] = self.stages[t][0].dEdY

        # Clear error and ready for next batch
        for t in range(self.timespan):
            for stage in self.stages[t]:
                stage.dEdY = 0.0
                #stage.tmpError = []
                #stage.dEdX = 0.0
        for stage in self.constStages:
            stage.dEdY = 0.0
            #stage.dEdX = 0.0
        
        # Sum error through time
        for s in range(1, len(self.stages[0]) - 1):
            if type(self.stages[0][s].W) is np.ndarray and self.stages[0][s].learningRate > 0.0:
                tmp = np.zeros((self.timespan, self.stages[0][s].W.shape[0], self.stages[0][s].W.shape[1]))
                for t in range(self.timespan):
                    tmp[t] = self.stages[t][s].getGradient()
                    self.stages[t][s].dEdW = 0.0
                self.dEdW[s] = np.sum(tmp, axis=0)

        # For gradient check purpose, synchronize the sum of gradient to the time=0 stage
        for s in range(1, len(self.stages[0]) - 1):
            if self.stages[0][s].learningRate > 0.0:
                self.stages[0][s].dEdW = self.dEdW[s]
        self.dEdX = dEdX
        return dEdX if self.outputdEdX else None

    def updateWeights(self):
        for s in range(1, len(self.stages[0])-1):
            # Because all stages are "shallow copied", the weights are shared.
            self.stages[0][s].updateWeights()
        for t in range(1, self.timespan):
            for s in range(1, len(self.stages[0])-1):
                if self.stages[t][s].learningRate > 0.0:
                    self.stages[t][s].W = self.stages[0][s].W

    def updateLearningParams(self, numEpoch):
        for s in range(1, len(self.stages[0])-1):
            # Since only the first stage updates the weights,
            # learning params just need to update in the first stage.
            self.stages[0][s].updateLearningParams(numEpoch)

    def getWeights(self):
        weights = []
        for s in range(1, len(self.stages[0])-1):
            weights.append(self.stages[0][s].getWeights())
        return np.array(weights, dtype=object)

    def loadWeights(self, W):
        i = 0
        for s in range(1, len(self.stages[0])-1):
            self.stages[0][s].loadWeights(W[i])
            i += 1
        for t in range(self.timespan):
            for s in range(1, len(self.stages[0])-1):
                self.stages[t][s].W = self.stages[0][s].W
