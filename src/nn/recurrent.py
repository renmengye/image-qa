import copy
from stage import *
from active_func import *

class RecurrentSubstage(Stage):
    def __init__(self, name,
                 inputsStr,
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
        self.inputs = None
        self.inputsStr = inputsStr # Before binding with actual objects
        self.dEdY = 0.0
        self.dEdX = 0.0
        pass

    def addInput(self, stage):
        if self.inputs is None:
            self.inputs = [stage]
        else:
            self.inputs.append(stage)

    def getInput(self):
        # fetch input from each input stage
        # concatenate input into one vector
        splX = []
        for stage in self.inputs:
            splX.append(stage.getValue())
        return np.concatenate(splX, axis=-1)

    def receiveError(self, error):
        self.dEdY += error

    def sendError(self):
        # iterate over input list and send dEdX
        if not self.outputdEdX:
            raise Exception('No input error avaialble. Set outputdEdX to be True.')
        for stage in self.inputs:
            stage.receiveError(self.dEdX)

    def getError(self):
        return self.dEdX

    def graphForward(self):
        self.forward(self.getInput())

    def graphBackward(self):
        self.dEdX = self.backward(self.dEdY)
        self.sendError()

    def forward(self, X):
        """Subclasses need to implement this"""
        pass

    def backward(self, dEdY):
        """Subclasses need to implement this"""
        pass

    def updateWeights(self):
        """Update weights is prohibited here"""
        raise Exception('Weights update not allowed in recurrent substages.')
        pass

    def updateWeightsSync(self, dEdW):
        self._updateWeights(dEdW)

class Map_Recurrent(RecurrentSubstage):
    def __init__(self,
                 inputDim,
                 outputDim,
                 activeFn,
                 inputsStr,
                 initRange=1.0,
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
                 name=None):
        RecurrentSubstage.__init__(self,
                 name=name,
                 inputsStr=inputsStr,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.activeFn = activeFn
        self.random = np.random.RandomState(initSeed)

        if needInit:
            self.W = self.random.uniform(
                -initRange/2.0, initRange/2.0, (outputDim, inputDim + 1))
        else:
            self.W = initWeights
        self.X = 0
        self.Y = 0
        self.Z = 0
        pass

    def forward(self, X):
        X2 = np.concatenate((X, np.ones(1)), axis=-1)
        Z = np.inner(X2, self.W)
        Y = self.activeFn.forward(Z)
        self.X = X2
        self.Z = Z
        self.Y = Y
        return Y

    def backward(self, dEdY):
        Y = self.Y
        Z = self.Z
        X = self.X
        dEdZ = self.activeFn.backward(dEdY, Y, Z)
        self.dEdW = np.dot(dEdZ.transpose(), X)
        dEdX = np.dot(dEdZ, self.W[:, :-1])
        return dEdX if self.outputdEdX else None

class Input_Recurrent(RecurrentSubstage):
    # Just forward X = X
    # All the recurrent stages need to be symbolic.
    # For now, we can assume that the order of the stages listed in the model follows the dependency
    # so we don't have to worry about the dependency.
    def __init__(self, name):
        RecurrentSubstage.__init__(self, name=name, inputsStr=[])
    def setValue(self, value):
        self.Y = value

class Output_Recurrent(RecurrentSubstage):
    def __init__(self, name):
        RecurrentSubstage.__init__(self, name=name, inputsStr=[])
    def forward(self, X):
        self.Y = X
        return self.Y

class Zero_Recurrent(RecurrentSubstage):
    def __init__(self, outputDim, name=None):
        self.outputDim = outputDim
        RecurrentSubstage.__init__(self, name=name, inputsStr=[])
    def getValue(self):
        return np.zeros(self.outputDim)

class Recurrent(Stage):
    """
    Recurrent container.
    Propagate through time.
    """
    def __init__(self, stages, timespan, outputStageName, multiOutput=True, name=None, outputdEdX=True):
        Stage.__init__(self, name=name, outputdEdX=outputdEdX)
        self.stages = []
        self.timespan = timespan
        self.multiOutput = multiOutput
        self.Xend = 0
        self.X = 0
        for t in range(timespan):
            self.stages.append({})
            inputStage = Input_Recurrent(name=('input-%d' % t))
            self.stages[t]['input'] = inputStage
            outputStage = Output_Recurrent(name=('output-%d' % t))
            self.stages[t]['output'] = outputStage
            for stage in stages:
                stageNew = copy.copy(stage)
                stageNew.name = '%s-%d' % (stage.name, t)
                self.stages[t][stage.name] = stageNew
            outputStage.addInput(self.stages[t][outputStageName])
        for t in range(timespan):
            for k in self.stages[t].keys():
                for inputStageStr in self.stages[t][k].inputsStr:
                    stageName = inputStageStr[:inputStageStr.index('(')]
                    stageTime = int(
                        inputStageStr[inputStageStr.index('(') + 1 : inputStageStr.index(')')])
                    if stageTime > 0:
                        raise Exception('Recurrent model definition is non-causal')
                    # stageNameTime = '%s-%d' % (stageName, stageTime)
                    if t + stageTime < 0:
                        stage = Zero_Recurrent(self.stages[0][stageName].outputDim)
                    else:
                        stage = self.stages[t + stageTime][stageName]
                    self.stages[t][k].addInput(stage)

    def forward(self, X):
        Y = None
        N = X.shape[0]
        self.Xend = np.zeros(N) + X.shape[1]
        for n in range(N):
            for t in range(X.shape[1]):
                if np.sum(X[n, t, :]) == 0.0:
                    self.Xend[n] = t
                    break
                self.stages[t]['input'].setValue(X[n, t, :])
                for stageName in self.stages[t].keys():
                    if stageName != 'input':
                        stage = self.stages[t][stageName]
                        stage.graphForward()
                if self.multiOutput:
                    Yt = self.stages[t]['output'].getValue()
                    if Y is None:
                        Y = np.zeros((N, self.timespan, Yt.shape[-1]))
                    Y[n, t, :] = Yt
            if not self.multiOutput:
                Yfinal = self.stages[self.Xend[n] - 1]['output']
                if Y is None:
                    Y = np.zeros((N, Yfinal.shape[1]))
                Y[n, :] = Yfinal
        self.Y = Y
        self.X = X
        return Y

    def backward(self, dEdY):
        self.dEdW = {}
        for stageName in self.stages[0].keys():
            if stageName != 'input' or 'output':
                self.dEdW[stageName] = 0.0
        if self.outputdEdX:
            dEdX = np.zeros(X.shape)
        for n in range(self.X.shape[0]):
            if not self.multiOutput:
                self.stages[self.Xend[n] - 1]['output'].receiveError(dEdY)
            for t in reversed(range(self.Xend[n])):
                if self.multiOutput:
                    self.stages[t]['output'].receiveError(dEdY)
                for stageName in reversed(self.stages[t].keys()):
                    stage = self.stages[t][stageName]
                    stage.graphBackward()
                    self.dEdW[stageName] += stage.getGradient()
            if self.outputdEdX:
                for t in range(self.Xend[n]):
                    dEdX[n, t] = self.stages[t]['input'].getError()
        return dEdX if self.outputdEdX else None

    def updateWeights(self):
        for stageName in self.stages[0].keys():
            if stageName != 'input' or 'output':
                # Because all stages are "shallow copied", the weights are shared.
                self.stages[0][stageName].updateWeightsSync(self.dEdW)

    def updateLearningParams(self, numEpoch):
        for stageName in self.stages[0].keys():
            if stageName != 'input' or 'output':
                # Since only the first stage updates the weights,
                # learning params just need to update in the first stage.
                self.stages[0][stageName].updateLearningParams()

    def getWeights(self):
        weights = []
        for stageName in self.stages[0].keys():
            weights.append(self.stages[0][stageName].getWeights())
        return np.array(weights, dtype=object)

    def loadWeights(self, W):
        i = 0
        for stageName in self.stages[0].keys():
            self.stages[0][stageName].loadWeights(W[i])
            i += 1

if __name__ == '__main__':
    sigm = Map_Recurrent(
            name='sigm',
            inputsStr=['input(0)','sigm(-1)'],
            inputDim=7,
            outputDim=2,
            activeFn=SigmoidActiveFn,
            initRange=1,
            initSeed=5
        )

    recurrent = Recurrent(
        stages=[sigm], 
        timespan=8,
        outputStageName='sigm',
        multiOutput=True,
        name='container', 
        outputdEdX=True)

    X = np.random.rand(10, 8, 5)
    Y = recurrent.forward(X)

    Y2 = np.zeros(Y.shape)
    for n in range(10):
        for t in range(8):
            Y2[n, t, :] = sigm.forward(np.concatenate((X[n, t, :], Y2[n, t-1, :])))

    print Y/Y2