from stage2 import *
from active_func import *
from map import *

class Input(GraphStage):
    def __init__(self, name, outputDim):
        GraphStage.__init__(self, 
            name=name, 
            inputNames=[],
            outputDim=outputDim)
    def setValue(self, value):
        self.Y = value

class Output(GraphStage):
    def __init__(self, name, inputNames):
        GraphStage.__init__(self, 
            name=name, 
            inputNames=inputNames, 
            outputDim=0)
    def graphForward(self):
        self.Y = self.getInput()

class GraphContainer(GraphStage):
    def __init__(self,
                 stages,
                 outputStageNames,
                 inputDim,
                 outputDim,
                 name=None,
                 inputNames=None,
                 inputType='float',
                 outputdEdX=True):
        GraphStage.__init__(self, 
                name=name, 
                inputNames=inputNames, 
                outputDim=outputDim, 
                outputdEdX=outputdEdX)
        self.stages = []
        self.stageDict = {}
        self.inputDim = inputDim
        self.inputType = inputType
        self.outputStageNames = outputStageNames

        inputStage = self.createInputStage()
        self.stages.append(inputStage)
        self.stageDict['input'] = inputStage

        for stage in stages:
            self.register(stage)
        
        outputStage = self.createOutputStage()
        self.stages.append(outputStage)
        self.stageDict['output'] = outputStage

        self.link()
        self.dEdW = []
        for stage in self.stages:
            self.dEdW.append(0.0)
        self.testRun()

    def createInputStage(self):
        return Input(name='input', outputDim=self.inputDim)

    def createOutputStage(self):
        return Output(name='output', inputNames=self.outputStageNames)

    def register(self, stage):
        """
        Register a substage
        :param stage: new recurrent substage
        :return:
        """
        stage.used = False
        self.stages.append(stage)
        self.stageDict[stage.name] = stage

    def link(self):
        """
        Link substages with their input strings
        :return:
        """
        for stage in self.stages:
            for stageName in stage.inputNames:
                stageInput = self.stageDict[stageName]
                stageInput.used = True
                stage.addInput(stageInput)

    def testRun(self):
        """Test run through the recurrent net to initialize all the weights."""
        if self.inputType == 'float':
            X = np.random.rand(2, self.inputDim)
        elif self.inputType == 'int':
            X = np.round(np.random.rand(2, self.inputDim) * 5).astype(int)
        self.forward(X)

    #@profile
    def forward(self, X, dropout=True):
        self.stages[0].Y = X
        for s in range(1, len(self.stages) - 1):
            if self.stages[s].used:
                if hasattr(self.stages[s], 'dropout'):
                    self.stages[s].dropout = dropout
                self.stages[s].graphForward()
        self.stages[-1].graphForward()
        Y = self.stages[-1].Y

        self.X = X
        return Y

    #@profile
    def backward(self, dEdY):
        self.stages[-1].sendError(dEdY)
        for s in reversed(range(0, len(self.stages) - 1)):
            if self.stages[s].used:
                self.stages[s].graphBackward()

        # Collect input error
        if self.outputdEdX:
            dEdX = self.stages[0].dEdY

        # Clear error and ready for next batch
        for stage in self.stages:
            stage.dEdY = 0.0

        return dEdX if self.outputdEdX else None

    def updateWeights(self):
        for s in range(1, len(self.stages)-1):
            # Because all stages are "shallow copied", the weights are shared.
            self.stages[s].updateWeights()

    def updateLearningParams(self, numEpoch):
        for s in range(1, len(self.stages)-1):
            # Since only the first stage updates the weights,
            # learning params just need to update in the first stage.
            self.stages[s].updateLearningParams(numEpoch)

    def setGradient(self, value):
        if type(value) is float:
            for s in range(1, len(self.stages) - 1):
                self.stages[s].setGradient(value)
        elif type(value) is np.ndarray:
            for s in range(1, len(self.stages) - 1):
                self.stages[s].setGradient(value[s - 1])
        else:
            raise Exception('Unknown type %s for setGradient' % type(value))

    def getWeights(self):
        weights = []
        for s in range(1, len(self.stages)-1):
            weights.append(self.stages[s].getWeights())
        return np.array(weights, dtype=object)

    def loadWeights(self, W):
        for s in range(1, len(self.stages) - 1):
            self.stages[s].loadWeights(W[s - 1])

if __name__ == '__main__':
    m1 = Map(name='hid1', 
            inputNames=['input'], 
            outputDim=50, 
            activeFn=SigmoidActiveFn,
            initRange=0.1,
            initSeed=1)
    m2 = Map(name='hid2',
            inputNames=['hid1'],
            outputDim=20,
            activeFn=SigmoidActiveFn,
            initRange=0.1,
            initSeed=2)
    container = GraphContainer(name='container',
            inputNames=[],
            inputDim=100,
            outputDim=20,
            outputStageNames=['hid2'],
            stages=[m1, m2]
        )
    random = np.random.RandomState(2)
    x = random.uniform(-0.1, 0.1, (2, 100))
    y1 = container.forward(x)

    # import map
    # import sequential
    # m12 = map.Map(name='hid1',
    #         inputDim=100,
    #         outputDim=50, 
    #         activeFn=SigmoidActiveFn,
    #         initRange=0.1,
    #         initSeed=1)
    # m22 = map.Map(name='hid2',
    #         inputDim=50,
    #         outputDim=20, 
    #         activeFn=SigmoidActiveFn,
    #         initRange=0.1,
    #         initSeed=2)
    # sequential = sequential.Sequential(stages=[m12, m22])
    # y2 = sequential.forward(x)
    # print y1/y2