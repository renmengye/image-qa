from layer import *

class DropoutLayer(Layer):
    def __init__(self,
                 name,
                 inputNames,
                 outputDim,
                 dropoutRate,
                 initSeed,
                 debug=False):
        Layer.__init__(self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim)
        self.dropout = True
        self.dropoutVec = 0
        self.dropoutRate = dropoutRate
        self.debug = debug
        self.random = np.random.RandomState(initSeed)
        self.seed = initSeed

    def forward(self, inputValue):
        if self.dropoutRate > 0.0 and self.isTraining:
            if self.debug:
                self.random = np.random.RandomState(self.seed)
            self.dropoutVec = (self.random.uniform(0, 1, (inputValue.shape[-1])) >
                               self.dropoutRate)
            Y = inputValue * self.dropoutVec
        else:
            Y = inputValue * (1 - self.dropoutRate)
        self._inputValue = inputValue
        return Y

    def backward(self, gradientToOutput):
        dEdX = None
        if self.outputdEdX:
            if self.isTraining:
                dEdX = gradientToOutput * self.dropoutVec
            else:
                dEdX = gradientToOutput / (1 - self.dropoutRate)
        return dEdX