from layer import *

class ElementWiseProduct(Layer):
    def __init__(self, name, inputNames, outputDim,
                 defaultValue=0.0):
        Layer.__init__(
            self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim,
            defaultValue=defaultValue)
    def forward(self, inputValue):
        # self.X = X
        # return X[:,:X.shape[1]/2] * X[:,X.shape[1]/2:]
        self._inputValue = inputValue
        return inputValue[0] * inputValue[1]

    def backward(self, gradientToOutput):
        self.dEdW = 0.0
        return [self._inputValue[1] * gradientToOutput, self._inputValue[0] * gradientToOutput]
        # return np.concatenate(
        #     (self.X[:,self.X.shape[1]/2:] * dEdY,
        #     self.X[:,:self.X.shape[1]/2] * dEdY),
        #     axis=-1)

class ElementWiseSum(Layer):
    def __init__(self, name, inputNames, outputDim,
                 defaultValue=0.0):
        Layer.__init__(
            self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim,
            defaultValue=defaultValue)

    def forward(self, inputValue):
        return inputValue[0] + inputValue[1]

    def backward(self, gradientToOutput):
        self.dEdW = 0.0
        return [gradientToOutput, gradientToOutput]