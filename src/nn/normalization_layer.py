from layer import *

class NormalizationLayer(Layer):
    def __init__(self,
                 outputDim,
                 mean,
                 std,
                 name=None,
                 inputNames=None,
                 outputdEdX=True):
        Layer.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 outputdEdX=outputdEdX)
        self.mean = mean
        self.std = std
        self._inputValue = 0
        self._outputValue = 0
        pass

    def forward(self, inputValue):
        return (inputValue - self.mean) / self.std

    def backward(self, gradientToOutput):
        return gradientToOutput / self.std