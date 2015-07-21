from layer import *

class ActivationLayer(Layer):
    def __init__(self,
                 activationFn,
                 inputNames,
                 outputDim,
                 defaultValue=0.0,
                 outputdEdX=True,
                 name=None):
        Layer.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 defaultValue=defaultValue,
                 outputdEdX=outputdEdX)
        self.activationFn = activationFn
    def forward(self, inputValue):
        self._outputValue = self.activationFn.forward(inputValue)
        return self._outputValue
    def backward(self, gradientToOutput):
        self.dEdW = 0
        return self.activationFn.backward(gradientToOutput, self._outputValue, 0)