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
    def forward(self, X):
        self.Y = self.activationFn.forward(X)
        return self.Y
    def backward(self, dEdY):
        self.dEdW = 0
        return self.activationFn.backward(dEdY, self.Y, 0)