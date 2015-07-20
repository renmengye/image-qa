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
    def forward(self, X):
        # self.X = X
        # return X[:,:X.shape[1]/2] * X[:,X.shape[1]/2:]
        self.X = X
        return X[0] * X[1]

    def backward(self, dEdY):
        self.dEdW = 0.0
        return [self.X[1] * dEdY, self.X[0] * dEdY]
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

    def forward(self, X):
        return X[0] + X[1]

    def backward(self, dEdY):
        self.dEdW = 0.0
        return [dEdY, dEdY]