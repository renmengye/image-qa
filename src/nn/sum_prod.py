from stage import *

class SumProduct(Stage):
    def __init__(self, 
                name, 
                inputNames, 
                sumAxis,
                outputDim,
                beta=1.0):
        Stage.__init__(self,
            name=name, 
            inputNames=inputNames, 
            outputDim=outputDim)
        self.sumAxis = sumAxis
        self.beta = beta

    def getInput(self):
        # Assume that the input size is always 2
        # Rewrite get input logic into two separate arrays
        return [self.inputs[0].Y, self.inputs[1].Y]

    def sendError(self, dEdX):
        self.inputs[0].dEdY += dEdX[0]
        self.inputs[1].dEdY += dEdX[1]

    def forward(self, X):
        self.X = X
        return self.beta * np.sum(X[0] * X[1], axis=self.sumAxis)

    def backward(self, dEdY):
        # Need to generalize, but now, let's assume it's the attention model.
        dEdX = []
        dEdY = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
        dEdX.append(self.beta * np.sum(dEdY * self.X[1], axis=2))
        dEdX.append(self.beta * dEdY * self.X[0])
        return dEdX