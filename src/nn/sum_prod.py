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
        if len(self.inputs) == 2:
            return [self.inputs[0].Y, self.inputs[1].Y]
        elif len(self.inputs) == 3:
            return [self.inputs[0].Y, self.inputs[1].Y, self.inputs[2].Y]

    def sendError(self, dEdX):
        self.inputs[0].dEdY += dEdX[0]
        self.inputs[1].dEdY += dEdX[1]
        if len(self.inputs) == 3:
            self.inputs[2].dEdY += dEdX[2]

    def forward(self, X):
        self.X = X
        if len(X) == 2:
            return self.beta * np.sum(X[0] * X[1], axis=self.sumAxis)
        elif len(X) == 3:
            self.Z = np.sum(X[0] * X[1], axis=self.sumAxis)
            return self.X[2] * self.Z

    def backward(self, dEdY):
        # Need to generalize, but now, let's assume it's the attention model.
        dEdX = []

        if len(self.X) == 2:
            dEdY = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
            dEdX.append(self.beta * np.sum(dEdY * self.X[1], axis=2))
            dEdX.append(self.beta * dEdY * self.X[0])
        elif len(self.X) == 3:
            dEdY2 = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
            dEdX.append(self.X[2] * np.sum(dEdY2 * self.X[1], axis=2))
            dEdX.append(self.X[2].reshape(self.X[2].shape[0], 1, 1) * dEdY2 * self.X[0])
            dEdX.append(np.sum(dEdY * self.Z, axis=-1).reshape(self.X[2].shape[0], 1))

        return dEdX