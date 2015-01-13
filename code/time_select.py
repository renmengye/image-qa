import numpy as np

class TimeSelect:
    def __init__(self, time):
        self.t = time
        self.W = 0
        self.X = 0
        self.Y = 0
        pass
    def forwardPass(self, X):
        if len(X.shape) == 3:
            return self.forwardPassAll(X)
        Y = X[self.t, :]
        self.X = X
        self.Y = Y
        return Y

    def forwardPassAll(self, X):
        # X(t, n, i)
        Y = X[self.t, :, :]
        self.X = X
        self.Y = Y
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        if len(self.X.shape) == 3:
            return self.backPropagateAll(dEdY, outputdEdX)
        dEdW = 0
        if outputdEdX:
            dEdX = np.zeros(self.X.shape)
            dEdX[self.t, :] = dEdY
        return dEdW, dEdX

    def backPropagateAll(self, dEdY, outputdEdX=True):
        dEdW = 0
        if outputdEdX:
            dEdX = np.zeros(self.X.shape)
            dEdX[self.t, :, :] = dEdY
        return dEdW, dEdX
