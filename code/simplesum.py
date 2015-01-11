import numpy as np

class SimpleSum:
    def __init__(self):
        self.W = 0
        self.X = 0
        pass
    def forwardPass(self, X):
        self.X = X
        Y = np.sum(X, axis=-1)
        Y = Y.reshape(Y.shape[0], 1)
        return Y

    def forwardPassAll(self, X):
        self.X = X
        Y = np.sum(X, axis=-1)
        Y = Y.reshape(Y.shape[0], Y.shape[1], 1)
        return Y

    def backPropagate(self, dEdY):
        dEdW = 0
        dEdX = dEdY.reshape(dEdY.shape[0], 1).repeat(self.X.shape[-1], axis=-1)
        return dEdW, dEdX