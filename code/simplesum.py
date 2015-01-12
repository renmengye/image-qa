import numpy as np

class SimpleSum:
    def __init__(self):
        self.W = 0
        self.X = 0
        pass
    def forwardPass(self, X):
        self.X = X
        Y = np.sum(X, axis=-1)
        Yshape = np.concatenate((Y.shape, np.ones(1)))
        Y = Y.reshape(Yshape)
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        dEdW = 0
        if outputdEdX:
            #dEdYshape = np.concatenate((dEdY.shape, np.ones(1)))
            #dEdX = dEdY.reshape(dEdYshape).repeat(self.X.shape[-1], axis=-1)
            dEdX = dEdY.repeat(self.X.shape[-1], axis=-1)
        else:
            dEdX = 0
        return dEdW, dEdX