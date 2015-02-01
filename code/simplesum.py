from stage import *

class SimpleSum(Stage):
    def __init__(self, name=None):
        Stage.__init__(self, name=name)
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
        self.dEdW = 0
        return dEdY.repeat(self.X.shape[-1], axis=-1) if outputdEdX else None