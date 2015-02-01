from stage import *

class TimeUnfold(Stage):
    def __init__(self, name=None):
        Stage.__init__(self, name=name)
        self.Xshape = 0
        pass

    def forwardPass(self, X):
        Y = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        self.Xshape = X.shape
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        self.dEdW = 0
        dEdX = dEdY.reshape(self.Xshape)
        return dEdX