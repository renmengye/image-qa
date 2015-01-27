import numpy as np

class TimeFold:
    def __init__(self, timespan):
        self.W = 0
        self.timespan = timespan
        self.Xshape = 0
        pass

    def forwardPass(self, X):
        if X.shape[0] == self.timespan:
            Y = X
        else:
            Y = np.reshape(X, (X.shape[0] / self.timespan, self.timespan, X.shape[1]))
        self.Xshape = X.shape

        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        dEdW = 0
        if outputdEdX:
            if len(dEdY.shape) == 3:
                dEdX = np.reshape(dEdY, self.Xshape)
            else:
                dEdX = dEdY
        else:
            dEdX = 0
        return dEdW, dEdX