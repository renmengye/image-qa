from stage import *

class TimeFold(Stage):
    def __init__(self, timespan):
        Stage.__init__(self)
        self.timespan = timespan
        self.Xshape = 0
        pass

    def forwardPass(self, X):
        Y = np.reshape(X, (X.shape[0] / self.timespan, self.timespan, X.shape[1]))
        self.Xshape = X.shape

        return Y

    def backPropagate(self, dEdY):
        self.dEdW = 0
        dEdX = np.reshape(dEdY, self.Xshape)
        return dEdX