from stage import *

class TimeFold(Stage):
    def __init__(self, timespan, name=None):
        Stage.__init__(self, name=name)
        self.timespan = timespan
        self.Xshape = 0
        pass

    def forwardPass(self, X):
        Y = X.reshape((X.shape[0] / self.timespan, self.timespan, X.shape[1]))
        self.Xshape = X.shape

        return Y

    def backPropagate(self, dEdY):
        self.dEdW = 0
        dEdX = dEdY.reshape(self.Xshape)
        return dEdX