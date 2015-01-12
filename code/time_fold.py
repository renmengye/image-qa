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
            Y = X.reshape(self.timespan, X.shape[0] / self.timespan, X.shape[1])
        self.Xshape = X.shape

        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        dEdW = 0
        if outputdEdX:
            if len(dEdY.shape) == 3:
                dEdX = dEdY.reshape(self.Xshape)
            else:
                dEdX = dEdY
        else:
            dEdX = 0
        return dEdW, dEdX