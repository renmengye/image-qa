class TimeUnfold:
    def __init__(self):
        self.W = 0
        self.Xshape = 0
        pass

    def forwardPass(self, X):
        if len(X.shape) == 3:
            Y = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
        else:
            Y = X
        self.Xshape = X.shape

        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        dEdW = 0
        if outputdEdX:
            if len(self.Xshape) == 3:
                dEdX = dEdY.reshape(self.Xshape)
            else:
                dEdX = dEdY
        else:
            dEdX = 0
        return dEdW, dEdX