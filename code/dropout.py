import numpy as np

class Dropout:
    def __init__(self, dropoutRate, initSeed, debug=False):
        self.W = 0
        self.X = 0
        self.dropout = True
        self.dropoutVec = 0
        self.dropoutRate = dropoutRate
        self.debug = debug
        self.random = np.random.RandomState(initSeed)
        self.seed = initSeed
        pass

    def forwardPass(self, X, dropout=False):
        Y = np.zeros(X.shape, float)
        if self.dropoutRate > 0.0 and dropout:
            if self.debug:
                self.random = np.random.RandomState(self.seed)
            self.dropoutVec = (self.random.uniform(0, 1, (X.shape[-1])) >
                               self.dropoutRate)
            for i in range(0, X.shape[-1]):
                if self.dropoutVec[i]:
                    if len(X.shape) == 1:
                        Y[i] = X[i]
                    elif len(X.shape) == 2:
                        Y[:, i] = X[:, i]
                    elif len(X.shape) == 3:
                        Y[:, :, i] = X[:, :, i]
        else:
            Y = X * (1 - self.dropoutRate)
        self.X = X
        self.dropout = dropout
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        dEdW = 0
        if outputdEdX:
            if self.dropout:
                dEdX = np.zeros(self.X.shape, float)
                for i in range(0, self.X.shape[-1]):
                    if self.dropoutVec[i]:
                        if len(self.X.shape) == 1:
                            dEdX[i] = dEdY[i]
                        elif len(self.X.shape) == 2:
                            dEdX[:, i] = dEdY[:, i]
                        elif len(self.X.shape) == 3:
                            dEdX[:, :, i] = dEdY[:, :, i]
            else:
                dEdX = dEdY / (1 - self.dropoutRate)

        return dEdW, dEdX