from stage import *
import numpy

class Dropout(Stage):
    def __init__(self,
                 dropoutRate,
                 initSeed,
                 debug=False,
                 name=None):
        Stage.__init__(self, name=name)
        self.W = 0
        self.X = 0
        self.dropout = True
        self.dropoutVec = 0
        self.dropoutRate = dropoutRate
        self.debug = debug
        self.random = numpy.random.RandomState(initSeed)
        self.seed = initSeed
        pass

    def forwardPass(self, X):
        Y = np.zeros(X.shape)
        if self.dropoutRate > 0.0 and self.dropout:
            if self.debug:
                self.random = numpy.random.RandomState(self.seed)
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
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        self.dEdW = 0
        dEdX = None
        if outputdEdX:
            if self.dropout:
                dEdX = np.zeros(self.X.shape)
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

        return dEdX