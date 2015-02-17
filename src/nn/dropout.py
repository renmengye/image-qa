from stage import *

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
        self.random = np.random.RandomState(initSeed)
        self.seed = initSeed
        pass

    def forward(self, X):
        if self.dropoutRate > 0.0 and self.dropout:
            if self.debug:
                self.random = np.random.RandomState(self.seed)
            self.dropoutVec = (self.random.uniform(0, 1, (X.shape[-1])) >
                               self.dropoutRate)
            Y = X * self.dropoutVec
        else:
            Y = X * (1 - self.dropoutRate)
        self.X = X
        return Y

    def backward(self, dEdY, outputdEdX=True):
        self.dEdW = 0
        dEdX = None
        if outputdEdX:
            if self.dropout:
                dEdX = dEdY * self.dropoutVec
            else:
                dEdX = dEdY / (1 - self.dropoutRate)

        return dEdX