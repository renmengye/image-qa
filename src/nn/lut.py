from stage import *

class LUT(Stage):
    def __init__(self,
                 inputNames,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 intConversion=False,
                 needInit=True,
                 initWeights=0,
                 sparse=False,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=False,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 learningRate=learningRate,
                 outputDim=outputDim,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        self.outputDim = outputDim
        self.inputDim = inputDim
        self.initRange = initRange
        self.random = np.random.RandomState(initSeed)
        self.needInit = needInit
        self.intConversion = intConversion

        # Zeroth rows of the weight matrix is reserved
        # for empty word at the end of a sentence.
        if needInit:
            self.W = None
        else:
            if sparse:
                initWeights = np.array(initWeights.todense())
                self.W = np.concatenate(
                    (np.zeros((1, outputDim)), initWeights), axis=0)
            else:
                self.W = np.concatenate(
                    (np.zeros((1, outputDim)), initWeights), axis=0)
        self.X = 0
        self.Y = 0
        self.sparse = sparse
        self.dEdW = 0.0

    def initWeights(self):
        self.W = np.concatenate(
            (np.zeros((1, self.outputDim)),
             self.random.uniform(
            -self.initRange/2.0, self.initRange/2.0,
            (self.inputDim, self.outputDim))), axis=0)

    def forward(self, X):
        if self.W is None: self.initWeights()
        if self.intConversion: X = X.astype(int)
        self.X = X
        X = X.reshape(X.size)
        Y = np.zeros((X.shape[0], self.outputDim))
        for n in range(0, X.shape[0]):
             Y[n] = self.W[X[n]]
        return Y

    def backward(self, dEdY):
        X = self.X
        if self.learningRate > 0.0:
            self.dEdW = np.zeros(self.W.shape)
            for n in range(0, X.shape[0]):
                self.dEdW[X[n]] += dEdY[n]
        if self.outputdEdX:
            return np.zeros(X.shape)
        else:
            return None

    def loadWeights(self, W):
        if self.learningRate == 0.0:
            return
        else:
            Stage.loadWeights(W)

    def getWeights(self):
        if self.learningRate == 0.0:
            return 0
        else:
            return self.W
