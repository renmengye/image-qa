from stage import *

class LUT(Stage):
    def __init__(self,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
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
                 name=None):
        Stage.__init__(self,
                 name=name,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=False)
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.random = np.random.RandomState(initSeed)

        # Zeroth dimension of the weight matrix is reserved
        # for empty word at the end of a sentence.
        if needInit:
            self.W = np.concatenate(
                (np.zeros((1, outputDim)),
                 self.random.uniform(
                -initRange/2.0, initRange/2.0,
                (inputDim, outputDim))), axis=0)
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
        pass

    def forward(self, X):
        X = X.reshape(X.size)
        if self.sparse:
            Y = np.zeros((X.shape[0], self.outputDim))
            for n in range(0, X.shape[0]):
                Y[n] = self.W[X[n]] if X[n] > 0 else np.zeros(self.outputDim)
        else:
            Y = self.W[X, :]
        self.X = X
        self.Y = Y
        return Y

    def backward(self, dEdY):
        X = self.X
        if self.learningRate > 0.0:
            self.dEdW = np.zeros(self.W.shape)
            # for n in range(0, X.shape[0]):
            #     self.dEdW[:, X[n]] += dEdY[n, :]
            self.dEdW[X] += dEdY
        return None

    def loadWeights(self, W):
        if self.learningRate == 0.0:
            return
        else:
            Stage.loadWeights(self, W)

    def getWeights(self):
        if self.learningRate == 0.0:
            return 0
        else:
            return W