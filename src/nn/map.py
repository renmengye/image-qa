from stage import *
class Map(Stage):
    def __init__(self,
                 inputDim,
                 outputDim,
                 activeFn,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
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
                 outputdEdX=outputdEdX)
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.activeFn = activeFn
        self.random = np.random.RandomState(initSeed)

        if needInit:
            self.W = self.random.uniform(
                -initRange/2.0, initRange/2.0, (outputDim, inputDim + 1))
        else:
            self.W = initWeights
        self.X = 0
        self.Y = 0
        self.Z = 0
        pass

    def forward(self, X):
        X2 = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        Z = np.inner(X2, self.W)
        Y = self.activeFn.forward(Z)
        self.X = X2
        self.Z = Z
        self.Y = Y
        return Y

    def backward(self, dEdY):
        Y = self.Y
        Z = self.Z
        X = self.X
        dEdZ = self.activeFn.backward(dEdY, Y, Z)
        self.dEdW = np.dot(dEdZ.transpose(), X)
        dEdX = np.dot(dEdZ, self.W[:, :-1])
        return dEdX if self.outputdEdX else None