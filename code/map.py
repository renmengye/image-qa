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

    def chkgrd(self, costFn):
        eps = 1e-3
        X = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, -0.3], [-0.1, -0.1]])
        T = np.array([[0], [1], [0], [1]])
        Y = self.forward(X)
        E, dEdY = costFn(Y, T)
        dEdX = self.backward(dEdY)
        dEdW = self.dEdW
        dEdWTmp = np.zeros(self.W.shape)
        dEdXTmp = np.zeros(X.shape)
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i,j] += eps
                Y = self.forward(X)
                Etmp1, d1 = costFn(Y, T)

                self.W[i,j] -= 2 * eps
                Y = self.forward(X)
                Etmp2, d2 = costFn(Y, T)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for t in range(0, X.shape[0]):
            for k in range(0, X.shape[-1]):
                X[t, k] += eps
                Y = self.forward(X)
                Etmp1, d1 = costFn(Y, T)

                X[t, k] -= 2 * eps
                Y = self.forward(X)
                Etmp2, d2 = costFn(Y, T)

                dEdXTmp[t, k] += (Etmp1 - Etmp2) / 2.0 / eps
                X[t, k] += eps
        print "haha"
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

if __name__ == '__main__':
    from active_func import *
    map_ = Map(
        inputDim=2,
        outputDim=1,
        activeFn=SigmoidActiveFn,
        initRange=0.01,
        initSeed=2
    )
    map_.chkgrd(costFn=crossEntOne)
    map_ = Map(
        inputDim=2,
        outputDim=2,
        activeFn=SoftmaxActiveFn,
        initRange=0.01,
        initSeed=2
    )
    map_.chkgrd(costFn=crossEntIdx)
    map_ = Map(
        inputDim=2,
        outputDim=1,
        activeFn=IdentityActiveFn,
        initRange=0.01,
        initSeed=2
    )
    map_.chkgrd(costFn=meanSqErr)
