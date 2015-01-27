from util_func import *

class LinearMap:
    def __init__(self,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0):
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.random = np.random.RandomState(initSeed)

        if needInit:
            self.W = self.random.uniform(
                -initRange/2.0, initRange/2.0, (outputDim, inputDim))
        else:
            self.W = initWeights
        self.X = 0
        self.Y = 0
        pass

    def chkgrd(self):
        X = np.array([0.1, 0.5])
        T = np.array([0])
        Y = self.forwardPass(X)
        E, dEdY = meanSqErr(Y, T)
        dEdW, dEdX = self.backPropagate(dEdY)
        eps = 1e-3
        dEdWTmp = np.zeros(self.W.shape)
        dEdXTmp = np.zeros(X.shape[-1])
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i,j] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = meanSqErr(Y, T)

                self.W[i,j] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = meanSqErr(Y, T)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for j in range(0, X.shape[-1]):
            X[j] += eps
            Y = self.forwardPass(X)
            Etmp1, d1 = meanSqErr(Y, T)

            X[j] -= 2 * eps
            Y = self.forwardPass(X)
            Etmp2, d2 = meanSqErr(Y, T)

            dEdXTmp[j] += (Etmp1 - Etmp2) / 2.0 / eps
            X[j] += eps

        X = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, -0.3], [-0.1, -0.1]])
        T = np.array([[0], [1], [0], [1]])
        Y = self.forwardPass(X)
        E, dEdY = meanSqErr(Y, T)
        dEdW, dEdX = self.backPropagate(dEdY)
        dEdWTmp = np.zeros(self.W.shape)
        dEdXTmp = np.zeros(X.shape)
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i,j] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = meanSqErr(Y, T)

                self.W[i,j] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = meanSqErr(Y, T)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for t in range(0, X.shape[0]):
            for k in range(0, X.shape[-1]):
                X[t, k] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = meanSqErr(Y, T)

                X[t, k] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = meanSqErr(Y, T)

                dEdXTmp[t, k] += (Etmp1 - Etmp2) / 2.0 / eps
                X[t, k] += eps

        print "haha"
        pass

    def forwardPass(self, X):
        Y = np.inner(X, self.W)
        self.X = X
        self.Y = Y
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        if len(dEdY.shape) == 2:
            return self.backPropagateAll(dEdY, outputdEdX)
        X = self.X
        # (1, k) * (j, 1) = (j, k)
        dEdW = X.reshape(1, self.inputDim) * dEdY.reshape(dEdY.shape[0], 1)

        if outputdEdX:
            # (j) * (j, k)
            dEdX = np.dot(dEdY, self.W)
        else:
            dEdX = 0
        return dEdW, dEdX

    def backPropagateAll(self, dEdY, outputdEdX=True):
        X = self.X
        # (j, t) * (t, k) = (j, k)
        dEdW = np.dot(dEdY.transpose(), X)

        if outputdEdX:
            # (t, j) * (j, k) = (t, k)
            dEdX = np.dot(dEdY, self.W)
        else:
            dEdX = 0

        return dEdW, dEdX

if __name__ == '__main__':
    map_ = LinearMap(
        inputDim=2,
        outputDim=1,
        initRange=0.01,
        initSeed=2)
    map_.chkgrd()