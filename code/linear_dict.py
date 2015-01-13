from util_func import *

class LinearDict:
    def __init__(self,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 W=0):
        self.inputDim = inputDim
        self.outputDim = outputDim

        if needInit:
            np.random.seed(initSeed)
            self.W = np.random.rand(outputDim, inputDim) * initRange - initRange / 2.0
            self.W[:, 0] = 0
        else:
            self.W = W
        self.X = 0
        self.Y = 0
        pass

    def chkgrd(self):
        X = np.array([3])
        T = np.array([0.1, 0.3])
        Y = self.forwardPass(X)
        E, dEdY = meanSqErr(Y, T)
        dEdW, dEdX = self.backPropagate(dEdY)
        eps = 1e-3
        dEdWTmp = np.zeros(self.W.shape)
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

        X = np.array([3, 4, 1, 0])
        T = np.array([[0.1, 0.4], [-1.2, 1.5], [3.3, -1.1], [2.0, 0.01]])
        Y = self.forwardPass(X)
        E, dEdY = meanSqErr(Y, T)
        dEdW, dEdX = self.backPropagate(dEdY)
        dEdWTmp = np.zeros(self.W.shape)
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

        print "haha"
        pass

    def forwardPass(self, X):
        if X.size > 1:
            X = X.reshape(X.size)
            Y = np.zeros((X.shape[0], self.outputDim))
            for n in range(0, X.shape[0]):
                Y[n, :] = self.W[:, X[n]]
        else:
            Y = self.W[:, X]
        self.X = X
        self.Y = Y
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        X = self.X
        dEdW = np.zeros(self.W.shape)
        if X.size > 1:
            for t in range(0, X.shape[0]):
                dEdW[:, X[t]] += dEdY[t, :]
        else:
            dEdW[:, X] = dEdY
        dEdX = 0
        return dEdW, dEdX

if __name__ == '__main__':
    dict = LinearDict(
        inputDim=5,
        outputDim=2,
        initRange=0.01,
        initSeed=2)
    dict.chkgrd()