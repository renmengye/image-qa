from util_func import *
from scipy import special

class Sigmoid:
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
            self.W = np.random.rand(outputDim, inputDim + 1) * initRange - initRange / 2.0
        else:
            self.W = W
        self.X = 0
        self.Y = 0
        pass

    def chkgrd(self):
        X = np.array([0.1, 0.5])
        T = 0
        Y = self.forwardPass(X)
        E, dEdY = crossEntIdx(Y, T)
        dEdW, dEdX = self.backPropagate(dEdY)
        eps = 1e-3
        dEdWTmp = np.zeros(self.W.shape)
        dEdXTmp = np.zeros(X.shape[-1])
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i,j] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = crossEntOne(Y, T)

                self.W[i,j] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = crossEntOne(Y, T)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for j in range(0, X.shape[-1]):
            X[j] += eps
            Y = self.forwardPass(X)
            Etmp1, d1 = crossEntOne(Y, T)

            X[j] -= 2 * eps
            Y = self.forwardPass(X)
            Etmp2, d2 = crossEntOne(Y, T)

            dEdXTmp[j] += (Etmp1 - Etmp2) / 2.0 / eps
            X[j] += eps

        X = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, -0.3], [-0.1, -0.1]])
        T = np.array([[0], [1], [0], [1]])
        Y = self.forwardPass(X)
        E, dEdY = crossEntIdx(Y, T)
        dEdW, dEdX = self.backPropagate(dEdY)
        dEdWTmp = np.zeros(self.W.shape)
        dEdXTmp = np.zeros(X.shape)
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i,j] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = crossEntOne(Y, T)

                self.W[i,j] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = crossEntOne(Y, T)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for t in range(0, X.shape[0]):
            for k in range(0, X.shape[-1]):
                X[t, k] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = crossEntOne(Y, T)

                X[t, k] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = crossEntOne(Y, T)

                dEdXTmp[t, k] += (Etmp1 - Etmp2) / 2.0 / eps
                X[t, k] += eps

        print "haha"
        pass

    def forwardPass(self, X):
        Y = np.inner(X, self.W)
        Y = special.expit(Y)
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        if len(self.X.shape) == 2:
            return self.backPropagateAll(dEdY, outputdEdX)

        Y = self.Y
        X = self.X

        dY_i__dZ_i = Y * (1 - Y).reshape(self.outputDim, 1)
        dY_i__dW_ij = dY_i__dZ_i * np.concatenate((X.reshape(1, self.inputDim), np.ones(1, 1)), axis=1)
        dEdW = np.dot(dEdY, dY_i__dW_ij)

        if outputdEdX:
            dY_i__dX_j = dY_i__dZ_i * self.W[:, 0:-1]
            dEdX = np.dot(dEdY, dY_i__dX_j)

        return dEdW, dEdX

    def backPropagateAll(self, dEdY, outputdEdX=True):
        Y = self.Y
        X = self.X
        numEx = X.shape[0]
        dY_ni__dZ_i = Y * (1 - Y).reshape(numEx, self.outputDim, 1)
        dZ_ni__dW_ij = X.reshape(numEx, 1, self.inputDim) + np.ones((numEx, 1, 1))
        dY_ni__dW_ij = dY_ni__dZ_i * dZ_ni__dW_ij
        dZ_ni__dX_j = self.W[:, 0:-1].reshape(1, self.outputDim, self.inputDim)
        dY_ni__dX_j = dY_ni__dZ_i * dZ_ni__dX_j

        dEdW = np.diagonal(dEdY.dot(dY_ni__dW_ij.transpose((1, 0, 2))), axis1=0, axis2=1).transpose()
        dEdX = np.diagonal(dEdY.dot(dY_ni__dX_j), axis1=0, axis2=1).transpose()

        return dEdW, dEdX

if __name__ == '__main__':
    sigmoid = Sigmoid(
        inputDim=2,
        outputDim=2,
        initRange=0.01,
        initSeed=2)
    sigmoid.chkgrd()
