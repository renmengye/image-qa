from util_func import *

class Softmax:
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
                Etmp1, d1 = crossEntIdx(Y, T)

                self.W[i,j] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = crossEntIdx(Y, T)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for j in range(0, X.shape[-1]):
            X[j] += eps
            Y = self.forwardPass(X)
            Etmp1, d1 = crossEntIdx(Y, T)

            X[j] -= 2 * eps
            Y = self.forwardPass(X)
            Etmp2, d2 = crossEntIdx(Y, T)

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
                Etmp1, d1 = crossEntIdx(Y, T)

                self.W[i,j] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = crossEntIdx(Y, T)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for t in range(0, X.shape[0]):
            for k in range(0, X.shape[-1]):
                X[t, k] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = crossEntIdx(Y, T)

                X[t, k] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = crossEntIdx(Y, T)

                dEdXTmp[t, k] += (Etmp1 - Etmp2) / 2.0 / eps
                X[t, k] += eps

        print "haha"
        pass

    def forwardPass(self, X):
        Y = np.inner(X, self.W)
        expY = np.exp(Y)
        expYshape = np.copy(Y.shape)
        expYshape[-1] = 1
        Y = expY / np.sum(expY, axis=-1).reshape(expYshape).repeat(Y.shape[-1], axis=-1)
        self.X = X
        self.Y = Y
        return Y

    def backPropagate(self, dEdY, outputdEdX=True):
        if len(self.X.shape) == 2:
            return self.backPropagateAll(dEdY, outputdEdX)
        Y = self.Y
        X = self.X
        # (j, i)
        dY_i__dZ_j = -Y.reshape(1, self.outputDim) * Y.reshape(self.outputDim, 1)
        dY_i__dZ_j += np.eye(self.outputDim) * Y.reshape(1, self.outputDim)
        # (j, k, i)
        dY_i__dW_jk = dY_i__dZ_j.reshape(self.outputDim, 1, self.outputDim) * X.reshape(1, self.inputDim, 1)
        dEdW = np.inner(dY_i__dW_jk, dEdY)
        if outputdEdX:
            # (k, i)
            dY_i__dX_k = np.inner(self.W.transpose(), dY_i__dZ_j)
            # (k, i) * (i) = (k)
            dEdX = np.inner(dY_i__dX_k, dEdY)
        return dEdW, dEdX

    def backPropagateAll(self, dEdY, outputdEdX=True):
        Y = self.Y
        X = self.X
        timespan = Y.shape[0]
        # (t, i, j)
        dY_ti__dZ_j = -Y.reshape(timespan, self.outputDim, 1) * Y.reshape(timespan, 1, self.outputDim)
        dY_ti__dZ_j += np.eye(self.outputDim).reshape(1, self.outputDim, self.outputDim) * Y.reshape(timespan, 1, self.outputDim)

        # (t, i, j, k)
        dY_ti__dW_jk = dY_ti__dZ_j.reshape(timespan, self.outputDim, self.outputDim, 1) * X.reshape(timespan, 1, 1, self.inputDim)

        # (t, i, j, k) * (t, i) = (j, k)
        dEdW = np.tensordot(dY_ti__dW_jk, dEdY, axes=([1, 0], [1, 0]))
        if outputdEdX:
            # (t, i, j) * (j, k) = (t, i, k)
            dY_ti__dX_k = np.dot(dY_ti__dZ_j, self.W)
            # (t, i) * (t, i, k) = (t, t, k) -> (k, t) -> (t, k)
            dEdX = np.diagonal(np.dot(dEdY, dY_ti__dX_k), axis1=0, axis2=1).transpose()

        return dEdW, dEdX

if __name__ == '__main__':
    softmax = Softmax(
        inputDim=2,
        outputDim=2,
        initRange=0.01,
        initSeed=2)
    softmax.chkgrd()