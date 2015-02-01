from util_func import *
from stage import *

class Softmax(Stage):
    def __init__(self,
                 inputDim,
                 outputDim,
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
                 outputdEdX=True):
        Stage.__init__(self,
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
        eps = 1e-3
        X = np.array([[0.1, 0.5], [0.2, 0.4], [0.3, -0.3], [-0.1, -0.1]])
        T = np.array([[0], [1], [0], [1]])
        Y = self.forwardPass(X)
        E, dEdY = crossEntIdx(Y, T)
        dEdX = self.backPropagate(dEdY)
        dEdW = self.dEdW
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

    def backPropagate(self, dEdY):
        Y = self.Y
        X = self.X
        timespan = Y.shape[0]
        # (t, i, j)
        dY_ti__dZ_j = -Y.reshape(timespan, self.outputDim, 1) * Y.reshape(timespan, 1, self.outputDim)
        dY_ti__dZ_j += np.eye(self.outputDim).reshape(1, self.outputDim, self.outputDim) * Y.reshape(timespan, 1, self.outputDim)

        # (t, i, j, k)
        dY_ti__dW_jk = dY_ti__dZ_j.reshape(timespan, self.outputDim, self.outputDim, 1) * X.reshape(timespan, 1, 1, self.inputDim)

        # (t, i, j, k) * (t, i) = (j, k)
        self.dEdW = np.tensordot(dY_ti__dW_jk, dEdY, axes=([1, 0], [1, 0]))
        if self.outputdEdX:
            # (t, i, j) * (j, k) = (t, i, k)
            dY_ti__dX_k = np.dot(dY_ti__dZ_j, self.W)
            # (t, i) * (t, i, k) = (t, t, k) -> (k, t) -> (t, k)
            dEdX = np.diagonal(np.dot(dEdY, dY_ti__dX_k), axis1=0, axis2=1).transpose()

        return dEdX if self.outputdEdX else None

if __name__ == '__main__':
    softmax = Softmax(
        inputDim=2,
        outputDim=2,
        initRange=0.01,
        initSeed=2)
    softmax.chkgrd()