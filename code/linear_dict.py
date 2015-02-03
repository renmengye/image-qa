from util_func import *
from stage import *
import numpy

class LinearDict(Stage):
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
        self.random = numpy.random.RandomState(initSeed)

        # Zeroth dimension of the weight matrix is reserved
        # for empty word at the end of a sentence.
        if needInit:
            self.W = self.random.uniform(
                -initRange/2.0, initRange/2.0, (outputDim, inputDim))
            self.W[:, 0] = 0
        else:
            self.W = np.concatenate(
                (np.zeros((outputDim, 1)), initWeights), axis=1)
        self.X = 0
        self.Y = 0
        pass

    def chkgrd(self):
        X = np.array([3])
        T = np.array([0.1, 0.3])
        Y = self.forwardPass(X)
        E, dEdY = meanSqErr(Y, T)
        dEdX = self.backPropagate(dEdY)
        dEdW = self.dEdW
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
        X = X.reshape(X.size)
        Y = np.zeros((X.shape[0], self.outputDim))
        for n in range(0, X.shape[0]):
            Y[n, :] = self.W[:, X[n]]
        self.X = X
        self.Y = Y
        return Y

    def backPropagate(self, dEdY):
        X = self.X
        self.dEdW = np.zeros(self.W.shape)
        for n in range(0, X.shape[0]):
            self.dEdW[:, X[n]] += dEdY[n, :]
        return None

if __name__ == '__main__':
    lindict = LinearDict(
        inputDim=5,
        outputDim=2,
        initRange=0.01,
        initSeed=2)
    lindict.chkgrd()
