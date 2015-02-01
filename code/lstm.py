from util_func import *
from stage import *
import lstmpy as lstmx

class LSTM(Stage):
    def __init__(self,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 cutOffZeroEnd=False,
                 multiErr=False,
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
        self.cutOffZeroEnd = cutOffZeroEnd
        self.multiErr = multiErr
        self.random = np.random.RandomState(initSeed)

        if needInit:
            np.random.seed(initSeed)
            start = -initRange / 2.0
            end = initRange / 2.0
            Wxi = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxf = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxc = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxo = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wyi = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyf = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyc = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyo = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wci = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wcf = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wco = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wbi = np.ones((self.outputDim, 1), dtype=FLOAT)
            Wbf = np.ones((self.outputDim, 1), dtype=FLOAT)
            Wbc = np.zeros((self.outputDim, 1), dtype=FLOAT)
            Wbo = np.ones((self.outputDim, 1), dtype=FLOAT)

            Wi = np.concatenate((Wxi, Wyi, Wci, Wbi), axis=1)
            Wf = np.concatenate((Wxf, Wyf, Wcf, Wbf), axis=1)
            Wc = np.concatenate((Wxc, Wyc, Wbc), axis=1)
            Wo = np.concatenate((Wxo, Wyo, Wco, Wbo), axis=1)
            self.W = np.concatenate((Wi, Wf, Wc, Wo), axis = 1)
        else:
            self.W = initWeights

        self.X = 0
        self.Xend = 0
        self.Y = 0
        self.C = 0
        self.Z = 0
        self.Gi = 0
        self.Gf = 0
        self.Go = 0
        pass

    def chkgrd(self):
        X = np.array([[[0.1, 1]], [[1, 0.5]], [[0.2, -0.2]], [[1, 0.3]], [[0.3, -0.2]], [[1, -1]], [[-0.1, 2.0]], [[1, -2]]])
        T = np.array([[[0]], [[0]], [[1.0]], [[1]], [[1]], [[1]], [[0.0]], [[1.0]]])
        Y = self.forwardPass(X)
        E, dEdY = simpleSumDeriv(T, Y)
        dEdX = self.backPropagate(dEdY)
        dEdW = self.dEdW
        eps = 1e-3
        dEdWTmp = np.zeros(self.W.shape)
        dEdXTmp = np.zeros(X.shape)
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i,j] += eps
                Y = self.forwardPass(X)
                Etmp1, d1 = simpleSumDeriv(T, Y)

                self.W[i,j] -= 2 * eps
                Y = self.forwardPass(X)
                Etmp2, d2 = simpleSumDeriv(T, Y)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        for n in range(0, X.shape[0]):
            for t in range(0, X.shape[1]):
                for j in range(0, X.shape[2]):
                    X[n, t, j] += eps
                    Y = self.forwardPass(X)
                    Etmp1, d1 = simpleSumDeriv(T, Y)

                    X[n, t, j] -= 2 * eps
                    Y = self.forwardPass(X)
                    Etmp2, d2 = simpleSumDeriv(T, Y)

                    dEdXTmp[n, t, j] = (Etmp1 - Etmp2) / 2.0 / eps
                    X[n, t, j] += eps

        print "haha"
        pass

    def forwardPass(self, X):
        Y, C, Z, Gi, Gf, Go, Xend = \
            lstmx.forwardPassN(
            X, self.cutOffZeroEnd, self.W)

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go
        self.Xend = Xend

        return Y if self.multiErr else Y[:,-1]

    def backPropagate(self, dEdY):
        self.dEdW, dEdX = lstmx.backPropagateN(dEdY,self.X,self.Y,
                                self.C,self.Z,self.Gi,
                                self.Gf,self.Go,
                                self.Xend,self.cutOffZeroEnd,
                                self.multiErr,self.outputdEdX,
                                self.W)
        return dEdX if self.outputdEdX else None

    def sliceWeights(
                    inputDim,
                    outputDim,
                    W):
        s1 = inputDim + outputDim * 2 + 1
        s2 = s1 * 2
        s3 = s2 + inputDim + outputDim + 1
        s4 = s3 + s1
        Wi = W[:, 0 : s1]
        Wf = W[:, s1 : s2]
        Wc = W[:, s2 : s3]
        Wo = W[:, s3 : s4]

        return Wi, Wf, Wc, Wo

if __name__ == '__main__':
    lstm = LSTM(
        inputDim=2,
        outputDim=3,
        initRange=0.01,
        initSeed=2,
        multiErr=True)
    lstm.chkgrd()
