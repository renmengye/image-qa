from util_func import *
import lstmx

FLOAT = np.float

class LSTM:
    def __init__(self,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 cutOffZeroEnd=False,
                 multiErr=False):
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

    def _chkgrd(self):
        X = np.array([[[0.1, 1]], [[1, 0.5]], [[0.2, -0.2]], [[1, 0.3]], [[0.3, -0.2]], [[1, -1]], [[-0.1, 2.0]], [[1, -2]]])
        T = np.array([[[0]], [[0]], [[1.0]], [[1]], [[1]], [[1]], [[0.0]], [[1.0]]])
        Y = self.forwardPass(X)
        E, dEdY = simpleSumDeriv(T, Y)
        dEdW, dEdX = self.backPropagate(dEdY)
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
        if len(X.shape) == 3:
            return self._forwardPassN(X)
        Wi, Wf, Wc, Wo = \
            self.sliceWeights(self.inputDim, self.outputDim, self.W)

        if self.cutOffZeroEnd:
            reachedEnd = np.sum(X, axis=-1) == 0.0
        else:
            reachedEnd = 0

        Y, C, Z, Gi, Gf, Go, Xend = \
                lstmx.forwardPass(
                    X, reachedEnd, self.cutOffZeroEnd, Wi, Wf, Wc, Wo)

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go
        self.Xend = Xend

        return Y if self.multiErr else Y[-1]

    def _forwardPassN(self, X):
        # X[n, t, i] -> n: example, t: time, i: input dimension
        numEx = X.shape[0]
        timespan = X.shape[1]
        self.Xend = np.zeros(numEx, dtype=int)
        myShape = (numEx, timespan, self.outputDim)
        if self.cutOffZeroEnd:
            self.Y = np.zeros((numEx, timespan + 1, self.outputDim),
                              dtype=FLOAT)
            reachedEnd = np.sum(X, axis=-1) == 0.0
        else:
            self.Y = np.zeros(myShape, dtype=FLOAT)
            reachedEnd = np.zeros((numEx, timespan), dtype=FLOAT)

        self.C = np.zeros(myShape, dtype=FLOAT)
        self.Z = np.zeros(myShape, dtype=FLOAT)
        self.Gi = np.zeros(myShape, dtype=FLOAT)
        self.Gf = np.zeros(myShape, dtype=FLOAT)
        self.Go = np.zeros(myShape, dtype=FLOAT)
        Wi, Wf, Wc, Wo = \
            self.sliceWeights(self.inputDim, self.outputDim, self.W)

        for n in range(0, numEx):
            self.Y[n], self.C[n], self.Z[n], \
            self.Gi[n], self.Gf[n], self.Go[n], \
            self.Xend[n] = \
                lstmx.forwardPass(
                    X[n], reachedEnd[n], self.cutOffZeroEnd, Wi, Wf, Wc, Wo)

        self.X = X

        return self.Y \
            if self.multiErr else self.Y[:, -1]

    def _forwardPassOneOld(self, X, reachedEnd, cutOffZeroEnd, Wi, Wf, Wc, Wo):
        timespan = X.shape[0]
        # Last time step is reserved for final output of the entire input.
        if cutOffZeroEnd:
            Y = np.zeros((timespan + 1, self.outputDim), dtype=FLOAT)
        else:
            Y = np.zeros((timespan, self.outputDim), dtype=FLOAT)
        C = np.zeros((timespan, self.outputDim), dtype=FLOAT)
        Z = np.zeros((timespan, self.outputDim), dtype=FLOAT)
        Gi = np.zeros((timespan, self.outputDim), dtype=FLOAT)
        Gf = np.zeros((timespan, self.outputDim), dtype=FLOAT)
        Go = np.zeros((timespan, self.outputDim), dtype=FLOAT)
        Xend = timespan

        for t in range(0, timespan):
            if cutOffZeroEnd and reachedEnd[t]:
                Xend = t
                Y[-1, :] = Y[t - 1, :]
                break

            states1 = np.concatenate((X[t, :], \
                                      Y[t-1, :], \
                                      C[t-1, :], \
                                      np.ones(1, dtype=FLOAT)))
            states2 = np.concatenate((X[t, :], \
                                      Y[t-1, :], \
                                      np.ones(1, dtype=FLOAT)))
            Gi[t, :] = sigmoidFn(np.dot(Wi, states1))
            Gf[t, :] = sigmoidFn(np.dot(Wf, states1))
            Z[t, :] = np.tanh(np.dot(Wc, states2))
            C[t, :] = Gf[t, :] * C[t-1, :] + Gi[t, :] * Z[t, :]
            states3 = np.concatenate((X[t, :], \
                                      Y[t-1, :], \
                                      C[t, :], \
                                      np.ones(1, dtype=FLOAT)))
            Go[t, :] = sigmoidFn(np.dot(Wo, states3))
            Y[t, :] = Go[t, :] * np.tanh(C[t, :])

        return Y, C, Z, Gi, Gf, Go, Xend

    def backPropagate(self, dEdY, outputdEdX=True):
        if len(self.X.shape) == 3:
            return self._backPropagateN(dEdY, outputdEdX)
        Wxi, Wyi, Wci, Wxf, Wyf, Wcf, Wxc, Wyc, Wxo, Wyo, Wco = \
            self.sliceWeightsSmall(self.inputDim, self.outputDim, self.W)
        return lstmx.backPropagate(dEdY,self.X,self.Y,
                                    self.C,self.Z,self.Gi,
                                    self.Gf,self.Go,
                                    self.Xend,self.cutOffZeroEnd,
                                    self.multiErr,outputdEdX,
                                    Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,
                                    Wyc,Wxo,Wyo,Wco,self.W.shape)

    def _backPropagateN(self, dEdY, outputdEdX):
        numEx = self.X.shape[0]
        dEdW = np.zeros(self.W.shape, dtype=FLOAT)
        dEdX = np.zeros(self.X.shape, dtype=FLOAT)
        Wxi, Wyi, Wci, Wxf, Wyf, Wcf, Wxc, Wyc, Wxo, Wyo, Wco = \
            self.sliceWeightsSmall(self.inputDim, self.outputDim, self.W)
        for n in range(0, numEx):
            dEdWtmp, dEdX[n] = \
                lstmx.backPropagate(dEdY[n],self.X[n],self.Y[n],
                                    self.C[n],self.Z[n],self.Gi[n],
                                    self.Gf[n],self.Go[n],
                                    self.Xend[n],self.cutOffZeroEnd,
                                    self.multiErr,outputdEdX,
                                    Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,
                                    Wyc,Wxo,Wyo,Wco,self.W.shape)
            dEdW += dEdWtmp

        return dEdW, dEdX

    def _backPropagateOneOld(
            self, dEdY, X, Y, C, Z, Gi, Gf, Go, Xend, cutOffZeroEnd, multiErr,outputdEdX,
                                    Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,
                                    Wyc,Wxo,Wyo,Wco, Wshape):
        if cutOffZeroEnd and multiErr:
            dEdY[Xend - 1] += dEdY[-1]

        dEdW = np.zeros(Wshape, dtype=FLOAT)
        dEdWi, dEdWf, dEdWc, dEdWo = \
            self.sliceWeights(self.inputDim, self.outputDim, dEdW)

        # (j, t)
        dEdGi = np.zeros((self.outputDim, Xend), dtype=FLOAT)
        dEdGf = np.zeros((self.outputDim, Xend), dtype=FLOAT)
        dEdZ = np.zeros((self.outputDim, Xend), dtype=FLOAT)
        dEdGo = np.zeros((self.outputDim, Xend), dtype=FLOAT)

        # (t, k)
        states1T = np.zeros((Xend,
                   self.inputDim + 2 * self.outputDim + 1), dtype=FLOAT)
        states2T = np.zeros((Xend,
                   self.inputDim + self.outputDim + 1), dtype=FLOAT)
        states3T = np.zeros((Xend,
                   self.inputDim + 2 * self.outputDim + 1), dtype=FLOAT)

        dEdX = np.zeros(X.shape, dtype=FLOAT)

        memEye = np.eye(self.outputDim)
        memCol = (self.outputDim, 1)

        for t in reversed(range(0, Xend)):
            if t == 0:
                Yt1 = np.zeros(self.outputDim, dtype=FLOAT)
                Ct1 = np.zeros(self.outputDim, dtype=FLOAT)
            else:
                Yt1 = Y[t-1]
                Ct1 = C[t-1]

            states1T[t] = \
                np.concatenate((X[t], Yt1, Ct1, np.ones(1, dtype=FLOAT)))
            states2T[t] = \
                np.concatenate((X[t], Yt1, np.ones(1, dtype=FLOAT)))
            states3T[t] = \
                np.concatenate((X[t], Yt1, C[t], np.ones(1, dtype=FLOAT)))


            # (k -> t)
            U = np.tanh(C[t])
            dU = 1 - np.power(U, 2)
            dZ = 1 - np.power(Z[t], 2)

            dGi = Gi[t] * (1 - Gi[t])
            dGf = Gf[t] * (1 - Gf[t])
            dGo = Go[t] * (1 - Go[t])
            dCtdGi = Z[t] * dGi
            dCtdGf = Ct1 * dGf
            dCtdZ = Gi[t] * dZ
            dYtdGo = U * dGo

            # (k, l)
            dYtdCt = (Go[t] * dU) * memEye+ \
                     dYtdGo.reshape(memCol) * Wco

            dEdYnow = dEdY[t] if multiErr else 0
            # (T, t)
            if t < Xend - 1:
                dEdYt = np.dot(dEdYt, dYtdYt1) + np.dot(dEdCt, dCtdYt1) + dEdYnow
                dEdCt = np.dot(dEdCt, dCtdCt1) + np.dot(dEdYt, dYtdCt)
            else:
                dEdYt = dEdYnow if multiErr else dEdY
                dEdCt = np.dot(dEdYt, dYtdCt)

            dEdGi[:, t] = dEdCt * dCtdGi
            dEdGf[:, t] = dEdCt * dCtdGf
            dEdZ[:, t] = dEdCt * dCtdZ
            dEdGo[:, t] = dEdYt * dYtdGo

            # (k -> t, l -> t-1)
            dCtdCt1 = dCtdGf.reshape(memCol) * Wcf + \
                      Gf[t] * memEye + \
                      dCtdGi.reshape(memCol) * Wci
            dCtdYt1 = dCtdGf.reshape(memCol) * Wyf + \
                      dCtdZ.reshape(memCol) * Wyc + \
                      dCtdGi.reshape(memCol) * Wyi
            dYtdYt1 = dYtdGo.reshape(memCol) * Wyo

        dEdWi += np.dot(dEdGi, states1T)
        dEdWf += np.dot(dEdGf, states1T)
        dEdWc += np.dot(dEdZ, states2T)
        dEdWo += np.dot(dEdGo, states3T)

        if outputdEdX:
            dEdX[0:Xend] = np.dot(dEdGi.transpose(), Wxi) + \
                           np.dot(dEdGf.transpose(), Wxf) + \
                           np.dot(dEdZ.transpose(), Wxc) + \
                           np.dot(dEdGo.transpose(), Wxo)

        return dEdW, dEdX

    @staticmethod
    def sliceWeights(inputDim, memoryDim, W):
        s1 = inputDim + memoryDim * 2 + 1
        s2 = s1 * 2
        s3 = s2 + inputDim + memoryDim + 1
        s4 = s3 + s1
        Wi = W[:, 0 : s1]
        Wf = W[:, s1 : s2]
        Wc = W[:, s2 : s3]
        Wo = W[:, s3 : s4]
        
        return Wi, Wf, Wc, Wo

    @staticmethod
    def sliceWeightsSmall(inputDim, memoryDim, W):
        Wi, Wf, Wc, Wo = LSTM.sliceWeights(inputDim, memoryDim, W)

        Wxi = Wi[:, 0 : inputDim]
        Wyi = Wi[:, inputDim : inputDim + memoryDim]
        Wci = Wi[:, inputDim + memoryDim : inputDim + memoryDim + memoryDim]
        Wxf = Wf[:, 0 : inputDim]
        Wyf = Wf[:, inputDim : inputDim + memoryDim]
        Wcf = Wf[:, inputDim + memoryDim : inputDim + memoryDim + memoryDim]
        Wxc = Wc[:, 0 : inputDim]
        Wyc = Wc[:, inputDim : inputDim + memoryDim]
        Wxo = Wo[:, 0 : inputDim]
        Wyo = Wo[:, inputDim : inputDim + memoryDim]
        Wco = Wo[:, inputDim + memoryDim : inputDim + memoryDim + memoryDim]

        return Wxi, Wyi, Wci, Wxf, Wyf, Wcf, Wxc, Wyc, Wxo, Wyo, Wco

if __name__ == '__main__':
    lstm = LSTM(
        inputDim=2,
        outputDim=3,
        initRange=0.01,
        initSeed=2,
        multiErr=True)
    lstm._chkgrd()