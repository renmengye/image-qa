from scipy import special
from util_func import *

FLOAT = np.float

class LSTM:
    def __init__(self,
                 inputDim,
                 memoryDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 W=0,
                 cutOffZeroEnd=False):
        self.inputDim = inputDim
        self.memoryDim = memoryDim
        self.cutOffZeroEnd = cutOffZeroEnd

        if needInit:
            np.random.seed(initSeed)
            Wxi = np.random.rand(self.memoryDim, self.inputDim) * initRange - initRange / 2.0
            Wxf = np.random.rand(self.memoryDim, self.inputDim) * initRange - initRange / 2.0
            Wxc = np.random.rand(self.memoryDim, self.inputDim) * initRange - initRange / 2.0
            Wxo = np.random.rand(self.memoryDim, self.inputDim) * initRange - initRange / 2.0
            Wyi = np.random.rand(self.memoryDim, self.memoryDim) * initRange - initRange / 2.0
            Wyf = np.random.rand(self.memoryDim, self.memoryDim) * initRange - initRange / 2.0
            Wyc = np.random.rand(self.memoryDim, self.memoryDim) * initRange - initRange / 2.0
            Wyo = np.random.rand(self.memoryDim, self.memoryDim) * initRange - initRange / 2.0
            Wci = np.random.rand(self.memoryDim, self.memoryDim) * initRange - initRange / 2.0
            Wcf = np.random.rand(self.memoryDim, self.memoryDim) * initRange - initRange / 2.0
            Wco = np.random.rand(self.memoryDim, self.memoryDim) * initRange - initRange / 2.0
            # Wbi = np.random.rand(self.memoryDim, 1) * initRange - initRange / 2.0
            # Wbf = np.random.rand(self.memoryDim, 1) * initRange - initRange / 2.0
            # Wbc = np.random.rand(self.memoryDim, 1) * initRange - initRange / 2.0
            # Wbo = np.random.rand(self.memoryDim, 1) * initRange - initRange / 2.0
            Wbi = np.ones((self.memoryDim, 1), dtype=FLOAT)
            Wbf = np.ones((self.memoryDim, 1), dtype=FLOAT)
            Wbc = np.zeros((self.memoryDim, 1), dtype=FLOAT)
            Wbo = np.ones((self.memoryDim, 1), dtype=FLOAT)

            Wi = np.concatenate((Wxi, Wyi, Wci, Wbi), axis=1)
            Wf = np.concatenate((Wxf, Wyf, Wcf, Wbf), axis=1)
            Wc = np.concatenate((Wxc, Wyc, Wbc), axis=1)
            Wo = np.concatenate((Wxo, Wyo, Wco, Wbo), axis=1)
            self.W = np.concatenate((Wi, Wf, Wc, Wo), axis = 1)
        else:
            self.W = W

        self.X = 0
        self.Y = 0
        self.C = 0
        self.Z = 0
        self.Gi = 0
        self.Gf = 0
        self.Go = 0
        pass

    def chkgrd(self):
        X = np.array([[[0.1]], [[1]], [[0.2]], [[1]], [[0.3]], [[1]], [[-0.1]], [[1]]])
        T = np.array([[[0]], [[0]], [[0]], [[1]], [[1]], [[1]], [[0]], [[1]]])
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
        for t in range(0, X.shape[0]):
            for n in range(0, X.shape[1]):
                for j in range(0, X.shape[2]):
                    X[t, n, j] += eps
                    Y = self.forwardPass(X)
                    Etmp1, d1 = simpleSumDeriv(T, Y)

                    X[t, n, j] -= 2 * eps
                    Y = self.forwardPass(X)
                    Etmp2, d2 = simpleSumDeriv(T, Y)

                    dEdXTmp[t, n, j] = (Etmp1 - Etmp2) / 2.0 / eps
                    X[t, n, j] += eps

        print "haha"
        pass

    @staticmethod
    def needsCutOff(X, t):
        return np.mean(X[t]) == 0.0

    def forwardPass(self, X):
        if len(X.shape) == 3:
            return self.forwardPassN(X)
        else:
            Y, C, Z, Gi, Gf, Go = self.forwardPass_(X)

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go
        return Y

    def forwardPassN(self, X):
        # X[t, n, i] -> t: time, n: example, i: input dimension
        timespan = X.shape[0]
        numEx = X.shape[1]
        Y = np.zeros((timespan, numEx, self.memoryDim), dtype=FLOAT)
        C = np.zeros((timespan, numEx, self.memoryDim), dtype=FLOAT)
        Z = np.zeros((timespan, numEx, self.memoryDim), dtype=FLOAT)
        Gi = np.zeros((timespan, numEx, self.memoryDim), dtype=FLOAT)
        Gf = np.zeros((timespan, numEx, self.memoryDim), dtype=FLOAT)
        Go = np.zeros((timespan, numEx, self.memoryDim), dtype=FLOAT)

        for n in range(0, numEx):
            Y[:, n, :], C[:, n, :], Z[:, n, :], \
            Gi[:, n, :], Gf[:, n, :], Go[:, n, :] = self.forwardPass_(X[:, n, :])

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go

        return Y

    def forwardPass_(self, X):
        timespan = X.shape[0]
        Y = np.zeros((timespan, self.memoryDim), dtype=FLOAT)
        C = np.zeros((timespan, self.memoryDim), dtype=FLOAT)
        Z = np.zeros((timespan, self.memoryDim), dtype=FLOAT)
        Gi = np.zeros((timespan, self.memoryDim), dtype=FLOAT)
        Gf = np.zeros((timespan, self.memoryDim), dtype=FLOAT)
        Go = np.zeros((timespan, self.memoryDim), dtype=FLOAT)
        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)

        for t in range(0, timespan):
            # Hack: copy the real sentence end to the fixed end
            if self.cutOffZeroEnd and self.needsCutOff(X, t):
                Y[-1, :] = Y[t - 1, :]
                break

            # In forward pass initial stage -1 is empty, equivalent to zero.
            states1 = np.concatenate((X[t, :], Y[t-1, :], C[t-1, :], np.ones(1, dtype=FLOAT)))
            states2 = np.concatenate((X[t, :], Y[t-1, :], np.ones(1, dtype=FLOAT)))
            Gi[t, :] = special.expit(np.dot(Wi, states1))
            Gf[t, :] = special.expit(np.dot(Wf, states1))
            Z[t, :] = np.tanh(np.dot(Wc, states2))
            C[t, :] = Gf[t, :] * C[t-1, :] + Gi[t, :] * Z[t, :]
            states3 = np.concatenate((X[t, :], Y[t-1, :], C[t, :], np.ones(1, dtype=FLOAT)))
            Go[t, :] = special.expit(np.dot(Wo, states3))
            Y[t, :] = Go[t, :] * np.tanh(C[t, :])

        return Y, C, Z, Gi, Gf, Go

    def backPropagate(self, dEdY, outputdEdX=True):
        if len(self.X.shape) == 3:
            return self.backPropagateN(dEdY, outputdEdX)
        X = self.X
        Y = self.Y
        C = self.C
        Z = self.Z
        Gi = self.Gi
        Gf = self.Gf
        Go = self.Go
        return self.backPropagate_new(dEdY, X, Y, C, Z, Gi, Gf, Go, outputdEdX)

    def backPropagate_new(self, dEdY, X, Y, C, Z, Gi, Gf, Go, outputdEdX):
        timespan = Y.shape[0]
        Wxi, Wyi, Wci, Wxf, Wyf, Wcf, Wxc, Wyc, Wxo, Wyo, Wco = \
            self.sliceWeightsSmall(self.inputDim, self.memoryDim, self.W)

        dEdWi = np.zeros((self.memoryDim, self.inputDim + 2 * self.memoryDim + 1), dtype=FLOAT)
        dEdWf = np.zeros((self.memoryDim, self.inputDim + 2 * self.memoryDim + 1), dtype=FLOAT)
        dEdWc = np.zeros((self.memoryDim, self.inputDim + self.memoryDim + 1), dtype=FLOAT)
        dEdWo = np.zeros((self.memoryDim, self.inputDim + 2 * self.memoryDim + 1), dtype=FLOAT)

        # (j -> T, k -> t)
        dYTdYt = np.ones((self.memoryDim, self.memoryDim), dtype=FLOAT)
        dYTdCt = np.zeros((self.memoryDim, self.memoryDim), dtype=FLOAT)

        timespan_x = timespan
        if self.cutOffZeroEnd:
            for t in range(0, timespan):
                if self.needsCutOff(X, t):
                    timespan_x = t
                    break

        for t in reversed(range(0, timespan_x)):
            if t == 0:
                Yt1 = np.zeros(self.memoryDim, dtype=FLOAT)
                Ct1 = np.zeros(self.memoryDim, dtype=FLOAT)
            else:
                Yt1 = Y[t-1]
                Ct1 = C[t-1]

            states1 = np.concatenate((X[t], Yt1, Ct1, np.ones(1, dtype=FLOAT)))
            states2 = np.concatenate((X[t], Yt1, np.ones(1, dtype=FLOAT)))
            states3 = np.concatenate((X[t], Yt1, C[t], np.ones(1, dtype=FLOAT)))

            dEdYT = dEdY[timespan - 1]

            # (k -> t)
            U = np.tanh(C[t])
            dU = 1 - np.power(U, 2)
            dZ = 1 - np.power(Z[t], 2)

            dGi = Gi[t] * (1 - Gi[t])
            dGf = Gf[t] * (1 - Gf[t])
            dGo = Go[t] * (1 - Go[t])

            # (k, l)
            dYtdCt = (Go[t] * dU) * np.eye(self.memoryDim) + \
                     (U * dGo).reshape((self.memoryDim, 1)) * Wco

            # (T, t)
            if t == timespan_x - 1:
                dYTdCt = dYtdCt
            else:
                dYTdCt = np.dot(dYTdCt, dCtdCt1) + np.dot(dYTdYt, dYtdCt)

            dYTdGi = dYTdCt * Z[t]
            dEdGi = np.dot(dEdYT, dYTdGi)
            dEdWi += np.outer(dEdGi * dGi, states1)

            dYTdGf = dYTdCt * Ct1
            dEdGf = np.dot(dEdYT, dYTdGf)
            dEdWf += np.outer(dEdGf * dGf, states1)

            dYTdZ = dYTdCt * Gi[t]
            dEdZ = np.dot(dEdYT, dYTdZ)
            dEdWc += np.outer(dEdZ * dZ, states2)

            if t == timespan_x - 1:
                dYTdGo = U
                dEdGo = dEdYT * dYTdGo
            else:
                dYTdGo = dYTdYt * U
                dEdGo = np.dot(dEdYT, dYTdGo)
            dEdWo += np.outer(dEdGo * dGo, states3)

            # (k -> t, l -> t-1)
            dCtdCt1 = (Ct1 * dGf).reshape((self.memoryDim, 1)) * Wcf + \
                      Gf[t] * np.eye(self.memoryDim) + \
                      (Z[t] * dGi).reshape((self.memoryDim, 1)) * Wci
            dCtdYt1 = (Ct1 * dGf).reshape((self.memoryDim, 1)) * Wyf + \
                      (Gi[t] * dZ).reshape((self.memoryDim, 1)) * Wyc + \
                      (Z[t] * dGi).reshape((self.memoryDim, 1)) * Wyi
            dYtdYt1 = (U * dGo).reshape((self.memoryDim, 1)) * Wyo

            # (j -> T, l -> t-1) = (j -> T, k -> t) * (k -> t, l -> t-1)
            dYTdYt = np.dot(dYTdYt, dYtdYt1) + np.dot(dYTdCt, dCtdYt1)

        dEdW = np.concatenate((dEdWi, dEdWf, dEdWc, dEdWo), axis=1)
        return dEdW, 0

    def backPropagate_(self, dEdY, X, Y, C, Z, Gi, Gf, Go, outputdEdX):
        timespan = Y.shape[0]
        Wxi, Wyi, Wci, Wxf, Wyf, Wcf, Wxc, Wyc, Wxo, Wyo, Wco = \
            self.sliceWeightsSmall(self.inputDim, self.memoryDim, self.W)
        dEdW = np.zeros(self.W.shape, dtype=FLOAT)

        # Calculate dEdW
        # dY_i__dW_kl_(t-1) -> (k, l, i)
        dYdW_t1 = np.zeros((self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, self.memoryDim), dtype=FLOAT)
        dCdW_t1 = np.zeros((self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, self.memoryDim), dtype=FLOAT)

        for t in range(0, timespan):
            if self.cutOffZeroEnd and self.needsCutOff(X, t):
                break
            if t == 0:
                Yt1 = np.zeros(self.memoryDim, dtype=FLOAT)
                Ct1 = np.zeros(self.memoryDim, dtype=FLOAT)
            else:
                Yt1 = Y[t-1, :]
                Ct1 = C[t-1, :]
            states1 = np.concatenate((X[t, :], Yt1, Ct1, np.ones(1, dtype=FLOAT)))
            states2 = np.concatenate((X[t, :], Yt1, np.ones(1, dtype=FLOAT)))
            states3 = np.concatenate((X[t, :], Yt1, C[t, :], np.ones(1, dtype=FLOAT)))

            dGi_i__dW_kl = np.inner(dYdW_t1, Wyi) + \
                           np.inner(dCdW_t1, Wci)
            dGi_i__dW_kl += np.eye(self.memoryDim)\
                            .reshape(self.memoryDim, 1, self.memoryDim) * \
                            np.concatenate((
                            states1.reshape(1, states1.size, 1),
                            np.zeros((1, states1.size + \
                            states2.size + states3.size, 1), dtype=FLOAT)),
                            axis=1)
            dGi_i__dW_kl *= Gi[t, :] * (1 - Gi[t, :])
            dGf_i__dW_kl = np.inner(dYdW_t1, Wyf) + \
                           np.inner(dCdW_t1, Wcf)
            dGf_i__dW_kl += np.eye(self.memoryDim)\
                            .reshape(self.memoryDim, 1, self.memoryDim) * \
                            np.concatenate((
                            np.zeros((1, states1.size, 1), dtype=FLOAT),
                            states1.reshape(1, states1.size, 1),
                            np.zeros((1, states2.size + \
                            states3.size, 1), dtype=FLOAT)),
                            axis=1)
            dGf_i__dW_kl *= Gf[t, :] * (1 - Gf[t, :])
            dZ_i__dW_kl = np.inner(dYdW_t1, Wyc)
            dZ_i__dW_kl += np.eye(self.memoryDim)\
                           .reshape(self.memoryDim, 1, self.memoryDim) * \
                           np.concatenate((
                           np.zeros((1, states1.size * 2, 1), dtype=FLOAT),
                           states2.reshape(1, states2.size, 1),
                           np.zeros((1, states3.size, 1), dtype=FLOAT)),
                           axis=1)
            dZ_i__dW_kl *= 1 - np.power(Z[t, :], 2)
            dCdW_t1 = dGf_i__dW_kl * Ct1 + \
                               Gf[t, :] * dCdW_t1 + \
                               dGi_i__dW_kl * Z[t, :] + \
                               Gi[t, :] * dZ_i__dW_kl
            dGo_i__dW_kl = np.inner(dYdW_t1, Wyo) + \
                           np.inner(dCdW_t1, Wco)
            dGo_i__dW_kl += np.eye(self.memoryDim)\
                            .reshape(self.memoryDim, 1, self.memoryDim) * \
                            np.concatenate((
                            np.zeros((1, states1.size * 2 + \
                            states2.size, 1), dtype=FLOAT),
                            states3.reshape(1, states3.size, 1)),
                            axis=1)
            dGo_i__dW_kl *= Go[t, :] * (1 - Go[t, :])
            U = np.tanh(C[t, :])
            dYdW_t1 = dGo_i__dW_kl * U + \
                      Go[t, :] * (1 - np.power(U, 2)) * dCdW_t1

            if not self.cutOffZeroEnd:
                dEdW += np.dot(dYdW_t1, dEdY[t, :])

        if self.cutOffZeroEnd:
            dEdW = np.dot(dYdW_t1, dEdY[timespan - 1, :])

        # Calculate dEdX
        # dY_ti__dXtau_j -> (t, tau, j, i)
        if outputdEdX:
            dYdX_t1 = np.zeros((timespan, self.inputDim, self.memoryDim), dtype=FLOAT)
            dCdX_t1 = np.zeros((timespan, self.inputDim, self.memoryDim), dtype=FLOAT)
            dEdX = np.zeros(X.shape, dtype=FLOAT)

            for t in range(0, timespan):
                if self.cutOffZeroEnd and self.needsCutOff(X, t):
                        break
                if t == 0:
                    Ct1 = np.zeros(self.memoryDim, dtype=FLOAT)
                else:
                    Ct1 = C[t-1, :]

                dGi_i__dX_tj = np.inner(dYdX_t1, Wyi) + \
                               np.inner(dCdX_t1, Wci)
                dGi_i__dX_tj[t, :, :] += np.transpose(Wxi)
                dGi_i__dX_tj *= Gi[t, :] * (1 - Gi[t, :])
                dGf_i__dX_tj = np.inner(dYdX_t1, Wyf) + \
                               np.inner(dCdX_t1, Wcf)
                dGf_i__dX_tj[t, :, :] += np.transpose(Wxf)
                dGf_i__dX_tj *= Gf[t, :] * (1 - Gf[t, :])
                dZ_i__dX_tj = np.inner(dYdX_t1, Wyc)
                dZ_i__dX_tj[t, :, :] += np.transpose(Wxc)
                dZ_i__dX_tj *= 1 - np.power(Z[t, :], 2)
                dCdX_t1 = dGf_i__dX_tj * Ct1 + \
                                   Gf[t, :] * dCdX_t1 + \
                                   dGi_i__dX_tj * Z[t, :] + \
                                   Gi[t, :] * dZ_i__dX_tj
                dGo_i__dX_tj = np.inner(dYdX_t1, Wyo) + \
                               np.inner(dCdX_t1, Wco)
                dGo_i__dX_tj[t, :, :] += np.transpose(Wxo)
                dGo_i__dX_tj *= Go[t, :] * (1 - Go[t, :])

                U = np.tanh(C[t, :])
                dYdX_t1 = dGo_i__dX_tj * U + \
                                   Go[t, :] * (1 - np.power(U, 2)) * dCdX_t1

                if not self.cutOffZeroEnd:
                    dEdX += np.dot(dYdX_t1, dEdY[t, :])

            if self.cutOffZeroEnd:
                dEdX = np.dot(dYdX_t1, dEdY[timespan - 1, :])
        else:
            dEdX = 0

        return dEdW, dEdX

    def backPropagateN(self, dEdY, outputdEdX):
        numEx = self.X.shape[1]
        dEdW = np.zeros(self.W.shape, dtype=FLOAT)
        dEdX = np.zeros(self.X.shape, dtype=FLOAT)
        X = self.X
        Y = self.Y
        C = self.C
        Z = self.Z
        Gi = self.Gi
        Gf = self.Gf
        Go = self.Go
        for n in range(0, numEx):
            dEdWtmp, dEdX[:, n, :] = \
                self.backPropagate_new(
                    dEdY[:, n, :], X[:, n, :], Y[:, n, :],
                    C[:, n, :], Z[:, n, :], Gi[:, n, :],
                    Gf[:, n, :], Go[:, n, :], outputdEdX)
            dEdW += dEdWtmp

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
        inputDim=1,
        memoryDim=1,
        initRange=0.01,
        initSeed=2)
    lstm.chkgrd()