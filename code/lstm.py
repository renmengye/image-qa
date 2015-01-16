from scipy import special
from util_func import *

class LSTM:
    def __init__(self,
                 inputDim,
                 memoryDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 W=0,
                 cutOffZeroEnd=False,
                 dropoutRate=0.0):
        self.inputDim = inputDim
        self.memoryDim = memoryDim
        self.cutOffZeroEnd = cutOffZeroEnd
        self.dropoutRate = dropoutRate

        if needInit:
            np.random.seed(initSeed)
            Wxi = np.random.rand(self.memoryDim, self.inputDim) * initRange
            Wxf = np.random.rand(self.memoryDim, self.inputDim) * initRange
            Wxc = np.random.rand(self.memoryDim, self.inputDim) * initRange
            Wxo = np.random.rand(self.memoryDim, self.inputDim) * initRange
            Wyi = np.random.rand(self.memoryDim, self.memoryDim) * initRange
            Wyf = np.random.rand(self.memoryDim, self.memoryDim) * initRange
            Wyc = np.random.rand(self.memoryDim, self.memoryDim) * initRange
            Wyo = np.random.rand(self.memoryDim, self.memoryDim) * initRange
            Wci = np.random.rand(self.memoryDim, self.memoryDim) * initRange
            Wcf = np.random.rand(self.memoryDim, self.memoryDim) * initRange
            Wco = np.random.rand(self.memoryDim, self.memoryDim) * initRange
            # Wbi = np.random.rand(self.memoryDim, 1) * initRange
            # Wbf = np.random.rand(self.memoryDim, 1) * initRange
            # Wbc = np.random.rand(self.memoryDim, 1) * initRange
            # Wbo = np.random.rand(self.memoryDim, 1) * initRange
            Wbi = np.ones((self.memoryDim, 1), float)
            Wbf = np.ones((self.memoryDim, 1), float)
            Wbc = np.zeros((self.memoryDim, 1), float)
            Wbo = np.ones((self.memoryDim, 1), float)

            Wi = np.concatenate((Wxi, Wyi, Wci, Wbi), axis=1)
            Wf = np.concatenate((Wxf, Wyf, Wcf, Wbf), axis=1)
            Wc = np.concatenate((Wxc, Wyc, Wbc), axis=1)
            Wo = np.concatenate((Wxo, Wyo, Wco, Wbo), axis=1)
            self.W = np.concatenate((Wi, Wf, Wc, Wo), axis = 1)
            #self.W = np.random.rand(self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4) * initRange - initRange / 2.0
        else:
            self.W = W

        self.X = 0
        self.Y = 0
        self.C = 0
        self.Z = 0
        self.Gi = 0
        self.Gf = 0
        self.Go = 0
        self.dropoutVec = 0
        self.dropout = False
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
    def needCutOff(X, t):
        cut = True
        for i in range(0, X.shape[1]):
            if X[t, i] != 0.0:
                cut = False
        return cut

    def forwardPass_(self, X, dropout, initDropout):
        if len(X.shape) == 3:
            return self.forwardPassN(X, dropout)
        timespan = X.shape[0]
        Y = np.zeros((timespan, self.memoryDim), float)
        C = np.zeros((timespan, self.memoryDim), float)
        Z = np.zeros((timespan, self.memoryDim), float)
        Gi = np.zeros((timespan, self.memoryDim), float)
        Gf = np.zeros((timespan, self.memoryDim), float)
        Go = np.zeros((timespan, self.memoryDim), float)
        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)

        # Dropout
        if self.dropoutRate > 0.0 and dropout:
            if initDropout:
                self.dropoutVec = (np.random.rand(self.memoryDim) > self.dropoutRate)
            for i in range(0, self.memoryDim):
                if self.dropoutVec[i]:
                    Wc[i, :] = 0

        for t in range(0, timespan):
            if self.cutOffZeroEnd and self.needCutOff(X, t):
                for t1 in range(t, timespan):
                    Y[t1, :] = Y[t1-1, :]
                break

            # In forward pass initial stage -1 is empty, equivalent to zero.
            # Need to explicitly pass zero in backward pass.
            states1 = np.concatenate((X[t, :], Y[t-1, :], C[t-1, :], np.ones(1, float)))
            states2 = np.concatenate((X[t, :], Y[t-1, :], np.ones(1, float)))
            Gi[t, :] = special.expit(np.dot(Wi, states1))
            Gf[t, :] = special.expit(np.dot(Wf, states1))
            Z[t, :] = np.tanh(np.dot(Wc, states2))
            C[t, :] = Gf[t, :] * C[t-1, :] + Gi[t, :] * Z[t, :]
            states3 = np.concatenate((X[t, :], Y[t-1, :], C[t, :], np.ones(1, float)))
            Go[t, :] = special.expit(np.dot(Wo, states3))
            Y[t, :] = Go[t, :] * np.tanh(C[t, :])

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go
        self.dropout = dropout

        return Y, C, Z, Gi, Gf, Go

    def forwardPass(self, X, dropout=False):
        self.forwardPass_(X, dropout, initDropout=True)
        return self.Y

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
        timespan = Y.shape[0]
        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)

        # Dropout
        if self.dropoutRate > 0.0 and self.dropout:
            for i in range(0, self.memoryDim):
                if self.dropoutVec[i]:
                    Wc[i, :] = 0

        Wxi = Wi[:, 0 : self.inputDim]
        Wyi = Wi[:, self.inputDim : self.inputDim + self.memoryDim]
        Wci = Wi[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wxf = Wf[:, 0 : self.inputDim]
        Wyf = Wf[:, self.inputDim : self.inputDim + self.memoryDim]
        Wcf = Wf[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wxc = Wc[:, 0 : self.inputDim]
        Wyc = Wc[:, self.inputDim : self.inputDim + self.memoryDim]
        Wxo = Wo[:, 0 : self.inputDim]
        Wyo = Wo[:, self.inputDim : self.inputDim + self.memoryDim]
        Wco = Wo[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]

        # Calculate dEdW
        # dY_ti__dW_kl -> (t, k, l, j)
        dYdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, self.memoryDim), float)
        dCdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, self.memoryDim), float)

        for t in range(0, timespan):
            if self.cutOffZeroEnd and self.needCutOff(X, t):
                break
            if t == 0:
                Yt1 = np.zeros(self.memoryDim, float)
                Ct1 = np.zeros(self.memoryDim, float)
            else:
                Yt1 = Y[t-1, :]
                Ct1 = C[t-1, :]
            states1 = np.concatenate((X[t, :], Yt1, Ct1, np.ones(1, float)))
            states2 = np.concatenate((X[t, :], Yt1, np.ones(1, float)))
            states3 = np.concatenate((X[t, :], Yt1, C[t, :], np.ones(1, float)))

            dGi_i__dW_kl = np.inner(dYdW[t-1, :, :, :], Wyi) + \
                           np.inner(dCdW[t-1, :, :, :], Wci)
            dGi_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, self.memoryDim) * \
                            np.concatenate((
                            states1.reshape(1, states1.size, 1),
                            np.zeros((1, states1.size + states2.size + states3.size, 1), float)),
                            axis=1)
            dGi_i__dW_kl *= Gi[t, :] * (1 - Gi[t, :])
            dGf_i__dW_kl = np.inner(dYdW[t-1, :, :, :], Wyf) + \
                           np.inner(dCdW[t-1, :, :, :], Wcf)
            dGf_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, self.memoryDim) * \
                            np.concatenate((
                            np.zeros((1, states1.size, 1), float),
                            states1.reshape(1, states1.size, 1),
                            np.zeros((1, states2.size + states3.size, 1), float)),
                            axis=1)
            dGf_i__dW_kl *= Gf[t, :] * (1 - Gf[t, :])
            dZ_i__dW_kl = np.inner(dYdW[t-1, :, :, :], Wyc)
            dZ_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, self.memoryDim) * \
                           np.concatenate((
                           np.zeros((1, states1.size * 2, 1), float),
                           states2.reshape(1, states2.size, 1),
                           np.zeros((1, states3.size, 1), float)),
                           axis=1)
            dZ_i__dW_kl *= 1 - np.power(Z[t, :], 2)
            dCdW[t, :, :, :] = dGf_i__dW_kl * Ct1 + \
                               Gf[t, :] * dCdW[t-1, :, :, :] + \
                               dGi_i__dW_kl * Z[t, :] + \
                               Gi[t, :] * dZ_i__dW_kl
            dGo_i__dW_kl = np.inner(dYdW[t-1, :, :, :], Wyo) + \
                           np.inner(dCdW[t, :, :, :], Wco)
            dGo_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, self.memoryDim) * \
                            np.concatenate((
                            np.zeros((1, states1.size * 2 + states2.size, 1), float),
                            states3.reshape(1, states3.size, 1)),
                            axis=1)
            dGo_i__dW_kl *= Go[t, :] * (1 - Go[t, :])
            U = np.tanh(C[t, :])
            dYdW[t, :, :, :] = dGo_i__dW_kl * U + \
                               Go[t, :] * (1 - np.power(U, 2)) * dCdW[t, :, :, :]

        dEdW = np.tensordot(dYdW, dEdY, axes=([3, 0], [1, 0]))

        # Dropout
        if self.dropoutRate > 0.0 and self.dropout:
            for i in range(0, self.memoryDim):
                if self.dropoutVec[i]:
                    dEdW[i, 2 * self.inputDim + 4 * self.memoryDim + 2 : 3 * self.inputDim + 5 * self.memoryDim + 3] = 0

        # Calculate dEdX
        # dY_ti__dXtau_j -> (t, tau, j, i)
        if outputdEdX:
            dYdX = np.zeros((timespan, timespan, self.inputDim, self.memoryDim), float)
            dCdX = np.zeros((timespan, timespan, self.inputDim, self.memoryDim), float)

            for t in range(0, timespan):
                if self.cutOffZeroEnd and self.needCutOff(X, t):
                    break
                if t == 0:
                    Ct1 = np.zeros(self.memoryDim, float)
                else:
                    Ct1 = C[t-1, :]

                dGi_i__dX_tj = np.inner(dYdX[t-1, :, :, :], Wyi) + \
                               np.inner(dCdX[t-1, :, :, :], Wci)
                dGi_i__dX_tj[t, :, :] += np.transpose(Wxi)
                dGi_i__dX_tj *= Gi[t, :] * (1 - Gi[t, :])
                dGf_i__dX_tj = np.inner(dYdX[t-1, :, :, :], Wyf) + \
                               np.inner(dCdX[t-1, :, :, :], Wcf)
                dGf_i__dX_tj[t, :, :] += np.transpose(Wxf)
                dGf_i__dX_tj *= Gf[t, :] * (1 - Gf[t, :])
                dZ_i__dX_tj = np.inner(dYdX[t-1, :, :, :], Wyc)
                dZ_i__dX_tj[t, :, :] += np.transpose(Wxc)
                dZ_i__dX_tj *= 1 - np.power(Z[t, :], 2)
                dCdX[t, :, :, :] = dGf_i__dX_tj * Ct1 + \
                                   Gf[t, :] * dCdX[t-1, :, :, :] + \
                                   dGi_i__dX_tj * Z[t, :] + \
                                   Gi[t, :] * dZ_i__dX_tj
                dGo_i__dX_tj = np.inner(dYdX[t-1, :, :, :], Wyo) + \
                               np.inner(dCdX[t, :, :, :], Wco)
                dGo_i__dX_tj[t, :, :] += np.transpose(Wxo)
                dGo_i__dX_tj *= Go[t, :] * (1 - Go[t, :])

                U = np.tanh(C[t, :])
                dYdX[t, :, :, :] = dGo_i__dX_tj * U + \
                                   Go[t, :] * (1 - np.power(U, 2)) * dCdX[t, :, :, :]

            dEdX = np.tensordot(dYdX, dEdY, axes=([3, 0], [1, 0]))
        else:
            dEdX = 0

        return dEdW, dEdX

    def forwardPassN(self, X, dropout):
        # X[t, n, i] -> t: time, n: example, i: input dimension
        timespan = X.shape[0]
        numEx = X.shape[1]
        Y = np.zeros((timespan, numEx, self.memoryDim), float)
        C = np.zeros((timespan, numEx, self.memoryDim), float)
        Z = np.zeros((timespan, numEx, self.memoryDim), float)
        Gi = np.zeros((timespan, numEx, self.memoryDim), float)
        Gf = np.zeros((timespan, numEx, self.memoryDim), float)
        Go = np.zeros((timespan, numEx, self.memoryDim), float)

        # Dropout
        if self.dropoutRate > 0.0 and dropout:
            self.dropoutVec = (np.random.rand(self.memoryDim) > self.dropoutRate)

        for n in range(0, numEx):
            Y[:, n, :], C[:, n, :], Z[:, n, :], \
            Gi[:, n, :], Gf[:, n, :], Go[:, n, :] = self.forwardPass_(X[:, n, :], dropout=True, initDropout=False)

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go
        self.dropout = dropout

        return Y

    def backPropagateN(self, dEdY, outputdEdX):
        timespan = self.X.shape[0]
        numEx = self.X.shape[1]
        dEdW = np.zeros((self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4), float)
        dEdX = np.zeros((timespan, numEx, self.inputDim), float)
        X = self.X
        Y = self.Y
        C = self.C
        Z = self.Z
        Gi = self.Gi
        Gf = self.Gf
        Go = self.Go
        for n in range(0, numEx):
            self.X = X[:, n, :]
            self.Y = Y[:, n, :]
            self.C = C[:, n, :]
            self.Z = Z[:, n, :]
            self.Gi = Gi[:, n, :]
            self.Gf = Gf[:, n, :]
            self.Go = Go[:, n, :]
            dEdWtmp, dEdX[:, n, :] = self.backPropagate(dEdY[:, n, :], outputdEdX)
            dEdW += dEdWtmp

        return dEdW, dEdX

    def forwardPassAll(self, X, dropout):
        # X[t, n, i] -> t: time, n: example, i: input dimension
        timespan = X.shape[0]
        numEx = X.shape[1]
        Y = np.zeros((timespan, numEx, self.memoryDim), float)
        C = np.zeros((timespan, numEx, self.memoryDim), float)
        Z = np.zeros((timespan, numEx, self.memoryDim), float)
        Gi = np.zeros((timespan, numEx, self.memoryDim), float)
        Gf = np.zeros((timespan, numEx, self.memoryDim), float)
        Go = np.zeros((timespan, numEx, self.memoryDim), float)
        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)

        if self.dropoutRate > 0.0 and dropout:
            self.dropoutVec = (np.random.rand(self.memoryDim) > self.dropoutRate)
            for i in range(0, self.memoryDim):
                if self.dropoutVec[i]:
                    Wc[i, :] = 0

        for t in range(0, timespan):
            # In forward pass initial stage -1 is empty, equivalent to zero.
            # Need to explicitly pass zero in backward pass.
            states1 = np.concatenate((X[t, :, :], Y[t-1, :, :], C[t-1, :, :], np.ones((numEx, 1), float)), axis=-1)
            states2 = np.concatenate((X[t, :, :], Y[t-1, :, :], np.ones((numEx, 1), float)), axis=-1)
            Gi[t, :, :] = special.expit(np.inner(states1, Wi))
            Gf[t, :, :] = special.expit(np.inner(states1, Wf))
            Z[t, :, :] = np.tanh(np.inner(states2, Wc))
            C[t, :, :] = Gf[t, :, :] * C[t-1, :, :] + Gi[t, :, :] * Z[t, :, :]
            states3 = np.concatenate((X[t, :, :], Y[t-1, :, :], C[t, :, :], np.ones((numEx, 1), float)), axis=-1)
            Go[t, :, :] = special.expit(np.inner(states3, Wo))
            Y[t, :, :] = Go[t, :, :] * np.tanh(C[t, :, :])

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go

        return Y

    def backPropagateAll(self, dEdY, outputdEdX):
        X = self.X
        Y = self.Y
        C = self.C
        Z = self.Z
        Gi = self.Gi
        Gf = self.Gf
        Go = self.Go
        timespan = Y.shape[0]
        numEx = Y.shape[1]

        # (t, k, l, n, i)
        dYdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, numEx, self.memoryDim), float)
        dCdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, numEx, self.memoryDim), float)

        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)

        # Dropout
        if self.dropoutRate > 0.0 and self.dropout:
            for i in range(0, self.memoryDim):
                if self.dropoutVec[i]:
                    Wc[i, :] = 0

        Wxi = Wi[:, 0 : self.inputDim]
        Wyi = Wi[:, self.inputDim : self.inputDim + self.memoryDim]
        Wci = Wi[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wxf = Wf[:, 0 : self.inputDim]
        Wyf = Wf[:, self.inputDim : self.inputDim + self.memoryDim]
        Wcf = Wf[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wxc = Wc[:, 0 : self.inputDim]
        Wyc = Wc[:, self.inputDim : self.inputDim + self.memoryDim]
        Wxo = Wo[:, 0 : self.inputDim]
        Wyo = Wo[:, self.inputDim : self.inputDim + self.memoryDim]
        Wco = Wo[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]

        for t in range(0, timespan):
            if t == 0:
                Yt1 = np.zeros((numEx, self.memoryDim), float)
                Ct1 = np.zeros((numEx, self.memoryDim), float)
            else:
                Yt1 = Y[t-1, :, :]
                Ct1 = C[t-1, :, :]
            states1 = np.concatenate((X[t, :, :], Yt1, Ct1, np.ones((numEx, 1), float)), axis=1)
            states2 = np.concatenate((X[t, :, :], Yt1, np.ones((numEx, 1), float)), axis=1)
            states3 = np.concatenate((X[t, :, :], Yt1, C[t, :, :], np.ones((numEx, 1), float)), axis=1)

            # W
            dGi_i__dW_kl = np.inner(dYdW[t-1, :, :, :, :], Wyi) + \
                           np.inner(dCdW[t-1, :, :, :, :], Wci)
            dGi_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, 1, self.memoryDim) * \
                            np.concatenate((
                            states1.transpose().reshape(1, states1.shape[-1], numEx, 1),
                            np.zeros((1, states1.shape[-1] + states2.shape[-1] + states3.shape[-1], numEx,  1), float)),
                            axis=1)
            dGi_i__dW_kl *= Gi[t, :, :] * (1 - Gi[t, :, :])
            dGf_i__dW_kl = np.inner(dYdW[t-1, :, :, :, :], Wyf) + \
                           np.inner(dCdW[t-1, :, :, :, :], Wcf)
            dGf_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, 1, self.memoryDim) * \
                            np.concatenate((
                            np.zeros((1, states1.shape[-1], numEx, 1), float),
                            states1.transpose().reshape(1, states1.shape[-1], numEx, 1),
                            np.zeros((1, states2.shape[-1] + states3.shape[1], numEx, 1), float)),
                            axis=1)
            dGf_i__dW_kl *= Gf[t, :, :] * (1 - Gf[t, :, :])
            dZ_i__dW_kl = np.inner(dYdW[t-1, :, :, :, :], Wyc)
            dZ_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, 1, self.memoryDim) * \
                           np.concatenate((
                           np.zeros((1, states1.shape[-1] * 2, numEx, 1), float),
                           states2.transpose().reshape(1, states2.shape[-1], numEx, 1),
                           np.zeros((1, states3.shape[-1], numEx, 1), float)),
                           axis=1)
            dZ_i__dW_kl *= 1 - np.power(Z[t, :, :], 2)
            dCdW[t, :, :, :] = dGf_i__dW_kl * Ct1 + \
                               Gf[t, :, :] * dCdW[t-1, :, :, :, :] + \
                               dGi_i__dW_kl * Z[t, :, :] + \
                               Gi[t, :, :] * dZ_i__dW_kl
            dGo_i__dW_kl = np.inner(dYdW[t-1, :, :, :, :], Wyo) + \
                           np.inner(dCdW[t, :, :, :, :], Wco)
            dGo_i__dW_kl += np.eye(self.memoryDim).reshape(self.memoryDim, 1, 1, self.memoryDim) * \
                            np.concatenate((
                            np.zeros((1, states1.shape[-1] * 2 + states2.shape[-1], numEx, 1), float),
                            states3.transpose().reshape(1, states3.shape[-1], numEx, 1)),
                            axis=1)
            dGo_i__dW_kl *= Go[t, :, :] * (1 - Go[t, :, :])
            U = np.tanh(C[t, :, :])
            dYdW[t, :, :, :, :] = dGo_i__dW_kl * U + \
                                  Go[t, :, :] * (1 - np.power(U, 2)) * dCdW[t, :, :, :, :]

        dEdW = np.tensordot(dYdW, dEdY, axes=([4, 3, 0], [2, 1, 0]))

        # Dropout
        if self.dropoutRate > 0.0 and self.dropout:
            for i in range(0, self.memoryDim):
                if self.dropoutVec[i]:
                    dEdW[i, 2 * self.inputDim + 4 * self.memoryDim + 2 : 3 * self.inputDim + 5 * self.memoryDim + 3] = 0

        # Calculate dEdX
        if outputdEdX:
            # (t, tau, n, j, i)
            # (t, tau, j, n, i)
            dYdX = np.zeros((timespan, timespan, self.inputDim, numEx, self.memoryDim), float)
            dCdX = np.zeros((timespan, timespan, self.inputDim, numEx, self.memoryDim), float)

            for t in range(0, timespan):
                if t == 0:
                    Ct1 = np.zeros(self.memoryDim, float)
                else:
                    Ct1 = C[t-1, :]

                dGi_i__dX_tj = np.inner(dYdX[t-1, :, :, :, :], Wyi) + \
                               np.inner(dCdX[t-1, :, :, :, ], Wci)
                dGi_i__dX_tj[t, :, :] += np.transpose(Wxi).reshape(self.inputDim, 1, self.memoryDim)
                dGi_i__dX_tj *= Gi[t, :, :] * (1 - Gi[t, :, :])
                dGf_i__dX_tj = np.inner(dYdX[t-1, :, :, :, :], Wyf) + \
                               np.inner(dCdX[t-1, :, :, :, :], Wcf)
                dGf_i__dX_tj[t, :, :] += np.transpose(Wxf).reshape(self.inputDim, 1, self.memoryDim)
                dGf_i__dX_tj *= Gf[t, :, :] * (1 - Gf[t, :, :])
                dZ_i__dX_tj = np.inner(dYdX[t-1, :, :, :, :], Wyc)
                dZ_i__dX_tj[t, :, :] += np.transpose(Wxc).reshape(self.inputDim, 1, self.memoryDim)
                dZ_i__dX_tj *= 1 - np.power(Z[t, :, :], 2)
                dCdX[t, :, :, :, :] = dGf_i__dX_tj * Ct1 + \
                                      Gf[t, :] * dCdX[t-1, :, :, :, :] + \
                                      dGi_i__dX_tj * Z[t, :] + \
                                      Gi[t, :] * dZ_i__dX_tj
                dGo_i__dX_tj = np.inner(dYdX[t-1, :, :, :, :], Wyo) + \
                               np.inner(dCdX[t, :, :, :, :], Wco)
                dGo_i__dX_tj[t, :, :] += np.transpose(Wxo).reshape(self.inputDim, 1, self.memoryDim)
                dGo_i__dX_tj *= Go[t, :, :] * (1 - Go[t, :, :])

                U = np.tanh(C[t, :])
                dYdX[t, :, :, :, :] = dGo_i__dX_tj * U + \
                                      Go[t, :] * (1 - np.power(U, 2)) * dCdX[t, :, :, :, :]

            # dYdX -> (t, tau, j, n, i) * dEdY -> (t, n, i) -> (tau, j, n, n)
            dEdX = np.diagonal(np.tensordot(dYdX, dEdY, axes=([4, 0], [2, 0])), axis1=2, axis2=3).transpose((0, 2, 1))
        else:
            dEdX = 0

        # dEdW -> (k, l)
        # dEdX -> (tau, n, j)
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

if __name__ == '__main__':
    lstm = LSTM(
        inputDim=1,
        memoryDim=1,
        initRange=0.01,
        initSeed=2)
    lstm.chkgrd()