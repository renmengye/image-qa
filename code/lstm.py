import time
import numpy as np
from scipy import special

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class LSTM:
    def __init__(self, inputDim, memoryDim, initRange=1.0, initSeed=2, needInit=True, W=0):
        self.inputDim = inputDim
        self.memoryDim = memoryDim

        if needInit:
            np.random.seed(initSeed)
            self.W = np.random.rand(self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4) * initRange - initRange / 2.0
        else:
            self.W = W

    def chkgrd(self):
        data = np.array([[0], [1], [0], [1], [0], [1], [0], [1]])
        target = np.array([[0], [0], [0], [1], [1], [1], [0], [1]])
        Y, E, dEdW = self.runAndBackOnce(data, target, simpleSumDeriv)
        eps = 1e-3
        dEdWTmp = np.zeros(self.W.shape)
        for i in range(0, self.W.shape[0]):
            for j in range(0, self.W.shape[1]):
                self.W[i,j] += eps
                Y, C, Z, Gi, Gf, Go = self.forwardPass(data)
                Etmp1, d1 = self.simpleSumDeriv(target, Y)

                self.W[i,j] -= 2 * eps
                Y, C, Z, Gi, Gf, Go = self.forwardPass(data)
                Etmp2, d2 = self.simpleSumDeriv(target, Y)

                dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                self.W[i,j] += eps
        print "haha"
        pass

    def train(self, trainInput, trainTarget, trainOpt):
        needValid =  trainOpt['needValid']
        if needValid:
            trainInput, trainTarget, validInput, validTarget = self.splitData(trainInput, trainTarget)
        numEpoch = trainOpt['numEpoch']
        lr = trainOpt['learningRate']
        lrDecay = trainOpt['learningRateDecay']
        mom = trainOpt['momentum']
        combineFnDeriv = trainOpt['combineFnDeriv']
        decisionFn = trainOpt['decisionFn']
        calcError = trainOpt['calcError']
        bat = trainOpt['batchSize']
        N = trainInput.shape[0]
        lastdW = np.zeros(self.W.shape, float)
        dMom = (mom - trainOpt['momentumEnd']) / float(numEpoch)

        Etotal = np.zeros(numEpoch, float)
        VEtotal = np.zeros(numEpoch, float)
        Rtotal = np.zeros(numEpoch, float)
        VRtotal = np.zeros(numEpoch, float)

        startTime = time.time()

        # Make num ex the second dimension for minibatch.
        if bat > 1:
            trainInputPerm = np.transpose(trainInput, (1, 0, 2))
            trainTargetPerm = np.transpose(trainTarget, (1, 0, 2))
        if needValid:
            validInputPerm = np.transpose(validInput, (1, 0, 2))
            validTargetPerm = np.transpose(validTarget, (1, 0, 2))

        # Train loop through epochs
        for epoch in range(0, numEpoch):
            E = 0
            rate = 0
            VE = 0
            Vrate = 0
            correct = 0
            total = 0
            n = 0

            # Stochastic
            if bat == 1:
                for n in range(0, N):
                    X = trainInput[n, :, :]
                    T = trainTarget[n, :, :]
                    Y, Etmp, dEdW = self.runAndBackOnce(X, T, combineFnDeriv)
                    E += Etmp / float(N)
                    self.W = self.W - lr * dEdW + mom * lastdW
                    lastdW = -lr * dEdW
                    if calcError:
                        rate_, correct_, total_ = self.calcRate(T, Y, decisionFn)
                        correct += correct_
                        total += total_

            # Mini-batch (using for-loop now, so need to differentiate)
            else:
                while n < N:
                    batchEnd = min(N, n + bat)
                    numEx = batchEnd - n
                    X = trainInputPerm[:, n:batchEnd, :]
                    T = trainTargetPerm[:, n:batchEnd, :]
                    Y, Etmp, dEdW = self.runAndBackAll(X, T, combineFnDeriv)

                    E += np.sum(Etmp) / float(N)
                    dEdW = np.sum(dEdW, axis=0) / float(numEx)
                    self.W = self.W - lr * dEdW + mom * lastdW
                    lastdW = -lr * dEdW
                    if calcError:
                        rate_, correct_, total_ = self.calcRate(T, Y, decisionFn)
                        correct += correct_
                        total += total_
                    n += bat

            # Store train statistics
            if calcError:
                rate = correct / float(total)
                Rtotal[epoch] = 1 - rate
            Etotal[epoch] = E

            # Run validation
            if needValid:
                VY, VE, dVEdW = self.runAndBackAll(validInputPerm, validTargetPerm, combineFnDeriv)
                VE = np.mean(VE)
                VEtotal[epoch] = VE
                if calcError:
                    Vrate, correct, total = self.calcRate(validTargetPerm, VY, decisionFn)
                    VRtotal[epoch] = 1 - Vrate

            # Adjust learning rate
            lr = lr * lrDecay
            mom -= dMom

            # Print statistics
            print "EP: %4d LR: %.2f M: %.2f E: %.4f R: %.4f VE: %.4f VR: %.4f TM: %4d" % \
                  (epoch, lr, mom, E, rate, VE, Vrate, (time.time() - startTime))

            if trainOpt['stoppingR'] - rate < 1e-3 and \
               trainOpt['stoppingR'] - Vrate < 1e-3 and \
               E < trainOpt['stoppingE'] and \
               VE < trainOpt['stoppingE']:
                break

        if trainOpt['plotFigs']:
            plt.figure(1);
            plt.clf()
            plt.plot(np.arange(numEpoch), Etotal, 'b-x')
            plt.plot(np.arange(numEpoch), VEtotal, 'g-o')
            plt.legend(['Train', 'Valid'])
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('LSTM Train/Valid MSE Curve')
            plt.draw()
            plt.savefig(trainOpt['name'] + '_mse.png')

            if calcError:
                plt.figure(2);
                plt.clf()
                plt.plot(np.arange(numEpoch), Rtotal, 'b-x')
                plt.plot(np.arange(numEpoch), VRtotal, 'g-o')
                plt.legend(['Train', 'Valid'])
                plt.xlabel('Epoch')
                plt.ylabel('Prediction Error')
                plt.title('LSTM Train/Valid Error Curve')
                plt.draw()
                plt.savefig(trainOpt['name'] + '_err.png')
        pass

    def runAndBackAllLoop(self, X, T, combineFnDeriv):
        timespan = X.shape[0]
        numEx = X.shape[1]
        Y = np.zeros((timespan, numEx, self.memoryDim), float)
        E = np.zeros(numEx, float)
        dEdW = np.zeros((self.W.shape[0], self.W.shape[1], numEx), float)
        for n in range(0, numEx):
            Y[:, n, :], E[n], dEdW[:, :, n] = self.runAndBackOnce(X[:, n, :], T[:, n, :], combineFnDeriv)
        dEdW = np.sum(dEdW, axis=2)
        return Y, E, dEdW

    def runAndBackAll(self, X, T, combineFnDeriv):
        Y, C, Z, Gi, Gf, Go = self.forwardPassAll(X)
        E, dEdW = self.backPropagateAll(T, Y, C, Z, Gi, Gf, Go, X, combineFnDeriv)
        return Y, E, dEdW

    def runAndBackOnce(self, X, T, combineFnDeriv):
        Y, C, Z, Gi, Gf, Go = self.forwardPass(X)
        E, dEdW = self.backPropagate(T, Y, C, Z, Gi, Gf, Go, X, combineFnDeriv)
        return Y, E, dEdW

    def forwardPassAll(self, X):
        timespan = X.shape[0]
        numEx = X.shape[1]
        Y = np.zeros((timespan, numEx, self.memoryDim), float)
        C = np.zeros((timespan, numEx, self.memoryDim), float)
        Z = np.zeros((timespan, numEx, self.memoryDim), float)
        Gi = np.zeros((timespan, numEx, self.memoryDim), float)
        Gf = np.zeros((timespan, numEx, self.memoryDim), float)
        Go = np.zeros((timespan, numEx, self.memoryDim), float)
        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)

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

        return Y, C, Z, Gi, Gf, Go

    def backPropagateAll(self, T, Y, C, Z, Gi, Gf, Go, X, combineFnDeriv):
        timespan = Y.shape[0]
        numEx = Y.shape[1]
        E, dEdY = combineFnDeriv(T, Y)
        dYdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, numEx, self.memoryDim), float)
        dCdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, numEx, self.memoryDim), float)

        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)
        Wyi = Wi[:, self.inputDim : self.inputDim + self.memoryDim]
        Wci = Wi[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wyf = Wf[:, self.inputDim : self.inputDim + self.memoryDim]
        Wcf = Wf[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wyc = Wc[:, self.inputDim : self.inputDim + self.memoryDim]
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
        return E, dEdW

    def forwardPass(self, X):
        # Online training for now
        timespan = X.shape[0]
        Y = np.zeros((timespan, self.memoryDim), float)
        C = np.zeros((timespan, self.memoryDim), float)
        Z = np.zeros((timespan, self.memoryDim), float)
        Gi = np.zeros((timespan, self.memoryDim), float)
        Gf = np.zeros((timespan, self.memoryDim), float)
        Go = np.zeros((timespan, self.memoryDim), float)
        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)

        for t in range(0, timespan):
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

        return Y, C, Z, Gi, Gf, Go

    def backPropagate(self, T, Y, C, Z, Gi, Gf, Go, X, combineFnDeriv):
        timespan = Y.shape[0]
        E, dEdY = combineFnDeriv(T, Y)
        dYdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, self.memoryDim), float)
        dCdW = np.zeros((timespan, self.memoryDim, self.inputDim * 4 + self.memoryDim * 7 + 4, self.memoryDim), float)

        Wi, Wf, Wc, Wo = self.sliceWeights(self.inputDim, self.memoryDim, self.W)
        Wyi = Wi[:, self.inputDim : self.inputDim + self.memoryDim]
        Wci = Wi[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wyf = Wf[:, self.inputDim : self.inputDim + self.memoryDim]
        Wcf = Wf[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]
        Wyc = Wc[:, self.inputDim : self.inputDim + self.memoryDim]
        Wyo = Wo[:, self.inputDim : self.inputDim + self.memoryDim]
        Wco = Wo[:, self.inputDim + self.memoryDim : self.inputDim + self.memoryDim + self.memoryDim]

        for t in range(0, timespan):
            if t == 0:
                Yt1 = np.zeros(self.memoryDim, float)
                Ct1 = np.zeros(self.memoryDim, float)
            else:
                Yt1 = Y[t-1, :]
                Ct1 = C[t-1, :]
            states1 = np.concatenate((X[t, :], Yt1, Ct1, np.ones(1, float)))
            states2 = np.concatenate((X[t, :], Yt1, np.ones(1, float)))
            states3 = np.concatenate((X[t, :], Yt1, C[t, :], np.ones(1, float)))

            # W
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
        return E, dEdW

    def testRate(self, X, T, decisionFn, printEx=False):
        Y, C, Z, Gi, Gf, Go = self.forwardPassAll(X.transpose((1, 0, 2)))
        if printEx:
            for n in range(0, min(X.shape[0], 10)):
                for j in range(0, X.shape[2]):
                    print "X:",
                    print X[n, :, j]
                for j in range(0, T.shape[2]):
                    print "T:",
                    print T[n, :, j]
                Yfinal = decisionFn(Y)
                print "Y:",
                print Yfinal[:, n].astype(float)

        return self.calcRate(T.transpose((1, 0, 2)), Y, decisionFn)

    @staticmethod
    def calcRate(T, Y, decisionFn):
        Yfinal = decisionFn(Y)
        correct = np.sum(Yfinal.reshape(Yfinal.size) == T.reshape(T.size))
        total = Yfinal.size
        rate = correct / float(total)
        return rate, correct, total

    @staticmethod
    def splitData(trainInput, trainTarget):
        s = trainInput.shape[0] / 2
        validInput = trainInput[0:s, :, :]
        validTarget = trainTarget[0:s, :, :]
        trainInput = trainInput[s:, :, :]
        trainTarget = trainTarget[s:, :, :]
        return trainInput, trainTarget, validInput, validTarget

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
    def flattenWeights(W):
        return W.reshape(W.size)

    @staticmethod
    def foldWeights(memoryDim, W):
        return W.reshape(memoryDim, W.size/memoryDim)

    def save(self, filename):
        lstmArray = np.concatenate((
            np.ones(1, float) * self.inputDim,
            np.ones(1, float) * self.memoryDim,
            self.flattenWeights(self.W)))
        np.save(filename, lstmArray)
        pass

    @staticmethod
    def read(filename):
        lstmArray = np.load(filename)
        inputDim = int(lstmArray[0])
        memoryDim = int(lstmArray[1])
        W = LSTM.foldWeights(memoryDim, lstmArray[2:])
        lstm = LSTM(
            inputDim=inputDim,
            memoryDim=memoryDim,
            reinit=False,
            W=W)
        return lstm

def simpleSum(Y):
    return np.sum(Y, axis=-1)

def simpleSumDeriv(T, Y):
    Yout = simpleSum(Y)
    diff =  Yout - T.reshape(Yout.shape)
    timespan = Y.shape[0]
    E = 0.5 * np.sum(np.power(diff, 2), axis=0) / float(timespan)

    if len(Y.shape) == 2:
        dEdY = np.repeat(diff.reshape(diff.shape[0], 1), Y.shape[1], axis=1) / float(timespan)
    elif len(Y.shape) == 3:
        dEdY = np.repeat(diff.reshape(diff.shape[0], diff.shape[1], 1), Y.shape[2], axis=2) / float(timespan)
    return E, dEdY

def simpleSumDecision(Y):
    return (simpleSum(Y) > 0.5).astype(int)
