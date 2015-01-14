import numpy as np
import time
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class Pipeline:
    def __init__(self, name, costFn, decisionFn=None):
        self.stages = []
        self.name = name + time.strftime("-%Y%m%d-%H%M%S")
        self.costFn = costFn
        self.decisionFn = decisionFn
        pass

    def clear(self):
        self.stages = []
        pass

    def addStage(self, stage):
        self.stages.append(stage)
        stage.lastdW = 0
        pass

    def train(self, trainInput, trainTarget, trainOpt):
        needValid =  trainOpt['needValid']
        if needValid:
            trainInput, trainTarget, validInput, validTarget = \
                self.splitData(trainInput, trainTarget, trainOpt['heldOutRatio'])
        if len(trainInput.shape) == 3:
            X = trainInput.transpose((1, 0, 2))
            VX = validInput.transpose((1, 0, 2))
        else:
            X = trainInput.transpose()
            VX = validInput.transpose()
        if len(trainTarget.shape) == 3:
            T = trainTarget.transpose((1, 0, 2))
            VT = validTarget.transpose((1, 0, 2))
        else:
            T = trainTarget
            VT = validTarget
        numEpoch = trainOpt['numEpoch']
        lr = trainOpt['learningRate']
        lrDecay = trainOpt['learningRateDecay']
        mom = trainOpt['momentum']
        calcError = trainOpt['calcError']
        bat = trainOpt['batchSize']
        N = trainInput.shape[0]
        dMom = (mom - trainOpt['momentumEnd']) / float(numEpoch)

        Etotal = np.zeros(numEpoch, float)
        VEtotal = np.zeros(numEpoch, float)
        Rtotal = np.zeros(numEpoch, float)
        VRtotal = np.zeros(numEpoch, float)

        startTime = time.time()

        # Train loop through epochs
        for epoch in range(0, numEpoch):
            E = 0
            rate = 0
            VE = 0
            Vrate = 0
            correct = 0
            total = 0

            # Stochastic only for now
            if bat == 1:
                for n in range(0, N):
                    Y_n = self.forwardPass(X[:, n])
                    if len(T.shape) == 3:
                        T_n = T[:, n, :]
                    else:
                        T_n = T[n, :]
                    Etmp, dEdY = self.costFn(Y_n, T_n)
                    E += Etmp / float(N)
                    if calcError:
                        rate_, correct_, total_ = self.calcRate(Y_n, T_n)
                        correct += correct_
                        total += total_

                    for stage in reversed(self.stages):
                        dEdW, dEdY = stage.backPropagate(dEdY,
                                                         outputdEdX=(stage!=self.stages[0]))
                        # if np.sum(dEdW) != 0 and np.min(dEdW) != 0:
                        #     lr = 1e-3 / np.min(dEdW)
                        # else:
                        #     lr = trainOpt['learningRate']
                        stage.W = stage.W - lr * dEdW + mom * stage.lastdW
                        stage.lastdW = -lr * dEdW
            else:
                batchStart = 0
                while batchStart < N:
                    batchEnd = min(N, batchStart + bat)
                    numEx = batchEnd - batchStart
                    Y_bat = self.forwardPass(X[:, batchStart:batchEnd])
                    if len(T.shape) == 3:
                        T_bat = T[:, batchStart:batchEnd, :]
                    else:
                        T_bat = T[batchStart:batchEnd, :]
                    Etmp, dEdY = self.costFn(Y_bat, T_bat)
                    E += np.sum(Etmp) * numEx / float(N)

                    for stage in reversed(self.stages):
                       dEdW, dEdY = stage.backPropagate(dEdY,
                                                        outputdEdX=(stage!=self.stages[0]))
                       stage.W = stage.W - lr * dEdW + mom * stage.lastdW
                       stage.lastdW = -lr * dEdW

                    if calcError:
                        rate_, correct_, total_ = self.calcRate(Y_bat, T_bat)
                        correct += correct_
                        total += total_
                    batchStart += bat

            # Store train statistics
            if calcError:
                rate = correct / float(total)
                Rtotal[epoch] = 1 - rate
            Etotal[epoch] = E

            # Run validation
            if needValid:
                VY = self.forwardPass(VX)
                VE, dVE = self.costFn(VY, VT)
                VE = np.sum(VE)
                VEtotal[epoch] = VE
                if calcError:
                    Vrate, correct, total = self.calcRate(VY, VT)
                    VRtotal[epoch] = 1 - Vrate

            # Adjust learning rate
            lr = lr * lrDecay
            mom -= dMom

            # Print statistics
            stats = 'EP: %4d LR: %.2f M: %.2f E: %.4f R: %.4f VE: %.4f VR: %.4f TM: %4d' % \
                    (epoch, lr, mom, E, rate, VE, Vrate, (time.time() - startTime))
            print stats
            with open(self.name + '.txt', 'a+') as f:
                f.write('%s\n' % stats)

            # Check stopping criterion
            if E < trainOpt['stopE'] and VE < trainOpt['stopE']:
                break

        # Plot train curves
        if trainOpt['plotFigs']:
            plt.figure(1);
            plt.clf()
            plt.plot(np.arange(numEpoch), Etotal, 'b-x')
            plt.plot(np.arange(numEpoch), VEtotal, 'g-o')
            plt.legend(['Train', 'Valid'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Train/Valid Loss Curve')
            plt.draw()
            plt.savefig(self.name + '_loss.png')

            if calcError:
                plt.figure(2);
                plt.clf()
                plt.plot(np.arange(numEpoch), Rtotal, 'b-x')
                plt.plot(np.arange(numEpoch), VRtotal, 'g-o')
                plt.legend(['Train', 'Valid'])
                plt.xlabel('Epoch')
                plt.ylabel('Prediction Error')
                plt.title('Train/Valid Error Curve')
                plt.draw()
                plt.savefig(self.name + '_err.png')
        pass

    def forwardPass(self, X):
        X1 = X
        for stage in self.stages:
            X1 = stage.forwardPass(X1)
        return X1

    def test(self, X, T, printEx=False):
        X = X.transpose((1, 0, 2))
        if len(T.shape) == 3:
            T = T.transpose((1, 0, 2))
        Y = self.forwardPass(X)
        if printEx:
            for n in range(0, min(X.shape[0], 10)):
                for j in range(0, X.shape[-1]):
                    print "X:",
                    print X[n, :, j]
                for j in range(0, T.shape[-1]):
                    print "T:",
                    print T[n, :, j]
                Yfinal = self.decisionFn(Y)
                print "Y:",
                print Yfinal[:, n].astype(float)

        rate, correct, total = self.calcRate(Y, T)
        print 'TR: %.4f' % rate

        return rate, correct, total

    def calcRate(self, Y, T):
        Yfinal = self.decisionFn(Y)
        correct = np.sum(Yfinal.reshape(Yfinal.size) == T.reshape(T.size))
        total = Yfinal.size
        rate = correct / float(total)
        return rate, correct, total

    def save(self, filename=None):
        if filename is None:
            filename = self.name + '.pip'
        with open(filename, 'w') as f:
            pickle.dump(self, f)
        pass

    @staticmethod
    def load(filename):
        with open(filename) as f:
            pipeline = pickle.load(f)
        return pipeline

    @staticmethod
    def splitData(trainInput, trainTarget, heldOutRatio):
        s = np.round(trainInput.shape[0] * heldOutRatio)
        validInput = trainInput[0:s]
        validTarget = trainTarget[0:s]
        trainInput = trainInput[s:]
        trainTarget = trainTarget[s:]
        return trainInput, trainTarget, validInput, validTarget