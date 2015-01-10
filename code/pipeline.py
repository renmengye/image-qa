import numpy as np
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class Pipeline:
    def __init__(self):
        self.stages = []
        pass
    def clear(self):
        self.stages = []
        pass
    def addStage(self, stage):
        self.stages.append(stage)
        pass
    def train(self, trainInput, trainTarget, trainOpt):
        needValid =  trainOpt['needValid']
        if needValid:
            trainInput, trainTarget, validInput, validTarget = self.splitData(trainInput, trainTarget)
        numEpoch = trainOpt['numEpoch']
        lr = trainOpt['learningRate']
        lrDecay = trainOpt['learningRateDecay']
        mom = trainOpt['momentum']
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
        lastdW = 0

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
                    X = trainInput[n, :, :]
                    T = trainTarget[n, :, :]
                    Y = self.forwardPass(X)
                    Etmp = self.costFunc(Y, T)
                    dEdY = self.costFuncDeriv(Y, T)
                    E += Etmp / float(N)

                    for stage in self.stages.reverse():
                        dEdY, dEdW = stage.backPropagate(dEdY)
                        stage.W = stage.W - lr * dEdW + mom * lastdW
                        lastdW = -lr * dEdW
                        if calcError:
                            rate_, correct_, total_ = self.calcRate(T, Y, decisionFn)
                            correct += correct_
                            total += total_

            # Store train statistics
            if calcError:
                rate = correct / float(total)
                Rtotal[epoch] = 1 - rate
            Etotal[epoch] = E

            # Run validation
            if needValid:
                VY = self.forwardPassAll(validInput)
                VE = self.costFunc(VY, validTarget)
                VE = np.mean(VE)
                VEtotal[epoch] = VE
                if calcError:
                    Vrate, correct, total = self.calcRate(validTarget, VY, decisionFn)
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
            plt.title('Train/Valid MSE Curve')
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
                plt.title('Train/Valid Error Curve')
                plt.draw()
                plt.savefig(trainOpt['name'] + '_err.png')
        pass

    def forwardPass(self, X):
        X1 = X
        for stage in self.stages:
            X1 = stage.forwardPass(X1)
        return X1

    def forwardPassAll(self, X):
        X1 = X
        for stage in self.stages:
            X1 = stage.forwardPassAll(X1)
        return X1

    def calcRate(self, T, Y, decisionFn):
        pass