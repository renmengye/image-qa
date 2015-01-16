import numpy as np
import time
import pickle
import sys

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

    def addStage(self, stage, learningRate=0.1):
        self.stages.append(stage)
        stage.lastdW = 0
        stage.learningRate = learningRate
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
        #lr = trainOpt['learningRate']
        lrDecay = trainOpt['learningRateDecay']
        mom = trainOpt['momentum']
        calcError = trainOpt['calcError']
        bat = trainOpt['batchSize']
        N = trainInput.shape[0]
        dMom = (mom - trainOpt['momentumEnd']) / float(numEpoch)
        writeRecord = trainOpt['writeRecord']
        everyEpoch = trainOpt['everyEpoch']
        plotFigs = trainOpt['plotFigs']

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
            progress = 0

            if trainOpt['shuffle']:
                shuffle = np.arange(0, X.shape[1])
                shuffle = np.random.permutation(shuffle)
                X = X[:, shuffle]
                if len(T.shape)==3:
                    T = T[:, shuffle]
                else:
                    T = T[shuffle]

            # Stochastic only for now
            if bat == 1:
                for n in range(0, N):
                    # Progress bar
                    while n/float(N) > progress / float(80):
                        sys.stdout.write('.')
                        progress += 1
                    Y_n = self.forwardPass(X[:, n], dropout=True)
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
                        stage.lastdW = -stage.learningRate * dEdW + mom * stage.lastdW
                        stage.W = stage.W + stage.lastdW
            else:
                batchStart = 0
                while batchStart < N:
                    # Progress bar
                    while batchStart/float(N) > progress / float(80):
                        sys.stdout.write('.')
                        progress += 1
                    batchEnd = min(N, batchStart + bat)
                    numEx = batchEnd - batchStart
                    Y_bat = self.forwardPass(X[:, batchStart:batchEnd], dropout=True)
                    if len(T.shape) == 3:
                        T_bat = T[:, batchStart:batchEnd, :]
                    else:
                        T_bat = T[batchStart:batchEnd, :]
                    Etmp, dEdY = self.costFn(Y_bat, T_bat)
                    E += np.sum(Etmp) * numEx / float(N)

                    for stage in reversed(self.stages):
                        dEdW, dEdY = stage.backPropagate(dEdY,
                                                        outputdEdX=(stage!=self.stages[0]))
                        stage.lastdW = -stage.learningRate * dEdW + mom * stage.lastdW
                        stage.W = stage.W + stage.lastdW

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
            #lr = lr * lrDecay
            mom -= dMom

            # Print statistics
            timeElapsed = time.time() - startTime
            stats = 'EP: %4d E: %.4f R: %.4f VE: %.4f VR: %.4f T:%4d' % \
                    (epoch, E, rate, VE, Vrate, timeElapsed)
            stats2 = '%d,%.4f,%.4f,%.4f,%.4f' % \
                    (epoch, E, rate, VE, Vrate)
            print stats
            if writeRecord:
                if everyEpoch:
                    with open(self.name + '.csv', 'a+') as f:
                        f.write('%s\n' % stats2)

            # Save pipeline
            if everyEpoch or epoch == numEpoch-1:
                self.save()

            # Check stopping criterion
            if E < trainOpt['stopE'] and VE < trainOpt['stopE']:
                break

            # Plot train curves
            if plotFigs and (everyEpoch or epoch == numEpoch-1):
                plt.figure(1);
                plt.clf()
                plt.plot(np.arange(epoch+1), Etotal[0:epoch+1], 'b-x')
                plt.plot(np.arange(epoch+1), VEtotal[0:epoch+1], 'g-o')
                plt.legend(['Train', 'Valid'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train/Valid Loss Curve')
                plt.draw()
                plt.savefig(self.name + '_loss.png')

                if calcError:
                    plt.figure(2);
                    plt.clf()
                    plt.plot(np.arange(epoch+1), Rtotal[0:epoch+1], 'b-x')
                    plt.plot(np.arange(epoch+1), VRtotal[0:epoch+1], 'g-o')
                    plt.legend(['Train', 'Valid'])
                    plt.xlabel('Epoch')
                    plt.ylabel('Prediction Error')
                    plt.title('Train/Valid Error Curve')
                    plt.draw()
                    plt.savefig(self.name + '_err.png')


        if writeRecord and not everyEpoch:
            with open(self.name + '.csv', 'w+') as f:
                for epoch in range(0, numEpoch):
                    stats2 = '%d,%.4f,%.4f,%.4f,%.4f' % \
                            (epoch,
                             Etotal[epoch],
                             1 - Rtotal[epoch],
                             VEtotal[epoch],
                             1 - VRtotal[epoch])
                    f.write('%s\n' % stats2)
        pass

    def forwardPass(self, X, dropout):
        X1 = X
        for stage in self.stages:
            if hasattr(stage, 'dropout'):
                X1 = stage.forwardPass(X1, dropout)
            else:
                X1 = stage.forwardPass(X1)
        return X1

    def test(self, X, T, printEx=False):
        X = X.transpose((1, 0, 2))
        if len(T.shape) == 3:
            T = T.transpose((1, 0, 2))
        Y = self.forwardPass(X, dropout=False)
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