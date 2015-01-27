import numpy as np
import time
import pickle
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class Pipeline:
    def __init__(self, name, costFn, outputFolder='', decisionFn=None):
        self.stages = []
        self.name = name + time.strftime("-%Y%m%d-%H%M%S")
        print 'Pipeline ' + self.name
        self.costFn = costFn
        self.decisionFn = decisionFn
        self.outputFolder = os.path.join(outputFolder, self.name)
        if not os.path.exists(self.outputFolder): os.makedirs(self.outputFolder)
        self.logFilename = os.path.join(self.outputFolder, self.name + '.csv')
        self.modelFilename = os.path.join(self.outputFolder, self.name + '.pipeline')
        self.lossFigFilename = os.path.join(self.outputFolder, self.name + '_loss.png')
        self.errFigFilename = os.path.join(self.outputFolder, self.name + '_err.png')
        pass

    def clear(self):
        self.stages = []
        pass

    def addStage(self, stage,
                 learningRate=0.1,
                 annealConst=0.0,
                 gradientClip=0.0,
                 weightClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True):
        self.stages.append(stage)
        stage.lastdW = 0
        stage.learningRate = learningRate
        stage.currentLearningRate = learningRate
        stage.gradientClip = gradientClip
        stage.annealConst = annealConst
        stage.weightClip = weightClip
        stage.weightRegConst = weightRegConst
        if len(self.stages) == 1:
            stage.outputdEdX = False
        else:
            stage.outputdEdX = outputdEdX
        pass

    def train(self, trainInput, trainTarget, trainOpt):
        needValid =  trainOpt['needValid'] if trainOpt.has_key('needValid') else False
        xvalidNo = trainOpt['xvalidNo'] if trainOpt.has_key('xvalidNo') else 0
        heldOutRatio = trainOpt['heldOutRatio'] if trainOpt.has_key('heldOutRatio') else 0.1
        if needValid:
            trainInput, trainTarget, validInput, validTarget = \
                self.splitData(trainInput, trainTarget, heldOutRatio, xvalidNo)
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
        lrDecay = trainOpt['learningRateDecay']
        mom = trainOpt['momentum']
        calcError = trainOpt['calcError']
        numExPerBat = trainOpt['batchSize']
        N = trainInput.shape[0]
        dMom = (mom - trainOpt['momentumEnd']) / float(numEpoch)
        writeRecord = trainOpt['writeRecord']
        saveModel = trainOpt['saveModel']
        everyEpoch = trainOpt['everyEpoch']
        plotFigs = trainOpt['plotFigs']
        printProgress = trainOpt['progress']
        shuffleTrainData = trainOpt['shuffle']

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

            if shuffleTrainData:
                shuffle = np.arange(0, X.shape[1])
                shuffle = np.random.permutation(shuffle)
                X = X[:, shuffle]
                if len(T.shape)==3:
                    T = T[:, shuffle]
                else:
                    T = T[shuffle]

            batchStart = 0
            while batchStart < N:
                # Progress bar
                if printProgress:
                    while batchStart/float(N) > progress / float(80):
                        sys.stdout.write('.')
                        sys.stdout.flush()
                        progress += 1

                # Batch info
                batchEnd = min(N, batchStart + numExPerBat)
                numEx = batchEnd - batchStart

                # Forward
                Y_bat = self.forwardPass(X[:, batchStart:batchEnd], dropout=True)
                if len(T.shape) == 3:
                    T_bat = T[:, batchStart:batchEnd, :]
                else:
                    T_bat = T[batchStart:batchEnd, :]

                # Loss
                Etmp, dEdY = self.costFn(Y_bat, T_bat)
                E += np.sum(Etmp) * numEx / float(N)

                # Backpropagate
                for stage in reversed(self.stages):
                    dEdW, dEdY = stage.backPropagate(dEdY, outputdEdX=stage.outputdEdX)
                    if stage.gradientClip > 0.0:
                        stage.dEdWnorm = np.sqrt(np.sum(np.power(dEdW, 2)))
                        if stage.dEdWnorm > stage.gradientClip:
                            dEdW *= stage.gradientClip / stage.dEdWnorm
                    if stage.learningRate > 0.0:
                        stage.lastdW = -stage.currentLearningRate * dEdW + \
                                       mom * stage.lastdW
                        stage.W = stage.W + stage.lastdW
                    if stage.weightRegConst > 0.0:
                        stage.Wnorm = np.sqrt(np.sum(np.power(stage.W, 2)))
                        stage.W -= stage.currentLearningRate * \
                                   stage.weightRegConst * stage.W
                    if stage.weightClip > 0.0:
                        stage.Wnorm = np.sqrt(np.sum(np.power(stage.W, 2)))
                        if stage.Wnorm > stage.weightClip:
                            stage.W *= stage.weightClip / stage.Wnorm
                        #stage.Wnorm = np.sqrt(np.sum(np.power(stage.W, 2)))
                        pass

                    # Stop backpropagate if frozen layers in the front.
                    if not stage.outputdEdX:
                        break

                # Prediction error
                if calcError:
                    rate_, correct_, total_ = self.calcRate(Y_bat, T_bat)
                    correct += correct_
                    total += total_

                batchStart += numExPerBat

            # Progress bar new line
            if printProgress:
                print

            # Store train statistics
            if calcError:
                rate = correct / float(total)
                Rtotal[epoch] = 1 - rate
            Etotal[epoch] = E

            # Run validation
            if needValid:
                VY = self.forwardPass(VX, dropout=False)
                VE, dVE = self.costFn(VY, VT)
                VE = np.sum(VE)
                VEtotal[epoch] = VE
                if calcError:
                    Vrate, correct, total = self.calcRate(VY, VT)
                    VRtotal[epoch] = 1 - Vrate

            # Adjust momentum
            mom -= dMom

            # Anneal learning rate
            for stage in self.stages:
                stage.currentLearningRate = stage.learningRate / (1.0 + stage.annealConst * epoch)

            # Print statistics
            timeElapsed = time.time() - startTime
            if trainOpt.has_key('displayDw'):
                stats = 'EP: %4d E: %.4f R: %.4f VE: %.4f VR: %.4f T:%4d DW:%.4f W:%.4f' % \
                        (epoch, E, rate, VE, Vrate, timeElapsed,
                         self.stages[trainOpt['displayDw']].dEdWnorm,
                         self.stages[trainOpt['displayDw']].Wnorm)
            else:
                stats = 'EP: %4d E: %.4f R: %.4f VE: %.4f VR: %.4f T:%4d' % \
                        (epoch, E, rate, VE, Vrate, timeElapsed)

            statsCsv = '%d,%.4f,%.4f,%.4f,%.4f' % \
                    (epoch, E, rate, VE, Vrate)
            print stats
            if writeRecord and everyEpoch:
                with open(self.logFilename, 'a+') as f:
                    f.write('%s\n' % statsCsv)

            # Save pipeline
            if saveModel and (everyEpoch or epoch == numEpoch - 1):
                self.save()

            # Check stopping criterion
            if E < trainOpt['stopE'] and VE < trainOpt['stopE']:
                break

            # Plot train curves
            if plotFigs and (everyEpoch or epoch == numEpoch-1):
                plt.figure(1);
                plt.clf()
                plt.plot(np.arange(epoch + 1), Etotal[0 : epoch + 1], 'b-x')
                plt.plot(np.arange(epoch + 1), VEtotal[0 : epoch + 1], 'g-o')
                plt.legend(['Train', 'Valid'])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train/Valid Loss Curve')
                plt.draw()
                plt.savefig(self.lossFigFilename)

                if calcError:
                    plt.figure(2);
                    plt.clf()
                    plt.plot(np.arange(epoch + 1), Rtotal[0 : epoch + 1], 'b-x')
                    plt.plot(np.arange(epoch + 1), VRtotal[0 : epoch + 1], 'g-o')
                    plt.legend(['Train', 'Valid'])
                    plt.xlabel('Epoch')
                    plt.ylabel('Prediction Error')
                    plt.title('Train/Valid Error Curve')
                    plt.draw()
                    plt.savefig(self.errFigFilename)

        if writeRecord and not everyEpoch:
            with open(self.logFilename, 'w+') as f:
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
            filename = self.modelFilename
        model = []
        for stage in self.stages:
            model.append(stage.W)
        np.save(filename, np.array(model, dtype=object))
        pass

    def savePickle(self, filename=None):
        if filename is None:
            filename = self.modelFilename
        with open(filename, 'w') as f:
            pickle.dump(self, f)
        pass

    @staticmethod
    def load(filename):
        with open(filename) as f:
            pipeline = pickle.load(f)
        return pipeline

    @staticmethod
    def splitData(trainInput, trainTarget, heldOutRatio, validNumber):
        s = np.round(trainInput.shape[0] * heldOutRatio)
        start = s * validNumber
        validInput = trainInput[start : start + s]
        validTarget = trainTarget[start : start + s]
        if validNumber == 0:
            trainInput = trainInput[s:]
            trainTarget = trainTarget[s:]
        else:
            trainInput = np.concatenate((trainInput[0:start], trainInput[s:]))
            trainTarget = np.concatenate((trainTarget[0:start], trainTarget[s:]))
        return trainInput, trainTarget, validInput, validTarget