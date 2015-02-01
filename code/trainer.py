from util_func import *
import time
import pickle
import sys
import os
import yaml
import shutil
import router

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class Trainer:
    def __init__(self,
                 name,
                 model,
                 trainOpt,
                 costFn,
                 decisionFn=None,
                 outputFolder='',
                 configFilename=None,
                 seed=1000):
        self.model = model
        self.name = name + time.strftime("-%Y%m%d-%H%M%S")
        print 'Trainer ' + self.name
        self.costFn = costFn
        self.decisionFn = decisionFn
        self.outputFolder = os.path.join(outputFolder, self.name)
        self.trainOpt = trainOpt
        if not os.path.exists(self.outputFolder): os.makedirs(self.outputFolder)
        if configFilename is not None:
            shutil.copyfile(configFilename, os.path.join(self.outputFolder, self.name + '.yaml'))
        self.logFilename = os.path.join(self.outputFolder, self.name + '.csv')
        self.modelFilename = os.path.join(self.outputFolder, self.name + '.w')
        self.lossFigFilename = os.path.join(self.outputFolder, self.name + '_loss.png')
        self.errFigFilename = os.path.join(self.outputFolder, self.name + '_err.png')
        self.startTime = time.time()
        self.random = np.random.RandomState(seed)
        pass

    @staticmethod
    def initFromConfig(name, configFilename, outputFolder=None):
        with open(configFilename) as f:
            pipeDict = yaml.load(f)

        for stageDict in pipeDict['stages']:
            stage = router.routeStage(stageDict)

        pipeline = Trainer(
            name=name,
            model=router.getStage(pipeDict['model']),
            trainOpt=pipeDict['trainOpt'],
            costFn=router.routeFn(pipeDict['costFn']),
            decisionFn=router.routeFn(pipeDict['decisionFn']),
            outputFolder=outputFolder,
            configFilename=configFilename,
            seed=pipeDict['seed'] if pipeDict.has_key('seed') else 1000
        )
        return pipeline

    def shuffleData(self, X, T):
        shuffle = numpy.arange(0, X.shape[0])
        shuffle = self.random.permutation(shuffle)
        X = X[shuffle]
        T = T[shuffle]
        return X, T

    def train(self,
              trainInput,
              trainTarget,
              testInput=None,
              testTarget=None):
        trainOpt = self.trainOpt
        needValid =  trainOpt['needValid'] if trainOpt.has_key('needValid') else False
        xvalidNo = trainOpt['xvalidNo'] if trainOpt.has_key('xvalidNo') else 0
        heldOutRatio = trainOpt['heldOutRatio'] if trainOpt.has_key('heldOutRatio') else 0.1
        trainInput, trainTarget = self.shuffleData(trainInput, trainTarget)
        if needValid:
            trainInput, trainTarget, validInput, validTarget = \
                self.splitData(trainInput, trainTarget, heldOutRatio, xvalidNo)
        X = trainInput
        VX = validInput
        T = trainTarget
        VT = validTarget
        N = trainInput.shape[0]
        numEpoch = trainOpt['numEpoch']
        calcError = trainOpt['calcError']
        numExPerBat = trainOpt['batchSize']
        needWriteRecord = trainOpt['writeRecord']
        saveModel = trainOpt['saveModel']
        everyEpoch = trainOpt['everyEpoch']
        needPlotFigs = trainOpt['plotFigs']
        printProgress = trainOpt['progress']
        shuffleTrainData = trainOpt['shuffle']

        Etotal = np.zeros(numEpoch)
        VEtotal = np.zeros(numEpoch)
        Rtotal = np.zeros(numEpoch)
        VRtotal = np.zeros(numEpoch)

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
                X, T = self.shuffleData(X, T)

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
                numExThisBat = batchEnd - batchStart

                # Forward
                Y_bat = self.model.forwardPass(X[batchStart:batchEnd], dropout=True)
                T_bat = T[batchStart:batchEnd]

                # Loss
                Etmp, dEdY = self.costFn(Y_bat, T_bat)
                E += np.sum(Etmp) * numExThisBat / float(N)

                # Backpropagate
                self.model.backPropagate(dEdY)

                # Update
                self.model.updateWeights()

                # Prediction error
                if calcError:
                    rate_, correct_, total_ = self.calcRate(Y_bat, T_bat)
                    correct += correct_
                    total += total_

                batchStart += numExPerBat

            # Progress bar new line
            if printProgress and progress < 80:
                print

            # Store train statistics
            if calcError:
                rate = correct / float(total)
                Rtotal[epoch] = 1 - rate
            Etotal[epoch] = E

            # Run validation
            if needValid:
                VY = self.model.forwardPass(VX, dropout=False)
                VE, dVE = self.costFn(VY, VT)
                VE = np.sum(VE)
                VEtotal[epoch] = VE
                if calcError:
                    Vrate, correct, total = self.calcRate(VY, VT)
                    VRtotal[epoch] = 1 - Vrate

            # Anneal learning rate
            self.model.updateLearningParams(epoch)

            # Print statistics
            displayDw = trainOpt['displayDw'] if trainOpt.has_key('displayDw') else None
            self.writeRecordEvery(epoch, E, rate, VE, Vrate, displayDw, needWriteRecord)

            # Save trainer
            if saveModel and (everyEpoch or epoch == numEpoch - 1):
                self.save()

            # Check stopping criterion
            if E < trainOpt['stopCost'] and VE < trainOpt['stopCost']:
                break

            # Plot train curves
            if needPlotFigs and (everyEpoch or epoch == numEpoch-1):
                self.plotFigs(epoch, Etotal, VEtotal, calcError, Rtotal, VRtotal)

        if needWriteRecord and not everyEpoch:
            self.writeRecordAll(numEpoch, Etotal, Rtotal, VEtotal, VRtotal)

        if testInput is not None and testTarget is not None:
            self.test(testInput, testTarget)

    def test(self, X, T):
        N = X.shape[0]
        numExPerBat = 100
        batchStart = 0
        correctT = 0
        totalT = 0
        while batchStart < N:
            # Batch info
            batchEnd = min(N, batchStart + numExPerBat)
            Y = self.model.forwardPass(X[batchStart:batchEnd], dropout=False)
            rate, correct, total = self.calcRate(Y, T)
            correctT += correct
            totalT += total

        print 'SR: %.4f' % correctT / float(totalT)

    def plotFigs(self, epoch, Etotal, VEtotal, calcError, Rtotal=None, VRtotal=None):
        plt.figure(1)
        plt.clf()
        plt.plot(numpy.arange(epoch + 1), Etotal[0 : epoch + 1], 'b-x')
        plt.plot(numpy.arange(epoch + 1), VEtotal[0 : epoch + 1], 'g-o')
        plt.legend(['Train', 'Valid'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train/Valid Loss Curve')
        plt.draw()
        plt.savefig(self.lossFigFilename)

        if calcError:
            plt.figure(2)
            plt.clf()
            plt.plot(numpy.arange(epoch + 1), Rtotal[0 : epoch + 1], 'b-x')
            plt.plot(numpy.arange(epoch + 1), VRtotal[0 : epoch + 1], 'g-o')
            plt.legend(['Train', 'Valid'])
            plt.xlabel('Epoch')
            plt.ylabel('Prediction Error')
            plt.title('Train/Valid Error Curve')
            plt.draw()
            plt.savefig(self.errFigFilename)

    def writeRecordAll(self, numEpoch, Etotal, Rtotal, VEtotal, VRtotal):
        with open(self.logFilename, 'w+') as f:
            for epoch in range(0, numEpoch):
                stats2 = '%d,%.4f,%.4f,%.4f,%.4f' % \
                        (epoch,
                         Etotal[epoch],
                         1 - Rtotal[epoch],
                         VEtotal[epoch],
                         1 - VRtotal[epoch])
                f.write('%s\n' % stats2)

    def writeRecordEvery(self, epoch, E, R, VE, VR, displayDw=None, writeToFile=True):
        # Print statistics
        timeElapsed = time.time() - self.startTime
        stats = 'N: %3d T: %5d  TE: %8.4f  TR: %8.4f  VE: %8.4f  VR: %8.4f' % \
                (epoch, timeElapsed, E, R, VE, VR)
        print stats

        if writeToFile:
            statsCsv = '%d,%.4f,%.4f,%.4f,%.4f' % \
                    (epoch, E, R, VE, VR)
            with open(self.logFilename, 'a+') as f:
                f.write('%s\n' % statsCsv)

    def calcRate(self, Y, T):
        Yfinal = self.decisionFn(Y)
        correct = np.sum(Yfinal.reshape(Yfinal.size) == T.reshape(T.size))
        total = Yfinal.size
        rate = correct / float(total)
        return rate, correct, total

    def save(self, filename=None):
        if filename is None:
            filename = self.modelFilename
        np.save(filename, self.model.getWeights())
        pass

    def loadWeights(self, weightsFilename):
        weights = numpy.load(weightsFilename)
        self.model.loadWeights(weights)
        pass

    def savePickle(self, filename=None):
        if filename is None:
            filename = self.modelFilename
        with open(filename, 'w') as f:
            pickle.dump(self, f)
        pass

    @staticmethod
    def loadPickle(filename):
        with open(filename) as f:
            pipeline = pickle.load(f)
        return pipeline

    @staticmethod
    def splitData(trainInput, trainTarget, heldOutRatio, validNumber):
        s = numpy.round(trainInput.shape[0] * heldOutRatio)
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