import time
import sys
import os
import shutil
import matplotlib
import valid_tool as vt
from tester import *
from func import *
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

class ProgressWriter:
    def __init__(self, total, width=80):
        self.total = total
        self.counter = 0
        self.progress = 0
        self.width = width

    def increment(self, amount=1):
        self.counter += amount
        while self.counter / float(self.total) > \
              self.progress / float(self.width):
            sys.stdout.write('.')
            sys.stdout.flush()
            self.progress += 1
        if self.counter == self.total and \
           self.progress < self.width:
            print

class Logger:
    def __init__(self, trainer, csv=True, filename=None):
        self.trainer = trainer
        self.startTime = time.time()
        self.epoch = 0
        self.saveCsv = csv
        if filename is None:
            self.outFilename = os.path.join(
                trainer.outputFolder, trainer.name + '.csv')
        else:
            self.outFilename = filename

    def logMsg(self, msg):
        print msg

    def logTrainStats(self):
        timeElapsed = time.time() - self.startTime
        stats = 'N: %3d T: %5d  TE: %8.4f  TR: %8.4f  VE: %8.4f  VR: %8.4f' % \
                (self.epoch,
                 timeElapsed,
                 self.trainer.loss[self.epoch],
                 self.trainer.rate[self.epoch],
                 self.trainer.validLoss[self.epoch],
                 self.trainer.validRate[self.epoch])
        print stats

        if self.saveCsv:
            statsCsv = '%d,%.4f,%.4f,%.4f,%.4f' % \
                    (self.epoch,
                     self.trainer.loss[self.epoch],
                     self.trainer.rate[self.epoch],
                     self.trainer.validLoss[self.epoch],
                     self.trainer.validRate[self.epoch])
            with open(self.outFilename, 'a+') as f:
                f.write('%s\n' % statsCsv)
        self.epoch += 1
    pass

class Plotter:
    def __init__(self, trainer):
        self.trainer = trainer
        self.startTime = time.time()
        self.epoch = 0
        self.lossFigFilename = \
            os.path.join(trainer.outputFolder, trainer.name + '_loss.png')
        self.errFigFilename = \
            os.path.join(trainer.outputFolder, trainer.name + '_err.png')
        self.epoch = 0

    def plot(self):
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(self.epoch + 1),
                 self.trainer.loss[0 : self.epoch + 1], 'b-x')
        plt.plot(np.arange(self.epoch + 1),
                 self.trainer.validLoss[0 : self.epoch + 1], 'g-o')
        plt.legend(['Train', 'Valid'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Train/Valid Loss Curve')
        plt.draw()
        plt.savefig(self.lossFigFilename)

        if self.trainer.trainOpt['calcError']:
            plt.figure(2)
            plt.clf()
            plt.plot(np.arange(self.epoch + 1),
                     1 - self.trainer.rate[0 : self.epoch + 1], 
                     'b-x')
            plt.plot(np.arange(self.epoch + 1),
                     1 - self.trainer.validRate[0 : self.epoch + 1], 
                     'g-o')
            plt.legend(['Train', 'Valid'])
            plt.xlabel('Epoch')
            plt.ylabel('Prediction Error')
            plt.title('Train/Valid Error Curve')
            plt.draw()
            plt.savefig(self.errFigFilename)
        self.epoch += 1

class Trainer:
    def __init__(self,
                 name,
                 model,
                 trainOpt,
                 outputFolder='',
                 seed=1000):
        self.model = model
        self.name = name + time.strftime("-%Y%m%d-%H%M%S")
        self.resultsFolder = outputFolder
        self.outputFolder = os.path.join(outputFolder, self.name)
        self.modelFilename = os.path.join(self.outputFolder, self.name + '.w.npy')
        self.trainOpt = trainOpt
        self.startTime = time.time()
        self.random = np.random.RandomState(seed)
        numEpoch = trainOpt['numEpoch']
        self.loss = np.zeros(numEpoch)
        self.validLoss = np.zeros(numEpoch)
        self.rate = np.zeros(numEpoch)
        self.validRate = np.zeros(numEpoch)
        self.stoppedEpoch = 0

    def initFolder(self):
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        if self.model.specFilename is not None:
            shutil.copyfile(
                self.model.specFilename,
                os.path.join(self.outputFolder, self.name + '.model.yml'))

    def initData(self, X, T):
        VX = None
        VT = None
        X, T = vt.shuffleData(X, T, self.random)
        if self.trainOpt['needValid']:
            X, T, VX, VT = \
                vt.splitData(X, T,
                            self.trainOpt['heldOutRatio'],
                            self.trainOpt['xvalidNo'])
        return X, T, VX, VT

    def train(self, trainInput, trainTarget):
        self.initFolder()
        trainOpt = self.trainOpt
        X, T, VX, VT = self.initData(trainInput, trainTarget)
        N = X.shape[0]
        numEpoch = trainOpt['numEpoch']
        calcError = trainOpt['calcError']
        numExPerBat = trainOpt['batchSize']
        progressWriter = ProgressWriter(N, width=80)
        logger = Logger(self, csv=trainOpt['writeRecord'])
        logger.logMsg('Trainer ' + self.name)
        plotter = Plotter(self)
        bestVE = None
        nAfterBest = 0

        # Train loop through epochs
        for epoch in range(0, numEpoch):
            E = 0
            correct = 0
            total = 0

            if trainOpt['shuffle']:
                X, T = vt.shuffleData(X, T, self.random)

            batchStart = 0
            while batchStart < N:
                # Batch info
                batchEnd = min(N, batchStart + numExPerBat)
                numExThisBat = batchEnd - batchStart

                # Write progress bar
                if trainOpt['progress']:
                    progressWriter.increment(amount=numExThisBat)

                # Forward
                Y_bat = self.model.forward(X[batchStart:batchEnd], dropout=True)
                T_bat = T[batchStart:batchEnd]

                # Loss
                Etmp, dEdY = self.model.getCost(Y_bat, T_bat)
                E += np.sum(Etmp) * numExThisBat / float(N)

                # Backward
                self.model.backward(dEdY)

                # Update
                self.model.updateWeights()

                # Prediction error
                if calcError:
                    rate_, correct_, total_ = calcRate(self.model, Y_bat, T_bat)
                    correct += correct_
                    total += total_

                batchStart += numExPerBat

            # Store train statistics
            if calcError:
                rate = correct / float(total)
                self.rate[epoch] = rate
            self.loss[epoch] = E

            # Run validation
            if trainOpt['needValid']:
                VY = self.model.forward(VX, dropout=False)
                VE, dVE = self.model.getCost(VY, VT)
                VE = np.sum(VE)
                self.validLoss[epoch] = VE
                if calcError:
                    Vrate, correct, total = calcRate(self.model, VY, VT)
                    self.validRate[epoch] = Vrate
                if bestVE is None or VE < bestVE:
                    bestVE = VE
                    nAfterBest = 0
                    # Save trainer if VE is best
                    if trainOpt['saveModel']:
                        self.save()
                else:
                    nAfterBest += 1
                    # Stop training if above tolerance level
                    if nAfterBest > trainOpt['tolerance']:
                        break

            # Anneal learning rate
            self.model.updateLearningParams(epoch)

            # Print statistics
            logger.logTrainStats()

            # Plot train curves
            if trainOpt['plotFigs']:
                plotter.plot()

        # Send email
        if trainOpt.has_key('sendEmail') and trainOpt['sendEmail']:
            print 'Appended to email list.'
            tosend = os.path.join(self.resultsFolder, 'tosend.txt')  
            with open(tosend, 'a+') as f:
                f.write(self.name + '\n')

        # Record final epoch number
        self.stoppedEpoch = epoch

    def save(self, filename=None):
        if filename is None:
            filename = self.modelFilename
        np.save(filename, self.model.getWeights())
        pass
