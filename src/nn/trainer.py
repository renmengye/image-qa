import time
import sys
import os
import shutil
import matplotlib
import valid_tool as vt
import tester
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
                (self.trainer.epoch,
                 timeElapsed,
                 self.trainer.loss[self.trainer.epoch],
                 self.trainer.rate[self.trainer.epoch],
                 self.trainer.validLoss[self.trainer.epoch],
                 self.trainer.validRate[self.trainer.epoch])
        print stats

        if self.saveCsv:
            statsCsv = '%d,%.4f,%.4f,%.4f,%.4f' % \
                    (self.trainer.epoch,
                     self.trainer.loss[self.trainer.epoch],
                     self.trainer.rate[self.trainer.epoch],
                     self.trainer.validLoss[self.trainer.epoch],
                     self.trainer.validRate[self.trainer.epoch])
            with open(self.outFilename, 'a+') as f:
                f.write('%s\n' % statsCsv)
    pass

class Plotter:
    def __init__(self, trainer):
        self.trainer = trainer
        self.startTime = time.time()
        self.trainer.epoch = 0
        self.lossFigFilename = \
            os.path.join(trainer.outputFolder, trainer.name + '_loss.png')
        self.errFigFilename = \
            os.path.join(trainer.outputFolder, trainer.name + '_err.png')
        self.trainer.epoch = 0

    def plot(self):
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(self.trainer.epoch + 1),
                 self.trainer.loss[0 : self.trainer.epoch + 1], 'b-x')
        if self.trainer.trainOpt['needValid']:
            plt.plot(np.arange(self.trainer.epoch + 1),
                     self.trainer.validLoss[0 : self.trainer.epoch + 1], 'g-o')
            plt.legend(['Train', 'Valid'])
            plt.title('Train/Valid Loss Curve')
        else:
            plt.legend(['Train'])
            plt.title('Train Loss Curve')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.draw()
        plt.savefig(self.lossFigFilename)

        if self.trainer.trainOpt['calcError']:
            plt.figure(2)
            plt.clf()
            plt.plot(np.arange(self.trainer.epoch + 1),
                     1 - self.trainer.rate[0 : self.trainer.epoch + 1], 
                     'b-x')
            if self.trainer.trainOpt['needValid']:
                plt.plot(np.arange(self.trainer.epoch + 1),
                         1 - self.trainer.validRate[0 : self.trainer.epoch + 1], 
                         'g-o')
                plt.legend(['Train', 'Valid'])
                plt.title('Train/Valid Error Curve')
            else:
                plt.legend(['Train'])
                plt.title('Train Error Curve')

            plt.xlabel('Epoch')
            plt.ylabel('Prediction Error')
            plt.draw()
            plt.savefig(self.errFigFilename)

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
        self.modelFilename = \
            os.path.join(self.outputFolder, self.name + '.w.npy')
        self.trainOpt = trainOpt
        self.startTime = time.time()
        self.random = np.random.RandomState(seed)
        numEpoch = trainOpt['numEpoch']
        self.loss = np.zeros(numEpoch)
        self.validLoss = np.zeros(numEpoch)
        self.rate = np.zeros(numEpoch)
        self.validRate = np.zeros(numEpoch)
        self.stoppedEpoch = 0
        self.epoch =  0

    def initFolder(self):
        if not os.path.exists(self.outputFolder):
            os.makedirs(self.outputFolder)
        if self.model.specFilename is not None:
            shutil.copyfile(
                self.model.specFilename,
                os.path.join(self.outputFolder, self.name + '.model.yml'))

    def initData(self, X, T, split=True):
        VX = None
        VT = None
        X, T = vt.shuffleData(X, T, self.random)
        if split:
            X, T, VX, VT = \
                vt.splitData(X, T,
                            self.trainOpt['heldOutRatio'],
                            self.trainOpt['xvalidNo'])
        return X, T, VX, VT

    def train(
                self, 
                trainInput, 
                trainTarget, 
                trainInputWeights=None,
                validInput=None, 
                validTarget=None,
                validInputWeights=None):
        self.initFolder()
        trainOpt = self.trainOpt
        if validInput is None and validTarget is None:
            X, T, VX, VT = self.initData(\
                trainInput, trainTarget, \
                split=self.trainOpt['needValid'])
        else:
            X = trainInput
            T = trainTarget
            VX = validInput
            VT = validTarget
        N = X.shape[0]
        numEpoch = trainOpt['numEpoch']
        calcError = trainOpt['calcError']
        numExPerBat = trainOpt['batchSize']
        progressWriter = ProgressWriter(N, width=80)
        logger = Logger(self, csv=trainOpt['writeRecord'])
        logger.logMsg('Trainer ' + self.name)
        plotter = Plotter(self)
        bestVscore = None
        bestTscore = None
        bestEpoch = 0
        nAfterBest = 0
        stop = False

        # Train loop through epochs
        for epoch in range(0, numEpoch):
            E = 0
            correct = 0
            total = 0
            self.epoch = epoch

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
                Etmp, dEdY = self.model.getCost(
                                    Y_bat, T_bat, weights=trainInputWeights)
                E += Etmp * numExThisBat / float(N)

                # Backward
                self.model.backward(dEdY)

                # Update
                self.model.updateWeights()

                # Prediction error
                if calcError:
                    rate_, correct_, total_ = \
                        tester.calcRate(self.model, Y_bat, T_bat)
                    correct += correct_
                    total += total_

                batchStart += numExPerBat
                if trainOpt.has_key('logNumBat') and \
                    np.mod(batchStart / numExPerBat, trainOpt['logNumBat']) == 0:
                    self.loss[epoch] = E / float(batchStart) * float(N)
                    logger.logTrainStats()

            # Store train statistics
            if calcError:
                rate = correct / float(total)
                self.rate[epoch] = rate
            self.loss[epoch] = E

            if not trainOpt.has_key('criterion'):
                Tscore = E
            else:
                if trainOpt['criterion'] == 'loss':
                    Tscore = E
                elif trainOpt['criterion'] == 'rate':
                    Tscore = 1 - rate
                else:
                    raise Exception('Unknown stopping criterion "%s"' % \
                        trainOpt['criterion'])

            # Run validation
            if trainOpt['needValid']:
                VY = tester.test(self.model, VX)
                VE, dVE = self.model.getCost(VY, VT, weights=validInputWeights)
                self.validLoss[epoch] = VE
                if calcError:
                    Vrate, correct, total = tester.calcRate(self.model, VY, VT)
                    self.validRate[epoch] = Vrate

                # Check stopping criterion
                if not trainOpt.has_key('criterion'):
                    Vscore = VE
                else:
                    if trainOpt['criterion'] == 'loss':
                        Vscore = VE
                    elif trainOpt['criterion'] == 'rate':
                        Vscore = 1 - Vrate
                    else:
                        raise Exception('Unknown stopping criterion "%s"' % \
                            trainOpt['criterion'])
                if (bestVscore is None) or (Vscore < bestVscore):
                    bestVscore = Vscore
                    bestTscore = Tscore
                    nAfterBest = 0
                    bestEpoch = epoch
                    # Save trainer if VE is best
                    if trainOpt['saveModel']:
                        self.save()
                else:
                    nAfterBest += 1
                    # Stop training if above patience level
                    if nAfterBest > trainOpt['patience']:
                        print 'Patience level reached, early stop.'
                        print 'Will stop at score ', bestTscore
                        stop = True
            else:
                if trainOpt['saveModel']:
                    self.save()
                if trainOpt.has_key('stopScore') and \
                    Tscore < trainOpt['stopScore']:
                    print 'Training score is lower than %.4f , ealy stop.' % \
                        trainOpt['stopScore'] 
                    stop = True                    

            # Anneal learning rate
            self.model.updateLearningParams(epoch)

            # Print statistics
            logger.logTrainStats()
            if trainOpt['needValid']:
                print 'BT: %.4f' % bestTscore
            print self.name
            
            # Plot train curves
            if trainOpt['plotFigs']:
                plotter.plot()

            # Terminate
            if stop:       
                break

        # Record final epoch number
        self.stoppedTrainScore = bestTscore
        self.stoppedEpoch = bestEpoch if trainOpt['needValid'] else epoch

    def save(self, filename=None):
        if filename is None:
            filename = self.modelFilename
        np.save(filename, self.model.getWeights())
