from environment import *

class gradientDescentController():
    def __init__(self,
                 learningRate,
                 momentum,
                 gradientClip,
                 weightClip,
                 weightRegConst,
                 learningRateAnnealConst,
                 deltaMomentum,
                 useGpu):
        self._learningRate = learningRate
        self._startLearningRate = learningRate
        self._momentum = momentum
        self._gradientClip = gradientClip
        self._weightClip = weightClip
        self._weightRegConst = weightRegConst
        self._learningRateAnnealConst = learningRateAnnealConst
        self._deltaMomentum = deltaMomentum
        self._useGpu = useGpu
        self._gradientHistory = 0
        self._gradientNorm = 0
        self._weightNorm = 0
        self._counter = 0

    def getGradientHistory(self):
        if self._useGpu:
            return gnp.as_numpy_array(self._gradientHistory)
        else:
            return self._gradientHistory

    def setGradientHistory(self, gradientHistory):
        if self._useGpu:
            self._gradientHistory = gnp.as_garray(self._gradientHistory)
        else:
            self._gradientHistory = gradientHistory

    def updateWeight(self, weight, gradient):
        if self._useGpu:
            if self._gradientClip > 0.0:
                gradientNorm = gnp.sqrt(gnp.sum(gradient ** 2))
                if gradientNorm > self._gradientClip:
                    gradient *= self._gradientClip / gradientNorm
            if self._learningRate > 0.0:
                self._gradientHistory = -self._learningRate * gradient + \
                           self._momentum * self._gradientHistory
                weight += self._gradientHistory
            if self._weightRegConst > 0.0:
                a = self._learningRate * self._weightRegConst
                weight -= a * weight
            if self._weightClip > 0.0:
                self._weightNorm = gnp.sqrt(gnp.sum(weight ** 2))
                if self._weightNorm > self._weightClip:
                    weight *= self._weightClip / self._weightNorm
        else:
            if self._gradientClip > 0.0:
                gradientNorm = np.sqrt(np.sum(gradient ** 2))
                if gradientNorm > self._gradientClip:
                    gradient *= self._gradientClip / gradientNorm
            if self._learningRate > 0.0:
                self._gradientHistory = -self._learningRate * gradient + \
                           self._momentum * self._gradientHistory
                weight += self._gradientHistory
            if self._weightRegConst > 0.0:
                a = self._learningRate * self._weightRegConst
                weight -= a * weight
            if self._weightClip > 0.0:
                self._weightNorm = np.sqrt(np.sum(weight ** 2))
                if self._weightNorm > self._weightClip:
                    weight *= self._weightClip / self._weightNorm
        return weight

    def updateLearningParams(self):
        self._learningRate = self._startLearningRate / \
                                   (1.0 + self._learningRateAnnealConst *
                                    self._counter)
        self._momentum -= self._deltaMomentum

        # Add monitor for this piece of code
        # if self._gradientClip > 0.0 or self._weightClip > 0.0:
        #     print 'ST: %11s ' % self.name,
        #     if self.gradientClip > 0.0:
        #         print 'GN: %8.4f ' % self.dEdWnorm,
        #         print 'GC: %8.4f ' % self.gradientClip,
        #     if self.weightClip > 0.0:
        #         print 'WN: %8.4f ' % self.Wnorm,
        #         print 'WC: %8.4f ' % self.weightClip,
        #     print

    def incrementCounter(self):
        self._counter += 1