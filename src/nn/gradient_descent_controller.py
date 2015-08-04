from environment import *

class GradientDescentControllerFactory():
    singletonControllers = {}
    def __init__(self):
        pass

    @staticmethod
    def createFromSpec(name, spec):
        if spec['name'] in \
                GradientDescentControllerFactory.singletonControllers:
            return GradientDescentControllerFactory.\
                singletonControllers[spec['name']]
        if spec['type'] != 'sgd':
            raise Exception(
                'Gradient descent controller type not implemented: ' +
                spec['type'])
        learningRate = 0.0 \
            if 'learningRate' not in spec else spec['learningRate']
        momentum = 0.0 if 'momentum' not in spec else spec['momentum']
        gradientClip = 0.0 if 'gradientClip' not in spec else spec['gradientClip']
        weightClip = 0.0 if 'weightClip' not in spec else spec['weightClip']
        weightRegConst = 0.0 \
            if 'weightRegConst' not in spec else spec['weightRegConst']
        gpuEnabled = USE_GPU if 'gpuEnabled' not in spec else spec['gpuEnabled']
        controller = SGDController(name=spec['name'],
                                   learningRate=learningRate,
                                   momentum=momentum,
                                   gradientClip=gradientClip,
                                   weightClip=weightClip,
                                   weightRegConst=weightRegConst,
                                   gpuEnabled=gpuEnabled)
        GradientDescentControllerFactory.\
            singletonControllers[spec['name']] = controller

class SGDController():
    def __init__(self,
                 name,
                 learningRate=0.0,
                 momentum=0.0,
                 gradientClip=0.0,
                 weightClip=0.0,
                 weightRegConst=0.0,
                 # learningRateAnnealConst,
                 # deltaMomentum,
                 gpuEnabled=USE_GPU):
        self._learningRate = learningRate
        self._startLearningRate = learningRate
        self._momentum = momentum
        self._gradientClip = gradientClip
        self._weightClip = weightClip
        self._weightRegConst = weightRegConst
        self._learningRateAnnealConst = 0.0
        self._deltaMomentum = 0.0
        #self._learningRateAnnealConst = learningRateAnnealConst
        #self._deltaMomentum = deltaMomentum
        self._gpuEnabled = gpuEnabled
        self._gradientHistory = 0
        self._gradientNorm = 0
        self._weightNorm = 0
        self._counter = 0

    def getGradientHistory(self):
        if self._gpuEnabled:
            return gnp.as_numpy_array(self._gradientHistory)
        else:
            return self._gradientHistory

    def setGradientHistory(self, gradientHistory):
        if self._gpuEnabled:
            self._gradientHistory = gnp.as_garray(self._gradientHistory)
        else:
            self._gradientHistory = gradientHistory

    def updateWeight(self, weight, gradient):
        if self._gpuEnabled:
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

    def toDict(self):
        return {
            'type': 'sgd',
            'learningRate': self._learningRate,
            'momentum': self._momentum,
            'gradientClip': self._gradientClip,
            'weightClip': self._weightClip,
            'weightRegConst': self._weightRegConst,
            'gpuEnabled': self._gpuEnabled
        }