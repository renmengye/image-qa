from func import *

class Stage:
    def __init__(self,
                 name=None,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True):
        self.name = name
        self.startLearningRate = learningRate
        self.learningRate = learningRate
        self.learningRateAnnealConst = learningRateAnnealConst
        self.momentum = momentum
        self.deltaMomentum = deltaMomentum
        self.weightClip = weightClip
        self.gradientClip = gradientClip
        self.weightRegConst = weightRegConst
        self.outputdEdX=outputdEdX
        self.dEdWnorm = 0.0
        self.Wnorm = 0.0
        self.dEdW = 0.0
        self.lastdW = 0.0
        self.W = 0.0
        self.Y = 0.0

    def getValue(self):
        return self.Y

    def getGradient(self):
        return self.dEdW

    def forward(self, X):
        """
        Abstract method. Forward pass input to the stage.
        :param X: The input. At least two dimensional numpy array.
        The first dimension is always the number of examples.
        :return: The output of the stage.
        """
        return

    def backward(self, dEdY):
        """
        Abstract method. Backward propagate error in the stage.
        :param dEdY: The error of the output.
        :return: The error of the input.
        """
        return

    def updateWeights(self):
        self._updateWeights(self.dEdW)

    def _updateWeights(self, dEdW):
        # print self.name
        # print self.learningRate
        # print self.momentum
        if self.gradientClip > 0.0:
            self.dEdWnorm = np.sqrt(np.sum(np.power(dEdW, 2)))
            if self.dEdWnorm > self.gradientClip:
                dEdW *= self.gradientClip / self.dEdWnorm
        if self.learningRate > 0.0:
            self.lastdW = -self.learningRate * dEdW + \
                           self.momentum * self.lastdW
            self.W += self.lastdW
        if self.weightRegConst > 0.0:
            a = self.learningRate * self.weightRegConst
            self.W -= a * self.W
        if self.weightClip > 0.0:
            self.Wnorm = np.sqrt(np.sum(np.power(self.W, 2)))
            if self.Wnorm > self.weightClip:
                self.W *= self.weightClip / self.Wnorm

    def updateLearningParams(self, numEpoch):
        self.learningRate = self.startLearningRate / \
                                   (1.0 + self.learningRateAnnealConst * numEpoch)
        self.momentum -= self.deltaMomentum

        if self.gradientClip > 0.0 or self.weightClip > 0.0:
            print 'ST: %11s ' % self.name,
            if self.gradientClip > 0.0:
                print 'GN: %8.4f ' % self.dEdWnorm,
                print 'GC: %8.4f ' % self.gradientClip,
            if self.weightClip > 0.0:
                print 'WN: %8.4f ' % self.Wnorm,
                print 'WC: %8.4f ' % self.weightClip,
            print
        return

    def getWeights(self):
        return self.W

    def loadWeights(self, W):
        self.W = W
