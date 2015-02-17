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
        self.currentLearningRate = learningRate
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
        if self.gradientClip > 0.0:
            self.dEdWnorm = np.sqrt(np.sum(np.power(self.dEdW, 2)))
            if self.dEdWnorm > self.gradientClip:
                self.dEdW *= self.gradientClip / self.dEdWnorm
        if self.currentLearningRate > 0.0:
            self.lastdW = -self.currentLearningRate * self.dEdW + \
                           self.momentum * self.lastdW
            self.W += self.lastdW
        if self.weightRegConst > 0.0:
            a = self.currentLearningRate * self.weightRegConst
            self.W -= a * self.W
            pass
        if self.weightClip > 0.0:
            self.Wnorm = np.sqrt(np.sum(np.power(self.W, 2)))
            if self.Wnorm > self.weightClip:
                self.W *= self.weightClip / self.Wnorm
        return

    def updateLearningParams(self, numEpoch):
        self.currentLearningRate = self.startLearningRate / \
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
        return
