import numpy as np
import copy

class Stage:
    def __init__(self,
                 name,
                 inputNames,
                 outputDim,
                 defaultValue=0.0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True):
        self.name = name
        self.inputNames = inputNames
        self.inputs = None
        self.outputDim = outputDim
        self.defaultValue = np.zeros(outputDim) + defaultValue
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
        self.X = 0.0
        self.dEdY = 0.0
        self.splX = None
    def __str__(self):
        return self.name

    def addInput(self, stage):
        if self.inputs is None:
            self.inputs = [stage]
        else:
            self.inputs.append(stage)

    def getInput(self):
        """
        Fetches input from each input stage.
        Concatenates input into one vector.
        """
        if len(self.inputs) > 1:
            self.splX = []
            for stage in self.inputs:
                X = stage.Y
                self.splX.append(X)
            return np.concatenate(self.splX, axis=-1)
        else:
            return self.inputs[0].Y

    def clearError(self):
        self.dEdY = 0.0

    def sendError(self, dEdX):
        """
        Iterates over input list and sends dEdX.
        """
        if len(self.inputs) > 1:
            s = 0
            for stage in self.inputs:
                s2 = s + stage.Y.shape[-1]
                stage.dEdY += dEdX[:, s : s2]
                s = s2
        else:
            #if type(self.inputs[0].dEdY) == np.ndarray:
            #    print self.name, self.inputs[0].name, self.inputs[0].dEdY.shape, dEdX.shape
            self.inputs[0].dEdY += dEdX

    def getValue(self):
        """
        Gets the output value.
        """
        return self.Y

    def getGradient(self):
        """
        Gets the gradient with regard to the weights.
        """
        return self.dEdW

    def setGradient(self, value):
        """
        Sets the gradient with regard to the weights.
        :param value: float or numpy array
        :return:
        """
        self.dEdW = value

    def graphForward(self):
        """
        Forward propagates.
        """
        self.X = self.getInput()
        # if hasattr(self.X, 'shape'):
        #     print 'forward', self.name, self.X.shape
        self.Y = self.forward(self.X)

    def forward(self, X):
        """
        Abstract method. Forward pass input to the stage.
        :param X: The input. At least two dimensional numpy array.
        The first dimension is always the number of examples.
        :return: The output of the stage.
        """
        return

    def graphBackward(self):
        """
        Backward propagates.
        """
        # if hasattr(self.dEdY, 'shape'):
        #     print 'backward', self.name, self.dEdY.shape
        dEdX = self.backward(self.dEdY)
        if self.outputdEdX:
            self.sendError(dEdX)

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

    def getWeights(self):
        return self.W

    def loadWeights(self, W):
        self.W = W

    def copy(self):
        return copy.copy(self)
