from environment import *

class Layer:
    def __init__(self,
                 name,
                 weight=None,
                 outputDim=0,
                 defaultValue=0.0,
                 useGpu=False,
                 outputGpu=False,
                 outputdEdX=True):
        self.name = name
        self.inputLayers = []
        self.outputDim = outputDim
        self.defaultValue = np.zeros(outputDim) + defaultValue
        self.weight = weight
        self._outputValue = 0.0
        self._inputValue = 0.0
        self._gradientToOutput = 0.0
        self._gradientToInput = 0.0
        self.useGpu = useGpu
        self.outputGpu = outputGpu
        self.receivedError = False
        self.isTraining = True

        # deprecated. Should be able to compute from isTraining
        self.outputdEdX = outputdEdX

    def __str__(self):
        return self.name

    def addInput(self, stage):
        if self.inputLayers is None:
            self.inputLayers = [stage]
        else:
            self.inputLayers.append(stage)

    def getInput(self):
        """
        Get inputs from previous layer
        :return:
        """
        if len(self.inputLayers) == 1:
            return self.inputLayers[0].Y
        else:
            return [inputLayer.Y for inputLayer in self.inputLayers]

    def clearError(self):
        self._gradientToOutput = 0.0
        self.receivedError = False

    def receiveError(self, gradientToOutput):
        self._gradientToOutput += gradientToOutput
        self.receivedError = True

    def sendError(self, gradientToInput):
        """
        :param gradientToInput:
        :return:
        """
        if len(self.inputLayers) == 1:
            self.inputLayers[0].receiveError(gradientToInput)
        else:
            for i in range(len(self.inputLayers)):
                self.inputLayers[i].receiveError(gradientToInput[i])

    def getValue(self):
        """
        Gets the output value.
        """
        return self._outputValue

    def graphForward(self):
        """
        Forward propagates.
        """
        self._inputValue = self.getInput()
        if VERBOSE and hasattr(self._inputValue, 'shape'):
            print 'forward in', self.name, self._inputValue.shape
        self._outputValue = self.forward(self._inputValue)
        if VERBOSE and hasattr(self._outputValue, 'shape'):
            print 'forward out', self.name, self._outputValue.shape

    def forward(self, inputValue):
        """
        Abstract method. Forward pass input to the stage.
        :param inputValue: The input. At least two dimensional numpy array.
        The first dimension is always the number of examples.
        :return: The output of the stage.
        """
        return

    def graphBackward(self):
        """
        Backward propagates.
        """
        if VERBOSE and hasattr(self._gradientToOutput, 'shape'):
            print 'backward in', self.name, self._gradientToOutput.shape, \
                np.mean(self._gradientToOutput)
        self._gradientToInput = self.backward(self._gradientToOutput)
        if self.outputdEdX:
            self.sendError(self._gradientToInput)
        if VERBOSE and hasattr(self._gradientToInput, 'shape'):
            print 'backward out', self.name, self._gradientToInput.shape, \
                np.mean(self._gradientToInput)

    def backward(self, gradientToOutput):
        """
        Abstract method. Backward propagate error in the stage.
        :param gradientToOutput: The error of the output.
        :return: The error of the input.
        """
        return

    def update(self):
        if self.isTraining:
            self.weight.update()

    def setIsTraining(self, isTraining):
        self.isTraining = isTraining