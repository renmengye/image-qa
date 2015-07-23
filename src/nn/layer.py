from environment import *

class Layer:
    def __init__(self,
                 name,
                 useGpu=USE_GPU,
                 outputGpu=USE_GPU,
                 # Deprecated
                 outputdEdX=True):
        self.name = name
        self.inputLayers = []
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

    def addInput(self, layer):
        """
        Add an input to this layer
        :param layer: Input layer
        :return:
        """
        if self.inputLayers is None:
            self.inputLayers = [layer]
        else:
            self.inputLayers.append(layer)
        return layer

    def connect(self, layer):
        """
        Connect this layer to next layer
        :param layer: Next layer
        :return:
        """
        layer.addInput(self)
        return layer

    def getInput(self):
        """
        Get inputs from previous layer
        :return:
        """
        if len(self.inputLayers) == 1:
            if self.useGpu and not self.inputLayers[0].outputGpu:
                if VERBOSE:
                    print 'Converting from CPU to GPU'
                return gnp.as_garray(self.inputLayers[0].getValue())
            elif not self.useGpu and self.inputLayers[0].outputGpu:
                if VERBOSE:
                    print 'Converting from GPU to CPU'
                return gnp.as_numpy_array(self.inputLayers[0].getValue())
            else:
                return self.inputLayers[0].getValue()
        else:
            result = []
            for inputLayer in self.inputLayers:
                if self.useGpu and not inputLayer.outputGpu:
                    result.append(gnp.as_garray(inputLayer.getValue()))
                elif not self.useGpu and inputLayer.outputGpu:
                    result.append(gnp.as_numpy_array(inputLayer.getValue()))
                else:
                    result.append(inputLayer.getValue())
            return result

    def clearError(self):
        self._gradientToOutput = 0.0
        self.receivedError = False

    def receiveError(self, gradientToOutput):
        ############################################################
        # Careful here to handle both GPU and CPU gradientToOutput #
        ############################################################
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

    def setValue(self, value):
        """
        Sets the output value.
        :param value: output value.
        :return:
        """
        self._outputValue = value

    def graphForward(self):
        """
        Forward propagates.
        """
        self._inputValue = self.getInput()
        if VERBOSE and hasattr(self._inputValue, 'shape'):
            print 'forward in', self.name, self._inputValue.shape, \
                type(self._inputValue)
        self._outputValue = self.forward(self._inputValue)
        if VERBOSE and hasattr(self._outputValue, 'shape'):
            print 'forward out', self.name, self._outputValue.shape, \
                type(self._outputValue)

    def forward(self, inputValue):
        """
        Abstract method. Forward pass input to the stage.
        :param inputValue: The input. At least two dimensional numpy array.
        The first dimension is always the number of examples.
        :return: The output of the stage.
        """
        raise Exception('Not implemented')

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
        raise Exception('Not implemented')

    # def update(self):
    #     if self.isTraining:
    #         self.weight.update()

    def setIsTraining(self, isTraining):
        self.isTraining = isTraining

    def toDict(self):
        """
        Serialize the layer specifications into a dicitonary. Subclasses are
        expected to override this method to add its own properties.
        :return: A dictionary containing properties.
        """
        return {
            'name': self.name,
            'useGpu': self.useGpu,
            'outputGpu': self.outputGpu
        }

    @staticmethod
    def fromDict(value):
        """
        Contruct a layer based on a dictionary. Subclasses are expected to
        override this method to contruct themselves.
        :param value: A Layer instance.
        :return:
        """
        return Layer(name=value['name'],
                     useGpu=value['useGpu'] if value.has_key('useGpu') else
                     USE_GPU,
                     outputGpu=value['outputGpu'] if value.has_key(
                         'outputGpu') else USE_GPU)