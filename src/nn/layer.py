from environment import *

class Layer:
    def __init__(self,
                 name,
                 numNode,
                 gpuEnabled=USE_GPU,
                 # Deprecated
                 outputdEdX=True):
        if ':' in name:
            raise Exception('Layer name does not allow character ":"')
        self.name = name
        self.inputLayers = []
        self.numNode = numNode
        self._outputValue = 0.0
        self._inputValue = 0.0
        self._gradientToOutput = 0.0
        self._gradientToInput = 0.0
        self.gpuEnabled = gpuEnabled
        self.isTraining = True

        # deprecated. Should be able to compute from isTraining
        self.outputdEdX = outputdEdX

    def __str__(self):
        return self.name

    def initialize(self):
        """
        Abstract method, for initializing weights, etc.
        :return:
        """
        pass

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
            return self._getInput(self.inputLayers[0])
        else:
            return [self._getInput(inputLayer )for inputLayer in
                    self.inputLayers]

    def _getInput(self, inputLayer):
        """
        Get input from one of the previous layers
        :param inputLayer:
        :return:
        """
        if self.gpuEnabled and not inputLayer.gpuEnabled:
            if VERBOSE:
                print 'Converting from CPU to GPU'
            return gnp.as_garray(inputLayer.getValue())
        elif not self.gpuEnabled and inputLayer.gpuEnabled:
            if VERBOSE:
                print 'Converting from GPU to CPU'
            return gnp.as_numpy_array(inputLayer.getValue())
        else:
            return inputLayer.getValue()

    def clearError(self):
        """

        :return:
        """
        self._gradientToOutput = 0.0

    def receiveError(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        self._gradientToOutput += gradientToOutput

    def sendError(self, gradientToInput):
        """
        :param gradientToInput:
        :return:
        """
        if len(self.inputLayers) == 1:
            inputLayer = self.inputLayers[0]
            self._sendError(inputLayer, gradientToInput)
        else:
            for inputLayer, gradient in zip(self.inputLayers, gradientToInput):
                self._sendError(inputLayer, gradient)

    def _sendError(self, inputLayer, gradientToInput):
        if inputLayer.gpuEnabled and not self.gpuEnabled:
            inputLayer.receiveError(
                gnp.as_garray(gradientToInput))
        elif not inputLayer.gpuEnabled and self.gpuEnabled:
            inputLayer.receiveError(
                gnp.as_numpy_array(gradientToInput))
        else:
            inputLayer.receiveError(gradientToInput)

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
        if VERBOSE:
            print '-->> in',
            self._printValueDebugString(self._inputValue)
        self._outputValue = self.forward(self._inputValue)
        if VERBOSE:
            print '-->> out',
            self._printValueDebugString(self._outputValue)

    def _printValueDebugString(self, value):
        print self.name, \
            (value.shape if hasattr(self._inputValue, 'shape') else ''), \
            self._getTypeString(value), \
            'mean', \
            '%.4f' % \
            (np.mean(value) if not self.gpuEnabled else gnp.mean(value))

    def _getTypeString(self, value):
        if type(value) is np.ndarray or type(value) is np.float64 or type(
                value) is np.float32:
            return 'CPU'
        elif self.gpuEnabled:
            if type(value) is gnp.garray:
                return 'GPU'
        else:
            return str(type(value))

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
        if VERBOSE:
            print '<<-- in',
            self._printValueDebugString(self._gradientToOutput)
        self._gradientToInput = self.backward(self._gradientToOutput)
        if self.outputdEdX:
            self.sendError(self._gradientToInput)
        if VERBOSE:
            print '<<-- out',
            self._printValueDebugString(self._gradientToInput)

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
            'numNode': self.numNode,
            'gpuEnabled': self.gpuEnabled
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
                     numNode=value['numNode'],
                     gpuEnabled=value['useGpu'] if value.has_key('useGpu') else
                     USE_GPU)


class WeightLayer(Layer):
    def __init__(self,
                 name,
                 numNode,
                 weight,
                 gpuEnabled=USE_GPU,
                 # Deprecated
                 outputdEdX=True):
        Layer.__init__(self,
                 name=name,
                 numNode=numNode,
                 gpuEnabled=gpuEnabled,
                 outputdEdX=outputdEdX)
        self.weight = weight

    def serializeWeight(self):
        """
        Save the weights to a dictionary if the weights.
        :return:
        """
        if self.weight.savedToFile:
            return self.weight.serialize()
        else:
            return {}

    def deserializeWeight(self, value):
        """
        Read the weights from a dictionary.
        :param value: A dictionary with keys to be names of the weights and
        values are the serialized version of the weights.
        :return:
        """
        if not self.weight.hasInitialized:
            self.weight.deserialize(value)