from environment import *
from layer import Layer

class FullyConnectedLayer(Layer):
    """

    """
    def __init__(self,
                 name,
                 activationFn,
                 numNode,
                 weight,
                 hasBias=True,
                 outputdEdX=True,
                 gpuEnabled=USE_GPU):
        """

        :param name:
        :param activationFn:
        :param numNode:
        :param weight:
        :param hasBias:
        :param outputdEdX:
        :param gpuEnabled:
        :return:
        """
        Layer.__init__(self,
                 name=name,
                 numNode=numNode,
                 gpuEnabled=gpuEnabled,
                 outputdEdX=outputdEdX)
        self._activationFn = activationFn
        self._hasBias = hasBias
        self.weight = weight
        if self._activationFn.gpuEnabled != self.gpuEnabled:
            raise Exception('Activation function does not have the same GPU '
                            'configuration as Layer ' + self.name)

    def init(self, inputNumNode=None):
        """

        :param inputNumNode:
        :return:
        """
        if len(self.inputLayers) > 0:
            inputNumNode = self.inputLayers[0].numNode
        else:
            if inputNumNode is None:
                raise Exception('Need to specify input node dimension if no '
                                'connection is present.')
        self._initWeight(inputNumNode)

    def _initWeight(self, inputNumNode):
        """

        :param inputNumNode:
        :return:
        """
        if not self.weight.hasInitialized:
            if self._hasBias:
                self.weight.initialize([inputNumNode + 1, self.numNode])
            else:
                self.weight.initialize([inputNumNode.shape[-1], self.numNode])

    def forward(self, inputValue):
        """

        :param inputValue:
        :return:
        """
        if self.gpuEnabled:
            if self._hasBias:
                self._inputValue = \
                    gnp.concatenate(
                        (inputValue, gnp.ones((inputValue.shape[0], 1))),
                        axis=-1)
            else:
                self._inputValue = inputValue
            weightedSum = gnp.dot(self._inputValue, self.weight.get())
            self._outputValue = self._activationFn.forward(weightedSum)
        else:
            if self._hasBias:
                self._inputValue = \
                    np.concatenate(
                        (inputValue, np.ones((inputValue.shape[0], 1),
                                             dtype=inputValue.dtype)), axis=-1)
            else:
                self._inputValue = inputValue
            weightedSum = np.dot(self._inputValue, self.weight.get())
            self._outputValue = self._activationFn.forward(weightedSum)
        return self._outputValue

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        gradientToWeightedSum = self._activationFn.backward(gradientToOutput)
        if self.gpuEnabled:
            gradient = gnp.dot(self._inputValue.transpose(),
                               gradientToWeightedSum)
            if self._hasBias:
                gradientToInput = gnp.dot(gradientToWeightedSum,
                                          self.weight.get()[:-1, :].transpose())
            else:
                gradientToInput = gnp.dot(gradientToWeightedSum,
                                          self.weight.get().transpose())
        else:
            gradient = np.dot(self._inputValue.transpose(),
                              gradientToWeightedSum)
            if self._hasBias:
                gradientToInput = np.dot(gradientToWeightedSum,
                                         self.weight.get()[:-1, :].transpose())
            else:
                gradientToInput = np.dot(gradientToWeightedSum,
                                         self.weight.get().transpose())

        self.weight.addGradient(gradient)
        return gradientToInput if self.outputdEdX else None
