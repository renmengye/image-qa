from environment import *
from layer import WeightLayer

class FullyConnectedLayer(WeightLayer):
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
        FC layer constructor.
        :param name: string, name of the layer.
        :param activationFn: object, subclass instance of ActivationFn.
        :param numNode: int or list, number of nodes.
        :param weight: object, subclass instance of Weight.
        :param hasBias: bool, whether the layer adds a bias.
        :param outputdEdX: bool, whether the layer back-propagate gradient.
        :param gpuEnabled: bool, whether the layer computation is on GPU.
        :return:
        """
        WeightLayer.__init__(self,
                 name=name,
                 numNode=numNode,
                 weight=weight,
                 gpuEnabled=gpuEnabled,
                 outputdEdX=outputdEdX)
        self._activationFn = activationFn
        self._hasBias = hasBias
        if self._activationFn.gpuEnabled != self.gpuEnabled:
            raise Exception('Activation function does not have the same GPU '
                            'configuration as Layer ' + self.name)

    def initialize(self, inputNumNode=None):
        """
        Initialize the weight matrix.
        :param inputNumNode: int or list, number of nodes from the input.
        :return:
        """
        skip = False
        if not self.weight.hasInitialized:
            if len(self.inputLayers) > 0:
                if self.inputLayers[0].numNode > 0:
                    inputNumNode = self.inputLayers[0].numNode
                else:
                    skip = True
            else:
                if inputNumNode is None:
                    skip = True
        if skip:
            # Will initiate lazily at run time.
            print 'Skipped weight initialization for layer', self.name
        else:
            self._initWeight(inputNumNode)

    def _initWeight(self, inputNumNode):
        """

        :param inputNumNode:
        :return:
        """
        print 'Initializing weights for layer', self.name,
        if self._hasBias:
            self.weight.initialize([inputNumNode + 1, self.numNode])
        else:
            self.weight.initialize([inputNumNode.shape[-1], self.numNode])
        print self.weight.get().shape

    def forward(self, inputValue):
        """
        Forward pass.
        :param inputValue:
        :return:
        """
        if not self.weight.hasInitialized:
            self._initWeight(inputValue.shape[-1])
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
        Backward pass. Compute gradient wrt. weights and input.
        :param gradientToOutput: gradient wrt. output.
        :return: numpy or gnumpy array, gradient wrt. input.
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
