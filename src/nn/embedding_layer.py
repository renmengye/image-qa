from layer import WeightLayer
from environment import *

class EmbeddingLayer(WeightLayer):
    """
    Embedding layer (i.e. look-up table.)
    This implementation of look-up table is 1-based index.
    The first row of the weight matrix are zeros.
    This layer can only stay in CPU.
    """
    def __init__(self,
                 name,
                 inputDim,
                 numNode,
                 weight,
                 sparse=False):
        """
        Construct an embedding layer.
        :param name: Layer name
        :param inputDim: Maximum input index.
        :param numNode: Number of output nodes.
        :param weight: Weight object.
        :param sparse: Whether the weight matrix is stored in row sparse
        representation.
        :return:
        """
        WeightLayer.__init__(self,
                             name=name,
                             numNode=numNode,
                             weight=weight,
                             gpuEnabled=False)

        # Zeroth rows of the weight matrix is reserved
        # for empty word at the end of a sentence.
        self.weight = weight
        self._sparse = sparse
        self._inputDim = inputDim

    def initialize(self):
        """

        :return:
        """
        if not self.weight.hasInitialized:
            print 'Initializing weights for layer', self.name,
            self.weight.initialize([self._inputDim + 1, self.numNode])
            print self.weight.get().shape
        self.weight.get()[0] *= 0.0

    def forward(self, inputValue):
        """

        :param inputValue:
        :return:
        """
        inputValue = inputValue.astype(int)
        self._inputValue = inputValue
        inputValue = inputValue.reshape(inputValue.size)
        weight = self.weight.get()
        numEx = inputValue.shape[0]
        outputValue = np.zeros((numEx, self.numNode), weight.dtype)
        for n in range(0, numEx):
            if self._sparse:
                if inputValue[n] != 0:
                    outputValue[n] = weight[inputValue[n] - 1].todense()
            else:
                if inputValue[n] != 0:
                    outputValue[n] = weight[inputValue[n] - 1]
        return outputValue

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        inputValue = self._inputValue
        weight = self.weight.get()
        gradient = np.zeros(weight.shape, weight.dtype)
        numEx = inputValue.shape[0]
        for n in range(0, numEx):
            gradient[inputValue[n] - 1] += gradientToOutput[n]
        self.weight.addGradient(gradient)
        return np.zeros(inputValue.shape)
