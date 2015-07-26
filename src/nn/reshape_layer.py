from layer import Layer
from environment import *

class ReshapeLayer(Layer):
    """
    A layer that reshapes matrix.
    """
    def __init__(self, name, reshapeFn, numNode=0, gpuEnabled=USE_GPU):
        """

        :param name:
        :param reshapeFn:
        :param numNode:
        :param gpuEnabled:
        :return:
        """
        Layer.__init__(self, name=name, numNode=numNode, gpuEnabled=gpuEnabled)
        self.reshapeFn = eval('lambda x: ' + reshapeFn)
        self.Xshape = 0

    def forward(self, inputValue):
        """

        :param inputValue:
        :return:
        """
        self.Xshape = inputValue.shape
        if self.gpuEnabled:
            return gnp.reshape(inputValue, self.reshapeFn(inputValue.shape))
        else:
            return np.reshape(inputValue, self.reshapeFn(inputValue.shape))

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        if self.gpuEnabled:
            return gnp.reshape(gradientToOutput, self.Xshape)
        else:
            return np.reshape(gradientToOutput, self.Xshape)