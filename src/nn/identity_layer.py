from layer import Layer
from environment import *

class IdentityLayer(Layer):
    """
    A layer that sends the input to the output.
    """
    def __init__(self, name, numNode):
        """

        :param name:
        :param numNode:
        :return:
        """
        Layer.__init__(self, name=name, numNode=numNode, gpuEnabled=USE_GPU)
        pass

    def graphForward(self):
        """

        :return:
        """
        pass

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        return gradientToOutput