from layer import Layer
from environment import *

class SliceLayer(Layer):
    def __init__(self,
                 name,
                 start,
                 end,
                 numNode,
                 axis=-1,
                 gpuEnabled=USE_GPU):
        """

        :param name:
        :param start:
        :param end:
        :param numNode:
        :param axis:
        :param gpuEnabled:
        :return:
        """
        Layer.__init__(self, name=name, numNode=numNode, gpuEnabled=gpuEnabled)
        self._start = start
        self._end = end
        self._axis = axis

    def forward(self, inputValue):
        """

        :param inputValue:
        :return:
        """
        self._inputValue = inputValue
        axis = np.mod(self._axis, len(self._inputValue[0].shape))
        if axis == 0:
            return inputValue[self._start:self._end]
        elif axis == 1:
            return inputValue[:, self._start:self._end]
        elif axis == 2:
            return inputValue[:, :, self._start:self._end]
        else:
            raise Exception('Slicing with axis larger than 2 is not '
                            'supported.')

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        gradientToInput = np.zeros(self._inputValue.shape)
        axis = np.mod(self._axis, len(self._inputValue[0].shape))
        if axis == 0:
            gradientToInput[self._start:self._end] = gradientToOutput
        elif axis == 1:
            gradientToInput[:, self._start:self._end] = gradientToOutput
        elif axis == 2:
            gradientToInput[:, :, self._start:self._end] = gradientToOutput
        else:
            raise Exception('Slicing with axis larger than 2 is not '
                            'supported.')
        return gradientToInput