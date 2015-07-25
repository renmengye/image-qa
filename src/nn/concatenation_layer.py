from environment import *
from layer import Layer

class ConcatenationLayer(Layer):
    """
    Layer that concatenates a few input arrays into one single array.
    """
    def __init__(self, name, numNode, axis, gpuEnabled=USE_GPU):
        """
        Construct a concatenation layer.
        :param name: string, layer name.
        :param numNode: int, number of nodes in the feature dimension.
        :param axis: int, on which axis will the concatenation happen,
        supports negative integers.
        :param gpuEnabled: bool, whether this layer will operate on GPU.
        :return:
        """
        Layer.__init__(self, name=name, numNode=numNode, gpuEnabled=gpuEnabled)
        self.axis = axis

    def forward(self, inputValue):
        """
        Run forward direction.
        :param inputValue: A list of numpy.ndarray or gnumpy.garray.
        :return: One numpy.ndarray or gnumpy.garray.
        """
        self._inputValue = inputValue
        if self.gpuEnabled:
            return gnp.concatenate(inputValue, axis=self.axis)
        else:
            return np.concatenate(inputValue, axis=self.axis)

    def backward(self, gradientToOutput):
        """
        Run backward direction
        :param gradientToOutput: One numpy.ndarray or gnumpy.garray.
        :return: A list of numpy.ndarray or gnumpy.garray.
        """
        gradientToInput = []
        s = 0
        axis = np.mod(self.axis, len(self._inputValue[0].shape))
        for inputItem in self._inputValue:
            s2 = s + inputItem.shape[self.axis]
            if axis == 0:
                gradientToInput.append(gradientToOutput[s : s2])
            elif axis == 1:
                gradientToInput.append(gradientToOutput[:, s : s2])
            elif axis == 2:
                gradientToInput.append(gradientToOutput[:, :, s : s2])
            else:
                raise Exception('Concatenation with axis larger than 2 is not '
                                'supported.')
            s = s2
        return gradientToInput
