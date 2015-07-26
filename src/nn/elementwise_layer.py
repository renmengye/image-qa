from layer import Layer

class ElementwiseProductLayer(Layer):
    """
    Currently only support element-wise product of two inputs.
    """
    def __init__(self, name, numNode):
        """

        :param name:
        :param numNode:
        :return:
        """
        Layer.__init__(
            self,
            name=name,
            numNode=numNode)

    def forward(self, inputValue):
        """

        :param inputValue:
        :return:
        """
        self._inputValue = inputValue
        return inputValue[0] * inputValue[1]

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        return [self._inputValue[1] * gradientToOutput,
                self._inputValue[0] * gradientToOutput]


class ElementwiseSumLayer(Layer):
    """
    Currently only support element-wise sum of two inputs.
    """
    def __init__(self, name, numNode):
        """

        :param name:
        :param numNode:
        :return:
        """
        Layer.__init__(
            self,
            name=name,
            numNode=numNode)

    def forward(self, inputValue):
        """

        :param inputValue:
        :return:
        """
        return inputValue[0] + inputValue[1]

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        return [gradientToOutput, gradientToOutput]