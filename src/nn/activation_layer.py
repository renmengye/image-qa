from layer import Layer

class ActivationLayer(Layer):
    """

    """
    def __init__(self,
                 name,
                 activationFn,
                 numNode):
        """

        :param name:
        :param activationFn:
        :param numNode:
        :return:
        """
        Layer.__init__(self, name=name, numNode=numNode)
        self.activationFn = activationFn

    def forward(self, inputValue):
        """

        :param inputValue:
        :return:
        """
        self._outputValue = self.activationFn.forward(inputValue)
        return self._outputValue

    def backward(self, gradientToOutput):
        """

        :param gradientToOutput:
        :return:
        """
        return self.activationFn.backward(gradientToOutput)