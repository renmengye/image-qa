from layer import *

class LossLayer(Layer):
    def __init__(self, name=None, useGpu=USE_GPU):
        Layer.__init__(
                 self,
                 name=name,
                 useGpu=useGpu,
                 outputGpu=False,
                 outputdEdX=True)

    def forward(self, inputValue):
        outputValue = inputValue[0]
        targetValue = inputValue[1]
        if len(inputValue) > 2:
            weights = outputValue[2]
        else:
            weights = None
        loss, gradientToInput = self.computeLossWithGrad(outputValue,
                                                         targetValue,
                                                         weights)
        self.gradientToInput = gradientToInput
        return loss

    def backward(self, gradientToOutput):
        return self.gradientToInput

    def computeLossWithGrad(self, outputValue, targetValue, weights=None):
        """
        Abstract method
        :param outputValue: Model output
        :param targetValue: Model target
        :return: Tuple of the loss value and the gradient of loss w.r.t. Y.
        """
        pass