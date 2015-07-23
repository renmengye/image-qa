from layer import *

class LossLayer(Layer):
    def __init__(self, name=None, gpuEnabled=USE_GPU):
        Layer.__init__(
                 self,
                 name=name,
                 numNode=1,
                 gpuEnabled=gpuEnabled,
                 outputdEdX=True)
        self._loss = 0.0

    def getLoss(self):
        return self._loss;

    def forward(self, inputValue):
        outputValue = inputValue[0]
        targetValue = inputValue[1]
        if len(inputValue) > 2:
            weights = outputValue[2]
        else:
            weights = None
        self._loss, self._gradientToInput = \
            self.computeLossWithGrad(outputValue, targetValue, weights)
        return self._loss

    def backward(self, gradientToOutput):
        return self._gradientToInput

    def computeLossWithGrad(self, outputValue, targetValue, weights=None):
        """
        Abstract method
        :param outputValue: Model output
        :param targetValue: Model target
        :return: Tuple of the loss value and the gradient of loss w.r.t. Y.
        """
        pass