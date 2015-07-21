from layer import *

class LossLayer(Layer):
    def __init__(self, inputNames, name=None):
        Layer.__init__(
                 self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=1,
                 defaultValue=0.0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 useGpu=False,
                 outputGpu=False,
                 outputdEdX=True)

    def forward(self, inputValue):
        Y = inputValue[0]
        T = inputValue[1]
        if len(inputValue) > 2:
            weights = inputValue[2]
        else:
            weights = None
        E, dEdX = self.computeLossWithGrad(Y, T, weights)
        self.dEdX = dEdX
        return E

    def backward(self, gradientToOutput):
        return self.dEdX

    def computeLossWithGrad(self, Y, T, weights=None):
        """
        Abstract method
        :param Y: Model output
        :param T: Model target
        :return: Tuple of the loss value and the gradient of loss w.r.t. Y.
        """
        pass