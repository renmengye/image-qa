from layer import *

class InnerProduct(Layer):
    """
    Inner product calculates the inner product of two input vectors.
    Two vectors aligns on the second axis (time-axis).
    """
    def __init__(self,
                name,
                inputNames,
                outputDim,
                learningRate=0.0,
                learningRateAnnealConst=0.0,
                momentum=0.0,
                deltaMomentum=0.0,
                weightClip=0.0,
                gradientClip=0.0,
                weightRegConst=0.0,
                outputdEdX=True):
        Layer.__init__(self,
                 name=name,
                 outputDim=outputDim,
                 inputNames=inputNames,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        self.W = 1
    def forward(self, inputValue):
        Y = np.sum(inputValue[:, 0, :] * inputValue[:, 1, :], axis=-1) + self.W
        self._outputValue = Y
        self._inputValue = inputValue
        return Y

    def backward(self, gradientToOutput):
        self.dEdW = np.sum(gradientToOutput,axis=0)
        #print dEdY
        dEdX = np.zeros(self._inputValue.shape)
        dEdX[:, 1, :] = gradientToOutput.reshape(gradientToOutput.size, 1) * self._inputValue[:, 0, :]
        dEdX[:, 0, :] = gradientToOutput.reshape(gradientToOutput.size, 1) * self._inputValue[:, 1, :]
        return dEdX