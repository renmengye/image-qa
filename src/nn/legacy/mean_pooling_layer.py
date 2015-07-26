from layer import *

class MeanPooling1DLayer(Layer):
    """
    1D mean pooling. 
    Padding no longer make sense now. 
    Make sure you have the right size.
    """
    def __init__(self,
                 outputDim,
                 windowSize,
                 inputNames=None,
                 defaultValue=0.0,
                 outputdEdX=True,
                 name=None):
        Layer.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 defaultValue=defaultValue,
                 outputdEdX=outputdEdX)
        self.windowSize = windowSize
        self._inputValue = 0
        self._outputValue = 0

    def forward(self, inputValue):
        inputValue = inputValue.reshape(inputValue.shape[0], self.windowSize, inputValue.shape[1] / self.windowSize, inputValue.shape[2])
        Y = np.mean(inputValue, axis=1)
        self._inputValue = inputValue
        return Y

    def backward(self, gradientToOutput):
        dEdX = np.tile(
            gradientToOutput.reshape(gradientToOutput.shape[0], 1, gradientToOutput.shape[1], gradientToOutput.shape[2]),
            (1, self.windowSize, 1, 1))
        dEdX /= float(self.windowSize)
        dEdX = dEdX.reshape(dEdX.shape[0], dEdX.shape[1] * dEdX.shape[2], dEdX.shape[3])
        return dEdX