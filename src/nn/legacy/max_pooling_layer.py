from layer import *

class MaxPooling1DLayer(Layer):
    """
    1D max pooling.
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
        mod = np.mod(inputValue.shape[1], self.windowSize)
        if mod > 0:
            inputValue = np.concatenate((inputValue, np.zeros((inputValue.shape[0], self.windowSize - mod, inputValue.shape[2]))), axis=1)
        inputValue = inputValue.reshape(inputValue.shape[0], self.windowSize, inputValue.shape[1] / self.windowSize, inputValue.shape[2])
        self.argX = np.argmax(inputValue, axis=1)
        Y = np.max(inputValue, axis=1)
        self._inputValue = inputValue
        self.mod = mod
        return Y

    def backward(self, gradientToOutput):
        """
        Assuming the last dimension is the largest.
        """
        self.dEdW = 0
        dEdX = np.zeros(self._inputValue.shape)
        for i in range(self._inputValue.shape[0]):
            for j in range(self._inputValue.shape[2]):
                dEdX[i, self.argX[i, j, :], j, range(0, self._inputValue.shape[3])] = gradientToOutput[i, j, :]
        dEdX = dEdX.reshape(dEdX.shape[0], dEdX.shape[1] * dEdX.shape[2], dEdX.shape[3])
        if self.mod > 0:
            dEdX = dEdX[:, :-(self.windowSize - self.mod), :]
        return dEdX