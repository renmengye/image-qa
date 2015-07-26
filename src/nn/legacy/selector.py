from layer import *

class Selector(Layer):
    def __init__(self, 
                 name, 
                 inputNames,
                 start, 
                 end, 
                 axis=-1):
        Layer.__init__(
                 self,
                 name=name, 
                 inputNames=inputNames,
                 outputDim=end-start)
        self.start = start
        self.end = end
        self.axis = axis
        if axis < -2 or axis > 2:
            raise Exception('Selector axis=%d not supported' % axis)

    def forward(self, inputValue):
        self._inputValue = inputValue
        if self.axis == -1:
            self.axis = len(inputValue.shape) - 1
        if self.axis == 0:
            return inputValue[self.start:self.end]
        elif self.axis == 1:
            return inputValue[:, self.start:self.end]
        elif self.axis == 2:
            return inputValue[:, :, self.start:self.end]

    def backward(self, gradientToOutput):
        dEdX = np.zeros(self._inputValue.shape)
        if self.axis == 0:
            dEdX[self.start:self.end] = gradientToOutput
        elif self.axis == 1:
            dEdX[:, self.start:self.end] = gradientToOutput
        elif self.axis == 2:
            dEdX[:, :, self.start:self.end] = gradientToOutput
        return dEdX