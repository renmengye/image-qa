from layer import *
use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'
if USE_GPU:
    import gnumpy as gpu
    import gnumpy as gnp

class SumProduct(Layer):
    def __init__(self, 
                name, 
                inputNames, 
                sumAxis,
                outputDim,
                useGpu=USE_GPU,
                beta=1.0):
        Layer.__init__(self,
            name=name, 
            inputNames=inputNames,
            useGpu=useGpu,
            outputDim=outputDim)
        self.sumAxis = sumAxis
        self.beta = beta

    def getInput(self):
        # Assume that the input size is always 2
        # Rewrite get input logic into two separate arrays
        if len(self.inputLayers) == 2:
            return [self.inputLayers[0].Y, self.inputLayers[1].Y]
        elif len(self.inputLayers) == 3:
            return [self.inputLayers[0].Y, self.inputLayers[1].Y, self.inputLayers[2].Y]

    def sendError(self, gradientToInput):
        self.inputLayers[0].dEdY += gradientToInput[0]
        self.inputLayers[0].receivedError = True
        self.inputLayers[1].dEdY += gradientToInput[1]
        self.inputLayers[1].receivedError = True
        if len(self.inputLayers) == 3:
            self.inputLayers[2].dEdY += gradientToInput[2]
            self.inputLayers[2].receivedError = True

    def forward(self, inputValue):
        if self.useGpu:
            self._inputValue = []
            self._inputValue.append(gpu.as_garray(inputValue[0].astype('float32')))
            self._inputValue.append(gpu.as_garray(inputValue[1].astype('float32')))
            if len(inputValue) == 2:
                Y = self.beta * gpu.sum(self._inputValue[0] * self._inputValue[1], axis=self.sumAxis)
            elif len(inputValue) == 3:
                self._inputValue.append(gpu.as_garray(inputValue[2].astype('float32')))
                self.Z = gpu.sum(self._inputValue[0] * self._inputValue[1], axis=self.sumAxis)
                Y = self._inputValue[2] * self.Z
            Y = Y.as_numpy_array(dtype='float32')
        else:
            self._inputValue = inputValue
            if len(self._inputValue) == 2:
                Y = self.beta * np.sum(self._inputValue[0] * self._inputValue[1], axis=self.sumAxis)
            elif len(self._inputValue) == 3:
                self.Z = np.sum(self._inputValue[0] * self._inputValue[1], axis=self.sumAxis)
                Y = self._inputValue[2] * self.Z
        return Y

    def backward(self, gradientToOutput):
        # Need to generalize, but now, let's assume it's the attention model.
        dEdX = []
        if self.useGpu:
            if len(self._inputValue) == 2:
                gradientToOutput = gradientToOutput.reshape(gradientToOutput.shape[0], 1, gradientToOutput.shape[1])
                gradientToOutput = gpu.as_garray(gradientToOutput)
                dEdX1 = self.beta * gpu.sum(gradientToOutput * self._inputValue[1], axis=2)
                dEdX2 = self.beta * gradientToOutput * self._inputValue[0]
                dEdX.append(dEdX1.as_numpy_array(dtype='float32'))
                dEdX.append(dEdX2.as_numpy_array(dtype='float32'))
            elif len(self._inputValue) == 3:
                gradientToOutput = gpu.as_garray(gradientToOutput)
                dEdY2 = gradientToOutput.reshape(gradientToOutput.shape[0], 1, gradientToOutput.shape[1])
                dEdY2 = gpu.as_garray(dEdY2)
                dEdX1 = self._inputValue[2] * gpu.sum(dEdY2 * self._inputValue[1], axis=2)
                dEdX2 = self._inputValue[2].reshape(self._inputValue[2].shape[0], 1, 1) * dEdY2 * self._inputValue[0]
                dEdX3 = gpu.sum(gradientToOutput * self.Z, axis=-1).reshape(self._inputValue[2].shape[0], 1)
                dEdX.append(dEdX1.as_numpy_array(dtype='float32'))
                dEdX.append(dEdX2.as_numpy_array(dtype='float32'))
                dEdX.append(dEdX3.as_numpy_array(dtype='float32'))
        else:
            if len(self._inputValue) == 2:
                gradientToOutput = gradientToOutput.reshape(gradientToOutput.shape[0], 1, gradientToOutput.shape[1])
                dEdX.append(self.beta * np.sum(gradientToOutput * self._inputValue[1], axis=2))
                dEdX.append(self.beta * gradientToOutput * self._inputValue[0])
            elif len(self._inputValue) == 3:
                dEdY2 = gradientToOutput.reshape(gradientToOutput.shape[0], 1, gradientToOutput.shape[1])
                dEdX.append(self._inputValue[2] * np.sum(dEdY2 * self._inputValue[1], axis=2))
                dEdX.append(self._inputValue[2].reshape(self._inputValue[2].shape[0], 1, 1) * dEdY2 * self._inputValue[0])
                dEdX.append(np.sum(gradientToOutput * self.Z, axis=-1).reshape(self._inputValue[2].shape[0], 1))
        return dEdX