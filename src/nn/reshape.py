from layer import *

class Reshape(Layer):
    def __init__(self, reshapeFn, inputNames=None, outputDim=0, name=None, outputdEdX=True):
        Layer.__init__(self, name=name, inputNames=inputNames, outputDim=outputDim, outputdEdX=outputdEdX)
        self.reshapeFn = eval('lambda x: ' + reshapeFn)
        self.Xshape = 0

    def forward(self, inputValue):
        self.Xshape = inputValue.shape
        # if self.inputs[0].outputGpu:
        #     return gpu
        return np.reshape(inputValue, self.reshapeFn(inputValue.shape))

    def backward(self, gradientToOutput):
        if self.outputdEdX:
            return np.reshape(gradientToOutput, self.Xshape)

class TimeUnfold(Reshape):
    def __init__(self, inputNames=None, name=None, outputdEdX=True):
        Reshape.__init__(self,
                         name=name,
                         inputNames=inputNames,
                         reshapeFn='(x[0] * x[1], x[2])',
                         outputdEdX=outputdEdX)

class TimeFold(Reshape):
    def __init__(self, timespan, inputNames=None, name=None, outputdEdX=True):
        self.timespan = timespan
        t = str(self.timespan)
        Reshape.__init__(self,
                         name=name,
                         inputNames=inputNames,
                         reshapeFn='(x[0] / '+t+','+t+', x[1])',
                         outputdEdX=outputdEdX)

class TimeReverse(Layer):
    def __init__(self, inputNames, outputDim=0, name=None, outputdEdX=True):
        Layer.__init__(self,
                       name=name,
                       inputNames=inputNames,
                       outputDim=outputDim,
                       outputdEdX=outputdEdX)

    def forward(self, inputValue):
        #print self.name, X.shape
        N = inputValue.shape[0]
        self.Xend = np.zeros(N, dtype=int) + inputValue.shape[1]
        reachedEnd = np.sum(inputValue, axis=-1) == 0.0
        Y = np.zeros(inputValue.shape)
        # Scan for the end of the sequence.
        for n in range(N):
            found = False
            for t in range(inputValue.shape[1]):
                if reachedEnd[n, t]:
                    self.Xend[n] = t
                    if t > 0:
                        found = True
                        Y[n, 0:t, :] = inputValue[n, t-1::-1, :]
                    break
            if found == False:
                self.Xend[n] = inputValue.shape[1]
                Y[n, :, :] = inputValue[n, ::-1, :]
        return Y

    def backward(self, gradientToOutput):
        if self.outputdEdX:
            dEdX = np.zeros(gradientToOutput.shape)
            for n in range(gradientToOutput.shape[0]):
                t = self.Xend[n]
                if t > 0:
                    dEdX[n, 0:t, :] = gradientToOutput[n, t-1::-1, :]
            return dEdX
        else:
            return None

class TimeRepeat(Layer):
    def __init__(self, numRepeats, inputNames=None, outputDim=0, name=None, outputdEdX=True):
        Layer.__init__(self, name=name, inputNames=inputNames, outputDim=outputDim, outputdEdX=outputdEdX)
        self.numRepeats = numRepeats

    def forward(self, inputValue):
        self.Xshape = inputValue.shape
        if len(inputValue.shape) == 2:
            inputValue = inputValue.reshape(inputValue.shape[0], 1, inputValue.shape[1])
        return np.tile(inputValue, (1, self.numRepeats, 1))

    def backward(self, gradientToOutput):
        if self.outputdEdX:
            gradientToOutput = gradientToOutput.reshape(
                gradientToOutput.shape[0], self.numRepeats, gradientToOutput.shape[1] / self.numRepeats, gradientToOutput.shape[2])
            dEdX = np.sum(gradientToOutput, axis=1)
            if len(self.Xshape) == 2:
                dEdX = dEdX.reshape(dEdX.shape[0], dEdX.shape[-1])
            return dEdX

class TimeFinal(Layer):
    """
    Scans and selects the last timestep.
    """
    def __init__(self, inputNames, outputDim=0, name=None, outputdEdX=True):
        Layer.__init__(self,
                       name=name, 
                       inputNames=inputNames, 
                       outputDim=outputDim, 
                       outputdEdX=outputdEdX)
        self.Xend = 0.0

    def forward(self, inputValue):
        N = inputValue.shape[0]
        self._inputValue = inputValue
        self.Xend = np.zeros(N, dtype=int) + inputValue.shape[1]
        reachedEnd = np.sum(inputValue, axis=-1) == 0.0
        Y = np.zeros((N, inputValue.shape[-1]))
        # Scan for the end of the sequence.
        for n in range(N):
            for t in range(inputValue.shape[1]):
                if reachedEnd[n, t]:
                    self.Xend[n] = t
                    break
        for n in range(N):
            if self.Xend[n] > 0:
                Y[n] = inputValue[n, self.Xend[n] - 1]
        return Y

    def backward(self, gradientToOutput):
        if self.outputdEdX:
            dEdX = np.zeros(self._inputValue.shape)
            for n in range(gradientToOutput.shape[0]):
                if self.Xend[n] > 0:
                    dEdX[n, self.Xend[n] - 1, :] = gradientToOutput[n]
            return dEdX
        else:
            return None

class ConcatenationLayer(Layer):
    def __init__(self, inputNames, axis, name=None):
        Layer.__init__(self, name=name, inputNames=inputNames, outputDim=0)
        self.axis = axis

    def getInput(self):
        if len(self.inputLayers) > 1:
            self.splX = []
            for stage in self.inputLayers:
                X = stage.Y
                self.splX.append(X)
            return np.concatenate(self.splX, axis=self.axis)
        else:
            return self.inputLayers[0].Y

    def sendError(self, gradientToInput):
        """
        Iterates over input list and sends dEdX.
        """
        axis = np.mod(self.axis, len(gradientToInput.shape))
        if len(self.inputLayers) > 1:
            s = 0
            for stage in self.inputLayers:
                s2 = s + stage.Y.shape[self.axis]
                if axis == 0:
                    stage.dEdY += gradientToInput[s : s2]
                elif axis == 1:
                    stage.dEdY += gradientToInput[:, s : s2]
                elif axis == 2:
                    stage.dEdY += gradientToInput[:, :, s : s2]
                s = s2
                stage.receivedError = True
        else:
            self.inputLayers[0].dEdY += gradientToInput
            self.inputLayers[0].receivedError = True

    def forward(self, inputValue):
        return inputValue

    def backward(self, gradientToOutput):
        return gradientToOutput
