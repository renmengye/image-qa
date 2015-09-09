from stage import *
#use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'
use_gpu = False
if use_gpu:
    import gnumpy as gpu
    import gnumpy as gnp

class SumProduct(Stage):
    def __init__(self, 
                name, 
                inputNames, 
                sumAxis,
                outputDim,
                gpu=use_gpu,
                beta=1.0):
        Stage.__init__(self,
            name=name, 
            inputNames=inputNames,
            gpu=gpu,
            outputDim=outputDim)
        self.sumAxis = sumAxis
        self.beta = beta
        if self.sumAxis != 1 and self.sumAxis != 2:
            raise Exception('Sum axis other than 2 or 3 is not implemented.')

    def getInput(self):
        # Assume that the input size is always 2
        # Rewrite get input logic into two separate arrays
        if len(self.inputs) == 2:
            return [self.inputs[0].Y, self.inputs[1].Y]
        elif len(self.inputs) == 3:
            return [self.inputs[0].Y, self.inputs[1].Y, self.inputs[2].Y]

    def sendError(self, dEdX):
        self.inputs[0].dEdY += dEdX[0]
        self.inputs[0].receivedError = True
        self.inputs[1].dEdY += dEdX[1]
        self.inputs[1].receivedError = True
        if len(self.inputs) == 3:
            self.inputs[2].dEdY += dEdX[2]
            self.inputs[2].receivedError = True

    def forward(self, X):
        if self.gpu:
            self.X = []
            self.X.append(gpu.as_garray(X[0].astype('float32')))
            self.X.append(gpu.as_garray(X[1].astype('float32')))
            if len(X) == 2:
                Y = self.beta * gpu.sum(self.X[0] * self.X[1], axis=self.sumAxis)
            elif len(X) == 3:
                self.X.append(gpu.as_garray(X[2].astype('float32')))
                self.Z = gpu.sum(self.X[0] * self.X[1], axis=self.sumAxis)
                Y = self.X[2] * self.Z
            Y = Y.as_numpy_array(dtype='float32')
        else:
            self.X = X
            if len(self.X) == 2:
                Y = self.beta * np.sum(self.X[0] * self.X[1], axis=self.sumAxis)
            elif len(self.X) == 3:
                self.Z = np.sum(self.X[0] * self.X[1], axis=self.sumAxis)
                Y = self.X[2] * self.Z
        return Y

    def backward(self, dEdY):
        # Need to generalize, but now, let's assume it's the attention model.
        dEdX = []
        if self.sumAxis == 1:
            newshape = (dEdY.shape[0], 1, dEdY.shape[1])
            newaxis = 2
        elif self.sumAxis == 2:
            newshape = (dEdY.shape[0], dEdY.shape[1], 1)
            newaxis = 1
        if self.gpu:
            if len(self.X) == 2:
                dEdY = dEdY.reshape(newshape)
                #dEdY = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
                # 100, 196, 512
                # dedy = 100, 196
                # dedy = 100, 196, 1
                # x0 = 100, 1, 512
                # x1 = 100, 196, 512
                dEdY = gpu.as_garray(dEdY)
                dEdX1 = self.beta * gpu.sum(dEdY * self.X[1], axis=newaxis)
                dEdX2 = self.beta * dEdY * self.X[0]
                dEdX.append(dEdX1.as_numpy_array(dtype='float32'))
                dEdX.append(dEdX2.as_numpy_array(dtype='float32'))
            elif len(self.X) == 3:
                raise Exception('Not supported')
                #dEdY = gpu.as_garray(dEdY)
                #dEdY2 = dEdY.reshape(newshape)
                ## dEdY2 = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
                #dEdY2 = gpu.as_garray(dEdY2)
                #dEdX1 = self.X[2] * gpu.sum(dEdY2 * self.X[1], axis=2)
                #dEdX2 = self.X[2].reshape(
                #    self.X[2].shape[0], 1, 1) * dEdY2 * self.X[0]
                #dEdX3 = gpu.sum(
                #    dEdY * self.Z, axis=-1).reshape(self.X[2].shape[0], 1)
                #dEdX.append(dEdX1.as_numpy_array(dtype='float32'))
                #dEdX.append(dEdX2.as_numpy_array(dtype='float32'))
                #dEdX.append(dEdX3.as_numpy_array(dtype='float32'))
        else:
            if len(self.X) == 2:
                dEdY = dEdY.reshape(newshape)
                # dEdY = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
                dEdX.append(self.beta * np.sum(dEdY * self.X[1], axis=newaxis))
                dEdX.append(self.beta * dEdY * self.X[0])
            elif len(self.X) == 3:
                raise Exception('Not supported')
                #dEdY2 = dEdY.reshape(newshape)
                ## dEdY2 = dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1])
                #dEdX.append(self.X[2] * np.sum(dEdY2 * self.X[1], axis=2))
                #dEdX.append(self.X[2].reshape(self.X[2].shape[0], 1, 1) * dEdY2 * self.X[0])
                #dEdX.append(np.sum(dEdY * self.Z, axis=-1).reshape(self.X[2].shape[0], 1))
        return dEdX
