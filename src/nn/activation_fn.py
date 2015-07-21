from func import *
from environment import *

class ActivationFn():
    def __init__(self, useGpu=USE_GPU):
        self._useGpu = useGpu
        self._inputValue = 0.0
        self._outputValue = 0.0

    def forward(self, inputValue):
        """
        Abstract method
        :param inputValue:
        :return:
        """
        pass

    def backward(self, gradientToOutput):
        """
        Abstract method
        :param gradientToOutput:
        :return:
        """
        pass


class SoftmaxActivationFn(ActivationFn):
    def __init__(self, useGpu=USE_GPU):
        ActivationFn.__init__(self, useGpu)

    def forward(self, inputValue):
        expYshape = np.copy(inputValue.shape)
        expYshape[-1] = 1
        if self._useGpu:
            expY = np.exp(inputValue)
            self._outputValue = expY / np.sum(expY, axis=-1).reshape(
                expYshape)
        else:
            expY = gnp.exp(inputValue)
            self._outputValue = expY / gnp.sum(expY, axis=-1).reshape(
                expYshape)

        return self._outputValue

    def backward(self, gradientToOutput):
        timespan = Y.shape[0]
        U = dEdY * Y
        dEdZ = U - np.sum(U, axis=-1).reshape(timespan, 1) * Y
        return dEdZ

class SigmoidActivationFn(ActivationFn):
    def __init__(self, useGpu):
        ActivationFn.__init__(self, useGpu)

    def forward(self, inputValue):
        Y = sigmoidFn(inputValue)
        return Y

    def backward(self, gradientToOutput):
        dEdZ = dEdY * Y * (1 - Y)
        return dEdZ

class TanhActivationFn(ActivationFn):
    def __init__(self):
        pass

    def forward(self, Z):
        Y = np.tanh(Z)
        return Y

    def backward(self, dEdY, Y, Z):
        dEdZ = dEdY * (1 - Y * Y)
        return dEdZ

class IdentityActivationFn(ActivationFn):
    def __init__(self):
        pass

    @staticmethod
    def forward(Z):
        return Z

    @staticmethod
    def backward(dEdY, Y, Z):
        return dEdY

class ReluActivationFn(ActivationFn):
    def __init__(self):
        pass

    def forward(self, Z):
        return np.maximum(0, Z)

    def backward(self, dEdY, Y, Z):
        return (Y > 0).astype(int) * dEdY
