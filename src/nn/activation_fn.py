from environment import *

class ActivationFn():
    """
    Abstract class for an activation function.
    """
    def __init__(self, gpuEnabled=USE_GPU):
        """
        Construct an activation function.
        :param gpuEnabled: Whether the computation is on GPU.
        :return:
        """
        self.gpuEnabled = gpuEnabled
        self._inputValue = 0.0
        self._outputValue = 0.0

    def forward(self, inputValue):
        """
        Abstract method
        :param inputValue: Before activation value.
        :return: After activation value
        """
        pass

    def backward(self, gradientToOutput):
        """
        Abstract method
        :param gradientToOutput: Gradient to after activation value.
        :return: Gradient to before activation value.
        """
        pass

class SoftmaxActivationFn(ActivationFn):
    def __init__(self, gpuEnabled=USE_GPU):
        ActivationFn.__init__(self, gpuEnabled)

    def forward(self, inputValue):
        expYshape = np.copy(inputValue.shape)
        expYshape[-1] = 1
        if self.gpuEnabled:
            expY = gnp.exp(inputValue)
            self._outputValue = expY / gnp.sum(expY, axis=-1).reshape(expYshape)
        else:
            expY = np.exp(inputValue)
            self._outputValue = expY / np.sum(expY, axis=-1).reshape(expYshape)
        return self._outputValue

    def backward(self, gradientToOutput):
        N = self._outputValue.shape[0]
        U = gradientToOutput * self._outputValue
        if self.gpuEnabled:
            gradientToInput = U - gnp.sum(U, axis=-1).reshape(N, 1) \
                                  * self._outputValue
        else:
            gradientToInput = U - np.sum(U, axis=-1).reshape(N, 1) \
                                  * self._outputValue
        return gradientToInput

class SigmoidActivationFn(ActivationFn):
    def __init__(self, gpuEnabled=USE_GPU):
        ActivationFn.__init__(self, gpuEnabled)

    def forward(self, inputValue):
        if self.gpuEnabled:
            self._outputValue = 1 / (1 + gnp.exp(inputValue))
        else:
            self._outputValue = 1 / (1 + np.exp(inputValue))
        return self._outputValue

    def backward(self, gradientToOutput):
        return gradientToOutput * self._outputValue * (1 - self._outputValue)

class TanhActivationFn(ActivationFn):
    def __init__(self, gpuEnabled=USE_GPU):
        ActivationFn.__init__(self, gpuEnabled)

    def forward(self, inputValue):
        if self.gpuEnabled:
            self._outputValue = gnp.tanh(inputValue)
        else:
            self._outputValue = np.tanh(inputValue)
        return self._outputValue

    def backward(self, gradientToOutput):
        return gradientToOutput * (1 - self._outputValue ** 2)

class IdentityActivationFn(ActivationFn):
    def __init__(self):
        ActivationFn.__init__(self, gpuEnabled=False)

    def forward(self, inputValue):
        return inputValue

    def backward(self, gradientToOutput):
        return gradientToOutput

class ReluActivationFn(ActivationFn):
    def __init__(self, gpuEnabled=USE_GPU):
        ActivationFn.__init__(self, gpuEnabled=gpuEnabled)

    def forward(self, inputValue):
        self._inputValue = inputValue
        if self.gpuEnabled:
            self._outputValue = 0.5 * inputValue + 0.5 * gnp.abs(inputValue)
        else:
            self._outputValue = 0.5 * inputValue + 0.5 * np.abs(inputValue)
        return self._outputValue

    def backward(self, gradientToOutput):
        return self._outputValue / self._inputValue * gradientToOutput
