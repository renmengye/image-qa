from environment import *

class Weight():
    """
    Designed as a weights matrix wrapper for the ease of sharing parameters.
    Currently only considered for gradient descent optimization.
    """
    def __init__(self, initializer, gdController, useGpu=USE_GPU):
        self._gdController = gdController
        self._initializer = initializer
        self._weight = 0
        self._gradient = 0
        self._gradientStack = []
        self._useGpu = useGpu
        weight = self._initializer.initialize()
        if self._useGpu:
            self._weight = gnp.as_garray(weight)
        else:
            self._weight = weight

    def get(self):
        return self._weight

    def getNumpy(self):
        if self._useGpu:
            return gnp.as_numpy_array(self._weight)
        else:
            return self._weight

    def load(self, value):
        if self._useGpu:
            self._weight = gnp.as_garray(value)
        else:
            self._weight = value

    def addGradient(self, gradient):
        self._gradientStack.append(gradient)

    def getGradient(self):
        if self._useGpu:
            return gnp.as_numpy_array(self._gradient)
        else:
            return self._gradient

    def update(self):
        if len(self._gradientStack) > 0:
            if self._useGpu:
                tmp = gnp.concatenate(self._gradientStack, axis=0)
                self._gradient = gnp.sum(tmp, axis=0)
            else:
                tmp = np.concatenate(self._gradientStack, axis=0)
                self._gradient = np.sum(tmp, axis=0)
            self._weight = self._gdController.updateWeight(self._weight,
                                                    self._gradient)