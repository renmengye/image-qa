from environment import *

class Weight():
    """
    Designed as a weights matrix wrapper for the ease of sharing parameters.
    Currently only considered for gradient descent optimization.
    """
    def __init__(self, name, initializer, gdController, useGpu=USE_GPU,
                 shared=False):
        self._gdController = gdController
        self._initializer = initializer
        self._weight = 0
        self._gradient = 0
        self._gradientStack = []
        self.name = name
        self.useGpu = useGpu
        self.shared = shared
        self.hasInitialized = False

    def initialize(self, shape):
        weight = self._initializer.initialize(shape)
        self.hasInitialized = True
        if self.useGpu:
            self._weight = gnp.as_garray(weight)
        else:
            self._weight = weight

    def get(self):
        return self._weight

    def set(self, value):
        self._weight = value

    def getNumpy(self):
        if self.useGpu:
            return gnp.as_numpy_array(self._weight)
        else:
            return self._weight

    def load(self, value):
        if self.useGpu:
            self._weight = gnp.as_garray(value)
        else:
            self._weight = value

    def addGradient(self, gradient):
        if self.shared:
            self._gradientStack.append(gradient)
        else:
            self._gradient = gradient

    def getGradient(self):
        return self._gradient

    def getGradientNumpy(self):
        if self.useGpu:
            return gnp.as_numpy_array(self._gradient)
        else:
            return self._gradient

    def update(self):
        if self.shared:
            if self.useGpu:
                tmp = gnp.concatenate(self._gradientStack, axis=0)
                self._gradient = gnp.sum(tmp, axis=0)
            else:
                tmp = np.concatenate(self._gradientStack, axis=0)
                self._gradient = np.sum(tmp, axis=0)
            self._gradientStack = []
        if self._gdController is not None:
            self._weight = self._gdController.updateWeight(self._weight,
                                                    self._gradient)
