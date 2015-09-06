from weight_initializer import WeightInitializerFactory
from gradient_descent_controller import GradientDescentControllerFactory
from environment import *

class Weight():
    """
    Designed as a weights matrix wrapper for the ease of sharing parameters.
    Currently only considered for gradient descent optimization.

    Currently there is no sparse update and sparse serialization.
    """
    def __init__(self, name, initializer, controller=None, gpuEnabled=USE_GPU,
                 shared=False, saveToFile=True):
        self._controller = controller
        self._initializer = initializer
        self._weight = 0
        self._gradient = 0
        self._gradientStack = []
        self.name = name
        self.gpuEnabled = gpuEnabled
        self.shared = shared
        self.hasInitialized = False
        self.savedToFile = saveToFile

    def initialize(self, shape):
        """
        Initialize the weight into certain shape.
        :param shape:
        :return:
        """
        weight = self._initializer.initialize(shape)
        self.hasInitialized = True
        if self.gpuEnabled:
            self._weight = gnp.as_garray(weight)
        else:
            self._weight = weight

    def get(self):
        """
        Get the real representation of the weight in memory.
        Do not use this method for serialization.
        :return: Numpy or Gnumpy array.
        """
        return self._weight

    def set(self, value):
        """
        Set the real representation of the weight in memory.
        Do not use this method for deserialization.
        :param value: Numpy or Gnumpy array.
        :return:
        """
        self._weight = value

    def serialize(self):
        """
        Serialize the weight value into permanent storage version.
        :return: A dictionary with keys and values that can be stored in Numpy
        or H5 files.
        """
        if self.savedToFile:
            return {self.name: self._getNumpy()}
        else:
            return None

    def deserialize(self, value):
        """
        Load the weight value from permanent storage version.
        :param value: A dictionary with keys and values that are from a Numpy
        or H5 file.
        :return:
        """
        if self.gpuEnabled:
            self._weight = gnp.as_garray(value[self.name])
        else:
            self._weight = value[self.name]

    def _getNumpy(self):
        if self.gpuEnabled:
            return gnp.as_numpy_array(self._weight)
        return self._weight

    def addGradient(self, gradient):
        if self.shared:
            self._gradientStack.append(gradient)
        else:
            self._gradient = gradient

    def getGradient(self):
        return self._gradient

    def update(self):
        if self.shared:
            if self.gpuEnabled:
                tmp = gnp.concatenate(self._gradientStack, axis=0)
                self._gradient = gnp.sum(tmp, axis=0)
            else:
                tmp = np.concatenate(self._gradientStack, axis=0)
                self._gradient = np.sum(tmp, axis=0)
            self._gradientStack = []
        if self._controller is not None:
            self._weight = self._controller.updateWeight(self._weight,
                                                         self._gradient)
    def toDict(self):
        return {
            'name': self.name,
            'initializerSpec': self._initializer.toDict(),
            'gpuEnabled': self.gpuEnabled,
            'shared': self.shared
        }

    @staticmethod
    def fromDict(value):
        return Weight(name=value['name'],
                      initializer=WeightInitializerFactory.createFromSpec(
                          value['initializerSpec']),
                      controller=None if 'controllerSpec' not in value else
                      GradientDescentControllerFactory.createFromSpec(
                          name=value['name'],
                          spec=value['controllerSpec']),
                      gpuEnabled=value['gpuEnabled'])

class SparseWeight(Weight):
    ####
    # Should handle sparse weight (de)serialization here.
    # Should handle sparse update here.
    ####
    pass