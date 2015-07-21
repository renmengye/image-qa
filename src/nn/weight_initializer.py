from environment import *

class WeightInitializer():
    def __init__(self, shape):
        pass

    def initialize(self):
        """
        :return:
        """
        pass

class UniformWeightInitializer(WeightInitializer):
    """

    """
    def __init__(self, limit, seed, shape, affine=True, biasInitConst=-1.0):
        """

        :param limit:
        :param seed:
        :param shape: Shape of the weight matrix, including the bias,
        i.e. inputDim + 1, outputDim + 1
        :param affine:
        :param biasInitConst:
        :return:
        """
        WeightInitializer.__init__(self, shape)
        self._limit = limit
        self._seed = seed
        self._shape = shape
        self._random = np.random.RandomState(seed)
        self._affine = affine
        self._biasInitConst = biasInitConst
        pass

    def initialize(self):
        r0 = self._limit[0]
        r1 = self._limit[1]
        if self._affine:
            if self._biasInitConst >= 0.0:
                return np.concatenate((self._random.uniform(
                    r0, r1, (self._shape[0] - 1, self._shape[1])),
                    np.ones((1, self._shape[1])) * self._biasInitConst), axis=0)
            else:
                return self._random.uniform(r0, r1, self._shape)
        else:
            return self._random.uniform(r0, r1, self._shape)

class GaussianWeightInitializer(WeightInitializer):
    def __init__(self, mean, std, seed, shape):
        WeightInitializer.__init__(self, shape)
        raise Exception("Not implemented")
        pass

class StaticWeightInitializer(WeightInitializer):
    def __init__(self, filename, filetype):
        shape = 0
        WeightInitializer.__init__(self, shape)
        pass