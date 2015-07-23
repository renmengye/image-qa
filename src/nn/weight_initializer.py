from environment import *

class WeightInitializer():
    def __init__(self):
        pass

    def initialize(self, shape):
        """
        Abstract method
        :param shape: Shape of the weights to be initialized
        :return: A weight matrix with given shape
        """

class UniformWeightConstBiasInitializer(WeightInitializer):
    pass

class UniformWeightInitializer(WeightInitializer):
    """
    Initialize weights from uniform distribution
    """
    def __init__(self, limit, seed):
        """

        :param limit:
        :param seed:
        :return:
        """
        WeightInitializer.__init__(self)
        self._limit = limit
        self._seed = seed
        self._random = np.random.RandomState(seed)

    def initialize(self, shape):
        r0 = self._limit[0]
        r1 = self._limit[1]
        # if self._affine:
        #     if self._biasInitConst >= 0.0:
        #         return np.concatenate((self._random.uniform(
        #             r0, r1, (shape[0] - 1, shape[1])),
        #             np.ones((1, shape[1])) * self._biasInitConst), axis=0)
        #     else:
        #         return self._random.uniform(r0, r1, shape)
        # else:
        return self._random.uniform(r0, r1, shape)

class GaussianWeightInitializer(WeightInitializer):
    def __init__(self, mean, std, seed):
        WeightInitializer.__init__(self)
        raise Exception("Not implemented")
        pass

    def initialize(self, shape=None):
        pass

class StaticWeightInitializer(WeightInitializer):
    def __init__(self, filename, filetype):
        shape = 0
        WeightInitializer.__init__(self)
        pass

    def initialize(self):
        pass