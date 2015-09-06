import h5py
import scipy.sparse
from environment import *

class WeightInitializerFactory():
    """
    Create a weight initializer from dictionary.
    """
    def __init__(self):
        pass

    @staticmethod
    def createFromSpec(spec):
        """
        Create a weight initializer from dictionary.
        :param spec: A dictionary that may contain the following fields:
        format:
        seed:
        limit:
        sparse:
        key:
        :return:
        """
        if not 'type' in spec:
            raise Exception('Must specify weight initializer type!')
        seed = 0 if 'seed' not in spec else spec['seed']
        dataFormat = 'numpy' if 'format' not in spec else spec['format']
        sparse = False if 'sparse' not in spec else spec['sparse']
        key = None if 'key' not in spec else spec['key']
        if spec['type'] == 'uniform':
            return UniformWeightInitializer(limit=eval(spec['limit']),
                                            seed=seed)
        elif spec['type'] == 'uniformConstBias':
            raise Exception('Not implemented!')
            # return UniformWeightConstBiasInitializer()
        elif spec['type'] == 'gaussian':
            raise Exception('Not implemented!')
        elif spec['type'] == 'static':
            return StaticWeightInitializer(filename=spec['filename'],
                                           format=dataFormat,
                                           key=key,
                                           sparse=sparse)
        else:
            raise Exception('Unknown weight initializer type: ' + spec['type'])

class WeightInitializer():
    def __init__(self):
        pass

    def initialize(self, shape):
        """
        Abstract method
        :param shape: Shape of the weights to be initialized
        :return: A weight matrix with given shape
        """

    def toDict(self):
        """
        Abstract method
        """
        pass

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
        """
        """
        r0 = self._limit[0]
        r1 = self._limit[1]
        return self._random.uniform(r0, r1, shape)

    def toDict(self):
        return {
            'type': 'uniform',
            'limit': str(self._limit),
            'seed': self._seed
        }

class GaussianWeightInitializer(WeightInitializer):
    def __init__(self, mean, std, seed):
        """
        """
        WeightInitializer.__init__(self)
        raise Exception("Not implemented")
        pass

    def initialize(self, shape=None):
        pass

class StaticWeightInitializer(WeightInitializer):
    """
    Loading weights from files. Supports dense matrix or row sparse matrix.
    Supported file types: npy, h5, plain.
    """
    def __init__(self, filename, format, key=None, sparse=False):
        """

        :param filename: Location of the weight file.
        :param format: Type of the weight file.
        :param sparse: Whether the weight is sparse.
        :return:
        """
        WeightInitializer.__init__(self)
        self._filename = filename
        self._format = format
        self._sparse = sparse
        if format != 'h5' and format != 'numpy' and format != 'plain':
            raise Exception(
                'Unknown static initialization file type: ' + format)
        if format == 'h5':
            self._key = key

    def initialize(self):
        """

        :return:
        """
        if self._format == 'plain':
            initWeights = np.loadtxt(self._filename)
        elif self._format == 'h5':
            initWeightsFile = h5py.File(self._filename)
            if self._sparse:
                key = self._key
                iwShape = initWeightsFile[key + '_shape'][:]
                iwData = initWeightsFile[key + '_data'][:]
                iwInd = initWeightsFile[key + '_indices'][:]
                iwPtr = initWeightsFile[key + '_indptr'][:]
                initWeights = scipy.sparse.csr_matrix(
                    (iwData, iwInd, iwPtr), shape=iwShape)
            else:
                initWeights = initWeightsFile[self._key][:]
            print initWeights.shape
        elif self._format == 'numpy':
            initWeights = np.load(self._filename)
        return initWeights

    def toDict(self):
        return {
            'type': 'static',
            'filename': self._filename,
            'format': self._format,
            'sparse': self._sparse,
            'key': self._key
        }