from stage import *

use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'
if use_gpu:
    import gnumpy as gpu
    import gnumpy as gnp

class Sum2(Stage):
    """Stage summing first half of the input with second half."""
    def __init__(self, name, inputNames, outputDim,
                 defaultValue=0.0):
        Stage.__init__(
            self,
            name=name,
            inputNames=inputNames,
            outputDim=outputDim,
            defaultValue=defaultValue)
    def forward(self, X):
        self.numComponents = X.shape[1]
        if use_gpu:
            return gnp.as_numpy_array(gnp.sum(gnp.as_garray(X), axis=1))
        else:
            return np.sum(X, axis=1)
    def backward(self, dEdY):
        self.dEdW = 0.0
        return np.tile(dEdY.reshape(dEdY.shape[0], 1, dEdY.shape[1]), (1, self.numComponents, 1))
