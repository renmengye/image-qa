from layer import *
import os
use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'
if USE_GPU:
    import gnumpy as gpu
    import gnumpy as gnp

class FullyConnectedLayer(Layer):
    def __init__(self,
                 activeFn,
                 initRange=1.0,
                 affine=True,
                 outputdEdX=True,
                 defaultValue=0.0,
                 useGpu=USE_GPU,
                 name=None):
        Layer.__init__(self,
                 name=name,
                 useGpu=useGpu,
                 outputdEdX=outputdEdX)
        self.activeFn = activeFn
        self._affine = affine

    def forward(self, inputValue):
        if self._affine:
            self._inputValue = \
                np.concatenate(
                    (inputValue, np.ones((inputValue.shape[0], 1),
                                         dtype=inputValue.dtype)), axis=-1)
        else:
            self._inputValue = inputValue
        if self.useGpu:
            self._inputValue = gpu.as_garray(self._inputValue.astype('float32'))
            weightedSum = gpu.dot(self._inputValue, self.weight.get())
            weightedSum = weightedSum.as_numpy_array(dtype='float32')
            self._outputValue = self.activeFn.forward(weightedSum)
        else:
            weightedSum = np.dot(self._inputValue, self.weight.get())
            self._outputValue = self.activeFn.forward(weightedSum)
        return self._outputValue

    def backward(self, gradientToOutput):
        #######################################################
        # Attention, may need to convert this to GPU as well! #
        #######################################################
        gradientToWeightedSum = self.activeFn.backward(gradientToOutput,
                                          self._outputValue, 0)
        if self.useGpu:
            #################################################################
            # Attention here, explicit conversion. Want to remove it in the #
            # future                                                        #
            #################################################################
            gradientToWeightedSumGpu = gnp.as_garray(
                gradientToWeightedSum.astype('float32'))
            gradient = gnp.dot(self._inputValue.transpose(),
                               gradientToWeightedSumGpu)
            self.weight.addGradient(gradient)
            if self._affine:
                gradientToInput = gnp.dot(gradientToWeightedSumGpu,
                                          self.weight.get()[:-1, :].transpose())
            else:
                gradientToInput = gnp.dot(gradientToWeightedSumGpu,
                                          self.weight.get().transpose())
            gradientToInput = gnp.as_numpy_array(gradientToInput)
        else:
            gradient = np.dot(self._inputValue.transpose(),
                              gradientToWeightedSum)
            self.weight.addGradient(gradient)
            if self._affine:
                gradientToInput = np.dot(gradientToWeightedSum,
                                         self.weight.get()[:-1, :].transpose())
            else:
                gradientToInput = np.dot(gradientToWeightedSum,
                                         self.weight.get().transpose())
        return gradientToInput if self.outputdEdX else None
