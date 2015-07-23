from layer import *
import os
use_gpu = os.environ.get('GNUMPY_USE_GPU', 'yes') == 'yes'
if USE_GPU:
    import gnumpy as gpu
    import gnumpy as gnp

class FullyConnectedLayer(Layer):
    def __init__(self,
                 name,
                 activationFn,
                 numNode,
                 weight,
                 hasBias=True,
                 outputdEdX=True,
                 useGpu=USE_GPU):
        Layer.__init__(self,
                 name=name,
                 useGpu=useGpu,
                 outputGpu=useGpu,
                 outputdEdX=outputdEdX)
        self._activationFn = activationFn
        self._hasBias = hasBias
        self.numNode = numNode
        self.weight = weight
        if self._activationFn.useGpu != self.useGpu:
            raise Exception('Activation function does not have the same GPU '
                            'configuration as Layer ' + self.name)

    def init(self):
        inputNumNode = self.inputLayers[0].numNode
        if not self.weight.hasInitialized:
            if self._hasBias:
                self.weight.initialize([inputNumNode + 1, self.numNode])
            else:
                self.weight.initialize([inputNumNode.shape[-1], self.numNode])

    def forward(self, inputValue):
        if self.useGpu:
            if self._hasBias:
                self._inputValue = \
                    gnp.concatenate(
                        (inputValue, gnp.ones((inputValue.shape[0], 1))),
                        axis=-1)
            else:
                self._inputValue = inputValue
            weightedSum = gpu.dot(self._inputValue, self.weight.get())
            self._outputValue = self._activationFn.forward(weightedSum)
        else:
            if self._hasBias:
                self._inputValue = \
                    np.concatenate(
                        (inputValue, np.ones((inputValue.shape[0], 1),
                                             dtype=inputValue.dtype)), axis=-1)
            else:
                self._inputValue = inputValue
            weightedSum = np.dot(self._inputValue, self.weight.get())
            self._outputValue = self._activationFn.forward(weightedSum)
        return self._outputValue

    def backward(self, gradientToOutput):
        #######################################################
        # Attention, may need to convert this to GPU as well! #
        #######################################################
        if self.useGpu and type(gradientToOutput) is np.ndarray:
            gradientToOutput = gnp.as_garray(gradientToOutput.astype('float32'))
        elif not self.useGpu and type(gradientToOutput) is not np.ndarray:
            gradientToOutput = gnp.as_numpy_array(gradientToOutput)

        gradientToWeightedSum = self._activationFn.backward(gradientToOutput)
        if self.useGpu:
            #################################################################
            # Attention here, explicit conversion. Want to remove it in the #
            # future                                                        #
            #################################################################
            gradient = gnp.dot(self._inputValue.transpose(),
                               gradientToWeightedSum)
            if self._hasBias:
                gradientToInput = gnp.dot(gradientToWeightedSum,
                                          self.weight.get()[:-1, :].transpose())
            else:
                gradientToInput = gnp.dot(gradientToWeightedSum,
                                          self.weight.get().transpose())
            # gradientToInput = gnp.as_numpy_array(gradientToInput)
        else:
            gradient = np.dot(self._inputValue.transpose(),
                              gradientToWeightedSum)
            if self._hasBias:
                gradientToInput = np.dot(gradientToWeightedSum,
                                         self.weight.get()[:-1, :].transpose())
            else:
                gradientToInput = np.dot(gradientToWeightedSum,
                                         self.weight.get().transpose())

        self.weight.addGradient(gradient)
        return gradientToInput if self.outputdEdX else None
