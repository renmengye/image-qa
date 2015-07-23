from loss_layer import *

class MSELossLayer(LossLayer):
    def __init__(self, name, useGpu=USE_GPU):
        LossLayer.__init__(self, name=name, useGpu=useGpu)
        pass

    def computeLossWithGrad(self, outputValue, targetValue, weights=None):
        diff =  outputValue - targetValue.reshape(outputValue.shape)
        if self._useGpu:
            diff2 = np.sum(diff ** 2, axis=-1)
            if weights is not None:
                diff2 *= weights
                weights = weights.reshape(weights.shape[0], 1)
                diff *= weights
            loss = 0.5 * np.sum(diff2) / float(outputValue.shape[0])
            gradientToOutput = diff / float(outputValue.shape[0])
        else:
            diff2 = gnp.sum(diff ** 2, axis=-1)
            if weights is not None:
                diff2 *= weights
                weights = weights.reshape(weights.shape[0], 1)
                diff *= weights
            loss = 0.5 * gnp.sum(diff2) / float(outputValue.shape[0])
            gradientToOutput = diff / float(outputValue.shape[0])
        return loss, gradientToOutput