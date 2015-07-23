from environment import *
from loss_layer import LossLayer

class MSELossLayer(LossLayer):
    def __init__(self, name, gpuEnabled=USE_GPU):
        LossLayer.__init__(self, name=name, gpuEnabled=gpuEnabled)

    def computeLossWithGrad(self, outputValue, targetValue,
                            exampleWeights=None):
        diff =  outputValue - targetValue.reshape(outputValue.shape)
        if not self.gpuEnabled:
            diff2 = np.sum(diff ** 2, axis=-1)
            if exampleWeights is not None:
                diff2 *= exampleWeights
                exampleWeights = exampleWeights.reshape(
                    exampleWeights.shape[0], 1)
                diff *= exampleWeights
            loss = 0.5 * np.sum(diff2) / float(outputValue.shape[0])
            gradientToOutput = diff / float(outputValue.shape[0])
        else:
            diff2 = gnp.sum(diff ** 2, axis=-1)
            if exampleWeights is not None:
                diff2 *= exampleWeights
                exampleWeights = exampleWeights.reshape(
                    exampleWeights.shape[0], 1)
                diff *= exampleWeights
            loss = 0.5 * gnp.sum(diff2) / float(outputValue.shape[0])
            gradientToOutput = diff / float(outputValue.shape[0])
        return loss, gradientToOutput