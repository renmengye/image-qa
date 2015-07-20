from loss_layer import *

class MSELossLayer(LossLayer):
    def __init__(self, inputNames, name=None):
        LossLayer.__init__(self, inputNames=inputNames, name=name)
        pass

    def computeLossWithGrad(self, Y, T, weights=None):
        diff =  Y - T.reshape(Y.shape)
        diff2 = np.sum(np.power(diff, 2), axis=-1)
        if weights is not None:
            diff2 *= weights
            weights = weights.reshape(weights.shape[0], 1)
            diff *= weights
        E = 0.5 * np.sum(diff2) / float(Y.shape[0])
        dEdY = diff / float(Y.shape[0])
        return E, dEdY