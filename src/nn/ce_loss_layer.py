from loss_layer import *

class CEBinaryLossLayer(LossLayer):
    """
    Cross entropy for binary classification.
    """
    def __init__(self, inputNames, name=None):
        LossLayer.__init__(self, inputNames=inputNames, name=name)
        pass

    def computeLossWithGrad(self, Y, T, weights=None):
        eps = 1e-8
        T = T.reshape(Y.shape)
        cost = -T * np.log(Y + eps) - (1 - T) * np.log(1 - Y + eps)
        dcost = -T / (Y + eps) + (1 - T) / (1 - Y + eps)
        if weights is not None:
            cost *= weights
            dcost *= weights.reshape(weights.shape[0], 1)
        if len(Y.shape) == 0:
            E = cost
            dEdY = dcost
        else:
            E = np.sum(cost) / float(Y.size)
            dEdY = dcost / float(Y.size)
        return E, dEdY

class CEMultiClassLossLayer(LossLayer):
    """
    Cross entropy for multi-class classification.
    """
    def __init__(self, inputNames, name=None):
        LossLayer.__init__(self, inputNames=inputNames, name=name)
        pass

    def computeLossWithGrad(self, Y, T, weights=None):
        eps = 1e-8
        Y2 = Y.reshape(Y.size / Y.shape[-1], Y.shape[-1])
        T2 = T.reshape(T.size)
        E = 0.0
        dEdY = np.zeros(Y2.shape, float)
        if weights is None:
            for n in range(0, Y2.shape[0]):
                E += -np.log(Y2[n, T2[n]] + eps)
                dEdY[n, T2[n]] = -1 / (Y2[n, T2[n]] + eps)
        else:
            for n in range(0, Y2.shape[0]):
                E += -np.log(Y2[n, T2[n]] + eps) * weights[n]
                dEdY[n, T2[n]] = -1 / (Y2[n, T2[n]] + eps) * weights[n]
        E /= Y2.shape[0]
        dEdY /= Y2.shape[0]
        dEdY = dEdY.reshape(Y.shape)
        return E, dEdY