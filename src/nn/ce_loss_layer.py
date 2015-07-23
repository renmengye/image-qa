from loss_layer import *

class CEBinaryLossLayer(LossLayer):
    """
    Cross entropy for binary classification.
    """
    def __init__(self, inputNames, name=None):
        LossLayer.__init__(self, inputNames=inputNames, name=name)
        pass

    def computeLossWithGrad(self, outputValue, targetValue, weights=None):
        eps = 1e-8
        targetValue = targetValue.reshape(outputValue.shape)
        cost = -targetValue * np.log(outputValue + eps) - (1 - targetValue) * np.log(1 - outputValue + eps)
        dcost = -targetValue / (outputValue + eps) + (1 - targetValue) / (1 - outputValue + eps)
        if weights is not None:
            cost *= weights
            dcost *= weights.reshape(weights.shape[0], 1)
        if len(outputValue.shape) == 0:
            E = cost
            dEdY = dcost
        else:
            E = np.sum(cost) / float(outputValue.size)
            dEdY = dcost / float(outputValue.size)
        return E, dEdY

class CEMultiClassLossLayer(LossLayer):
    """
    Cross entropy for multi-class classification.
    """
    def __init__(self, inputNames, name=None):
        LossLayer.__init__(self, inputNames=inputNames, name=name)
        pass

    def computeLossWithGrad(self, outputValue, targetValue, weights=None):
        eps = 1e-8
        Y2 = outputValue.reshape(outputValue.size / outputValue.shape[-1], outputValue.shape[-1])
        T2 = targetValue.reshape(targetValue.size)
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
        dEdY = dEdY.reshape(outputValue.shape)
        return E, dEdY