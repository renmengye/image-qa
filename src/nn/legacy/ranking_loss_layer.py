from loss_layer import *

class PairWiseRankingLossLayer(LossLayer):
    """
    Pair-wise ranking loss, usually used after cosine similarity layer.
    """
    def __init__(self, inputNames, name=None, alpha=0.1):
        LossLayer.__init__(self, inputNames=inputNames, name=name)
        self.alpha = alpha
        pass

    def computeLossWithGrad(self, outputValue, targetValue, weights=None):
        dEdY = np.zeros(outputValue.shape)
        E = 0.0
        for n in range(targetValue.size):
            cost = outputValue[n] - outputValue[n, targetValue[n]] + self.alpha
            valid = (cost > 0).astype(int)
            nvalid = np.sum(valid) - 1
            cost = cost * valid
            dEdY[n] = valid
            dEdY[n, targetValue[n]] = -nvalid
            if weights is not None:
                cost *= weights[n]
                dEdY[n] *= weights[n]
            E += np.sum(cost) - self.alpha
        E /= float(targetValue.size)
        dEdY /= float(targetValue.size)
        return E, dEdY