from loss_layer import *

class PairWiseRankingLossLayer(LossLayer):
    """
    Pair-wise ranking loss, usually used after cosine similarity layer.
    """
    def __init__(self, inputNames, name=None, alpha=0.1):
        LossLayer.__init__(self, inputNames=inputNames, name=name)
        self.alpha = alpha
        pass

    def computeLossWithGrad(self, Y, T, weights=None):
        dEdY = np.zeros(Y.shape)
        E = 0.0
        for n in range(T.size):
            cost = Y[n] - Y[n, T[n]] + self.alpha
            valid = (cost > 0).astype(int)
            nvalid = np.sum(valid) - 1
            cost = cost * valid
            dEdY[n] = valid
            dEdY[n, T[n]] = -nvalid
            if weights is not None:
                cost *= weights[n]
                dEdY[n] *= weights[n]
            E += np.sum(cost) - self.alpha
        E /= float(T.size)
        dEdY /= float(T.size)
        return E, dEdY