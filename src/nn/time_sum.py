from stage import *

class TimeSum(Stage):
    def __init__(self,
                name=None,
                learningRate=0.0,
                learningRateAnnealConst=0.0,
                momentum=0.0,
                deltaMomentum=0.0,
                weightClip=0.0,
                gradientClip=0.0,
                weightRegConst=0.0,
                outputdEdX=True):
        Stage.__init__(self,
                 name=name,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
    def forward(self, X):
        self.X = X
        Y = np.sum(X, axis=1)
        self.Y = Y
        return Y

    def backward(self, dEdY):
        self.dEdW = 0
        X = self.X
        dEdX = dEdY.reshape(X.shape[0], 1, X.shape[2]).repeat(X.shape[1], axis=1)
        return dEdX