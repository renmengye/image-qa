from stage2 import *

class ConstWeights(GraphStage):
    def __init__(self,
                 name,
                 outputDim=0,
                 inputDim=0,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0):
        GraphStage.__init__(self,
                 name=name,
                 outputDim=outputDim,
                 inputNames=None,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=False)
        if needInit:
            self.random = np.random.RandomState(initSeed)
            self.W = self.random.uniform(
                    -initRange/2.0, initRange/2.0, (outputDim, inputDim))
        else:
            self.W = initWeights
        self.dEdW = 0

    def forward(self, X):
        return self.W

    def backward(self, dEdY):
        self.dEdW = dEdY
        return 0