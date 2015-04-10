from stage import *
import os
#import gnumpy as gpu

class Map(Stage):
    def __init__(self,
                 outputDim,
                 activeFn,
                 inputNames=None,
                 initRange=1.0,
                 biasInitConst=-1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
                 defaultValue=0.0,
                 name=None):
        Stage.__init__(self,
                 name=name,
                 inputNames=inputNames,
                 outputDim=outputDim,
                 defaultValue=defaultValue,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 weightClip=weightClip,
                 gradientClip=gradientClip,
                 weightRegConst=weightRegConst,
                 outputdEdX=outputdEdX)
        self.activeFn = activeFn
        self.inputDim = None
        self.random = np.random.RandomState(initSeed)
        if not needInit:
            self.W = initWeights
        else:
            # Lazy initialize the weights until the first data arrives
            self.W = None
        self.initRange = initRange
        self.biasInitConst = biasInitConst
        self.X = 0
        self.Y = 0
        pass

    def initWeights(self):
        # if self.biasInitConst >= 0.0:
        #     self.W = np.concatenate((self.random.uniform(
        #         -self.initRange/2.0, self.initRange/2.0, (self.outputDim, self.inputDim)),
        #         np.ones((self.outputDim, 1)) * self.biasInitConst), axis=-1)
        # else:
        #     self.W = self.random.uniform(
        #         -self.initRange/2.0, self.initRange/2.0, (self.outputDim, self.inputDim + 1))
        if self.biasInitConst >= 0.0:
            self.W = np.concatenate((self.random.uniform(
                -self.initRange/2.0, self.initRange/2.0, (self.inputDim, self.outputDim)),
                np.ones((1, self.outputDim)) * self.biasInitConst), axis=0)
        else:
            self.W = self.random.uniform(
                -self.initRange/2.0, self.initRange/2.0, (self.inputDim + 1, self.outputDim))

        # is os.environ.has_key('GNUMPY_USE_GPU') and os.environ['GNUMPY_USE_GPU'] == 'yes':
        #     self.gpu = True
        #     self.W = gpu.array(self.W)
    
    def forward(self, X):
        if self.inputDim is None: self.inputDim = X.shape[-1]
        if self.W is None: self.initWeights()
        self.X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=-1)
        Z = np.dot(self.X, self.W)
        self.Y = self.activeFn.forward(Z)
        return self.Y

    def backward(self, dEdY):
        dEdZ = self.activeFn.backward(dEdY, self.Y, 0)
        # self.dEdW = np.dot(dEdZ.transpose(), self.X)
        # dEdX = np.dot(dEdZ, self.W[:, :-1])
        self.dEdW = np.dot(self.X.transpose(), dEdZ)
        dEdX = np.dot(dEdZ, self.W[:-1, :].transpose())
        return dEdX if self.outputdEdX else None
