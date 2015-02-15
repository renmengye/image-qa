from stage import *
import lstmpy as lstmx

class LSTM(Stage):
    def __init__(self,
                 inputDim,
                 outputDim,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 cutOffZeroEnd=False,
                 multiErr=False,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
                 name=None):
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
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.cutOffZeroEnd = cutOffZeroEnd
        self.multiErr = multiErr
        self.random = np.random.RandomState(initSeed)

        if needInit:
            start = -initRange / 2.0
            end = initRange / 2.0
            Wxi = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxf = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxc = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wxo = self.random.uniform(start, end, (self.outputDim, self.inputDim))
            Wyi = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyf = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyc = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wyo = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wci = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wcf = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wco = self.random.uniform(start, end, (self.outputDim, self.outputDim))
            Wbi = np.ones((self.outputDim, 1))
            Wbf = np.ones((self.outputDim, 1))
            Wbc = np.zeros((self.outputDim, 1))
            Wbo = np.ones((self.outputDim, 1))

            Wi = np.concatenate((Wxi, Wyi, Wci, Wbi), axis=1)
            Wf = np.concatenate((Wxf, Wyf, Wcf, Wbf), axis=1)
            Wc = np.concatenate((Wxc, Wyc, Wbc), axis=1)
            Wo = np.concatenate((Wxo, Wyo, Wco, Wbo), axis=1)
            self.W = np.concatenate((Wi, Wf, Wc, Wo), axis = 1)
        else:
            self.W = initWeights
        self.X = 0
        self.Xend = 0
        self.Y = 0
        self.C = 0
        self.Z = 0
        self.Gi = 0
        self.Gf = 0
        self.Go = 0
        pass

    def forward(self, X):
        Y, C, Z, Gi, Gf, Go, Xend = \
            lstmx.forwardPassN(
            X, self.cutOffZeroEnd, self.W)

        self.X = X
        self.Y = Y
        self.C = C
        self.Z = Z
        self.Gi = Gi
        self.Gf = Gf
        self.Go = Go
        self.Xend = Xend

        return Y if self.multiErr else Y[:,-1]

    def backward(self, dEdY):
        self.dEdW, dEdX = lstmx.backPropagateN(dEdY,self.X,self.Y,
                                self.C,self.Z,self.Gi,
                                self.Gf,self.Go,
                                self.Xend,self.cutOffZeroEnd,
                                self.multiErr,self.outputdEdX,
                                self.W)
        return dEdX if self.outputdEdX else None
