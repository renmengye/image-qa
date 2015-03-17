from recurrent import *

class LSTM_Recurrent(Recurrent):
    def __init__(self,
                 inputDim,
                 outputDim,
                 timespan,
                 defaultValue=None,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 multiOutput=False,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
                 name=None):
        D = inputDim
        D2 = outputDim
        Time = timespan
        multiOutput = multiOutput
        if name is None: print 'Warning: name is None.'

        self.I = Map_Recurrent(
                 name=name + '.I',
                 inputNames=['input(0)', name + '.H(-1)', name + '.C(-1)'],
                 #inputDim=D+D2+D2,
                 outputDim=D2,
                 activeFn=SigmoidActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 momentum=momentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst)

        self.F = Map_Recurrent(
                 name=name + '.F',
                 inputNames=['input(0)', name + '.H(-1)', name + '.C(-1)'],
                 #inputDim=D+D2+D2,
                 outputDim=D2,
                 activeFn=SigmoidActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed+1,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 momentum=momentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst)

        self.Z = Map_Recurrent(
                 name=name + '.Z',
                 inputNames=['input(0)', name + '.H(-1)'],
                 #inputDim=D+D2,
                 outputDim=D2,
                 activeFn=TanhActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed+2,
                 biasInitConst=0.0,
                 learningRate=learningRate,
                 momentum=momentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst)

        self.O = Map_Recurrent(
                 name=name + '.O',
                 inputNames=['input(0)', name + '.H(-1)', name + '.C(0)'],
                 #inputDim=D+D2+D2,
                 outputDim=D2,
                 activeFn=SigmoidActiveFn(),
                 initRange=initRange,
                 initSeed=initSeed+3,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 momentum=momentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst)
        
        if not needInit:
            self.I.W, self.F.W, self.Z.W, self.O.W = self.splitWeights(initWeights)

        self.FC = ComponentProduct_Recurrent(
                  name=name + '.F*C',
                  inputNames=[name + '.F(0)', name + '.C(-1)'],
                  outputDim=D2)

        self.IZ = ComponentProduct_Recurrent(
                  name=name + '.I*Z',
                  inputNames=[name + '.I(0)', name + '.Z(0)'],
                  outputDim=D2)  

        self.C = Sum_Recurrent(
                 name=name + '.C',
                 inputNames=[name + '.F*C(0)', name + '.I*Z(0)'],
                 numComponents=2,
                 outputDim=D2)

        self.U = Active_Recurrent(
                 name=name + '.U',
                 inputNames=[name + '.C(0)'],
                 outputDim=D2,
                 activeFn=TanhActiveFn())

        self.H = ComponentProduct_Recurrent(
                 name=name + '.H',
                 inputNames=[name + '.O(0)', name + '.U(0)'],
                 outputDim=D2,
                 defaultValue=defaultValue)

        stages = [self.I, self.F, self.Z, self.FC, self.IZ, self.C, self.O, self.U, self.H]
        Recurrent.__init__(self,
                           stages=stages,
                           timespan=timespan,
                           outputStageName=name + '.H',
                           inputDim=inputDim,
                           outputDim=outputDim,
                           multiOutput=multiOutput,
                           name=name,
                           outputdEdX=outputdEdX)

    def getWeights(self):
        return np.concatenate((self.I.W, self.F.W, self.Z.W, self.O.W), axis=-1)

    def getGradient(self):
        return np.concatenate((self.I.dEdW, self.F.dEdW, self.Z.dEdW, self.O.dEdW), axis=-1)

    def splitWeights(self, W):
        D = self.inputDim
        D2 = self.outputDim
        s = D + D2 + D2 + 1
        s2 = D + D2 + 1
        IW = W[:, :s]
        FW = W[:, s:s + s]
        ZW = W[:, s + s:s + s + s2]
        OW = W[:, s + s +s2:s + s + s2 + s]
        return IW, FW, ZW, OW

    def loadWeights(self, W):
        IW, FW, ZW, OW = self.splitWeights(W)
        self.I.W= IW
        self.F.W = FW
        self.Z.W = ZW
        self.O.W = OW
        for t in range(1, self.timespan):
            for s in range(1, len(self.stages[t]) -1):
                self.stages[t][s].W = self.stages[0][s].W