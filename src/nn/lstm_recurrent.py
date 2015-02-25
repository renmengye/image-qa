from recurrent import *

class LSTM_Recurrent(Recurrent):
    def __init__(self,
                 inputDim,
                 outputDim,
                 timespan,
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

        if needInit:
            self.I = Map_Recurrent(
                    name=name + '.I',
                    inputsStr=['input(0)', name + '.Y(-1)', name + '.C(-1)'],
                    inputDim=D+D2+D2,
                    outputDim=D2,
                    activeFn=SigmoidActiveFn(),
                    initRange=initRange,
                    initSeed=initSeed,
                    biasInitConst=1.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )

            self.F = Map_Recurrent(
                    name=name + '.F',
                    inputsStr=['input(0)', name + '.Y(-1)', name + '.C(-1)'],
                    inputDim=D+D2+D2,
                    outputDim=D2,
                    activeFn=SigmoidActiveFn(),
                    initRange=initRange,
                    initSeed=initSeed+1,
                    biasInitConst=1.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )

            self.Z = Map_Recurrent(
                    name=name + '.Z',
                    inputsStr=['input(0)', name + '.Y(-1)'],
                    inputDim=D+D2,
                    outputDim=D2,
                    activeFn=TanhActiveFn(),
                    initRange=initRange,
                    initSeed=initSeed+2,
                    biasInitConst=0.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )

            self.O = Map_Recurrent(
                    name=name + '.O',
                    inputsStr=['input(0)', name + '.Y(-1)', name + '.C(0)'],
                    inputDim=D+D2+D2,
                    outputDim=D2,
                    activeFn=SigmoidActiveFn(),
                    initRange=initRange,
                    initSeed=initSeed+3,
                    biasInitConst=1.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )
        else:
            IW, FW, ZW, OW = self.splitWeights(initWeights)
            self.I = Map_Recurrent(
                    name=name + '.I',
                    inputsStr=['input(0)', name + '.Y(-1)', name + '.C(-1)'],
                    inputDim=D+D2+D2,
                    outputDim=D2,
                    activeFn=SigmoidActiveFn(),
                    needInit=False,
                    initWeights=IW,
                    biasInitConst=1.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )

            self.F = Map_Recurrent(
                    name=name + '.F',
                    inputsStr=['input(0)', name + '.Y(-1)', name + '.C(-1)'],
                    inputDim=D+D2+D2,
                    outputDim=D2,
                    activeFn=SigmoidActiveFn(),
                    needInit=False,
                    initWeights=FW,
                    biasInitConst=1.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )

            self.Z = Map_Recurrent(
                    name=name + '.Z',
                    inputsStr=['input(0)', name + '.Y(-1)'],
                    inputDim=D+D2,
                    outputDim=D2,
                    activeFn=TanhActiveFn(),
                    needInit=False,
                    initWeights=ZW,
                    biasInitConst=0.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )

            self.O = Map_Recurrent(
                    name=name + '.O',
                    inputsStr=['input(0)', name + '.Y(-1)', name + '.C(0)'],
                    inputDim=D+D2+D2,
                    outputDim=D2,
                    activeFn=SigmoidActiveFn(),
                    needInit=False,
                    initWeights=OW,
                    biasInitConst=1.0,
                    learningRate=learningRate,
                    momentum=momentum,
                    gradientClip=gradientClip,
                    weightClip=weightClip,
                    weightRegConst=weightRegConst
                )

        self.FC = ComponentProduct_Recurrent(
                name=name + '.F*C',
                inputsStr=[name + '.F(0)', name + '.C(-1)'],
                outputDim=D2
            )

        self.IZ = ComponentProduct_Recurrent(
                name=name + '.I*Z',
                inputsStr=[name + '.I(0)', name + '.Z(0)'],
                outputDim=D2
            )

        self.C = Sum_Recurrent(
                name=name + '.C',
                inputsStr=[name + '.F*C(0)', name + '.I*Z(0)'],
                numComponents=2,
                outputDim=D2
            )


        self.U = Active_Recurrent(
                name=name + '.U',
                inputsStr=[name + '.C(0)'],
                inputDim=D2,
                activeFn=TanhActiveFn()
            )

        self.Y = ComponentProduct_Recurrent(
                name=name + '.Y',
                inputsStr=[name + '.O(0)', name + '.U(0)'],
                outputDim=D2
            )
        stages = [self.I, self.F, self.Z, self.FC, self.IZ, self.C, self.O, self.U, self.Y]
        Recurrent.__init__(self,
                           stages=stages,
                           timespan=timespan,
                           outputStageName=name + '.Y',
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
        s = D + D2 + D2
        IW = W[:, :s]
        FW = W[:, s:s + s]
        s = s + s
        ZW = W[:, s:s + D + D2]
        s = D + D2
        OW = W[:, s:s + D + D2 + D2]
        return IW, FW, ZW, OW

    def loadWeights(self, W):
        IW, FW, ZW, OW = self.splitWeights(W)
        self.I.W= IW
        self.F.W = FW
        self.Z.W = ZW
        self.O.W = OW