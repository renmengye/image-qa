from recurrent_container import *
from element_wise import *
from sum import *
from activation_layer import *
from reshape import *

class LSTM(RecurrentContainer):
    def __init__(self,
                 inputDim,
                 outputDim,
                 timespan,
                 inputNames,
                 defaultValue=0.0,
                 initRange=1.0,
                 initSeed=2,
                 needInit=True,
                 initWeights=0,
                 multiInput=True,
                 multiOutput=False,
                 cutOffZeroEnd=True,
                 learningRate=0.0,
                 learningRateAnnealConst=0.0,
                 momentum=0.0,
                 deltaMomentum=0.0,
                 weightClip=0.0,
                 gradientClip=0.0,
                 weightRegConst=0.0,
                 outputdEdX=True,
                 name=None):
        D2 = outputDim
        multiOutput = multiOutput
        if name is None: print 'Warning: name is None.'
        self.inputDim = inputDim
        self.outputDim = outputDim

        self.state1 = RecurrentAdapter(ConcatenationLayer(
                      name=name + '.st1',
                      inputNames=['input(0)', name +'.H(-1)', name +'.C(-1)'],
                      axis=-1))

        self.I = RecurrentAdapter(FullyConnectedLayer(
                 name=name + '.I',
                 inputNames=[name + '.st1'],
                 outputDim=D2,
                 activationFn=SigmoidActivationFn(),
                 initRange=initRange,
                 initSeed=initSeed,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))

        self.F = RecurrentAdapter(FullyConnectedLayer(
                 name=name + '.F',
                 inputNames=[name + '.st1'],
                 outputDim=D2,
                 activationFn=SigmoidActivationFn(),
                 initRange=initRange,
                 initSeed=initSeed+1,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))

        self.state2 = RecurrentAdapter(ConcatenationLayer(
                      name=name + '.st2',
                      inputNames=['input(0)', name +'.H(-1)'],
                      axis=-1))

        self.Z = RecurrentAdapter(FullyConnectedLayer(
                 name=name + '.Z',
                 inputNames=[name + '.st2'],
                 outputDim=D2,
                 activationFn=TanhActivationFn(),
                 initRange=initRange,
                 initSeed=initSeed+2,
                 biasInitConst=0.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))

        self.state3 = RecurrentAdapter(ConcatenationLayer(
                      name=name + '.st3',
                      inputNames=['input(0)', name +'.H(-1)', name +'.C(0)'],
                      axis=-1))

        self.O = RecurrentAdapter(FullyConnectedLayer(
                 name=name + '.O',
                 inputNames=[name + '.st3'],
                 outputDim=D2,
                 activationFn=SigmoidActivationFn(),
                 initRange=initRange,
                 initSeed=initSeed+3,
                 biasInitConst=1.0,
                 learningRate=learningRate,
                 learningRateAnnealConst=learningRateAnnealConst,
                 momentum=momentum,
                 deltaMomentum=deltaMomentum,
                 gradientClip=gradientClip,
                 weightClip=weightClip,
                 weightRegConst=weightRegConst))
        
        if not needInit:
            self.I.W, self.F.W, self.Z.W, self.O.W = self.splitWeights(initWeights)

        self.FC = RecurrentAdapter(ElementWiseProduct(
                  name=name + '.F*C',
                  inputNames=[name + '.F', name + '.C(-1)'],
                  outputDim=D2))

        self.IZ = RecurrentAdapter(ElementWiseProduct(
                  name=name + '.I*Z',
                  inputNames=[name + '.I', name + '.Z'],
                  outputDim=D2))

        self.C = RecurrentAdapter(ElementWiseSum(
                 name=name + '.C',
                 inputNames=[name + '.F*C', name + '.I*Z'],
                 outputDim=D2))

        self.U = RecurrentAdapter(ActivationLayer(
                 name=name + '.U',
                 inputNames=[name + '.C'],
                 outputDim=D2,
                 activationFn=TanhActivationFn()))

        self.H = RecurrentAdapter(ElementWiseProduct(
                 name=name + '.H',
                 inputNames=[name + '.O', name + '.U'],
                 outputDim=D2,
                 defaultValue=defaultValue))

        stages = [self.state1, self.state2, self.I, self.F, self.Z, self.FC,
                  self.IZ, self.C, self.state3, self.O, self.U, self.H]

        RecurrentContainer.__init__(self,
                           stages=stages,
                           timespan=timespan,
                           inputNames=inputNames,
                           outputStageNames=[name + '.H'],
                           inputDim=inputDim,
                           outputDim=outputDim,
                           multiInput=multiInput,
                           multiOutput=multiOutput,
                           cutOffZeroEnd=cutOffZeroEnd,
                           name=name,
                           outputdEdX=outputdEdX)

    def getWeights(self):
        if self.I.stages[0].useGpu:
            return np.concatenate((
                    gpu.as_numpy_array(self.I.getWeights()),
                    gpu.as_numpy_array(self.F.getWeights()),
                    gpu.as_numpy_array(self.Z.getWeights()),
                    gpu.as_numpy_array(self.O.getWeights())), axis=0)
        else:
            return np.concatenate((self.I.getWeights(),
                               self.F.getWeights(),
                               self.Z.getWeights(),
                               self.O.getWeights()), axis=0)
            
    def getGradient(self):
        if self.I.stages[0].useGpu:
            return np.concatenate((
                gpu.as_numpy_array(self.I.getGradient()),
                gpu.as_numpy_array(self.F.getGradient()),
                gpu.as_numpy_array(self.Z.getGradient()),
                gpu.as_numpy_array(self.O.getGradient())), axis=0)
        else:
            return np.concatenate((self.I.getGradient(),
                                   self.F.getGradient(),
                                   self.Z.getGradient(),
                                   self.O.getGradient()), axis=0)

    def splitWeights(self, W):
        D = self.inputDim
        D2 = self.outputDim
        s = D + D2 + D2 + 1
        s2 = D + D2 + 1
        IW = W[:s, :]
        FW = W[s:s + s, :]
        ZW = W[s + s:s + s + s2, :]
        OW = W[s + s +s2:s + s + s2 + s, :]
        return IW, FW, ZW, OW

    def loadWeights(self, W):
        IW, FW, ZW, OW = self.splitWeights(W)
        if self.I.stages[0].useGpu:
            self.I.loadWeights(gpu.as_garray(IW))
            self.F.loadWeights(gpu.as_garray(FW))
            self.Z.loadWeights(gpu.as_garray(ZW))
            self.O.loadWeights(gpu.as_garray(OW))
        else:
            self.I.loadWeights(IW)
            self.F.loadWeights(FW)
            self.Z.loadWeights(ZW)
            self.O.loadWeights(OW)
