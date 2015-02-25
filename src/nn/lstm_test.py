from sequential import *
from lstm import *
from map import *
from dropout import *
from time_fold import *
from time_unfold import *
from lut import *
from model import *
from recurrent import *
import unittest

class LSTM_Recurrent_Real_Tests(unittest.TestCase):
    def test_all(self):
        data = np.load('../../data/sentiment3/train-1.npy')
        wordEmbed = np.load('../../data/sentiment3/word-embed-0.npy')
        trainInput = data[0]
        trainTarget = data[1]
        D = 300
        D2 = 50
        N = 25
        Time = trainInput.shape[1]
        multiOutput = False
        time_unfold = TimeUnfold()
        lut = LUT(
            inputDim=np.max(trainInput)+1,
            outputDim=D,
            needInit=False,
            initWeights=wordEmbed
        )

        time_fold = TimeFold(
            timespan=Time
        )

        dropout = Dropout(
            dropoutRate=0.2,
            initSeed=2
        )
        dropout2 = Dropout(
            dropoutRate=0.2,
            initSeed=2
        )

        I = Map_Recurrent(
                name='I',
                inputsStr=['input(0)', 'Y(-1)', 'C(-1)'],
                inputDim=D+D2+D2,
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=0.1,
                initSeed=5,
                biasInitConst=1.0,
                learningRate=0.8,
                momentum=0.9
            )

        F = Map_Recurrent(
                name='F',
                inputsStr=['input(0)', 'Y(-1)', 'C(-1)'],
                inputDim=D+D2+D2,
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=0.1,
                initSeed=6,
                biasInitConst=1.0,
                learningRate=0.8,
                momentum=0.9
            )

        Z = Map_Recurrent(
                name='Z',
                inputsStr=['input(0)', 'Y(-1)'],
                inputDim=D+D2,
                outputDim=D2,
                activeFn=TanhActiveFn(),
                initRange=0.1,
                initSeed=7,
                biasInitConst=0.0,
                learningRate=0.8,
                momentum=0.9
            )

        FC = ComponentProduct_Recurrent(
                name='F.C',
                inputsStr=['F(0)', 'C(-1)'],
                outputDim=D2
            )

        IZ = ComponentProduct_Recurrent(
                name='I.Z',
                inputsStr=['I(0)', 'Z(0)'],
                outputDim=D2
            )

        C = Sum_Recurrent(
                name='C',
                inputsStr=['F.C(0)', 'I.Z(0)'],
                numComponents=2,
                outputDim=D2
            )

        O = Map_Recurrent(
                name='O',
                inputsStr=['input(0)', 'Y(-1)', 'C(0)'],
                inputDim=D+D2+D2,
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=0.1,
                initSeed=8,
                biasInitConst=1.0,
                learningRate=0.8,
                momentum=0.9
            )

        U = Active_Recurrent(
                name='U',
                inputsStr=['C(0)'],
                inputDim=D2,
                activeFn=TanhActiveFn()
            )

        Y = ComponentProduct_Recurrent(
                name='Y',
                inputsStr=['O(0)', 'U(0)'],
                outputDim=D2
            )

        lstm = Recurrent(
                name='lstm',
                stages=[I, F, Z, FC, IZ, C, O, U, Y],
                timespan=Time,
                inputDim=D,
                outputDim=D2,
                outputStageName='Y',
                multiOutput=multiOutput,
                outputdEdX=True)

        W = np.concatenate((I.W, F.W, Z.W, O.W), axis=-1)
        lstm2 = LSTM(
            name='lstm',
            inputDim=D,
            outputDim=D2,
            needInit=False,
            initWeights=W,
            cutOffZeroEnd=True,
            multiErr=multiOutput,
            learningRate=0.8,
            momentum=0.9
        )

        sig = Map(
            name='sig',
            inputDim=D2,
            outputDim=1,
            activeFn=SigmoidActiveFn(),
            initRange=0.1,
            initSeed=5,
            learningRate=0.01,
            momentum=0.9,
            weightClip=10.0,
            gradientClip=0.1,
            weightRegConst=0.00005
        )
        sig2 = Map(
            name='sig',
            inputDim=D2,
            outputDim=1,
            activeFn=SigmoidActiveFn(),
            initRange=0.1,
            initSeed=5,
            learningRate=0.01,
            momentum=0.9,
            weightClip=10.0,
            gradientClip=0.1,
            weightRegConst=0.00005
        )

        model1 = Model(Sequential(
            stages=[
                time_unfold,
                lut,
                time_fold,
                dropout,
                lstm,
                sig
            ]
        ), crossEntOne, hardLimit)

        model2 = Model(Sequential(
            stages=[
                time_unfold,
                lut,
                time_fold,
                dropout2,
                lstm2,
                sig2
            ]
        ), crossEntOne, hardLimit)

        input_ = trainInput[0:N, 0:Time]
        target_ = trainTarget[0:N]
        Y1 = model1.forward(input_)
        Y2 = model2.forward(input_)
        self.chkEqual(Y1, Y2)

        E, dEdY1 = model1.getCost(Y1, target_)
        E, dEdY2 = model2.getCost(Y2, target_)
        model1.backward(dEdY1)
        model2.backward(dEdY2)
        dEdX1 = lstm.dEdX
        dEdX2 = lstm2.dEdX
        self.chkEqual(dEdX1, dEdX2)

        dEdW = np.concatenate((I.dEdW, F.dEdW, Z.dEdW, O.dEdW), axis=-1)
        self.chkEqual(dEdW, lstm2.dEdW)
        lstm.updateWeights()
        lstm2.updateWeights()
        W = np.concatenate((I.W, F.W, Z.W, O.W), axis=-1)
        self.chkEqual(W, lstm2.W)

    def chkEqual(self, a, b):
        tolerance = 1e-4
        a = a.reshape(a.size)
        b = b.reshape(b.size)
        for i in range(a.size):
            self.assertTrue(
                (a[i] == 0 and b[i] == 0) or
                (np.abs(a[i] / b[i] - 1) < tolerance))

if __name__ == '__main__':
    unittest.main()
