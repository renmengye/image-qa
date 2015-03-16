from sequential import *
from lstm_old import *
from map import *
from dropout import *
from reshape import *
from lut import *
from model import *
from lstm_recurrent import *
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
            inputNames=None,
            needInit=False,
            initWeights=wordEmbed
        )

        time_fold = TimeFold(
            timespan=Time
        )

        dropout = Dropout(
            name='d1',
            dropoutRate=0.2,
            initSeed=2,
            inputNames=None,
            outputDim=D2
        )
        dropout2 = Dropout(
            name='d2',
            dropoutRate=0.2,
            initSeed=2,
            inputNames=None,
            outputDim=D2
        )
        lstm = LSTM_Recurrent(
                name='lstm',
                timespan=Time,
                inputDim=D,
                outputDim=D2,
                multiOutput=multiOutput,
                learningRate=0.8,
                momentum=0.9,
                outputdEdX=True)

        W = lstm.getWeights()
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

        lstm2.W = W
        Y2 = model2.forward(input_)
        self.chkEqual(Y1, Y2)

        E, dEdY1 = model1.getCost(Y1, target_)
        E, dEdY2 = model2.getCost(Y2, target_)
        model1.backward(dEdY1)
        model2.backward(dEdY2)
        dEdX1 = lstm.dEdX
        dEdX2 = lstm2.dEdX
        self.chkEqual(dEdX1, dEdX2)

        #dEdW = np.concatenate((I.dEdW, F.dEdW, Z.dEdW, O.dEdW), axis=-1)
        dEdW = lstm.getGradient()
        self.chkEqual(dEdW, lstm2.dEdW)
        lstm.updateWeights()
        lstm2.updateWeights()
        #W = np.concatenate((I.W, F.W, Z.W, O.W), axis=-1)
        W = lstm.getWeights()
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
