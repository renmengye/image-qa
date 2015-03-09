from sequential import *
from lstm import *
from map import *
from dropout import *
from reshape import *
from lut import *
from active_func import *
from model import *
import unittest

class Sequential_Tests(unittest.TestCase):
    """Sequential stacks of stages tests"""
    def setUp(self):
        random = np.random.RandomState(2)
        self.trainInput = random.uniform(0, 10, (5, 5, 1)).astype(int)
        self.trainTarget = random.uniform(0, 1, (5, 1)).astype(int)

    def test_grad(self):
        wordEmbed = np.random.rand(5, np.max(self.trainInput))
        timespan = self.trainInput.shape[1]
        time_unfold = TimeUnfold()

        lut = LUT(
            inputDim=np.max(self.trainInput)+1,
            outputDim=5,
            needInit=False,
            initWeights=wordEmbed
        )

        time_fold = TimeFold(
            timespan=timespan
        )

        lstm = LSTM(
            inputDim=5,
            outputDim=5,
            initRange=.1,
            initSeed=3,
            cutOffZeroEnd=True,
            multiErr=True
        )

        dropout = Dropout(
            dropoutRate=0.5,
            initSeed=2,
            debug=True
        )

        lstm_second = LSTM(
            inputDim=5,
            outputDim=5,
            initRange=.1,
            initSeed=3,
            cutOffZeroEnd=True,
            multiErr=False
        )

        soft = Map(
            inputDim=5,
            outputDim=2,
            activeFn=SoftmaxActiveFn,
            initRange=1,
            initSeed=5
        )

        self.model = Model(Sequential(
            stages=[
                time_unfold,
                lut,
                time_fold,
                lstm,
                dropout,
                lstm_second,
                soft
            ]
        ), crossEntIdx, argmax)
        self.hasDropout = True
        output = self.model.forward(self.trainInput, dropout=self.hasDropout)
        E, dEdY = self.model.getCost(output, self.trainTarget)
        dEdX = self.model.backward(dEdY)
        self.chkgrd(soft.dEdW, self.evaluateGrad(soft.getWeights()))
        self.chkgrd(lstm_second.dEdW, self.evaluateGrad(lstm_second.getWeights()))
        self.chkgrd(lstm.dEdW, self.evaluateGrad(lstm.getWeights()))

    def chkgrd(self, dE, dETmp):
        dE = dE.reshape(dE.size)
        dETmp = dETmp.reshape(dE.size)
        tolerance = 1e-1
        for i in range(dE.size):
            # print 'DE',
            # print dE[i],
            # print 'DETMP',
            # print dETmp[i],
            # print 'R',
            # print dE[i] / dETmp[i]
            self.assertTrue(
                (dE[i] == 0 and dETmp[i] == 0) or
                (np.abs(dE[i] / dETmp[i] - 1) < tolerance))

    def evaluateGrad(self, W):
        eps = 1e-2
        dEdW = np.zeros(W.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i,j] += eps
                output = self.model.forward(self.trainInput, dropout=self.hasDropout)
                Etmp1, d1 = self.model.getCost(output, self.trainTarget)

                W[i,j] -= 2 * eps
                output = self.model.forward(self.trainInput, dropout=self.hasDropout)
                Etmp2, d2 = self.model.getCost(output, self.trainTarget)

                dEdW[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                W[i,j] += eps
        return dEdW

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(Sequential_Tests))
    unittest.TextTestRunner(verbosity=2).run(suite)