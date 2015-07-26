import unittest

from src.nn.legacy.sequential_container import *
from lstm_old import *
from fully_connected_layer import *
from dropout_layer import *
from reshape import *
from embedding_layer import *
from activation_fn import *


class Sequential_Tests(unittest.TestCase):
    """Sequential stacks of stages tests"""
    def setUp(self):
        random = np.random.RandomState(2)
        self.trainInput = random.uniform(0, 10, (5, 5, 1)).astype(int)
        self.trainTarget = random.uniform(0, 1, (5, 1)).astype(int)

    def test_grad(self):
        wordEmbed = np.random.rand(np.max(self.trainInput), 5)
        timespan = self.trainInput.shape[1]
        time_unfold = TimeUnfold()

        lut = EmbeddingLayer(
            inputDim=np.max(self.trainInput)+1,
            outputDim=5,
            inputNames=None,
            needInit=False,
            initWeights=wordEmbed
        )

        m = FullyConnectedLayer(
            outputDim=5,
            activationFn=IdentityActivationFn(),
            inputNames=None,
            initRange=0.1,
            initSeed=1,
        )

        time_fold = TimeFold(
            timespan=timespan
        )

        lstm = LSTM_Old(
            inputDim=5,
            outputDim=5,
            initRange=.1,
            initSeed=3,
            cutOffZeroEnd=True,
            multiErr=True
        )

        dropout = DropoutLayer(
            name='d1',
            dropoutRate=0.5,
            inputNames=None,
            outputDim=5,
            initSeed=2,
            debug=True
        )

        lstm_second = LSTM_Old(
            inputDim=5,
            outputDim=5,
            initRange=.1,
            initSeed=3,
            cutOffZeroEnd=True,
            multiErr=False
        )

        soft = FullyConnectedLayer(
            outputDim=2,
            activationFn=SoftmaxActivationFn,
            initRange=0.1,
            initSeed=5
        )

        self.model = SequentialContainer(
            stages=[
                time_unfold,
                lut,
                m,
                time_fold,
                lstm,
                dropout,
                lstm_second,
                soft
            ])
        self.hasDropout = True
        costFn = crossEntIdx
        self.model.setIsTraining(self.hasDropout) # Activates dropout
        output = self.model.forward(self.trainInput)
        E, dEdY = costFn(output, self.trainTarget)
        dEdX = self.model.backward(dEdY)
        self.chkgrd(soft.dEdW, self.evaluateGrad(soft.getWeights(), costFn))
        self.chkgrd(m.dEdW, self.evaluateGrad(m.getWeights(), costFn))

    def chkgrd(self, dE, dETmp):
        dE = dE.reshape(dE.size)
        dETmp = dETmp.reshape(dE.size)
        tolerance = 5e-1
        for i in range(dE.size):
            self.assertTrue(
                (dE[i] == 0 and dETmp[i] == 0) or
                (np.abs(dE[i] / dETmp[i] - 1) < tolerance))

    def evaluateGrad(self, W, costFn):
        eps = 1
        dEdW = np.zeros(W.shape)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i,j] += eps
                output = self.model.forward(self.trainInput)
                Etmp1, d1 = costFn(output, self.trainTarget)

                W[i,j] -= 2 * eps
                output = self.model.forward(self.trainInput)
                Etmp2, d2 = costFn(output, self.trainTarget)

                dEdW[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                W[i,j] += eps
        return dEdW

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(Sequential_Tests))
    unittest.TextTestRunner(verbosity=2).run(suite)
