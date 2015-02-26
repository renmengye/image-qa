from lstm import *
from map import *
from lut import *
from inner_prod import *
from time_sum import *
from recurrent import *
import unittest
import numpy as np

class StageTests(unittest.TestCase):
    def calcgrd(self, X, T):
        Y = self.model.forward(X)
        W = self.stage.W
        E, dEdY = self.costFn(Y, T)
        dEdX = self.model.backward(dEdY)
        dEdW = self.stage.dEdW
        eps = 1e-3
        dEdXTmp = np.zeros(X.shape)

        if hasattr(W, 'shape'):
            dEdWTmp = np.zeros(W.shape)
            for i in range(0, self.stage.W.shape[0]):
                for j in range(0, self.stage.W.shape[1]):
                    self.stage.W[i,j] += eps
                    Y = self.model.forward(X)
                    Etmp1, d1 = self.costFn(Y, T)

                    self.stage.W[i,j] -= 2 * eps
                    Y = self.model.forward(X)
                    Etmp2, d2 = self.costFn(Y, T)

                    dEdWTmp[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
                    self.stage.W[i,j] += eps
        else:
            dEdW = 0
            dEdWTmp = 0  
        if self.testInputErr:
            if len(X.shape) == 3:
                for n in range(0, X.shape[0]):
                    for t in range(0, X.shape[1]):
                        for j in range(0, X.shape[2]):
                            X[n, t, j] += eps
                            Y = self.model.forward(X)
                            Etmp1, d1 = self.costFn(Y, T)

                            X[n, t, j] -= 2 * eps
                            Y = self.model.forward(X)
                            Etmp2, d2 = self.costFn(Y, T)

                            dEdXTmp[n, t, j] = (Etmp1 - Etmp2) / 2.0 / eps
                            X[n, t, j] += eps

            elif len(X.shape) == 2:
                for n in range(0, X.shape[0]):
                    for j in range(0, X.shape[1]):
                        X[n, j] += eps
                        Y = self.model.forward(X)
                        Etmp1, d1 = self.costFn(Y, T)

                        X[n, j] -= 2 * eps
                        Y = self.model.forward(X)
                        Etmp2, d2 = self.costFn(Y, T)

                        dEdXTmp[n, j] = (Etmp1 - Etmp2) / 2.0 / eps
                        X[n, j] += eps

            elif len(X.shape) == 1:
                for j in range(0, X.shape[0]):
                    X[j] += eps
                    Y = self.model.forward(X)
                    Etmp1, d1 = self.costFn(Y, T)

                    X[j] -= 2 * eps
                    Y = self.model.forward(X)
                    Etmp2, d2 = self.costFn(Y, T)

                    dEdXTmp[j] = (Etmp1 - Etmp2) / 2.0 / eps
                    X[j] += eps
        else:
            dEdX = None
            dEdXTmp = None
        return dEdW, dEdWTmp, dEdX, dEdXTmp

    def chkgrd(self, dE, dETmp):
        dE = dE.reshape(dE.size)
        dETmp = dETmp.reshape(dE.size)
        tolerance = 1e-4
        for i in range(dE.size):
            self.assertTrue(
                (dE[i] == 0 and dETmp[i] == 0) or
                (np.abs(dE[i] / dETmp[i] - 1) < tolerance))

class LSTM_MultiErr_Tests(StageTests):
    """LSTM multi error tests"""
    def setUp(self):
        self.stage = LSTM(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            multiErr=True,
            cutOffZeroEnd=False)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,4,5))
        T = random.uniform(-0.1, 0.1, (6,4,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class LSTM_MultiErrCutZero_Tests(StageTests):
    """LSTM single error tests"""
    def setUp(self):
        self.stage = LSTM(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            multiErr=True,
            cutOffZeroEnd=True)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = np.concatenate(
            (random.uniform(-0.1, 0.1, (6,4,5)),
            np.zeros((6,3,5))), axis=1)
        T = np.concatenate(
            (random.uniform(-0.1, 0.1, (6,4,3)),
            np.zeros((6,4,3))), axis=1) # Need one more time dimension for cut off.
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX[:,0:4], dEdXTmp[:,0:4])

class LSTM_SingleErr_Tests(StageTests):
    """LSTM single error tests"""
    def setUp(self):
        self.stage = LSTM(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            multiErr=False,
            cutOffZeroEnd=False)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,4,5))
        T = random.uniform(-0.1, 0.1, (6,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class LSTM_SingleErrCutZero_Tests(StageTests):
    """LSTM single error tests"""
    def setUp(self):
        self.stage = LSTM(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            multiErr=False,
            cutOffZeroEnd=True)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = np.concatenate(
            (random.uniform(-0.1, 0.1, (6,4,5)),
            np.zeros((6,3,5))), axis=1)
        T = random.uniform(-0.1, 0.1, (6,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX[:,0:4], dEdXTmp[:,0:4])

class MapIdentity_Tests(StageTests):
    """Linear map tests"""
    def setUp(self):
        self.stage = Map(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            activeFn=IdentityActiveFn)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,5))
        T = random.uniform(-0.1, 0.1, (6,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class MapSigmoid_Tests(StageTests):
    """Sigmoid map tests"""
    def setUp(self):
        self.stage = Map(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            activeFn=SigmoidActiveFn)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,5))
        T = random.uniform(-0.1, 0.1, (6,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class MapSigmoid_CrossEnt_Tests(StageTests):
    """Sigmoid map tests"""
    def setUp(self):
        self.stage = Map(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            activeFn=SigmoidActiveFn)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = crossEntOne
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,5))
        T = random.uniform(0, 1, (6,3)).astype(int)
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class MapSoftmax_Tests(StageTests):
    """Sigmoid map tests"""
    def setUp(self):
        self.stage = Map(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            activeFn=SoftmaxActiveFn)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,5))
        T = random.uniform(-0.1, 0.1, (6,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class MapSoftmax_CrossEnt_Tests(StageTests):
    """Linear map tests"""
    def setUp(self):
        self.stage = Map(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1,
            activeFn=SoftmaxActiveFn)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = crossEntIdx
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,5))
        T = random.uniform(0, 2, (6)).astype(int)
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class LUT_Tests(StageTests):
    """Lookup table tests"""
    def setUp(self):
        self.stage = LUT(
            inputDim=5,
            outputDim=3,
            initRange=0.1,
            initSeed=1)
        self.model = self.stage
        self.testInputErr = False
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = np.array([1,2,3,4,5], dtype=int)
        T = random.uniform(-0.1, 0.1, (5,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)

class InnerProduct_Tests(StageTests):
    """Inner product tests"""
    def setUp(self):
        self.stage = InnerProduct()
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,2,5))
        T = random.uniform(-0.1, 0.1, (6,1))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdX, dEdXTmp)

class TimeSum_Tests(StageTests):
    def setUp(self):
        self.stage = TimeSum()
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (6,3,5))
        T = random.uniform(-0.1, 0.1, (6,5))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdX, dEdXTmp)

class Sum_Recurrent_Tests(StageTests):
    def setUp(self):
        self.stage = Sum_Recurrent(
            outputDim=3,
            name='sum',
            inputsStr=[],
            numComponents=2)
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (3,6))
        T = random.uniform(-0.1, 0.1, (3,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdX, dEdXTmp)

class ComponentProduct_Recurrent_Tests(StageTests):
    def setUp(self):
        self.stage = ComponentProduct_Recurrent(
            outputDim=3,
            name='product',
            inputsStr=[])
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (3,6))
        T = random.uniform(-0.1, 0.1, (3,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdX, dEdXTmp)

class MapSigmoid_Recurrent_Tests(StageTests):
    def setUp(self):
        self.stage = Map_Recurrent(
            name='sigmoid',
            activeFn=SigmoidActiveFn(),
            outputDim=3,
            inputsStr=[],
            initRange=0.1,
            initSeed=1,
            )
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (3,6))
        T = random.uniform(-0.1, 0.1, (3,3))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

class Active_Recurrent_Tests(StageTests):
    def setUp(self):
        self.stage = Active_Recurrent(
            outputDim=6,
            name='active',
            inputsStr=[],
            activeFn=TanhActiveFn())
        self.model = self.stage
        self.testInputErr = True
        self.costFn = meanSqErr
    def test_grad(self):
        random = np.random.RandomState(2)
        X = random.uniform(-0.1, 0.1, (3,6))
        T = random.uniform(-0.1, 0.1, (3,6))
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdX, dEdXTmp)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(LSTM_MultiErr_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(LSTM_MultiErrCutZero_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(LSTM_SingleErr_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(LSTM_SingleErrCutZero_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(MapIdentity_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(MapSigmoid_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(MapSigmoid_CrossEnt_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(MapSoftmax_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(MapSoftmax_CrossEnt_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(LUT_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(InnerProduct_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(TimeSum_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(Sum_Recurrent_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(ComponentProduct_Recurrent_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(MapSigmoid_Recurrent_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(Active_Recurrent_Tests))
    unittest.TextTestRunner(verbosity=2).run(suite)