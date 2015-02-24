from recurrent import *
from func import *
from lstm import *
import stage_tests
import unittest

class Recurrent_Tests(stage_tests.StageTests):
    def setUp(self):
        self.N = 5
        self.T = 5
        self.D = 10
        self.D2 = 5
        self.sigm = Map_Recurrent(
                name='sigm',
                inputsStr=['input(0)', 'sigm(-1)', 'sigm(-2)'],
                inputDim=self.D+self.D2+self.D2,
                outputDim=self.D2,
                activeFn=SigmoidActiveFn(),
                initRange=1,
                initSeed=5
            )
        self.stage = self.sigm
        self.model = Recurrent(
            stages=[self.sigm],
            timespan=self.T,
            inputDim=self.D,
            outputDim=self.D2,
            outputStageName='sigm',
            multiOutput=True,
            name='container',
            outputdEdX=True)

        self.testInputErr = True
        self.costFn = meanSqErr

    def test_grad(self):
        random = np.random.RandomState(1)
        X = random.rand(self.N, self.T, self.D)
        T = random.rand(self.N, self.T, self.D2)
        dEdW, dEdWTmp, dEdX, dEdXTmp = self.calcgrd(X, T)
        self.chkgrd(dEdW, dEdWTmp)
        self.chkgrd(dEdX, dEdXTmp)

    def test_forward(self):
        random = np.random.RandomState(1)
        X = random.rand(self.N, self.T, self.D)
        tolerance = 1e-4
        Y2 = self.realForward(X)
        Y2 = Y2.reshape(Y2.size)
        Y = self.model.forward(X)
        Y = Y.reshape(Y.size)
        for i in range(Y.size):
            self.assertTrue((Y[i] == 0 and Y2[i] == 0) or (np.abs(Y[i] / Y2[i] - 1) < tolerance))

    def realForward(self, X):
        Y2 = np.zeros((self.N, self.T, self.D2))
        self.sigm.setDimension(self.N)
        for n in range(self.N):
            for t in range(self.T):
                Y2[n, t, :] = self.sigm.forward(np.concatenate((X[n, t, :], Y2[n, t-1, :], Y2[n, t-2, :])))
        return Y2

class LSTM_Recurrent_Tests(unittest.TestCase):
    def test_singleErr(self):
        self.func(False)

    def test_multiErr(self):
        self.func(True)

    def func(self, multiOutput):
        N = 5
        D = 10
        D2 = 5
        Time = 5
        I = Map_Recurrent(
                name='I',
                inputsStr=['input(0)', 'Y(-1)', 'C(-1)'],
                inputDim=D+D2+D2,
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=1,
                initSeed=5
            )
        F = Map_Recurrent(
                name='F',
                inputsStr=['input(0)', 'Y(-1)', 'C(-1)'],
                inputDim=D+D2+D2,
                outputDim=D2,
                activeFn=SigmoidActiveFn(),
                initRange=1,
                initSeed=6
            )
        Z = Map_Recurrent(
                name='Z',
                inputsStr=['input(0)', 'Y(-1)'],
                inputDim=D+D2,
                outputDim=D2,
                activeFn=TanhActiveFn(),
                initRange=1,
                initSeed=7
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
                initRange=1,
                initSeed=8
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
        random = np.random.RandomState(1)
        X = random.rand(N, Time, D)
        if multiOutput:
            T = random.rand(N, Time, D2)
        else:
            T = random.rand(N, D2)
        #start = time.time()
        Y = lstm.forward(X)
        #print time.time()-start
        E, dEdY = meanSqErr(Y, T)
        #start = time.time()
        dEdX = lstm.backward(dEdY)
        #print time.time()-start
        W = np.concatenate((I.getWeights(), F.getWeights(), Z.getWeights(), O.getWeights()), axis=-1)
        lstm2 = LSTM(
            inputDim=D,
            outputDim=D2,
            needInit=False,
            initWeights=W,
            cutOffZeroEnd=True,
            multiErr=multiOutput
        )
        #start = time.time()
        if multiOutput:
            Y2 = lstm2.forward(X)[:,:-1]
        else:
            Y2 = lstm2.forward(X)
        #print time.time()-start
        #start = time.time()
        E, dEdY2 = meanSqErr(Y2, T)
        if multiOutput:
            dEdX2 = lstm2.backward(np.concatenate((dEdY2, np.zeros((N, 1, D2))), axis=1))
        else:
            dEdX2 = lstm2.backward(dEdY2)
        #print time.time()-start

        dEdW = np.concatenate((I.getGradient(), F.getGradient(), Z.getGradient(), O.getGradient()), axis=-1)
        dEdW2 = lstm2.getGradient()
        self.chkEqual(Y, Y2)
        self.chkEqual(dEdX, dEdX2)
        self.chkEqual(dEdW, dEdW2)

    def chkEqual(self, a, b):
        tolerance = 1e-4
        a = a.reshape(a.size)
        b = b.reshape(b.size)
        for i in range(a.size):
            self.assertTrue(
                (a[i] == 0 and b[i] == 0) or
                (np.abs(a[i] / b[i] - 1) < tolerance))

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(Recurrent_Tests))
    suite.addTests(
        unittest.TestLoader().loadTestsFromTestCase(LSTM_Recurrent_Tests))
    unittest.TextTestRunner(verbosity=2).run(suite)