from recurrent import *
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
                outputDim=self.D2,
                activeFn=SigmoidActiveFn(),
                initRange=1,
                initSeed=5,
                learningRate=0.9
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
        for t in range(self.T):
            Y2[:, t, :] = self.sigm.forward(
                np.concatenate((X[:, t, :], Y2[:, t-1, :], Y2[:, t-2, :]), axis=-1))
        return Y2

class LSTM_Recurrent_Random_Tests(unittest.TestCase):
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
                outputDim=D2,
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

        W = np.concatenate((I.getWeights(), F.getWeights(), Z.getWeights(), O.getWeights()), axis=-1)
        lstm2 = LSTM(
            name='lstm2',
            inputDim=D,
            outputDim=D2,
            needInit=False,
            initWeights=W,
            cutOffZeroEnd=True,
            multiErr=multiOutput,
            learningRate=0.8,
            momentum=0.9
        )
        W2 = lstm2.W
        self.chkEqual(W, W2)

        random = np.random.RandomState(1)
        costFn = crossEntOne
        for i in range(3):
            X = random.rand(N, Time, D)
            if multiOutput:
                T = random.rand(N, Time, D2)
            else:
                T = random.rand(N, D2)

            Y = lstm.forward(X)
            E, dEdY = costFn(Y, T)
            dEdX = lstm.backward(dEdY)
            if multiOutput:
                Y2 = lstm2.forward(X)[:,:-1]
            else:
                Y2 = lstm2.forward(X)

            E, dEdY2 = costFn(Y2, T)
            if multiOutput:
                dEdX2 = lstm2.backward(np.concatenate((dEdY2, np.zeros((N, 1, D2))), axis=1))
            else:
                dEdX2 = lstm2.backward(dEdY2)

            # print i, 'Y', Y/Y2
            # print i, 'dEdY', dEdY/dEdY2

            I2 = lstm.stageDict['I-4']
            F2 = lstm.stageDict['F-4']
            Z2 = lstm.stageDict['Z-4']
            O2 = lstm.stageDict['O-4']
            dEdW = np.concatenate((I.dEdW, F.dEdW, Z.dEdW, O.dEdW), axis=-1)
            dEdW2 = lstm2.dEdW
            lstm.updateWeights()
            lstm2.updateWeights()
            self.chkEqual(Y, Y2)

            #print i, 'haha', dEdX/dEdX2
            self.chkEqual(dEdX, dEdX2)
            # print i, '1', dEdW
            # print i, '2', dEdW2
            # print i, '3', dEdW/dEdW2
            self.chkEqual(dEdW, dEdW2)
            W = np.concatenate((I2.W, F2.W, Z2.W, O2.W), axis=-1)
            W2 = lstm2.W
            # print i, '4', W/W2
            self.chkEqual(W, W2)

    def chkEqual(self, a, b):
        tolerance = 1e-4
        a = a.reshape(a.size)
        b = b.reshape(b.size)
        for i in range(a.size):
            self.assertTrue(
                (a[i] == 0 and b[i] == 0) or
                (np.abs(a[i] / b[i] - 1) < tolerance))

if __name__ == '__main__':
    # suite = unittest.TestSuite()
    # suite.addTests(
    #     unittest.TestLoader().loadTestsFromTestCase(Recurrent_Tests))
    # suite.addTests(
    #     unittest.TestLoader().loadTestsFromTestCase(LSTM_Recurrent_Random_Tests))
    # unittest.TextTestRunner(verbosity=2).run(suite)
    unittest.main()