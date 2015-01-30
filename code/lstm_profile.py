from lstm import *
import sys

for i in range(0, 10):
    multiErr = len(sys.argv) > 1 and sys.argv[1] == 'm'
    dx = len(sys.argv) > 2 and sys.argv[2] == 'dx'
    lstm = LSTM(
        inputDim=100,
        outputDim=100,
        initRange=.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=multiErr)
    X = np.random.rand(10, 100, 100)
    Y = lstm.forwardPass(X)
    dEdY = np.random.rand(10, 100, 100) if multiErr else np.random.rand(100, 100)
    dEdY = lstm.backPropagate(dEdY, outputdEdX=dx)
