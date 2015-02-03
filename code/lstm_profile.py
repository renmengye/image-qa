import lstm as l
import lstmg as lg
import numpy as np
import sys
import time

start = time.time()
timespan = 100
multiErr = len(sys.argv) > 1 and sys.argv[1] == 'm'
for i in range(0, 10):
    lstm = l.LSTM(
        inputDim=100,
        outputDim=100,
        initRange=.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=multiErr,
        outputdEdX=True)
    X = np.random.rand(10, timespan, 100)
    Y = lstm.forwardPass(X)
    dEdY = np.random.rand(10, timespan, 100) if multiErr else np.random.rand(10, 100)
    dEdY = lstm.backPropagate(dEdY)
print '%.4f s' % (time.time() - start)


start = time.time()
for i in range(0, 10):
    lstm = lg.LSTM(
        inputDim=100,
        outputDim=100,
        initRange=.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=multiErr,
        outputdEdX=True)
    X = np.random.rand(10, timespan, 100)
    Y = lstm.forwardPass(X)
    dEdY = np.random.rand(10, timespan, 100) if multiErr else np.random.rand(10, 100)
    dEdY = lstm.backPropagate(dEdY)
print '%.4f s' % (time.time() - start)
