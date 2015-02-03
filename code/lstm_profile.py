import lstm as l
import lstmg as lg
import numpy as np
import sys
import time

#start = time.time()
#for i in range(0, 10):
#    multiErr = len(sys.argv) > 1 and sys.argv[1] == 'm'
#    lstm = l.LSTM(
#        inputDim=100,
#        outputDim=100,
#        initRange=.1,
#        initSeed=3,
#        cutOffZeroEnd=True,
#        multiErr=multiErr,
#        outputdEdX=True)
#    X = np.random.rand(10, 100, 100)
#    Y = lstm.forwardPass(X)
#    dEdY = np.random.rand(10, 100, 100) if multiErr else np.random.rand(10, 100)
#    dEdY = lstm.backPropagate(dEdY)
#
#print '%.4f ms' % (time.time() - start)


start = time.time()
for i in range(0, 10):
    multiErr = len(sys.argv) > 1 and sys.argv[1] == 'm'
    lstm = lg.LSTM(
        inputDim=100,
        outputDim=100,
        initRange=.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=multiErr,
        outputdEdX=True)
    X = np.random.rand(10, 100, 100)
    Y = lstm.forwardPass(X)
    dEdY = np.random.rand(10, 100, 100) if multiErr else np.random.rand(10, 100)
    dEdY = lstm.backPropagate(dEdY)
print '%.4f ms' % (time.time() - start)
