import numpy as np

def test(model, X, numExPerBat=100):
    N = X.shape[0]
    batchStart = 0
    Y = None
    while batchStart < N:
        # Batch info
        batchEnd = min(N, batchStart + numExPerBat)
        Ytmp = model.forward(X[batchStart:batchEnd], dropout=False)
        if Y is None:
            Yshape = np.copy(Ytmp.shape)
            Yshape[0] = N
            Y = np.zeros(Yshape)
        Y[batchStart:batchEnd] = Ytmp
        batchStart += numExPerBat
    return Y

def calcRate(model, Y, T):
    Yfinal = model.predict(Y)
    correct = np.sum(Yfinal.reshape(Yfinal.size) == T.reshape(T.size))
    total = Yfinal.size
    rate = correct / float(total)
    return rate, correct, total