import numpy as np

def meanSqErr(Y, T):
    diff =  Y - T.reshape(Y.shape)
    N = 1
    timespan = 1
    if len(Y.shape) > 1:
        N = Y.shape[0]
        timespan = 1
    if len(Y.shape) > 2:
        N = Y.shape[0]
        timespan = Y.shape[1]
    E = 0.5 * np.sum(np.power(diff, 2)) / float(timespan) / float(N)
    dEdY = diff / float(timespan) / float(N)
    return E, dEdY

def hardLimit(Y):
    return (Y > 0.5).astype(int)

def sigmoidFn(X):
    return 1 / (1 + np.exp(-X))

def crossEntIdx(Y, T):
    eps = 1e-5
    if len(Y.shape) == 1:
        E = -np.log(Y[T] + eps)
        dEdY = np.zeros(Y.shape)
        dEdY[T] = -1 / (Y[T] + eps)
    elif len(Y.shape) == 2:
        T = T.reshape(T.size)
        N = Y.shape[0]
        E = 0.0
        for n in range(0, N):
            E  += -np.log(Y[n, T[n]] + eps)
        E /= float(N)
        dEdY = np.zeros(Y.shape)
        for n in range(0, N):
            dEdY[n, T[n]] = -1 / (Y[n, T[n]] + eps)
        dEdY /= float(N)
    elif len(Y.shape) == 3:
        N = Y.shape[0]
        timespan = Y.shape[1]
        T = T.reshape(T.shape[0], T.shape[1])
        E = np.zeros(N)
        for n in range(0, N):
            for t in range(0, timespan):
                E[n] += -np.log(Y[n, t, T[n, t]] + eps)
        E /= float(N) * float(timespan)
        dEdY = np.zeros(Y.shape, float)
        for n in range(0, N):
            for t in range(0, timespan):
                dEdY[n, t, T[n, t]] += -1 / (Y[n, t, T[n, t]] + eps)
        dEdY /= float(N) * float(timespan)
    return E, dEdY

def crossEntOne(Y, T):
    eps = 1e-5
    T = T.reshape(Y.shape)
    cost = -T * np.log(Y + eps) - (1 - T) * np.log(1 - Y + eps)
    dcost = -T / (Y + eps) + (1 - T) / (1 - Y + eps)
    if len(Y.shape) == 0:
        E = cost
        dEdY = dcost
    else:
        E = np.sum(cost) / float(Y.size)
        dEdY = dcost / float(Y.size)

    return E, dEdY

def argmax(Y):
    return np.argmax(Y, axis=-1)

def simpleSum(Y):
    return np.sum(Y, axis=-1)

def simpleSumDeriv(T, Y):
    Yout = simpleSum(Y)
    diff =  Yout - T.reshape(Yout.shape)
    timespan = Y.shape[-2]
    E = 0.5 * np.sum(np.power(diff, 2), axis=0) / float(timespan)

    if len(Y.shape) == 2:
        dEdY = np.repeat(diff.reshape((diff.shape[0], 1)), Y.shape[1], axis=1) / float(timespan)
    elif len(Y.shape) == 3:
        dEdY = np.repeat(diff.reshape((diff.shape[0], diff.shape[1], 1)), Y.shape[2], axis=2) / float(timespan)
    return E, dEdY

def simpleSumDecision(Y):
    return (simpleSum(Y) > 0.5).astype(int)
