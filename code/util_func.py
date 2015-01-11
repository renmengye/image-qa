import numpy as np

def meanSqErr(Y, T):
    diff =  Y - T.reshape(Y.shape)
    timespan = Y.shape[0]
    E = 0.5 * np.sum(np.power(diff, 2), axis=0) / float(timespan)
    dEdY = diff / float(timespan)
    return E, dEdY

def hardLimit(Y):
    return (Y > 0.5).astype(int)

def simpleSum(Y):
    return np.sum(Y, axis=-1)

def simpleSumDeriv(T, Y):
    Yout = simpleSum(Y)
    diff =  Yout - T.reshape(Yout.shape)
    timespan = Y.shape[0]
    E = 0.5 * np.sum(np.power(diff, 2), axis=0) / float(timespan)

    if len(Y.shape) == 2:
        dEdY = np.repeat(diff.reshape(diff.shape[0], 1), Y.shape[1], axis=1) / float(timespan)
    elif len(Y.shape) == 3:
        dEdY = np.repeat(diff.reshape(diff.shape[0], diff.shape[1], 1), Y.shape[2], axis=2) / float(timespan)
    return E, dEdY

def simpleSumDecision(Y):
    return (simpleSum(Y) > 0.5).astype(int)
