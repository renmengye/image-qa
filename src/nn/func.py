import numpy as np

def meanSqErr(Y, T):
    diff =  Y - T.reshape(Y.shape)
    E = 0.5 * np.sum(np.power(diff, 2)) / float(Y.shape[0])
    dEdY = diff / float(Y.shape[0])
    return E, dEdY

def hardLimit(Y):
    return (Y > 0.5).astype(int)

def sigmoidFn(X):
    return 1 / (1 + np.exp(-X))

def crossEntIdx(Y, T):
    eps = 1e-8
    Y2 = Y.reshape(Y.size / Y.shape[-1], Y.shape[-1])
    T2 = T.reshape(T.size)
    E = 0.0
    dEdY = np.zeros(Y2.shape, float)
    for n in range(0, Y2.shape[0]):
        E += -np.log(Y2[n, T2[n]] + eps)
        dEdY[n, T2[n]] = -1 / (Y2[n, T2[n]] + eps)
    E /= Y2.shape[0]
    dEdY /= Y2.shape[0]
    dEdY = dEdY.reshape(Y.shape)
    return E, dEdY

def crossEntOne(Y, T):
    eps = 1e-8
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

def roundInt(Y):
    return np.round(Y, axis=-1).astype('int')

def rankingLoss(Y, T):
    alpha = 0.1
    dEdY = np.zeros(Y.shape)
    E = 0.0
    for n in range(T.size):
        cost = Y[n] - Y[n, T[n]] + alpha
        valid = (cost > 0).astype(int)
        nvalid = np.sum(valid) - 1
        cost = cost * valid
        dEdY[n] = valid
        dEdY[n, T[n]] = -nvalid
        E += np.sum(cost) - alpha
    E /= float(T.size)
    dEdY /= float(T.size)
    return E, dEdY