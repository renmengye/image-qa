import gnumpy as np
cimport numpy as np

FLOAT = np.float
ctypedef np.float_t FLOAT_t
import cython
@cython.boundscheck(False)
def sigmoidFn(X):
    Y = 1 / (1 + np.exp(-X))
    return Y

def sliceWeights(int inputDim, 
                int outputDim, 
                np.ndarray[FLOAT_t, ndim=2] W):
    cdef int s1 = inputDim + outputDim * 2 + 1
    cdef int s2 = s1 * 2
    cdef int s3 = s2 + inputDim + outputDim + 1
    cdef int s4 = s3 + s1
    Wi = W[:, 0 : s1]
    Wf = W[:, s1 : s2]
    Wc = W[:, s2 : s3]
    Wo = W[:, s3 : s4]
    
    return Wi, Wf, Wc, Wo

def sliceWeightsSmall(int inputDim, 
                    int outputDim, 
                    np.ndarray[FLOAT_t, ndim=2] W):
    Wi, Wf, Wc, Wo = sliceWeights(inputDim, outputDim, W)

    Wxi = Wi[:, 0 : outputDim]
    Wyi = Wi[:, inputDim : inputDim + outputDim]
    Wci = Wi[:, inputDim + outputDim : inputDim + outputDim + outputDim]
    Wxf = Wf[:, 0 : inputDim]
    Wyf = Wf[:, inputDim : inputDim + outputDim]
    Wcf = Wf[:, inputDim + outputDim : inputDim + outputDim + outputDim]
    Wxc = Wc[:, 0 : inputDim]
    Wyc = Wc[:, inputDim : inputDim + outputDim]
    Wxo = Wo[:, 0 : inputDim]
    Wyo = Wo[:, inputDim : inputDim + outputDim]
    Wco = Wo[:, inputDim + outputDim : inputDim + outputDim + outputDim]

    return Wxi, Wyi, Wci, Wxf, Wyf, Wcf, Wxc, Wyc, Wxo, Wyo, Wco

def forwardPassN(
                np.ndarray[FLOAT_t, ndim=3] X,
                cutOffZeroEnd,
                np.ndarray[FLOAT_t, ndim=2] W):
    cdef int numEx = X.shape[0]
    cdef int timespan = X.shape[1]
    cdef int inputDim = X.shape[2]
    cdef int outputDim = W.shape[0]
    Wi, Wf, Wc, Wo = sliceWeights(inputDim, outputDim, W)
    cdef np.ndarray Xend = np.zeros(numEx, dtype=int)
    cdef np.ndarray Y
    cdef np.ndarray reachedEnd
    cdef np.ndarray Gi = np.zeros((numEx,timespan,outputDim), dtype=FLOAT)
    cdef np.ndarray Gf = np.zeros((numEx,timespan,outputDim), dtype=FLOAT)
    cdef np.ndarray Go = np.zeros((numEx,timespan,outputDim), dtype=FLOAT)
    cdef np.ndarray Z = np.zeros((numEx,timespan,outputDim), dtype=FLOAT)
    cdef np.ndarray C = np.zeros((numEx,timespan,outputDim), dtype=FLOAT)
    myShape = (numEx,timespan,outputDim)
    if cutOffZeroEnd:
        Y = np.zeros((numEx, timespan + 1, outputDim),
                          dtype=FLOAT)
        reachedEnd = np.sum(X, axis=-1) == 0.0
    else:
        Y = np.zeros(myShape, dtype=FLOAT)
        reachedEnd = np.zeros((numEx, timespan), dtype=FLOAT)
    
    for n in range(0, numEx):
        Y[n], C[n], Z[n], \
        Gi[n], Gf[n], Go[n], \
        Xend[n] = \
            forwardPassOne(
                X[n], reachedEnd[n], cutOffZeroEnd, Wi, Wf, Wc, Wo)

    return Y, C, Z, Gi, Gf, Go, Xend
    
def forwardPassOne(
                np.ndarray[FLOAT_t, ndim=2] X, 
                np.ndarray reachedEnd,
                cutOffZeroEnd,
                np.ndarray[FLOAT_t, ndim=2] Wi, 
                np.ndarray[FLOAT_t, ndim=2] Wf, 
                np.ndarray[FLOAT_t, ndim=2] Wc, 
                np.ndarray[FLOAT_t, ndim=2] Wo):
    timespan = X.shape[0]
    outputDim = Wi.shape[0]
    # Last time step is reserved for final output of the entire input.
    cdef np.ndarray Y
    if cutOffZeroEnd:
        Y = np.zeros((timespan + 1, outputDim), dtype=FLOAT)
    else:
        Y = np.zeros((timespan, outputDim), dtype=FLOAT)
    cdef np.ndarray C = np.zeros((timespan, outputDim), dtype=FLOAT)
    cdef np.ndarray Z = np.zeros((timespan, outputDim), dtype=FLOAT)
    cdef np.ndarray Gi = np.zeros((timespan, outputDim), dtype=FLOAT)
    cdef np.ndarray Gf = np.zeros((timespan, outputDim), dtype=FLOAT)
    cdef np.ndarray Go = np.zeros((timespan, outputDim), dtype=FLOAT)
    cdef int Xend = timespan
    cdef np.ndarray states1
    cdef np.ndarray states2

    for t in range(0, timespan):
        if cutOffZeroEnd and reachedEnd[t]:
            Xend = t
            Y[-1, :] = Y[t - 1, :]
            break

        states1 = np.concatenate((X[t, :], \
                                  Y[t-1, :], \
                                  C[t-1, :], \
                                  np.ones(1, dtype=FLOAT)))
        states2 = np.concatenate((X[t, :], \
                                  Y[t-1, :], \
                                  np.ones(1, dtype=FLOAT)))
        Gi[t, :] = sigmoidFn(np.dot(Wi, states1))
        Gf[t, :] = sigmoidFn(np.dot(Wf, states1))
        Z[t, :] = np.tanh(np.dot(Wc, states2))
        C[t, :] = Gf[t, :] * C[t-1, :] + Gi[t, :] * Z[t, :]
        states3 = np.concatenate((X[t, :], \
                                  Y[t-1, :], \
                                  C[t, :], \
                                  np.ones(1, dtype=FLOAT)))
        Go[t, :] = sigmoidFn(np.dot(Wo, states3))
        Y[t, :] = Go[t, :] * np.tanh(C[t, :])

    return Y, C, Z, Gi, Gf, Go, Xend
    
def backPropagateN(
                   np.ndarray dEdY, 
                   np.ndarray[FLOAT_t, ndim=3] X, 
                   np.ndarray[FLOAT_t, ndim=3] Y, 
                   np.ndarray[FLOAT_t, ndim=3] C, 
                   np.ndarray[FLOAT_t, ndim=3] Z, 
                   np.ndarray[FLOAT_t, ndim=3] Gi, 
                   np.ndarray[FLOAT_t, ndim=3] Gf, 
                   np.ndarray[FLOAT_t, ndim=3] Go, 
                   Xend, 
                   cutOffZeroEnd,
                   multiErr,
                   outputdEdX,
                   np.ndarray[FLOAT_t, ndim=2] W):
    cdef int numEx = X.shape[0]
    cdef int inputDim = X.shape[2]
    cdef int outputDim = Y.shape[2]
    Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,Wyc,Wxo,Wyo,Wco = sliceWeightsSmall(inputDim, outputDim, W)
    cdef np.ndarray dEdW = np.zeros((W.shape[0], W.shape[1]), dtype=FLOAT)
    cdef np.ndarray dEdX = np.zeros((X.shape[0], X.shape[1], X.shape[2]), dtype=FLOAT)
    cdef np.ndarray dEdWtmp
    for n in range(0, numEx):
        dEdWtmp, dEdX[n] = \
            backPropagateOne(dEdY[n],X[n],Y[n],
                        C[n],Z[n],Gi[n],
                        Gf[n],Go[n],
                        Xend[n],cutOffZeroEnd,
                        multiErr,outputdEdX,
                        Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,
                        Wyc,Wxo,Wyo,Wco,(W.shape[0], W.shape[1]))
        dEdW += dEdWtmp
    return dEdW, dEdX

def backPropagateOne(
                   np.ndarray dEdY, 
                   np.ndarray[FLOAT_t, ndim=2] X, 
                   np.ndarray[FLOAT_t, ndim=2] Y, 
                   np.ndarray[FLOAT_t, ndim=2] C, 
                   np.ndarray[FLOAT_t, ndim=2] Z, 
                   np.ndarray[FLOAT_t, ndim=2] Gi, 
                   np.ndarray[FLOAT_t, ndim=2] Gf, 
                   np.ndarray[FLOAT_t, ndim=2] Go, 
                   Xend, 
                   cutOffZeroEnd,
                   multiErr,
                   outputdEdX,
                   np.ndarray[FLOAT_t, ndim=2] Wxi, 
                   np.ndarray[FLOAT_t, ndim=2] Wyi, 
                   np.ndarray[FLOAT_t, ndim=2] Wci, 
                   np.ndarray[FLOAT_t, ndim=2] Wxf, 
                   np.ndarray[FLOAT_t, ndim=2] Wyf, 
                   np.ndarray[FLOAT_t, ndim=2] Wcf, 
                   np.ndarray[FLOAT_t, ndim=2] Wxc, 
                   np.ndarray[FLOAT_t, ndim=2] Wyc, 
                   np.ndarray[FLOAT_t, ndim=2] Wxo, 
                   np.ndarray[FLOAT_t, ndim=2] Wyo, 
                   np.ndarray[FLOAT_t, ndim=2] Wco, 
                   Wshape):
    if cutOffZeroEnd and multiErr:
        dEdY[Xend - 1] += dEdY[-1]
    cdef int inputDim = X.shape[1]
    cdef int outputDim = Y.shape[1]
    cdef np.ndarray dEdW = np.zeros(Wshape, dtype=FLOAT)
    dEdWi,dEdWf,dEdWc,dEdWo = sliceWeights(inputDim, outputDim, dEdW)
    cdef ddim = (outputDim, Xend)

    # (j, t)
    cdef np.ndarray dEdGi = np.zeros(ddim, dtype=FLOAT)
    cdef np.ndarray dEdGf = np.zeros(ddim, dtype=FLOAT)
    cdef np.ndarray dEdZ = np.zeros(ddim, dtype=FLOAT)
    cdef np.ndarray dEdGo = np.zeros(ddim, dtype=FLOAT)

    # (t, k)
    cdef np.ndarray states1T = np.zeros((Xend,
               inputDim + 2 * outputDim + 1), dtype=FLOAT)
    cdef np.ndarray states2T = np.zeros((Xend,
               inputDim + outputDim + 1), dtype=FLOAT)
    cdef np.ndarray states3T = np.zeros((Xend,
               inputDim + 2 * outputDim + 1), dtype=FLOAT)

    cdef np.ndarray dEdX = np.zeros((X.shape[0], X.shape[1]), dtype=FLOAT)

    cdef np.ndarray memEye = np.eye(outputDim)
    cdef memCol = (outputDim, 1)
    cdef np.ndarray Yt1
    cdef np.ndarray Ct1
    
    for t in reversed(range(0, Xend)):
        if t == 0:
            Yt1 = np.zeros(outputDim, dtype=FLOAT)
            Ct1 = np.zeros(outputDim, dtype=FLOAT)
        else:
            Yt1 = Y[t-1]
            Ct1 = C[t-1]

        states1T[t] = \
            np.concatenate((X[t], Yt1, Ct1, np.ones(1, dtype=FLOAT)))
        states2T[t] = \
            np.concatenate((X[t], Yt1, np.ones(1, dtype=FLOAT)))
        states3T[t] = \
            np.concatenate((X[t], Yt1, C[t], np.ones(1, dtype=FLOAT)))

        # (k -> t)
        U = np.tanh(C[t])
        dU = 1 - np.power(U, 2)
        dZ = 1 - np.power(Z[t], 2)

        dGi = Gi[t] * (1 - Gi[t])
        dGf = Gf[t] * (1 - Gf[t])
        dGo = Go[t] * (1 - Go[t])
        dCtdGi = Z[t] * dGi
        dCtdGf = Ct1 * dGf
        dCtdZ = Gi[t] * dZ
        dYtdGo = U * dGo

        # (k, l)
        dYtdCt = (Go[t] * dU) * memEye+ \
                 dYtdGo.reshape(memCol) * Wco

        dEdYnow = dEdY[t] if multiErr else 0
        # (T, t)
        if t < Xend - 1:
            dEdYt = np.dot(dEdYt, dYtdYt1) + np.dot(dEdCt, dCtdYt1) + dEdYnow
            dEdCt = np.dot(dEdCt, dCtdCt1) + np.dot(dEdYt, dYtdCt)
        else:
            dEdYt = dEdYnow if multiErr else dEdY
            dEdCt = np.dot(dEdYt, dYtdCt)

        dEdGi[:, t] = dEdCt * dCtdGi
        dEdGf[:, t] = dEdCt * dCtdGf
        dEdZ[:, t] = dEdCt * dCtdZ
        dEdGo[:, t] = dEdYt * dYtdGo

        # (k -> t, l -> t-1)
        dCtdCt1 = dCtdGf.reshape(memCol) * Wcf + \
                  Gf[t] * memEye + \
                  dCtdGi.reshape(memCol) * Wci
        dCtdYt1 = dCtdGf.reshape(memCol) * Wyf + \
                  dCtdZ.reshape(memCol) * Wyc + \
                  dCtdGi.reshape(memCol) * Wyi
        dYtdYt1 = dYtdGo.reshape(memCol) * Wyo

    dEdWi += np.dot(dEdGi, states1T)
    dEdWf += np.dot(dEdGf, states1T)
    dEdWc += np.dot(dEdZ, states2T)
    dEdWo += np.dot(dEdGo, states3T)

    if outputdEdX:
        dEdX[0:Xend] = np.dot(dEdGi.transpose(), Wxi) + \
                       np.dot(dEdGf.transpose(), Wxf) + \
                       np.dot(dEdZ.transpose(), Wxc) + \
                       np.dot(dEdGo.transpose(), Wxo)

    return dEdW, dEdX
