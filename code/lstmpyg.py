import os
os.environ['GNUMPY_USE_GPU'] = 'yes'
import gnumpy as gnp
import numpy as np

def sigmoidFn(X):
    return 1 / (1 + np.exp(-X))

def sliceWeights(
                inputDim,
                outputDim,
                W):
    s1 = inputDim + outputDim * 2 + 1
    s2 = s1 * 2
    s3 = s2 + inputDim + outputDim + 1
    s4 = s3 + s1
    Wi = W[:, 0 : s1]
    Wf = W[:, s1 : s2]
    Wc = W[:, s2 : s3]
    Wo = W[:, s3 : s4]

    return Wi, Wf, Wc, Wo

def sliceWeightsSmall(
                    inputDim,
                    outputDim,
                    W):
    Wi, Wf, Wc, Wo = sliceWeights(inputDim, outputDim, W)

    Wxi = Wi[:, 0 : inputDim]
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
                X,
                cutOffZeroEnd,
                W):
    numEx = X.shape[0]
    timespan = X.shape[1]
    inputDim = X.shape[2]
    outputDim = W.shape[0]
    Wi, Wf, Wc, Wo = sliceWeights(inputDim, outputDim, W)
    Wig = gnp.as_garray(Wi.transpose())
    Wfg = gnp.as_garray(Wf.transpose())
    Wcg = gnp.as_garray(Wc.transpose())
    Wog = gnp.as_garray(Wo.transpose())
    Xend = np.zeros(numEx)
    Gi = np.zeros((numEx,timespan,outputDim))
    Gf = np.zeros((numEx,timespan,outputDim))
    Go = np.zeros((numEx,timespan,outputDim))
    Z = np.zeros((numEx,timespan,outputDim))
    C = np.zeros((numEx,timespan,outputDim))
    myShape = (numEx,timespan,outputDim)
    if cutOffZeroEnd:
        Y = np.zeros((numEx, timespan + 1, outputDim),)
        reachedEnd = np.sum(X, axis=-1) == 0.0
    else:
        Y = np.zeros(myShape,)
        reachedEnd = np.zeros((numEx, timespan))
    for t in range(0, timespan):
            states1 = np.concatenate((X[:,t], \
                                      Y[:,t-1], \
                                      C[:,t-1], \
                                      np.ones((numEx, 1))), axis=-1)
            states2 = np.concatenate((X[:,t], \
                                      Y[:,t-1], \
                                      np.ones((numEx, 1))), axis=-1)
            res = gnp.dot(gnp.as_garray(states1), Wig)
            res2 = res.as_numpy_array()
            Gi[:,t] = sigmoidFn(res2)
            res = gnp.dot(gnp.as_garray(states1), Wfg)
            res2 = res.as_numpy_array()
            Gf[:,t] = sigmoidFn(res2)

            Zt = gnp.tanh(gnp.dot(gnp.as_garray(states2), Wcg))
            Z[:,t] = gnp.as_numpy_array(Zt)
            C[:,t] = Gf[:,t] * C[:,t-1] + Gi[:,t] * Z[:,t]
            states3 = np.concatenate((X[:,t], \
                                      Y[:,t-1], \
                                      C[:,t], \
                                      np.ones((numEx, 1))), axis=-1)
            res = gnp.dot(gnp.as_garray(states3), Wog)
            res2 = res.as_numpy_array()
            Go[:,t] = sigmoidFn(res2)
            Y[:,t] = Go[:,t] * np.tanh(C[:,t])
    if cutOffZeroEnd:
        for n in range(0, numEx):
            for t in range(0, timespan):
                if reachedEnd[n, t]:
                    Y[n, -1] = Y[n, t - 1]
                    break
    return Y, C, Z, Gi, Gf, Go, Xend

def backPropagateN(
                   dEdY,
                   X,
                   Y,
                   C,
                   Z,
                   Gi,
                   Gf,
                   Go,
                   Xend,
                   cutOffZeroEnd,
                   multiErr,
                   outputdEdX,
                   W):
    numEx = X.shape[0]
    inputDim = X.shape[2]
    outputDim = Y.shape[2]
    Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,Wyc,Wxo,Wyo,Wco = sliceWeightsSmall(inputDim, outputDim, W)
    dEdW = np.zeros((W.shape[0], W.shape[1]))
    dEdX = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
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
                    dEdY,
                    X,
                    Y,
                    C,
                    Z,
                    Gi,
                    Gf,
                    Go,
                    Xend,
                    cutOffZeroEnd,
                    multiErr,
                    outputdEdX,
                    Wxi,
                    Wyi,
                    Wci,
                    Wxf,
                    Wyf,
                    Wcf,
                    Wxc,
                    Wyc,
                    Wxo,
                    Wyo,
                    Wco,
                   Wshape):
    Xend = int(Xend)
    if cutOffZeroEnd and multiErr:
        dEdY[Xend - 1] += dEdY[-1]
    inputDim = X.shape[1]
    outputDim = Y.shape[1]
    dEdW = np.zeros(Wshape)
    # dEdWi,dEdWf,dEdWc,dEdWo = sliceWeights(inputDim, outputDim, dEdW)
    # dEdWi = gnp.zeros(dEdWi.shape)
    # dEdWf = gnp.zeros(dEdWf.shape)
    # dEdWc = gnp.zeros(dEdWc.shape)
    # dEdWo = gnp.zeros(dEdWo.shape)
    ddim = (outputDim, Xend)

    # (j, t)
    dEdGi = np.zeros(ddim)
    dEdGf = np.zeros(ddim)
    dEdZ = np.zeros(ddim)
    dEdGo = np.zeros(ddim)

    # (t, k)
    states1T = np.zeros((Xend,
               inputDim + 2 * outputDim + 1))
    states2T = np.zeros((Xend,
               inputDim + outputDim + 1))
    states3T = np.zeros((Xend,
               inputDim + 2 * outputDim + 1))

    dEdX = np.zeros((X.shape[0], X.shape[1]))

    memEye = np.eye(outputDim)
    memCol = (outputDim, 1)

    for t in reversed(range(0, int(Xend))):
        if t == 0:
            Yt1 = np.zeros(outputDim)
            Ct1 = np.zeros(outputDim)
        else:
            Yt1 = Y[t-1]
            Ct1 = C[t-1]

        states1T[t] = \
            np.concatenate((X[t], Yt1, Ct1, np.ones(1)))
        states2T[t] = \
            np.concatenate((X[t], Yt1, np.ones(1)))
        states3T[t] = \
            np.concatenate((X[t], Yt1, C[t], np.ones(1)))

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
        dYtdCt = gnp.as_garray((Go[t] * dU) * memEye+ \
                 dYtdGo.reshape(memCol) * Wco)

        dEdYnow = gnp.as_garray(dEdY[t]) if multiErr else 0
        # (TT, t)
        if t < Xend - 1:
            dEdYt = gnp.dot(dEdYt, dYtdYt1) + gnp.dot(dEdCt, dCtdYt1) + dEdYnow
            dEdCt = gnp.dot(dEdCt, dCtdCt1) + gnp.dot(dEdYt, dYtdCt)
        else:
            dEdYt = dEdYnow if multiErr else dEdY
            dEdCt = gnp.dot(dEdYt, dYtdCt)

        dEdGi[:, t] = dEdCt * dCtdGi
        dEdGf[:, t] = dEdCt * dCtdGf
        dEdZ[:, t] = dEdCt * dCtdZ
        dEdGo[:, t] = dEdYt * dYtdGo

        # (k -> t, l -> t-1)
        dCtdCt1 = gnp.as_garray(dCtdGf.reshape(memCol) * Wcf + \
                  Gf[t] * memEye + \
                  dCtdGi.reshape(memCol) * Wci)
        dCtdYt1 = gnp.as_garray(dCtdGf.reshape(memCol) * Wyf + \
                  dCtdZ.reshape(memCol) * Wyc + \
                  dCtdGi.reshape(memCol) * Wyi)
        dYtdYt1 = gnp.as_garray(dYtdGo.reshape(memCol) * Wyo)

    dEdWi = gnp.dot(gnp.as_garray(dEdGi), gnp.as_garray(states1T)).as_numpy_array()
    dEdWf = gnp.dot(gnp.as_garray(dEdGf), gnp.as_garray(states1T)).as_numpy_array()
    dEdWc = gnp.dot(gnp.as_garray(dEdZ), gnp.as_garray(states2T)).as_numpy_array()
    dEdWo = gnp.dot(gnp.as_garray(dEdGo), gnp.as_garray(states3T)).as_numpy_array()

    if outputdEdX:
        dEdX[0:Xend] = (gnp.dot(gnp.as_garray(dEdGi.transpose()), gnp.as_garray(Wxi)) + \
                       gnp.dot(gnp.as_garray(dEdGf.transpose()), gnp.as_garray(Wxf)) + \
                       gnp.dot(gnp.as_garray(dEdZ.transpose()), gnp.as_garray(Wxc)) + \
                       gnp.dot(gnp.as_garray(dEdGo.transpose()), gnp.as_garray(Wxo))).as_numpy_array()

    dEdW = np.concatenate((dEdWi, dEdWf, dEdWc, dEdWo), axis=-1)
    return dEdW, dEdX

# def backPropagateN(
#                    dEdY,
#                    X,
#                    Y,
#                    C,
#                    Z,
#                    Gi,
#                    Gf,
#                    Go,
#                    Xend,
#                    cutOffZeroEnd,
#                    multiErr,
#                    outputdEdX,
#                    W):
#     numEx = X.shape[0]
#     inputDim = X.shape[2]
#     outputDim = Y.shape[2]
#     Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,Wyc,Wxo,Wyo,Wco = sliceWeightsSmall(inputDim, outputDim, W)
#     Wxi = gnp.as_garray(Wxi)
#     Wyi = gnp.as_garray(Wyi)
#     Wci = gnp.as_garray(Wci)
#     Wxf = gnp.as_garray(Wxf)
#     Wyf = gnp.as_garray(Wyf)
#     Wcf = gnp.as_garray(Wcf)
#     Wxc = gnp.as_garray(Wxc)
#     Wyc = gnp.as_garray(Wyc)
#     Wxo = gnp.as_garray(Wxo)
#     Wyo = gnp.as_garray(Wyo)
#     Wco = gnp.as_garray(Wco)
#     dEdYg = gnp.as_garray(dEdY)
#     Xg = gnp.as_garray(X)
#     Yg = gnp.as_garray(Y)
#     Cg = gnp.as_garray(C)
#     Zg = gnp.as_garray(Z)
#     Gig = gnp.as_garray(Gi)
#     Gfg = gnp.as_garray(Gf)
#     Gog = gnp.as_garray(Go)
#     dEdW = gnp.zeros((W.shape[0], W.shape[1]))
#     dEdX = gnp.zeros((X.shape[0], X.shape[1], X.shape[2]))
#     for n in range(0, numEx):
#         dEdWtmp, dEdX[n] = \
#             backPropagateOne(dEdYg[n],Xg[n],Yg[n],
#                         Cg[n],Zg[n],Gig[n],
#                         Gfg[n],Gog[n],
#                         Xend[n],cutOffZeroEnd,
#                         multiErr,outputdEdX,
#                         Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,
#                         Wyc,Wxo,Wyo,Wco,(W.shape[0], W.shape[1]))
#         dEdW += dEdWtmp
#     return dEdW.as_numpy_array(), dEdX.as_numpy_array()
#
# def backPropagateOne(
#                     dEdY,
#                     X,
#                     Y,
#                     C,
#                     Z,
#                     Gi,
#                     Gf,
#                     Go,
#                     Xend,
#                     cutOffZeroEnd,
#                     multiErr,
#                     outputdEdX,
#                     Wxi,
#                     Wyi,
#                     Wci,
#                     Wxf,
#                     Wyf,
#                     Wcf,
#                     Wxc,
#                     Wyc,
#                     Wxo,
#                     Wyo,
#                     Wco,
#                    Wshape):
#     Xend = int(Xend)
#     if cutOffZeroEnd and multiErr:
#         dEdY[Xend - 1] += dEdY[-1]
#     inputDim = X.shape[1]
#     outputDim = Y.shape[1]
#     #dEdW = gnp.zeros(Wshape)
#     #dEdWi,dEdWf,dEdWc,dEdWo = sliceWeights(inputDim, outputDim, dEdW)
#     dEdWi = 0
#     dEdWf = 0
#     dEdWc = 0
#     dEdWo = 0
#     ddim = (outputDim, Xend)
#
#     # (j, t)
#     dEdGi = gnp.zeros(ddim)
#     dEdGf = gnp.zeros(ddim)
#     dEdZ = gnp.zeros(ddim)
#     dEdGo = gnp.zeros(ddim)
#
#     # (t, k)
#     states1T = gnp.zeros((Xend,
#                inputDim + 2 * outputDim + 1))
#     states2T = gnp.zeros((Xend,
#                inputDim + outputDim + 1))
#     states3T = gnp.zeros((Xend,
#                inputDim + 2 * outputDim + 1))
#
#     dEdX = gnp.zeros((X.shape[0], X.shape[1]))
#
#     memEye = gnp.eye(outputDim)
#     memCol = (outputDim, 1)
#
#     for t in reversed(range(0, int(Xend))):
#         if t == 0:
#             Yt1 = gnp.zeros(outputDim)
#             Ct1 = gnp.zeros(outputDim)
#         else:
#             Yt1 = Y[t-1]
#             Ct1 = C[t-1]
#
#         states1T[t] = \
#             gnp.concatenate((X[t], Yt1, Ct1, gnp.ones(1)))
#         states2T[t] = \
#             gnp.concatenate((X[t], Yt1, gnp.ones(1)))
#         states3T[t] = \
#             gnp.concatenate((X[t], Yt1, C[t], gnp.ones(1)))
#
#         # (k -> t)
#         U = gnp.tanh(C[t])
#         dU = 1 - gnp.power(U, 2)
#         dZ = 1 - gnp.power(Z[t], 2)
#
#         dGi = Gi[t] * (1 - Gi[t])
#         dGf = Gf[t] * (1 - Gf[t])
#         dGo = Go[t] * (1 - Go[t])
#         dCtdGi = Z[t] * dGi
#         dCtdGf = Ct1 * dGf
#         dCtdZ = Gi[t] * dZ
#         dYtdGo = U * dGo
#
#         # (k, l)
#         dYtdCt = (Go[t] * dU) * memEye+ \
#                  dYtdGo.reshape(memCol) * Wco
#
#         dEdYnow = dEdY[t] if multiErr else 0
#         # (TT, t)
#         if t < Xend - 1:
#             dEdYt = gnp.dot(dEdYt, dYtdYt1) + gnp.dot(dEdCt, dCtdYt1) + dEdYnow
#             dEdCt = gnp.dot(dEdCt, dCtdCt1) + gnp.dot(dEdYt, dYtdCt)
#         else:
#             dEdYt = dEdYnow if multiErr else dEdY
#             dEdCt = gnp.dot(dEdYt, dYtdCt)
#
#         dEdGi[:, t] = dEdCt * dCtdGi
#         dEdGf[:, t] = dEdCt * dCtdGf
#         dEdZ[:, t] = dEdCt * dCtdZ
#         dEdGo[:, t] = dEdYt * dYtdGo
#
#         # (k -> t, l -> t-1)
#         dCtdCt1 = dCtdGf.reshape(memCol) * Wcf + \
#                   Gf[t] * memEye + \
#                   dCtdGi.reshape(memCol) * Wci
#         dCtdYt1 = dCtdGf.reshape(memCol) * Wyf + \
#                   dCtdZ.reshape(memCol) * Wyc + \
#                   dCtdGi.reshape(memCol) * Wyi
#         dYtdYt1 = dYtdGo.reshape(memCol) * Wyo
#
#     dEdWi += gnp.dot(dEdGi, states1T)
#     dEdWf += gnp.dot(dEdGf, states1T)
#     dEdWc += gnp.dot(dEdZ, states2T)
#     dEdWo += gnp.dot(dEdGo, states3T)
#
#     if outputdEdX:
#         dEdX[0:Xend] = gnp.dot(dEdGi.transpose(), Wxi) + \
#                        gnp.dot(dEdGf.transpose(), Wxf) + \
#                        gnp.dot(dEdZ.transpose(), Wxc) + \
#                        gnp.dot(dEdGo.transpose(), Wxo)
#
#     dEdW = gnp.concatenate((dEdWi, dEdWf, dEdWc, dEdWo), axis=-1)
#     return dEdW, dEdX
