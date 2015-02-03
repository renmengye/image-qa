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
    Xend = np.zeros(numEx) + timespan
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
                    Y[n, t] = 0.0
                    Xend[n] = t
                    break
    return Y, C, Z, Gi, Gf, Go, Xend

# def backPropagateN(dEdY,X,Y,C,Z,Gi,Gf,Go,Xend,
#                    cutOffZeroEnd,multiErr,outputdEdX,W):
#     numEx = X.shape[0]
#     inputDim = X.shape[2]
#     outputDim = Y.shape[2]
#     Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,Wyc,Wxo,Wyo,Wco = sliceWeightsSmall(inputDim, outputDim, W)
#     dEdW = np.zeros((W.shape[0], W.shape[1]))
#     dEdX = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
#     for n in range(0, numEx):
#         dEdWtmp, dEdX[n] = \
#             backPropagateOne(dEdY[n],X[n],Y[n],
#                         C[n],Z[n],Gi[n],
#                         Gf[n],Go[n],
#                         Xend[n],cutOffZeroEnd,
#                         multiErr,outputdEdX,
#                         Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,
#                         Wyc,Wxo,Wyo,Wco,(W.shape[0], W.shape[1]))
#         dEdW += dEdWtmp
#     return dEdW, dEdX

def backPropagateN(
                   dEdY,X,Y,C,Z,Gi,Gf,Go,
                   Xend,cutOffZeroEnd,multiErr,
                   outputdEdX,W):
    numEx = X.shape[0]
    inputDim = X.shape[2]
    outputDim = Y.shape[2]
    Wg = gnp.as_garray(W)
    Wxig,Wyig,Wcig,Wxfg,Wyfg,Wcfg,Wxcg,Wycg,Wxog,Wyog,Wcog = sliceWeightsSmall(inputDim, outputDim, Wg)
    dEdW = gnp.zeros((W.shape[0], W.shape[1]))
    dEdX = gnp.zeros((X.shape[0], X.shape[1], X.shape[2]))
    dEdWi,dEdWf,dEdWc,dEdWo = sliceWeights(inputDim, outputDim, dEdW)
    Xg = gnp.as_garray(X)
    Yg = gnp.as_garray(Y)
    Cg = gnp.as_garray(C)
    Zg = gnp.as_garray(Z)
    Gig = gnp.as_garray(Gi)
    Gfg = gnp.as_garray(Gf)
    Gog = gnp.as_garray(Go)
    dEdYg = gnp.as_garray(dEdY)

    for n in range(0, numEx):
        dEdWitmp, dEdWftmp, dEdWctmp, dEdWotmp, dEdX[n] = \
            backPropagateOne(dEdYg[n],Xg[n],Yg[n],
                        Cg[n],Zg[n],Gig[n],
                        Gfg[n],Gog[n],
                        Xend[n],cutOffZeroEnd,
                        multiErr,outputdEdX,
                        Wxig,Wyig,Wcig,Wxfg,Wyfg,Wcfg,Wxcg,
                        Wycg,Wxog,Wyog,Wcog)
        dEdWi += dEdWitmp
        dEdWf += dEdWftmp
        dEdWc += dEdWctmp
        dEdWo += dEdWotmp
    dEdW = dEdW.as_numpy_array()
    dEdX = dEdX.as_numpy_array()
    return dEdW, dEdX

def backPropagateOne(
                    dEdY,X,Y,C,Z,Gi,Gf,Go,Xend,cutOffZeroEnd,
                    multiErr,outputdEdX,Wxi,Wyi,Wci,Wxf,Wyf,Wcf,
                    Wxc,Wyc,Wxo,Wyo,Wco):
    Xend = int(Xend)
    if cutOffZeroEnd and multiErr:
        dEdY[Xend - 1] += dEdY[-1]
    inputDim = X.shape[1]
    outputDim = Y.shape[1]
    ddim = (outputDim, Xend)

    # (j, t)
    dEdGig = gnp.zeros(ddim)
    dEdGfg = gnp.zeros(ddim)
    dEdZg = gnp.zeros(ddim)
    dEdGog = gnp.zeros(ddim)
    dEdX = gnp.zeros((X.shape[0], X.shape[1]))
    memEyeT = gnp.eye(outputDim).reshape(1, outputDim, outputDim)

    # (k -> t)
    one = gnp.ones((Xend, 1))
    Yt1g = gnp.concatenate((np.zeros((1, outputDim)), Y[:Xend-1]))
    Ct1g = gnp.concatenate((np.zeros((1, outputDim)), C[:Xend-1]))
    states1T = gnp.concatenate((X[:Xend], Yt1g, Ct1g, one), axis=-1)
    states2T = gnp.concatenate((X[:Xend], Yt1g, one), axis=-1)
    states3T = gnp.concatenate((X[:Xend], Yt1g, C[:Xend], one), axis=-1)

    Cg = C[:Xend]
    Zg = Z[:Xend]
    Gig = Gi[:Xend]
    Gfg = Gf[:Xend]
    Gog = Go[:Xend]
    Ug = gnp.tanh(Cg)
    dU = 1 - Ug * Ug
    dZ = 1 - Zg * Zg
    dGi = Gig * (1 - Gig)
    dGf = Gfg * (1 - Gfg)
    dGo = Gog * (1 - Gog)

    # (j, t)
    dCdGig = (Zg * dGi)
    dCdGfg = (Ct1g * dGf)
    dCdZg = (Gig * dZ)
    dYdGog = (Ug * dGo)

    dYdCg = (Gog * dU).reshape(Xend, outputDim, 1) * memEyeT + \
            dYdGog.reshape(Xend, outputDim, 1) * Wco.reshape(1, Wco.shape[0], Wco.shape[1])
    dCdCg = dCdGfg.reshape(Xend, outputDim, 1) * Wcf.reshape(1, Wcf.shape[0], Wcf.shape[1]) + \
            Gfg.reshape(Xend, outputDim, 1) * memEyeT + \
            dCdGig.reshape(Xend, outputDim, 1) * Wci.reshape(1, Wci.shape[0], Wci.shape[1])
    dCdYg = dCdGfg.reshape(Xend, outputDim, 1) * Wyf.reshape(1, Wyf.shape[0], Wyf.shape[1]) + \
            dCdZg.reshape(Xend, outputDim, 1) * Wyc.reshape(1, Wyc.shape[0], Wyc.shape[1]) + \
            dCdGig.reshape(Xend, outputDim, 1) * Wyi.reshape(1, Wyi.shape[0], Wyi.shape[1])
    dYdYg = dYdGog.reshape(Xend, outputDim, 1) * Wyo.reshape(1, Wyo.shape[0], Wyo.shape[1])

    for t in reversed(range(0, Xend)):
        dEdYnow = dEdY[t] if multiErr else 0
        if t < Xend - 1:
            dEdYt = gnp.dot(dEdYt, dYdYg[t]) + gnp.dot(dEdCt, dCdYg[t]) + dEdYnow
            dEdCt = gnp.dot(dEdCt, dCdCg[t]) + gnp.dot(dEdYt, dYdCg[t])
        else:
            dEdYt = dEdYnow if multiErr else dEdY
            dEdCt = gnp.dot(dEdYt, dYdCg[t])
        dEdGig[:, t] = dEdCt * dCdGig[t]
        dEdGfg[:, t] = dEdCt * dCdGfg[t]
        dEdZg[:, t] = dEdCt * dCdZg[t]
        dEdGog[:, t] = dEdYt * dYdGog[t]

    dEdWi = gnp.dot(dEdGig, states1T)
    dEdWf = gnp.dot(dEdGfg, states1T)
    dEdWc = gnp.dot(dEdZg, states2T)
    dEdWo = gnp.dot(dEdGog, states3T)

    if outputdEdX:
        dEdX[:Xend] = (gnp.dot(dEdGig.transpose(), Wxi) + \
                      gnp.dot(dEdGfg.transpose(), Wxf) + \
                      gnp.dot(dEdZg.transpose(), Wxc) + \
                      gnp.dot(dEdGog.transpose(), Wxo))

    return dEdWi, dEdWf, dEdWc, dEdWo, dEdX