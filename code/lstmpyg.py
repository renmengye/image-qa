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

def backPropagateN(
                   dEdY,X,Y,C,Z,Gi,Gf,Go,
                   Xend,cutOffZeroEnd,multiErr,
                   outputdEdX,W):
    numEx = X.shape[0]
    inputDim = X.shape[2]
    outputDim = Y.shape[2]
    Wxi,Wyi,Wci,Wxf,Wyf,Wcf,Wxc,Wyc,Wxo,Wyo,Wco = sliceWeightsSmall(inputDim, outputDim, W)
    Wg = gnp.as_garray(W)
    Wxig,Wyig,Wcig,Wxfg,Wyfg,Wcfg,Wxcg,Wycg,Wxog,Wyog,Wcog = sliceWeightsSmall(inputDim, outputDim, Wg)
    dEdW = np.zeros(W.shape)
    dEdX = np.zeros(X.shape)
    dEdWi,dEdWf,dEdWc,dEdWo = sliceWeights(inputDim, outputDim, dEdW)
    Cg = gnp.as_garray(C)
    Zg = gnp.as_garray(Z)
    Gig = gnp.as_garray(Gi)
    Gfg = gnp.as_garray(Gf)
    Gog = gnp.as_garray(Go)
    memEyeT = gnp.as_garray(
        np.eye(outputDim).reshape(1, outputDim, outputDim))

    for n in range(0, numEx):
        dEdWitmp, dEdWftmp, dEdWctmp, dEdWotmp, dEdX[n] = \
            backPropagateOne(dEdY[n],X[n],Y[n],
                        C[n],Cg[n,:Xend[n]],Zg[n,:Xend[n]],Gig[n,:Xend[n]],
                        Gfg[n,:Xend[n]],Gog[n,:Xend[n]],
                        Xend[n],cutOffZeroEnd,
                        multiErr,outputdEdX,
                        Wxi,Wyig,Wcig,Wxf,Wyfg,Wcfg,Wxc,
                        Wycg,Wxo,Wyog,Wcog, memEyeT)
        dEdWi += dEdWitmp
        dEdWf += dEdWftmp
        dEdWc += dEdWctmp
        dEdWo += dEdWotmp
    return dEdW, dEdX

def backPropagateOne(
                    dEdY,X,Y,C,Cg,Z,Gi,Gf,Go,Xend,cutOffZeroEnd,
                    multiErr,outputdEdX,Wxi,Wyi,Wci,Wxf,Wyf,Wcf,
                    Wxc,Wyc,Wxo,Wyo,Wco,memEyeT):
    Xend = int(Xend)
    if cutOffZeroEnd and multiErr:
        dEdY[Xend - 1] += dEdY[-1]
    inputDim = X.shape[1]
    outputDim = Y.shape[1]
    ddim = (outputDim, Xend)

    # (j, t)
    dEdGi = np.zeros(ddim)
    dEdGf = np.zeros(ddim)
    dEdZ = np.zeros(ddim)
    dEdGo = np.zeros(ddim)
    dEdX = np.zeros(X.shape)

    # (k -> t)
    one = np.ones((Xend, 1))
    Yt1 = np.concatenate((np.zeros((1, outputDim)), Y[:Xend-1]))
    Ct1 = np.concatenate((np.zeros((1, outputDim)), C[:Xend-1]))
    Ct1g = gnp.concatenate((gnp.zeros((1, outputDim)), Cg[:Xend-1]))
    states1T = np.concatenate((X[:Xend], Yt1, Ct1, one), axis=-1)
    states2T = np.concatenate((X[:Xend], Yt1, one), axis=-1)
    states3T = np.concatenate((X[:Xend], Yt1, C[:Xend], one), axis=-1)

    Ug = gnp.tanh(Cg)
    dU = 1 - Ug * Ug
    dZ = 1 - Z * Z
    dGi = Gi * (1 - Gi)
    dGf = Gf * (1 - Gf)
    dGo = Go * (1 - Go)

    # (j, t)
    dCdGig = (Z * dGi)
    dCdGfg = (Ct1g * dGf)
    dCdZg = (Gi * dZ)
    dYdGog = (Ug * dGo)

    dYdCg = (Go * dU).reshape(Xend, outputDim, 1) * memEyeT + \
            dYdGog.reshape(Xend, outputDim, 1) * Wco.reshape(1, Wco.shape[0], Wco.shape[1])
    dCdCg = dCdGfg.reshape(Xend, outputDim, 1) * Wcf.reshape(1, Wcf.shape[0], Wcf.shape[1]) + \
            Gf.reshape(Xend, outputDim, 1) * memEyeT + \
            dCdGig.reshape(Xend, outputDim, 1) * Wci.reshape(1, Wci.shape[0], Wci.shape[1])
    dCdYg = dCdGfg.reshape(Xend, outputDim, 1) * Wyf.reshape(1, Wyf.shape[0], Wyf.shape[1]) + \
            dCdZg.reshape(Xend, outputDim, 1) * Wyc.reshape(1, Wyc.shape[0], Wyc.shape[1]) + \
            dCdGig.reshape(Xend, outputDim, 1) * Wyi.reshape(1, Wyi.shape[0], Wyi.shape[1])
    dYdYg = dYdGog.reshape(Xend, outputDim, 1) * Wyo.reshape(1, Wyo.shape[0], Wyo.shape[1])

    dYdC = dYdCg.as_numpy_array()
    dCdC = dCdCg.as_numpy_array()
    dCdY = dCdYg.as_numpy_array()
    dYdY = dYdYg.as_numpy_array()
    dCdGi = dCdGig.as_numpy_array()
    dCdGf = dCdGfg.as_numpy_array()
    dYdGo = dYdGog.as_numpy_array()
    dCdZ = dCdZg.as_numpy_array()

    for t in reversed(range(0, Xend)):
        dEdYnow = dEdY[t] if multiErr else 0
        if t < Xend - 1:
            dEdYt = np.dot(dEdYt, dYdY[t]) + np.dot(dEdCt, dCdY[t]) + dEdYnow
            dEdCt = np.dot(dEdCt, dCdC[t]) + np.dot(dEdYt, dYdC[t])
        else:
            dEdYt = dEdYnow if multiErr else dEdY
            dEdCt = np.dot(dEdYt, dYdC[t])
        dEdGi[:, t] = dEdCt * dCdGi[t]
        dEdGf[:, t] = dEdCt * dCdGf[t]
        dEdZ[:, t] = dEdCt * dCdZ[t]
        dEdGo[:, t] = dEdYt * dYdGo[t]

    dEdWi = np.dot(dEdGi, states1T)
    dEdWf = np.dot(dEdGf, states1T)
    dEdWc = np.dot(dEdZ, states2T)
    dEdWo = np.dot(dEdGo, states3T)

    if outputdEdX:
        dEdX[:Xend] = (np.dot(dEdGi.transpose(), Wxi) + \
                      np.dot(dEdGf.transpose(), Wxf) + \
                      np.dot(dEdZ.transpose(), Wxc) + \
                      np.dot(dEdGo.transpose(), Wxo))

    return dEdWi, dEdWf, dEdWc, dEdWo, dEdX