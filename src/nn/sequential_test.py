from sequential import *
from lstm import *
from map import *
from dropout import *
from time_fold import *
from time_unfold import *
from lut import *
from active_func import *

def evaluateGrad(model, W, eps, hasDropout):
    dEdW = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] += eps
            Y = model.forward(X, dropout=hasDropout)
            Etmp1, d1 = model.costFn(Y, T)

            W[i,j] -= 2 * eps
            Y = model.forward(X, dropout=hasDropout)
            Etmp2, d2 = model.costFn(Y, T)

            dEdW[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
            W[i,j] += eps
    return dEdW

if __name__ == '__main__':
    data = np.load('../../data/sentiment3/train-1.npy')
    trainInput = data[0]
    trainTarget = data[1]
    wordEmbed = np.random.rand(5,np.max(trainInput))
    timespan = trainInput.shape[1]

    time_unfold = TimeUnfold()

    lut = LUT(
        inputDim=np.max(trainInput)+1,
        outputDim=5,
        needInit=False,
        initWeights=wordEmbed
    )

    time_fold = TimeFold(
        timespan=timespan
    )

    lstm = LSTM(
        inputDim=5,
        outputDim=5,
        initRange=.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=True
    )

    dropout = Dropout(
        dropoutRate=0.5,
        initSeed=2,
        debug=True
    )

    lstm_second = LSTM(
        inputDim=5,
        outputDim=5,
        initRange=.1,
        initSeed=3,
        cutOffZeroEnd=True,
        multiErr=False
    )

    sig = Map(
        inputDim=5,
        outputDim=1,
        activeFn=SigmoidActiveFn,
        initRange=1,
        initSeed=4
    )

    soft = Map(
        inputDim=5,
        outputDim=2,
        activeFn=SoftmaxActiveFn,
        initRange=1,
        initSeed=5
    )

    model = Sequential(
        stages=[
            time_unfold,
            lut,
            time_fold,
            lstm,
            dropout,
            lstm_second,
            soft
        ]
    )

    model.costFn = crossEntIdx
    model.decisionFn = argmax
    hasDropout = True
    X = trainInput[0:1]
    T = trainTarget[0:1]
    Y = model.forward(X, dropout=hasDropout)
    E, dEdY = model.costFn(Y, T)
    dEdX = model.backward(dEdY)

    dEdWsoft = soft.dEdW
    dEdWsig = sig.dEdW
    dEdWlstm_second = lstm_second.dEdW
    dEdWlstm = lstm.dEdW
    dEdWdict = lut.dEdW

    eps = 1e-3
    dEdWsoft2 = evaluateGrad(model, soft.getWeights(), eps, hasDropout)
    dEdWlstm_second2 = evaluateGrad(model, lstm_second.getWeights(), eps, hasDropout)
    dEdWlstm2 = evaluateGrad(model, lstm.getWeights(), eps, hasDropout)

    print 'hello'
    print dEdWlstm / dEdWlstm2
    pass
