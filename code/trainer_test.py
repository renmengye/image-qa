from sentiment3_vec import *
from sequential import *

def evaluateGrad(model, W, eps, costFn, hasDropout):
    dEdW = np.zeros(W.shape)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i,j] += eps
            Y = model.forwardPass(X, dropout=hasDropout)
            Etmp1, d1 = costFn(Y, T)

            W[i,j] -= 2 * eps
            Y = model.forwardPass(X, dropout=hasDropout)
            Etmp2, d2 = costFn(Y, T)

            dEdW[i,j] = (Etmp1 - Etmp2) / 2.0 / eps
            W[i,j] += eps
    return dEdW

if __name__ == '__main__':
    trainInput, trainTarget = getTrainData()          # 2250 records

    np.random.seed(1)
    subset = np.arange(0, 129 * 2)                    # 522 records, 129 positive
    trainInput = trainInput[subset]
    trainInput = trainInput.reshape(trainInput.shape[0], trainInput.shape[1], 1)
    trainTarget = trainTarget[subset]
    timespan = trainInput.shape[1]

    trainOpt = {
        'numEpoch': 2000,
        'heldOutRatio': 0.1,
        'momentum': 0.9,
        'batchSize': 15,
        'learningRateDecay': 1.0,
        'momentumEnd': 0.9,
        'shuffle': True,
        'needValid': True,
        'writeRecord': True,
        'plotFigs': True,
        'everyEpoch': True,
        'calcError': True,
        'stopE': 0.01
    }

    time_unfold = TimeUnfold()

    lindict = LinearDict(
        inputDim=np.max(trainInput)+1,
        outputDim=5,
        needInit=False,
        initWeights=getWordEmbedding(
            initSeed=2,
            initRange=0.42,
            pcaDim=5)
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

    sig = Sigmoid(
        inputDim=5,
        outputDim=1,
        initRange=1,
        initSeed=4
    )

    soft = Softmax(
        inputDim=5,
        outputDim=2,
        initRange=1,
        initSeed=5
    )

    model = Sequential(
        stages=[
            time_unfold,
            lindict,
            time_fold,
            lstm,
            dropout,
            lstm_second,
            soft
        ]
    )

    costFn = crossEntIdx
    decisionFn = argmax

    trainer = Trainer(
        name='sentiment',
        model = model,
        costFn=costFn,
        decisionFn=decisionFn,
        outputFolder='../debug',
        trainOpt=trainOpt)

    hasDropout = True
    X = trainInput[0:1]
    T = trainTarget[0:1]
    Y = model.forwardPass(X, dropout=hasDropout)
    E, dEdY = costFn(Y, T)
    dEdX = model.backPropagate(dEdY)

    dEdWsoft = soft.dEdW
    dEdWsig = sig.dEdW
    dEdWlstm_second = lstm_second.dEdW
    dEdWlstm = lstm.dEdW
    dEdWdict = lindict.dEdW

    eps = 1e-3
    #dEdWsig2 = evaluateGrad(model, sig.getWeights(), eps, costFn, hasDropout)
    dEdWsoft2 = evaluateGrad(model, soft.getWeights(), eps, costFn, hasDropout)
    dEdWlstm_second2 = evaluateGrad(model, lstm_second.getWeights(), eps, costFn, hasDropout)
    dEdWlstm2 = evaluateGrad(model, lstm.getWeights(), eps, costFn, hasDropout)
    #dEdWdict2 = evaluateGrad(model, lindict.getWeights(), eps, costFn, hasDropout)

    print 'hello'
    print dEdWlstm2
    pass
