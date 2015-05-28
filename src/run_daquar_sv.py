import sklearn.linear_model
import h5py
import scipy
import numpy as np

imgFeatFile = '/ais/gobi3/u/mren/data/nyu-depth/hidden_oxford.h5'
sentVecFile = '/ais/gobi3/u/mren/data/daquar-37/skip_sent_vec.npy'
trainDataFile = '../data/daquar-37-sv/train.npy'
validDataFile = '../data/daquar-37-sv/valid.npy'
testDataFile = '../data/daquar-37-sv/test.npy'

def packData(inputs, imgFeats, sentVecs):
    imgIds = inputs[:, 0] - 1
    sentIds = inputs[:, 1] - 1
    imgSel = imgFeats[imgIds, :]
    sentVecSel = sentVecs[sentIds, :]
    return np.concatenate((sentVecSel, imgSel), axis=1)

if __name__ == '__main__':
    trainData = np.load(trainDataFile)
    validData = np.load(validDataFile)
    testData = np.load(testDataFile)
    imgFeatsH5 = h5py.File(imgFeatFile)
    key = 'hidden7'
    imgFeats = imgFeatsH5[key][:]
    sentVecs = np.load(sentVecFile)
    trainInput = packData(trainData[0], imgFeats, sentVecs)
    trainTarget = trainData[1].reshape(trainData[1].size)
    validInput = packData(validData[0], imgFeats, sentVecs)
    validTarget = validData[1].reshape(validData[1].size)
    testInput = packData(testData[0], imgFeats, sentVecs)
    testTarget = testData[1].reshape(testData[1].size)
    bestC = 0.0
    bestRate = 0.0

    for c in range(-9, 10):
        # From 2^9 to 2^-9
        C = np.power(2.0, -c)
        lr = sklearn.linear_model.LogisticRegression(
                C=C
            )
        lr.fit(trainInput, trainTarget)
        rate = lr.score(validInput, validTarget)
        print '%.f, %.4f' % (C, rate)
        if rate > bestRate:
            bestC = C
            bestRate = rate

    lr = sklearn.linear_model.LogisticRegression(
                C=bestC
            )
    print 'bestC:', bestC
    allInput = np.concatenate((trainInput, validInput), axis=0)
    allTarget = np.concatenate((trainTarget, validTarget), axis=0)
    lr.fit(trainInput, allTarget)
    rate = lr.score(testInput, testTarget)
    print 'Final rate:', rate