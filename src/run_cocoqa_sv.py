import sklearn.linear_model
import h5py
import scipy
import numpy as np

imgFeatFile = '/ais/gobi3/u/mren/data/cocoqa-full/hidden_oxford.h5'
sentVecFile = '/ais/gobi3/u/mren/data/cocoqa-full/skip_sent_vec.npy'
trainDataFile = '../data/cocoqa-sv/train.npy'
validDataFile = '../data/cocoqa-sv/valid.npy'
testDataFile = '../data/cocoqa-sv/test.npy'

def packData(inputs, imgFeats, sentVecs):
    imgIds = inputs[:, 0] - 1
    sentIds = inputs[:, 1] - 1
    imgSel = imgFeats[imgIds, :].todense()
    sentVecSel = sentVecs[sentIds, :]
    return np.concatenate((sentVecSel, imgSel), axis=1)

if __name__ == '__main__':
    trainData = np.load(trainDataFile)
    validData = np.load(validDataFile)
    testData = np.load(testDataFile)
    imgFeatsH5 = h5py.File(imgFeatFile)
    key = 'hidden7'
    iwShape = imgFeatsH5[key + '_shape'][:]
    iwData = imgFeatsH5[key + '_data'][:]
    iwInd = imgFeatsH5[key + '_indices'][:]
    iwPtr = imgFeatsH5[key + '_indptr'][:]
    imgFeats = scipy.sparse.csr_matrix(
        (iwData, iwInd, iwPtr), shape=iwShape)
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
        # lr = sklearn.linear_model.LogisticRegression(
        #         C=C
        #     )
        lr = sklearn.linear_model.SGDClassifier(
            loss='log', penalty='l2', alpha=C, n_jobs=30)
        lr.fit(trainInput, trainTarget)
        rate = lr.score(validInput, validTarget)
        print '%.f, %.4f' % (C, rate)
        if rate > bestRate:
            bestC = C
            bestRate = rate

    # lr = sklearn.linear_model.LogisticRegression(
    #             C=bestC
    #         )
    lr = sklearn.linear_model.SGDClassifier(
        loss='log', penalty='l2', alpha=bestC, n_jobs=30)
    print 'bestC:', bestC
    allInput = np.concatenate((trainInput, validInput), axis=0)
    allTarget = np.concatenate((trainTarget, validTarget), axis=0)
    lr.fit(trainInput, allTarget)
    rate = lr.score(testInput, testTarget)
    print 'Final rate:', rate