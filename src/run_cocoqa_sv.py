import sklearn
import h5py
import numpy as np

imgFeatFile = '/ais/gobi3/u/mren/data/cocoqa-full/hidden_oxford.h5'
sentVecFile = '/ais/gobi3/u/mren/data/cocoqa-full/skip_sent_vec.npy'
trainDataFile = '../data/cocoqa-sv/train.npy'
validDataFile = '../data/cocoqa-sv/valid.npy'
testDataFile = '../data/cocoqa-sv/test.npy'

def packData(inputs, imgFeats, sentVecs):
    imgIds = inputs[:, 0] - 1
    sentIds = inputs[:, 1] - 1
    imgSel = imgFeats[imgIds, :]
    sentVecSel = sentVecs[sentIds, :]
    return np.concatnate((sentVecSel, imgSel), axis=1)

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
    imgFeats = sparse.csr_matrix(
        (iwData, iwInd, iwPtr), shape=iwShape)
    sentVecs = np.load(sentVecFile)
    trainInput = packData(trainData[0], imgFeats, sentVecs)
    trainTarget = trainData[1]
    validInput = packData(validData[0], imgFeats, sentVecs)
    validTarget = validData[1]
    testInput = packData(testData[0], imgFeats, sentVecs)
    testTarget = testData[1]
    bestC = 0.0
    bestRate = 0.0

    for c in range(6):
        # From 1 to 1e-6
        C = np.power(10, -c)
        lr = sklearn.linear_model.LogisticRegression(
                penalty='l2',
                dual=False,
                tol=0.0001,
                C=C,
                fit_intercept=True, 
                intercept_scaling=1, 
                class_weight=None, 
                random_state=None, 
                solver='liblinear', 
                max_iter=100,
                multi_class='multinomial', 
                verbose=0
            )
        lr.fit(trainInput, trainTarget)
        rate = lr.score(validInput, validTarget)
        print '%.f, %.4f' % (C, rate)
        if rate > bestRate:
            bestC = C
            bestRate = rate

    lr = sklearn.linear_model.LogisticRegression(
                penalty='l2',
                dual=False,
                tol=0.0001,
                C=bestC,
                fit_intercept=True, 
                intercept_scaling=1, 
                class_weight=None, 
                random_state=None, 
                solver='liblinear', 
                max_iter=100,
                multi_class='multinomial', 
                verbose=0
            )
    print 'bestC:', bestC
    allInput = np.concatenate((trainInput, validInput), axis=0)
    allTarget = np.concatenate((trainTarget, validTarget), axis=0)
    lr.fit(trainInput, allTarget)
    rate = lr.score(testInput, testTarget)
    print 'Final rate:', rate