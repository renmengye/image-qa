import sys
import os

import numpy as np
import nn
import imageqa_test

def buildObjDict(trainData, questionIdict):
    objDict = {}
    objIdict = []
    for n in range(trainData[0].shape[0]):
        objId = trainData[0][n, -2, 0]
        obj = questionIdict[objId - 1]
        colorId = trainData[1][n, 0]
        if not objDict.has_key(obj):
            objDict[obj] = len(objIdict)
            objIdict.append(obj)
    objDict['UNK'] = len(objIdict)
    objIdict.append('UNK')
    return objDict, objIdict

def trainCount(trainData, questionIdict, objDict, objIdict, numAns):
    """
    Calculates count(w, a), count(a)
    """
    count_wa = np.zeros((len(objIdict), numAns))
    count_a = np.zeros((numAns))
    for n in range(trainData[0].shape[0]):
        objId = trainData[0][n, -2, 0]
        obj = questionIdict[objId - 1]
        colorId = trainData[1][n, 0]
        objId2 = objDict[obj]
        count_wa[objId2, colorId] += 1
        count_a[colorId] += 1
    # Add UNK count
    count_a[-1] += 1
    return count_wa, count_a

if __name__ == '__main__':
    """
    Usage:
    python imageqa_visprior.py
                                -cid {colorClassifierId}
                                -d[ata] {dataFolder}
                                -r[esults] {resultsFolder}
    """
    for i, flag in enumerate(sys.argv):
        if flag == '-cid':
            colorClassifierId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]

    delta = 0.01
    trainDataFile = os.path.join(dataFolder, 'train.npy')
    trainData = np.load(trainDataFile)
    testDataFile = os.path.join(dataFolder, 'test.npy')
    testData = np.load(testDataFile)
    vocabDictFile = os.path.join(dataFolder, 'vocab-dict.npy')
    vocabDict = np.load(vocabDictFile)
    questionDict = vocabDict[0]
    questionIdict = vocabDict[1]
    ansDict = vocabDict[2]
    ansIdict = vocabDict[3]

    print questionIdict

    objDict, objIdict = buildObjDict(trainData, questionIdict)
    count_wa, count_a = trainCount(trainData, 
                                questionIdict,
                                objDict,
                                objIdict,
                                len(ansIdict))

    for obj in objIdict:
        objId = objDict[obj]
        print obj,
        for i in range(count_wa.shape[1]):
            print ansIdict[i], count_wa[objId, i],
        print
    
    testInput = testData[0]
    testTarget = testData[1]

    testObjId = testInput[:, -2, 0].astype('int')
    questionIdictArray = np.array(questionIdict, dtype='object')
    testObjId = testObjId - 1
    testObj = questionIdictArray[testObjId]
    testObjId2 = np.zeros(testObjId.shape, dtype='int')
    for i in range(testObj.shape[0]):
        testObjId2[i] = objDict[testObj[i]]
    print testObjId2
    testColor = testTarget[:, 0]
    print np.max(testObjId2)
    print np.max(testColor)

    model = imageqa_test.loadModel(colorClassifierId, resultsFolder)
    testOutput = nn.test(model, testInput)

    # (n, c)
    P_w_a = count_wa[testObjId2, :]
    P_w_a /= count_a[:] 
    P_w_a += delta
    P_w_a /= (len(ansDict) * delta + 1)

    # (n, c)
    P_a_i = testOutput

    # (n, c)
    P_wai = P_w_a * P_a_i
    P_a_wi = P_wai / np.sum(P_wai, axis=1).reshape(P_wai.shape[0], 1)

    print 'Output probs:'
    print P_a_wi

    outputMax = np.argmax(P_a_wi, axis=-1)
    outputMax = outputMax.reshape(outputMax.size)
    testTarget = testTarget.reshape(testTarget.size)
    
    print 'Accuracy:',
    print np.sum((outputMax == testTarget).astype('int')) / \
            float(testTarget.size)