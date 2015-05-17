import sys
import os

import numpy as np
import nn
import imageqa_test

def locateObjNumber(data, questionDict):
    """
    Locate the object of how many questions.
    Very naive heuristic: take the word immediately after "how many".
    """
    how = questionDict['how']
    many = questionDict['many']
    for t in range(data.shape[0] - 2):
        if data[t, 0] == how and \
            data[t + 1, 0] == many:
            #print 'found', data[t + 2, 0]
            return data[t + 2, 0]
    print 'not found'

def locateObjColor(data):
    return data[-2, 0]

def buildObjDict(trainData, questionIdict, 
                questionType='color', questionDict=None):
    objDict = {}
    objIdict = []
    for n in range(trainData[0].shape[0]):
        if questionType == 'color':
            objId = locateObjColor(trainData[0][n])
        elif questionType == 'number':
            objId = locateObjNumber(trainData[0][n], questionDict)
        obj = questionIdict[objId - 1]
        colorId = trainData[1][n, 0]
        if not objDict.has_key(obj):
            objDict[obj] = len(objIdict)
            objIdict.append(obj)
    objDict['UNK'] = len(objIdict)
    objIdict.append('UNK')
    return objDict, objIdict

def trainCount(trainData, questionIdict, objDict, 
                objIdict, numAns, 
                questionType='color', questionDict=None):
    """
    Calculates count(w, a), count(a)
    """
    count_wa = np.zeros((len(objIdict), numAns))
    count_a = np.zeros((numAns))
    for n in range(trainData[0].shape[0]):
        if questionType == 'color':
            objId = locateObjColor(trainData[0][n])
        elif questionType == 'number':
            objId = locateObjNumber(trainData[0][n], questionDict)
        obj = questionIdict[objId - 1]
        colorId = trainData[1][n, 0]
        objId2 = objDict[obj]
        count_wa[objId2, colorId] += 1
        count_a[colorId] += 1
    # Add UNK count
    count_a[-1] += 1
    return count_wa, count_a

def testVisPrior(
                testData, 
                visModel,
                questionDict, 
                questionIdict,
                questionType='color',
                delta=0.01):
    objDict, objIdict = buildObjDict(trainData, 
                                questionIdict,
                                questionType,
                                questionDict)

    count_wa, count_a = trainCount(trainData, 
                                questionIdict,
                                objDict,
                                objIdict,
                                len(ansIdict),
                                questionType,
                                questionDict)
    print count_wa

    for obj in objIdict:
        objId = objDict[obj]
        print obj,
        for i in range(count_wa.shape[1]):
            print ansIdict[i], count_wa[objId, i],
        print

    testInput = testData[0]
    testObjId = np.zeros((testInput.shape[0]), dtype='int')

    if questionType == 'color':
        for i in range(testInput.shape[0]):
            testObjId[i] = locateObjColor(testInput[i])
    elif questionType == 'number':
        for i in range(testInput.shape[0]):
            testObjId[i] = locateObjNumber(testInput[i], questionDict)

    questionIdictArray = np.array(questionIdict, dtype='object')
    testObjId = testObjId - 1
    testObj = questionIdictArray[testObjId]
    testObjId2 = np.zeros(testObjId.shape, dtype='int')
    for i in range(testObj.shape[0]):
        if objDict.has_key(testObj[i]):
            testObjId2[i] = objDict[testObj[i]]
        else:
            testObjId2[i] = objDict['UNK']
    testOutput = nn.test(visModel, testInput)

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

    return P_a_wi


if __name__ == '__main__':
    """
    Usage:
    python imageqa_visprior.py
                                -cid {colorClassifierId}
                                -id {mainModelId}
                                -cd[ata] {colorDataFolder}
                                -d[ata] {mainDataFolder}
                                -r[esults] {resultsFolder}
                                -color/-number
    """
    questionType = 'color'
    for i, flag in enumerate(sys.argv):
        if flag == '-cid':
            colorClassifierId = sys.argv[i + 1]
        elif flag == '-cid':
            modelId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-color':
            questionType = 'color'
        elif flag == '-number':
            questionType = 'number'

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
    testInput = testData[0]
    testTarget = testData[1]

    visModel = imageqa_test.loadModel(colorClassifierId, resultsFolder)
    testOutput = testVisPrior(
                                testData, 
                                visModel,
                                questionDict,
                                questionIdict,
                                questionType,
                                delta)
    outputMax = np.argmax(testOutput, axis=-1)
    outputMax = outputMax.reshape(outputMax.size)
    testTarget = testTarget.reshape(testTarget.size)
    
    print 'Accuracy:',
    print np.sum((outputMax == testTarget).astype('int')) / \
            float(testTarget.size)

    if modelId is not None:
        # re-index the test set...
        otherModel = imageqa_test.loadModel(modelId, resultsFolder)
        testOutput2 = nn.test(testInput, otherModel)
        # Need to extract the color output from testOutput2