import sys
import os

import numpy as np
import nn
import imageqa_test
from nltk.corpus import wordnet

lexnameDict = {}

def lookupLexname(word):
    if lexnameDict.has_key(word):
        return lexnameDict[word]
    else:
        synsets = wordnet.synsets(word)
        # Just pick the first definition
        if len(synsets) > 0:
            lexname = synsets[0].lexname()
            lexnameDict[word] = lexname
            return lexname
        else:
            return None

def locateObjLocation(data, questionDict, questionIdict):
    """
    Locate the object of where questions.
    Very naive heuristic: take the noun immediately after "where".
    """
    where = questionDict['where']
    for t in range(data.shape[0] - 1):
        if data[t, 0] == where:
            for u in range(t + 1, data.shape[0]):
                word = questionIdict[data[u, 0] - 1]
                lexname = lookupLexname(word)
                if (lexname is not None and lexname.startswith('noun')) or \
                    (lexname is None):
                    return data[u, 0]
    print 'not found'
    return data[-1, 0]

def locateObjNumberNoun(data, questionDict, questionIdict):
    """
    Locate the object of how many questions.
    Very naive heuristic: take the noun immediately after "how many".
    """
    how = questionDict['how']
    many = questionDict['many']
    for t in range(data.shape[0] - 2):
        if data[t, 0] == how and \
            data[t + 1, 0] == many:
            for u in range(t + 2, data.shape[0]):
                word = questionIdict[data[u, 0] - 1]
                lexname = lookupLexname(word)
                if (lexname is not None and lexname.startswith('noun')) or \
                    (lexname is None):
                    return data[u, 0]
    print 'not found'
    return data[-1, 0]

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
            return data[t + 2, 0]
    print 'not found'

def locateObjColor(data):
    tmp = 0
    for i in range(data.shape[0]):
        if data[i, 0] != 0:
            tmp = data[i, 0]
        else:
            return tmp

def extractObjId(data):
    objIds = []
    for n in range(data.shape[0]):
        if questionType == 'color':
            objId = locateObjColor(data[n])
        elif questionType == 'number':
            #objId = locateObjNumber(data[n], questionDict)
            objId = locateObjNumberNoun(data[n], questionDict, questionIdict)
        elif questionType == 'location':
            objId = locateObjLocation(data[n], questionDict, questionIdict)
        objIds.append(objId)
    return np.array(objIds, dtype='int')

def buildObjDict(trainData, questionIdict, 
                questionType='color', questionDict=None):
    objDict = {}
    objIdict = []
    print questionType
    objIds = extractObjId(trainData[0])
    objIds = objIds - 1
    print objIds
    objList = questionIdict[objIds]
    for obj in objList:
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
    objIds = extractObjId(trainData[0])
    for i in range(objIds.shape[0]):
        objId = objIds[i]
        obj = questionIdict[objId - 1]
        ansId = trainData[1][n, 0]
        objId2 = objDict[obj]
        count_wa[objId2, ansId] += 1
        count_a[ansId] += 1
    # Add UNK count
    count_a[-1] += 1
    return count_wa, count_a

def runVisPriorOnce(objId, count_wa, count_a, modelOutput, delta):
    P_w_a = count_wa[objId, :]
    P_w_a /= count_a[:] 
    P_w_a += delta
    P_w_a /= (modelOutput.shape[1] * delta + 1)

    # (n, c)
    P_a_i = modelOutput

    # (n, c)
    P_wai = P_w_a * P_a_i
    P_a_wi = P_wai / np.sum(P_wai, axis=1).reshape(P_wai.shape[0], 1)
    return P_a_wi

def getObjId(inputData, objDict, questionDict, questionIdict, questionType):
    questionIdictArray = np.array(questionIdict, dtype='object')
    objIds = extractObjId(inputData)
    objIds = objIds - 1
    obj = questionIdictArray[objIds]
    objIds2 = np.zeros(objIds.shape, dtype='int')
    for i in range(obj.shape[0]):
        if objDict.has_key(obj[i]):
            objIds2[i] = objDict[obj[i]]
        else:
            objIds2[i] = objDict['UNK']
    return objIds2

def calcRate(output, target):
    outputMax = np.argmax(output, axis=-1)
    outputMax = outputMax.reshape(outputMax.size)
    targetReshape = target.reshape(target.size)
    rate = np.sum((outputMax == targetReshape).astype('int')) / \
            float(target.size)
    return rate

def runVisPrior(
                trainData,
                validData,
                testData,
                preVisModel,
                visModel,
                questionDict,
                questionIdict,
                deltas,
                questionType='color'):
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

    # Reindex test set
    testInput = testData[0]
    testTarget = testData[1]
    testTargetReshape = testTarget.reshape(testTarget.size)
    testObjId = getObjId(testInput, objDict, questionDict, questionIdict, questionType)

    # Run vis model on test set
    testOutput = nn.test(visModel, testInput)

    # Reindex valid set
    validInput = validData[0]
    validTarget = validData[1]
    validTargetReshape = validTarget.reshape(validTarget.size)
    validObjId = getObjId(validInput, objDict, questionDict, questionIdict, questionType)

    # Run vis model on valid set
    validOutput = nn.test(preVisModel, validInput)
    print 'Before Prior Valid Rate:',
    print calcRate(validOutput, validTarget)

    # Determine best delta
    bestRate = 0.0
    bestDelta = 0.0
    for delta in deltas:
        visPriorOutput = runVisPriorOnce(
                                validObjId, 
                                count_wa, 
                                count_a, 
                                validOutput, 
                                delta)        
        print 'delta=%f Valid Accuracy:' % delta,
        rate = calcRate(visPriorOutput, validTarget)
        print rate
        if rate > bestRate:
            bestRate = rate
            bestDelta = delta
    print 'Best Delta:', bestDelta
    
    # Run on test set
    visPriorOutput = runVisPriorOnce(
                            testObjId, 
                            count_wa, 
                            count_a, 
                            testOutput, 
                            bestDelta)
    print 'delta=%f Test Accuracy:' % bestDelta,
    rate = calcRate(visPriorOutput, testTarget)
    print rate
    return visPriorOutput

def loadData(dataFolder):
    trainDataFile = os.path.join(dataFolder, 'train.npy')
    trainData = np.load(trainDataFile)
    validDataFile = os.path.join(dataFolder, 'valid.npy')
    validData = np.load(validDataFile)
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
    return trainData, validData, testData, questionDict, questionIdict, ansDict, ansIdict

if __name__ == '__main__':
    """
    Usage:
    python imageqa_visprior.py
                                -pvid {preVisModelId}
                                -vid {visModelId}
                                -mid {mainModelId}
                                -vd[ata] {visDataFolder}
                                -md[ata] {mainDataFolder}
                                -r[esults] {resultsFolder}
                                -qtype {color/number/location}
    """
    questionType = 'color'
    visModelId = None
    mainModelId = None
    for i, flag in enumerate(sys.argv):
        if flag == '-pvid':
            preVisModelId = sys.argv[i + 1]
        elif flag == '-vid':
            visModelId = sys.argv[i + 1]
        elif flag == '-mid':
            mainModelId = sys.argv[i + 1]
        elif flag == '-vd' or flag == '-vdata':
            visDataFolder = sys.argv[i + 1]
        elif flag == '-md' or flag == '-mdata':
            mainDataFolder = sys.argv[i + 1]
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-qtype':
            questionType = sys.argv[i + 1]

    trainData, validData, testData, questionDict, questionIdict, ansDict, ansIdict = \
        loadData(visDataFolder)
    testInput = testData[0]
    testTarget = testData[1]
    deltas = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    preVisModel = imageqa_test.loadModel(preVisModelId, resultsFolder)
    visModel = imageqa_test.loadModel(visModelId, resultsFolder)
    visTestOutput = runVisPrior(trainData,
                                validData,
                                testData,
                                preVisModel,
                                visModel,
                                questionDict,
                                questionIdict,
                                deltas,
                                questionType)

    visModelFolder = os.path.join(resultsFolder, visModelId)
    answerFilename = os.path.join(visModelFolder, visModelId + '_prior.test.o.txt')
    truthFilename = os.path.join(visModelFolder, visModelId + '_prior.test.t.txt')
    imageqa_test.outputTxt(
                            visTestOutput, 
                            testTarget, 
                            ansIdict, 
                            answerFilename, 
                            truthFilename, 
                            topK=1, 
                            outputProb=False)
    imageqa_test.runWups(answerFilename, truthFilename)

    if mainModelId is not None:
        trainData_m, testData_m, questionDict_m, questionIdict_m, \
            ansDict_m, ansIdict_m = \
            loadData(mainDataFolder)

        newTestInput = np.zeros(testInput.shape, dtype='int')
        for n in range(testInput.shape[0]):
            for t in range(testInput.shape[1]):
                newTestInput[n, t, 0] = \
                    ansDict_m[ansIdict[testInput[n, t, 0] - 1]]
        mainModel = imageqa_test.loadModel(mainModelId, resultsFolder)
        mainTestOutput = nn.test(newTestInput, mainModel)

        # Need to extract the class output from mainTestOutput
        classNewId = []
        for ans in ansIdict:
            classNewId.append(ansDict_m[ans])
        classNewId = np.array(classNewId, dtype='int')
        mainTestOutput = mainTestOutput[:, classNewId]
        ensTestOutput = 0.5 * visTestOutput + 0.5 * mainTestOutput
        ensOutputMax = np.argmax(ensTestOutput, axis=-1)
        ensOutputMax = ensOutputMax.reshape(ensOutputMax.size)
        print '0.5 VIS+PRIOR & 0.5 VIS+BLSTM Accuracy:'
        print np.sum((ensOutputMax == testTargetReshape).astype('int')) / \
            float(testTarget.size)