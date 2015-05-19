import numpy as np
import calculate_wups
import os
import nn
import sys

def decodeQuestion(
                    modelInput, 
                    questionArray):
    sentence = ''
    for t in range(1, modelInput.shape[0]):
        if modelInput[t, 0] == 0:
            break
        sentence += questionArray[modelInput[t, 0]- 1] + ' '
    sentence += '?'
    return sentence

def estimateQuestionType(question):
    if 'how many' in question:
        typ = 1
    elif question.startswith('what is the color') or \
        question.startswith('what') and 'colour' in question:
        typ = 2
    elif question.startswith('where'):
        typ = 3
    else:
        typ = 0
    return typ


def calcRate(
                modelInput, 
                modelOutput, 
                target, 
                questionArray=None, 
                questionTypeArray=None):
    correct = np.zeros(4, dtype=int)
    total = np.zeros(4, dtype=int)
    for n in range(0, modelOutput.shape[0]):        
        sortIdx = np.argsort(modelOutput[n], axis=0)
        sortIdx = sortIdx[::-1]
        answer = sortIdx[0]
        if questionTypeArray is None:
            question = decodeQuestion(modelInput[n], questionArray)
            typ = estimateQuestionType(question)
        else:
            typ = questionTypeArray[n]
        total[typ] += 1
        if answer == target[n, 0]:
            correct[typ] += 1
    rate = correct / total.astype('float')
    print 'object: %.4f' % rate[0]
    print 'number: %.4f' % rate[1]
    print 'color: %.4f' % rate[2]
    print 'scene: %.4f' % rate[3]
    return correct, total

def calcPrecision(
                    modelOutput, 
                    target):
    # Calculate precision
    correctAt1 = 0
    correctAt5 = 0
    correctAt10 = 0
    for n in range(0, modelOutput.shape[0]):
        sortIdx = np.argsort(modelOutput[n], axis=0)
        sortIdx = sortIdx[::-1]
        for i in range(0, 10):
            if sortIdx[i] == target[n, 0]:
                if i == 0:
                    correctAt1 += 1
                if i <= 4:
                    correctAt5 += 1
                correctAt10 += 1
    r1 = correctAt1 / float(modelOutput.shape[0])
    r5 = correctAt5 / float(modelOutput.shape[0])
    r10 = correctAt10 / float(modelOutput.shape[0])
    print 'rate @ 1: %.4f' % r1
    print 'rate @ 5: %.4f' % r5
    print 'rate @ 10: %.4f' % r10
    return (r1, r5, r10)

def outputTxt(
                modelOutput, 
                target, 
                answerArray, 
                answerFilename, 
                truthFilename, 
                topK=1, 
                outputProb=False):
    """
    Output the results of all examples into a text file.
    topK: top k answers, separated by comma.
    outputProb: whether to output the probability of the answer as well.

    Format will look like this:
    q1ans1,0.99,a1ans2,0.01...
    q2ans1,0.90,q2ans2,0.02...
    """
    with open(truthFilename, 'w+') as f:
        for n in range(0, target.shape[0]):
            f.write(answerArray[target[n, 0]] + '\n')
    with open(answerFilename, 'w+') as f:
        for n in range(0, modelOutput.shape[0]):
            if topK == 1:
                f.write(answerArray[np.argmax(modelOutput[n, :])])
                if outputProb:
                    f.write(',%.4f' % modelOutput[n, np.argmax(modelOutput[n, :])])
                f.write('\n')
            else:
                sortIdx = np.argsort(modelOutput[n], axis=0)
                sortIdx = sortIdx[::-1]
                for i in range(0, topK):
                    f.write(answerArray[sortIdx[i]])
                    if outputProb:
                        f.write(',%.4f' % modelOutput[n, sortIdx[i]])
                    f.write('\n')

def runWups(
            answerFilename, 
            truthFilename):
    w10 = calculate_wups.runAll(truthFilename, answerFilename, -1)
    w09 = calculate_wups.runAll(truthFilename, answerFilename, 0.9)
    w00 = calculate_wups.runAll(truthFilename, answerFilename, 0.0)
    print 'WUPS @ 1.0: %.4f' % w10
    print 'WUPS @ 0.9: %.4f' % w09
    print 'WUPS @ 0.0: %.4f' % w00
    return (w10, w09, w00)

def getAnswerFilename(
                        taskId, 
                        resultsFolder):
    folder = os.path.join(resultsFolder, taskId)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(
                    folder, 
                    '%s.test.o.txt' % taskId)

def getTruthFilename(
                        taskId, 
                        resultsFolder):
    folder = os.path.join(resultsFolder, taskId)
    if not os.path.exists(folder):
        os.makedirs(folder)
    return os.path.join(
                    folder,
                    '%s.test.t.txt' % taskId)

def loadDataSet(dataFolder):
    trainDataFile = os.path.join(dataFolder, 'train.npy')
    testDataFile = os.path.join(dataFolder, 'test.npy')
    vocabDictFile = os.path.join(dataFolder, 'vocab-dict.npy')
    qtypeFile = os.path.join(dataFolder, 'test-qtype.npy')
    trainData = np.load(trainDataFile)
    testData = np.load(testDataFile)
    vocabDict = np.load(vocabDictFile)
    questionTypeArray = np.load(qtypeFile)
    inputTest = testData[0]
    targetTest = testData[1]
    qDict = vocabDict[0]
    qIdict = vocabDict[1]
    aDict = vocabDict[2]
    aIDict = vocabDict[3]
    return (trainData,
            testData,
            qDict,
            qIdict,
            aDict,
            aIdict,
            qTypeArray)

def loadTestSet(dataFolder):
    trainData,\
    testData,\
    qDict,\
    qIdict,\
    aDict,\
    aIdict,\
    qTypeArray = loadDataSet(dataFolder)
    inputTest = testData[0]
    targetTest = testData[1]
    return (inputTest, 
            targetTest,
            qIdict, 
            aIdict, 
            qTypeArray)

def loadModel(
                taskId,
                resultsFolder):
    modelSpecFile = '%s/%s/%s.model.yml' % (resultsFolder, taskId, taskId)
    modelWeightsFile = '%s/%s/%s.w.npy' % (resultsFolder, taskId, taskId)
    model = nn.load(modelSpecFile)
    model.loadWeights(np.load(modelWeightsFile))
    return model

def testAll(
            taskId, 
            model, 
            dataFolder, 
            resultsFolder):
    testAnswerFile = getAnswerFilename(taskId, resultsFolder)
    testTruthFile = getTruthFilename(taskId, resultsFolder)
    inputTest, \
    targetTest, \
    questionArray, \
    answerArray, \
    questionTypeArray = loadTestSet(dataFolder)
    outputTest = nn.test(model, inputTest)
    resultsRank, \
    resultsCategory, \
    resultsWups = runAllMetrics(
                                inputTest,
                                outputTest,
                                targetTest,
                                answerArray,
                                questionTypeArray,
                                testAnswerFile,
                                testTruthFile)
    writeMetricsToFile(
                        taskId,
                        resultsRank,
                        resultsCategory,
                        resultsWups,
                        resultsFolder)
    return outputTest

def runAllMetrics(
                    inputTest,
                    outputTest, 
                    targetTest, 
                    answerArray, 
                    questionTypeArray, 
                    testAnswerFile, 
                    testTruthFile):
    outputTxt(outputTest, targetTest, answerArray, 
              testAnswerFile, testTruthFile)
    resultsRank = calcPrecision(outputTest, targetTest)
    correct, total = calcRate(inputTest, 
        outputTest, targetTest, questionTypeArray=questionTypeArray)
    resultsCategory = correct / total.astype(float)
    resultsWups = runWups(testAnswerFile, testTruthFile)
    return (resultsRank, resultsCategory, resultsWups)

def writeMetricsToFile(
                        taskId, 
                        resultsRank, 
                        resultsCategory, 
                        resultsWups, 
                        resultsFolder):
    resultsFile = os.path.join(resultsFolder, taskId, 'result.txt')
    with open(resultsFile, 'w') as f:
        f.write('rate @ 1: %.4f\n' % resultsRank[0])
        f.write('rate @ 5: %.4f\n' % resultsRank[1])
        f.write('rate @ 10: %.4f\n' % resultsRank[2])
        f.write('object: %.4f\n' % resultsCategory[0])
        f.write('number: %.4f\n' % resultsCategory[1])
        f.write('color: %.4f\n' % resultsCategory[2])
        f.write('scene: %.4f\n' % resultsCategory[3])
        f.write('WUPS 1.0: %.4f\n' % resultsWups[0])
        f.write('WUPS 0.9: %.4f\n' % resultsWups[1])
        f.write('WUPS 0.0: %.4f\n' % resultsWups[2])

def loadEnsemble(
                    taskIds, 
                    resultsFolder):
    """
    Load class specific models.
    """
    models = []
    for taskId in taskIds:
        taskFolder = os.path.join(resultsFolder, taskId)
        modelSpec = os.path.join(taskFolder, '%s.model.yml' % taskId)
        modelWeights = os.path.join(taskFolder, '%s.w.npy' % taskId)
        model = nn.load(modelSpec)
        model.loadWeights(np.load(modelWeights))
        models.append(model)
    return models

def __runEnsemble(
                inputTest,
                models,
                ansDict,
                classAnsIdict,
                questionTypeArray):
    allOutput = []
    for i, model in enumerate(models):
        print 'Running test set on model #%d' % i
        outputTest = nn.test(model, inputTest)
        allOutput.append(outputTest)
    ensembleOutputTest = np.zeros((inputTest.shape[0], len(ansDict)))
    for n in range(allOutput[0].shape[0]):
        for i in range(allOutput[questionTypeArray[n]].shape[1]):
            ensembleOutputTest[n, ansDict[classAnsIdict[i]]] = \
                allOutput[questionTypeArray[n]][n, i]
    return ensembleOutputTest

def getClassDataFolders(dataset):
    """
    Get different original data folder name for class specific models.
    """
    if dataset == 'daquar':
        classDataFolders = [
            dataFolder + '-object',
            dataFolder + '-number',
            dataFolder + '-color'
        ]
    elif dataset == 'cocoqa':
        classDataFolders = [
            dataFolder + '-object',
            dataFolder + '-number',
            dataFolder + '-color',
            dataFolder + '-location'
        ]
    return classDataFolders

def runEnsemble(
                inputTest,
                models, 
                dataFolder, 
                classDataFolders,
                questionTypeArray):
    """
    Run a class specific model on any dataset.
    """
    trainData, \
    testData, \
    qDict, \
    qIdict, \
    aDict, \
    aIdict, \
    qTypeArray = loadDataSet(dataFolder)
    classAnsIdict = []
    for df in classDataFolders:
        trainData_c, \
        testData_c, \
        qDict_c, \
        qIdict_c, \
        aDict_c, \
        aIdict_c, \
        qTypeArray_c = loadDataSet(df)
        classAnsIdict.append(aIdict_c)

    ensembleOutputTest = __runEnsemble(
                                        inputTest, 
                                        models,
                                        aDict,
                                        classAnsIdict,
                                        questionTypeArray)
    return ensembleOutputTest

def testEnsemble(
                    ensembleId,
                    models,
                    dataFolder,
                    classDataFolders,
                    resultsFolder):
    """
    Test a class specific model in its original dataset.
    """
    trainData, \
    testData, \
    qDict, \
    qIdict, \
    aDict, \
    aIdict, \
    qTypeArray = loadDataSet(dataFolder)
    inputTest = testData[0]
    targetTest = testData[1]

    ensembleOutputTest = runEnsemble(
                                    inputTest,
                                    models, 
                                    dataFolder, 
                                    classDataFolders,
                                    qTypeArray)
    ensembleAnswerFile = getAnswerFilename(ensembleId, resultsFolder)
    ensembleTruthFile = getTruthFilename(ensembleId, resultsFolder)

    resultsRank, \
    resultsCategory, \
    resultsWups = runAllMetrics(
                                inputTest,
                                ensembleOutputTest,
                                targetTest,
                                aIdict,
                                qTypeArray,
                                ensembleAnswerFile,
                                ensembleTruthFile)
    writeMetricsToFile(
                        ensembleId,
                        resultsRank,
                        resultsCategory,
                        resultsWups,
                        resultsFolder)

    return ensembleOutputTest

if __name__ == '__main__':
    """
    Usage python imageqa_test.py -i[d] {taskId} 
                                 -d[ata] {dataFolder} 
                                 [-r[esults] {resultsFolder}]
    """
    dataFolder = None
    resultsFolder = '../results'
    for i, flag in enumerate(sys.argv):
        if flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-i' or flag == '-id':
            taskId = sys.argv[i + 1]
        elif flag == '-r' or flag == '-result':
            resultsFolder = sys.argv[i + 1]
    print taskId
    model = loadModel(taskId, resultsFolder)
    testAll(taskId, model, dataFolder, resultsFolder)