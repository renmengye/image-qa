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
    return os.path.join(
                    resultsFolder, 
                    taskId, 
                    '%s.test.o.txt' % taskId)

def getTruthFilename(
                        taskId, 
                        resultsFolder):
    return os.path.join(
                    resultsFolder, 
                    taskId, 
                    '%s.test.t.txt' % taskId)

def testAll(
            taskId, 
            model, 
            dataFolder, 
            resultsFolder):
    testAnswerFile = getAnswerFilename(taskId, resultsFolder)
    testTruthFile = getTruthFilename(taskId, resultsFolder)
    testDataFile = os.path.join(dataFolder, 'test.npy')
    vocabDictFile = os.path.join(dataFolder, 'vocab-dict.npy')
    qtypeFile = os.path.join(dataFolder, 'test-qtype.npy')
    vocabDict = np.load(vocabDictFile)
    testData = np.load(testDataFile)
    inputTest = testData[0]
    outputTest = nn.test(model, inputTest)
    targetTest = testData[1]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]
    questionTypeArray = np.load(qtypeFile)
    outputTxt(outputTest, targetTest, answerArray, 
              testAnswerFile, testTruthFile)
    resultsRank = calcPrecision(outputTest, targetTest)
    correct, total = calcRate(inputTest, 
        outputTest, targetTest, questionTypeArray=questionTypeArray)
    resultsCategory = correct / total.astype(float)
    resultsFile = os.path.join(resultsFolder, taskId, 'result.txt')
    resultsWups = runWups(testAnswerFile, testTruthFile)
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
    return outputTest

def testEnsemble(
                    ensembleId,
                    taskIds, 
                    models, 
                    dataFolder, 
                    resultsFolder):
    testDataFile = os.path.join(dataFolder, 'test.npy')
    vocabDictFile = os.path.join(dataFolder, 'vocab-dict.npy')
    questionTypesFile = os.path.join(dataFolder, 'test-qtype.npy')
    vocabDict = np.load(vocabDictFile)
    testData = np.load(testDataFile)
    inputTest = testData[0]
    targetTest = testData[1]
    vocabDict = np.load(vocabDictFile)
    answerArray = vocabDict[3]
    questionTypes = np.load(questionTypesFile)
    allOutput = []
    for i, taskId in enumerate(taskIds):
        testAnswerFile = getAnswerFilename(taskId, resultsFolder)
        testTruthFile = getTruthFilename(taskId, resultsFolder)
        print 'Running test set on model #%d' % i
        outputTest = nn.test(models[i], inputTest)
        allOutput.append(outputTest)
    ensembleOutput = np.zeros(allOutput[0].shape)
    for i in range(allOutput[0].shape[0]):
        ensembleOutput[i] = allOutput[questionTypes[i]]

    ensembleAnswerFile = getAnswerFilename(ensembleId, resultsFolder)
    ensembleTruthFile = getTruthFilename(ensembleId, resultsFolder)
    outputTxt(outputTest, targetTest, answerArray, 
              testAnswerFile, testTruthFile)
    resultsRank = calcPrecision(outputTest, targetTest)
    correct, total = calcRate(None, 
        ensembleOutput, targetTest, questionTypeArray=questionTypes)
    resultsCategory = correct / total.astype(float)
    resultsFile = os.path.join(resultsFolder, taskId, 'result.txt')
    resultsWups = runWups(testAnswerFile, testTruthFile)
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
    return ensembleOutput

if __name__ == '__main__':
    """
    Usage python imageqa_test.py {taskId} 
                                 -d[ata] {dataFolder} 
                                 [-r[esults] {resultsFolder}]
    """
    taskId = sys.argv[1]
    print taskId
    dataFolder = None
    resultsFolder = None
    for i in range(len(sys.argv)):
        if sys.argv[i] == '-d' or sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-r' or sys.argv[i] == '-result':
            resultsFolder = sys.argv[i + 1]
    if resultsFolder is None:
        resultsFolder = '../results'
    modelSpecFile = '%s/%s/%s.model.yml' % (resultsFolder, taskId, taskId)
    modelWeightsFile = '%s/%s/%s.w.npy' % (resultsFolder, taskId, taskId)
    model = nn.load(modelSpecFile)
    model.loadWeights(np.load(modelWeightsFile))
    testAll(taskId, model, dataFolder, resultsFolder)
