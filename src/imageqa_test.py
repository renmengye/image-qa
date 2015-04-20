import numpy as np
import calculate_wups
import os
import nn

def decodeQuestion(X, questionArray):
    sentence = ''
    for t in range(1, X.shape[0]):
        if X[t, 0] == 0:
            break
        sentence += questionArray[X[t, 0]- 1] + ' '
    sentence += '?'
    return sentence

def calcRate(X, Y, T, questionArray):
    correct = np.zeros(4, dtype=int)
    total = np.zeros(4, dtype=int)
    for n in range(0, X.shape[0]):        
        sortIdx = np.argsort(Y[n], axis=0)
        sortIdx = sortIdx[::-1]
        A = sortIdx[0]
        question = decodeQuestion(X[n], questionArray)
        if 'how many' in question:
            typ = 1
        elif question.startswith('what is the color') or \
            question.startswith('what') and 'colour' in question:
            typ = 2
        elif question.startswith('where'):
            typ = 3
        else:
            typ = 0
        total[typ] += 1
        if A == T[n, 0]:
            correct[typ] += 1
    rate = correct / total.astype('float')
    print 'object: %.4f' % rate[0]
    print 'number: %.4f' % rate[1]
    print 'color: %.4f' % rate[2]
    print 'scene: %.4f' % rate[3]
    return correct, total

def calcPrecision(Y, T):
    # Calculate precision
    correctAt1 = 0
    correctAt5 = 0
    correctAt10 = 0
    for n in range(0, Y.shape[0]):
        sortIdx = np.argsort(Y[n], axis=0)
        sortIdx = sortIdx[::-1]
        for i in range(0, 10):
            if sortIdx[i] == T[n, 0]:
                if i == 0:
                    correctAt1 += 1
                if i <= 4:
                    correctAt5 += 1
                correctAt10 += 1
    r1 = correctAt1 / float(Y.shape[0])
    r5 = correctAt5 / float(Y.shape[0])
    r10 = correctAt10 / float(Y.shape[0])
    print 'rate @ 1: %.4f' % r1
    print 'rate @ 5: %.4f' % r5
    print 'rate @ 10: %.4f' % r10
    return (r1, r5, r10)

def outputTxt(Y, T, answerArray, answerFilename, truthFilename):
    with open(truthFilename, 'w+') as f:
        for n in range(0, T.shape[0]):
            f.write(answerArray[T[n, 0]] + '\n')
    with open(answerFilename, 'w+') as f:
        for n in range(0, Y.shape[0]):
            f.write(answerArray[np.argmax(Y[n, :])] + '\n')

def runWups(answerFilename, truthFilename):
    w1 = calculate_wups.runAll(truthFilename, answerFilename, -1)
    w09 = calculate_wups.runAll(truthFilename, answerFilename, 0.9)
    w0 = calculate_wups.runAll(truthFilename, answerFilename, 0.0)
    print 'WUPS @ 1.0: %.4f' % w1
    print 'WUPS @ 0.9: %.4f' % w09
    print 'WUPS @ 0.0: %.4f' % w0
    return (w1, w09, w0)

def testAll(taskId, model, dataFolder, resultsFolder):
    testAnswerFile = os.path.join(resultsFolder, taskId, '%s.test.o.txt' % taskId)
    testTruthFile = os.path.join(resultsFolder, taskId, '%s.test.t.txt' % taskId)
    testDataFile = os.path.join(dataFolder, 'test.npy')
    vocabDictFile = os.path.join(dataFolder, 'vocab-dict.npy')
    vocabDict = np.load(vocabDictFile)
    testData = np.load(testDataFile)
    inputTest = testData[0]
    outputTest = nn.test(model, inputTest)
    targetTest = testData[1]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]
    print len(answerArray)
    print outputTest.shape
    outputTxt(outputTest, targetTest, answerArray, testAnswerFile, testTruthFile)
    resultsRank = calcPrecision(outputTest, targetTest)
    correct, total = calcRate(inputTest, outputTest, targetTest, questionArray)
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
