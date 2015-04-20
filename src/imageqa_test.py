import numpy as np
import calculate_wups

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
        elif question.startswith('what is the color'):
            typ = 2
        elif question.startswith('where'):
            typ = 3
        else:
            typ = 0
        total[typ] += 1
        if A == T[n, 0]:
            correct[typ] += 1
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
    print 'rate @ 1: %.4f' % (correctAt1 / float(Y.shape[0]))
    print 'rate @ 5: %.4f' % (correctAt5 / float(Y.shape[0]))
    print 'rate @ 10: %.4f' % (correctAt10 / float(Y.shape[0]))
    return (correctAt1, correctAt5, correctAt10)

def outputTxt(Y, T, answerArray, answerFilename, truthFilename):
    # testAnswerFile = os.path.join('../results/%s' % taskId, '%s.test.o.txt' % taskId)
    # testTruthFile = os.path.join('../results/%s' % taskId, '%s.test.t.txt' % taskId)
    with open(truthFilename, 'w+') as f:
        for n in range(0, T.shape[0]):
            f.write(answerArray[T[n, 0]] + '\n')
    with open(answerFilename, 'w+') as f:
        for n in range(0, Y.shape[0]):
            f.write(answerArray[np.argmax(Y[n, :])] + '\n')

def testAll(taskId, model, dataFolder, resultsFolder):
    testAnswerFile = os.path.join(resultsFolder, taskId, '%s.test.o.txt' % taskId)
    testTruthFile = os.path.join(resultsFolder, taskId, '%s.test.t.txt' % taskId)
    testDataFile = os.path.join(dataFolder, 'test.npy')
    vocabDictFile = os.path.join(dataFolder, 'vocab-dict.npy')
    vocabDict = np.load(vocabDictFile)
    testData = np.load(testDataFile)
    inputTest = testData[0]
    outputTest = nn.test(model, TX)
    targetTest = testData[1]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]
    outputTxt(outputTest, targetTest, answerArray, testAnswerFile, testTruthFile)
    resultsRank = calcPrecision(outputTest, targetTest)
    correct, total = calcRate(inputTest, outputTest, targetTest, questionArray)
    resultsCategory = (correct, total, correct / total.astype(float))
    resultsFile = os.path.join(resultsFolder, taskId, 'result.txt')
    with open(resultsFile, 'w') as f:
        f.write('rate @ 1: %.4f\n' % (resultsRank[0] / float(outputTest.shape[0])))
        f.write('rate @ 5: %.4f\n' % (resultsRank[1] / float(outputTest.shape[0])))
        f.write('rate @ 10: %.4f\n' % (resultsRank[2] / float(outputTest.shape[0])))
        f.write('object: %.4f\n' % resultsCategory[2][0])
        f.write('number: %.4f\n' % resultsCategory[2][1])
        f.write('color: %.4f\n' % resultsCategory[2][2])
        f.write('scene: %.4f\n' % resultsCategory[2][3])
        f.write('WUPS -1: %.4f\n' % calculate_wups.runAll(testTruthFile, testAnswerFile, -1))
        f.write('WUPS 0.9: %.4f\n' % calculate_wups.runAll(testTruthFile, testAnswerFile, 0.9))
        f.write('WUPS 0.0: %.4f\n' % calculate_wups.runAll(testTruthFile, testAnswerFile, 0.0))