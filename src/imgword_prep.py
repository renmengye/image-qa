import numpy as np
import re
import sys
import os

def escapeNumber(line):
    line = re.sub('^21$', 'twenty_one', line)
    line = re.sub('^22$', 'twenty_two', line)
    line = re.sub('^23$', 'twenty_three', line)
    line = re.sub('^24$', 'twenty_four', line)
    line = re.sub('^25$', 'twenty_five', line)
    line = re.sub('^26$', 'twenty_six', line)
    line = re.sub('^27$', 'twenty_seven', line)
    line = re.sub('^28$', 'twenty_eight', line)
    line = re.sub('^29$', 'twenty_nine', line)
    line = re.sub('^30$', 'thirty', line)
    line = re.sub('^11$', 'eleven', line)
    line = re.sub('^12$', 'twelve', line)
    line = re.sub('^13$', 'thirteen', line)
    line = re.sub('^14$', 'fourteen', line)
    line = re.sub('^15$', 'fifteen', line)
    line = re.sub('^16$', 'sixteen', line)
    line = re.sub('^17$', 'seventeen', line)
    line = re.sub('^18$', 'eighteen', line)
    line = re.sub('^19$', 'nineteen', line)
    line = re.sub('^20$', 'twenty', line)
    line = re.sub('^10$', 'ten', line)
    line = re.sub('^0$', 'zero', line)
    line = re.sub('^1$', 'one', line)
    line = re.sub('^2$', 'two', line)
    line = re.sub('^3$', 'three', line)
    line = re.sub('^4$', 'four', line)
    line = re.sub('^5$', 'five', line)
    line = re.sub('^6$', 'six', line)
    line = re.sub('^7$', 'seven', line)
    line = re.sub('^8$', 'eight', line)
    line = re.sub('^9$', 'nine', line)

    return line

def buildDict(lines, keystart):
    # From word to number.
    word_dict = {}
    # From number to word, numbers need to minus one to convert to list indices.
    word_array = []
    # Word frequency
    word_freq = []
    # Key is 1-based, 0 is reserved for sentence end.
    key = keystart
    for i in range(0, len(lines)):
        line = lines[i].replace(',', '')
        words = line.split(' ')
        for j in range(0, len(words)):
            if not word_dict.has_key(words[j]):
                word_dict[words[j]] = key
                word_array.append(words[j])
                word_freq.append(1)
                key += 1
            else:
                k = word_dict[words[j]]
                word_freq[k - 1] += 1

    return  word_dict, word_array

def buildInputTarget(question, answer, numEx, lineMax, wordDict, answerDict):
    input_ = np.zeros((numEx, lineMax, 1), dtype=int)
    target_ = np.zeros((numEx, 1), dtype=int)
    for i in range(0, numEx):
        words = question[i].split(' ')
        target_[i, 0] = answerDict[answer[i]]
        for t in range(0, len(words)):
            input_[i, t, 0] = wordDict[words[t]]
    return input_, target_

if __name__ == '__main__':

    """
    Usage: imgword_prep.py -train trainQAFile -test testQAFile -o outputFolder
    """
    # Image ID
    imgIds = []

    if len(sys.argv) > 6:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == '-train':
                trainQAFilename = sys.argv[i + 1]
            elif sys.argv[i] == '-test':
                testQAFilename = sys.argv[i + 1]
            elif sys.argv[i] == '-o':
                outputFolder = sys.argv[i + 1]
    else:
        trainQAFilename = '../data/mpi-qa/qa.37.raw.train.txt'
        testQAFilename = '../data/mpi-qa/qa.37.raw.test.txt'
        outputFolder = '../data/imgword'

    with open(trainQAFilename) as f:
        lines = f.readlines()

    newlines = []
    # Purge multi answer question for now.
    for i in range(len(lines) / 2):
        if ',' in lines[2 * i + 1]:
            continue
        newlines.append(lines[2 * i])
        newlines.append(lines[2 * i + 1])
    numTrain = len(newlines) / 2

    with open(testQAFilename) as f:
        lines = f.readlines()
    # Purge multi answer question for now.
    for i in range(len(lines) / 2):
        if ',' in lines[2 * i + 1]:
            continue
        newlines.append(lines[2 * i])
        newlines.append(lines[2 * i + 1])
    numTest = len(newlines) / 2 - numTrain

    imgIdPattern = re.compile('image')
    pureQ = []
    pureA = []
    lineMax = 0
    for i in range(0, len(newlines) / 2):
        n = i * 2
        match = re.search('image(\d+)', newlines[n])
        number = int((re.search('\d+', match.group())).group())
        line = newlines[n]
        line = re.sub(' in the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in this image(\d+)( \?\s)?', '' , line)
        line = re.sub(' on the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' of the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in image(\d+)( \?\s)?', '' , line)
        line = re.sub(' image(\d+)( \?\s)?', '' , line)
        pureQ.append(line)
        answer = escapeNumber(re.sub('\s$', '', newlines[n + 1]))
        pureA.append(answer)
        imgIds.append(number)
        l = len(pureQ[i].split())
        if l > lineMax: lineMax = l

    questionDict, questionVocab = buildDict(pureQ, keystart=1)
    answerDict, answerVocab = buildDict(pureA, keystart=0)
    trainInput, trainTarget = buildInputTarget(pureQ[:numTrain], pureA[:numTrain], numTrain, lineMax, questionDict, answerDict)
    trainInputAll = np.concatenate((np.reshape(imgIds[:numTrain], (numTrain, 1, 1)), trainInput), axis=1)
    testInput, testTarget = buildInputTarget(pureQ[numTrain:], pureA[numTrain:], numTest, lineMax, questionDict, answerDict)
    testInputAll = np.concatenate((np.reshape(imgIds[numTrain:], (numTest, 1, 1)), testInput), axis=1)
    trainData = np.array((trainInputAll, trainTarget, 0), dtype=object)
    testData = np.array((testInputAll, testTarget, 0), dtype=object)
    vocabDict = np.array((questionDict, questionVocab, answerDict, answerVocab, 0), dtype=object)
    np.save(os.path.join(outputFolder, 'train-37.npy'), trainData)
    np.save(os.path.join(outputFolder, 'test-37.npy'), testData)
    np.save(os.path.join(outputFolder, 'vocab-dict.npy'), vocabDict)

    trainInputAll = np.concatenate((trainInput, np.reshape(imgIds[:numTrain], (numTrain, 1, 1))), axis=1)
    testInputAll = np.concatenate((testInput, np.reshape(imgIds[numTrain:], (numTest, 1, 1))), axis=1)
    trainData = np.array((trainInputAll, trainTarget, 0), dtype=object)
    testData = np.array((testInputAll, testTarget, 0), dtype=object)
    np.save(os.path.join(outputFolder, 'train-last-37.npy'), trainData)
    np.save(os.path.join(outputFolder, 'test-last-37.npy'), testData)

    with open(os.path.join(outputFolder, 'question_vocabs.txt'), 'w+') as f:
        for word in questionVocab:
            f.write(word + '\n')

    with open(os.path.join(outputFolder, 'answer_vocabs.txt'), 'w+') as f:
        for word in answerVocab:
            f.write(word + '\n')

    #imgW = np.loadtxt('../data/nyu-depth-v2/oxford_hidden7.txt')
    #np.save(os.path.join(outputFolder, 'oxford-feat.npy'), imgW.transpose())
    print 'haha'