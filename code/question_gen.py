import os
import sys
import numpy as np
import imgword_prep

factFolder = '../data/mpi-qa/facts-37-human_seg'
split = []
def getTrainTestSplit():
    trainIdFilename = '../data/mpi-qa/train.txt'
    testIdFilename = '../data/mpi-qa/test.txt'
    trainSet = {}
    testSet = {}
    result = [1]

    numMax = 0
    with open(trainIdFilename) as f:
        lines = f.readlines()

    for line in lines:
        n = int(line[5:])
        if n > numMax: numMax = n
        trainSet[n] = None

    with open(testIdFilename) as f:
        lines = f.readlines()

    for line in lines:
        n = int(line[5:])
        if n > numMax: numMax = n
        testSet[n] = None

    for i in range(1, numMax + 1):
        if trainSet.has_key(i):
            result.append(1)
        elif testSet.has_key(i):
            result.append(0)
        else:
            raise Exception('haha')

    return result

def parseFact(str):
    parts = str.split(',')
    obj = parts[0].split('(')[0]
    img = parts[1].split('\'')
    imgno = int(img[1][5:])
    color = parts[2].split('\'')[1]
    return imgno, obj, color

def genSingleObjColor(database, singleWord=False):
    qaTrainList = []
    qaTestList = []
    colorTrainStats = {}
    colorQA = {}
    np.random.seed(1)
    sum = 0
    for i in range(1, len(database) + 1):
        if database.has_key(i):
            for obj in database[i].iteritems():
                if len(obj[1]) == 1:
                    objname = obj[0]
                    objcolor = obj[1][0]
                    if singleWord:
                        pair = ('%s in the image%d ?\n' % (objname, i), objcolor + '\n', i)
                    else:
                        pair = ('what is the color of the %s in the image%d ?\n' % (objname, i), objcolor + '\n', i)
                    if colorTrainStats.has_key(objcolor):
                        colorTrainStats[objcolor] += 1
                        colorQA[objcolor].append(pair)
                    else:
                        colorTrainStats[objcolor] = 1
                        colorQA[objcolor] = [pair]
                    sum += 1
    print 'Total color stats:'
    print colorTrainStats
    for item in colorTrainStats.iteritems():
        print item[0] + ': ',
        print item[1] / float(sum)

    # Select color stage
    colorTrainStats = {}
    colorTestStats = {}
    sumTrain = 0
    sumTest = 0
    for item in colorQA.iteritems():
        N = len(item[1])
        ind = np.random.permutation(N)
        count = 0
        colorTrainStats[item[0]] = 0
        colorTestStats[item[0]] = 0
        for i in ind:
            pair = item[1][i]
            if split[pair[2]] == 1:
                qaTrainList.append(pair[0])
                qaTrainList.append(pair[1])
                colorTrainStats[item[0]] += 1
                sumTrain += 1
            else:
                qaTestList.append(pair[0])
                qaTestList.append(pair[1])
                colorTestStats[item[0]] += 1
                sumTest += 1
            if count > 400: break # At most generate 400 questions for a single color to avoid uneven distribution.
            count += 1

    print 'Train color stats:'
    print colorTrainStats
    for item in colorTrainStats.iteritems():
        print item[0] + ': ',
        print item[1] / float(sumTrain)
    print 'Test color stats:'
    print colorTestStats
    for item in colorTestStats.iteritems():
        print item[0] + ': ',
        print item[1] / float(sumTest)
    return qaTrainList, qaTestList

def genCounting(database, singleWord=False):
    qaTrainList = []
    qaTestList = []
    numberStats = {}
    numberQA = {}
    np.random.seed(1)
    sum = 0
    for i in range(1, len(database) + 1):
        if database.has_key(i):
            for obj in database[i].iteritems():
                objname = obj[0]
                objnum = len(obj[1])
                objnums = imgword_prep.escapeNumber(str(objnum))
                if singleWord:
                    pair = ('%s in the image%d ?\n' % (objname, i), objnums  + '\n', i)
                else:
                    pair = ('how many %s in the image%d ?\n' % (objname, i), objnums  + '\n', i)
                if numberStats.has_key(objnum):
                    numberStats[objnum] += 1
                    numberQA[objnum].append(pair)
                else:
                    numberStats[objnum] = 1
                    numberQA[objnum] = [pair]
                sum += 1
    print 'Total number stats:'
    print numberStats
    for item in numberStats.iteritems():
        print str(item[0]) + ': ',
        print item[1] / float(sum)

    # Select number stage
    numberTrainStats = {}
    numberTestStats = {}
    sumTrain = 0
    sumTest = 0
    for item in numberQA.iteritems():
        N = len(item[1])
        ind = np.random.permutation(N)
        count = 0
        numberTrainStats[item[0]] = 0
        numberTestStats[item[0]] = 0
        for i in ind:
            pair = item[1][i]
            if split[pair[2]] == 1:
                qaTrainList.append(pair[0])
                qaTrainList.append(pair[1])
                numberTrainStats[item[0]] += 1
                sumTrain += 1
            else:
                qaTestList.append(pair[0])
                qaTestList.append(pair[1])
                numberTestStats[item[0]] += 1
                sumTest += 1
            if count > 400: break # At most generate 400 questions for a single color to avoid uneven distribution.
            count += 1

    print 'Train number stats:'
    print numberTrainStats
    for item in numberTrainStats.iteritems():
        print str(item[0]) + ': ',
        print item[1] / float(sumTrain)
    print 'Test number stats:'
    print numberTestStats
    for item in numberTestStats.iteritems():
        print str(item[0]) + ': ',
        print item[1] / float(sumTest)
    return qaTrainList, qaTestList

def writeToFile(trainTestList, trainFile, testFile):
    with open(trainFile, 'w+') as f:
        f.writelines(trainTestList[0])
    with open(testFile, 'w+') as f:
        f.writelines(trainTestList[1])

if __name__ == '__main__':
    """
    Usage: question_gen {type} {output folder}
    type:
        1. color: What is the color of xxx in the image### ?
        2. number: How many xxx in the image### ?
        3. all: existing human QA + color + number as train set. Human QA test as test.
    """
    split = getTrainTestSplit()
    factFilenames = next(os.walk(factFolder))[2]
    imgDatabase = {}
    qtype = 'color'
    if len(sys.argv) > 2:
        qtype = sys.argv[1]
        outputFolder = sys.argv[2]
        outputTrainFile = os.path.join(outputFolder, '%s.train.txt' % qtype)
        outputTestFile = os.path.join(outputFolder, '%s.test.txt' % qtype)
    else:
        outputTrainFile = '../data/synth-qa/color2/color.train.txt'
        outputTestFile = '../data/synth-qa/color2/color.test.txt'
    for factFilename in factFilenames:
        with open(os.path.join(factFolder,factFilename)) as f:
            lines = f.readlines()
        for line in lines:
            if line[0] != '%':
                imgno, obj, color = parseFact(line)
                if imgDatabase.has_key(imgno):
                    if imgDatabase[imgno].has_key(obj):
                        imgDatabase[imgno][obj].append(color)
                    else:
                        imgDatabase[imgno][obj] = [color]
                else:
                    imgDatabase[imgno] = {obj: [color]}
    if qtype == 'color':
        writeToFile(genSingleObjColor(imgDatabase, singleWord=True), outputTrainFile, outputTestFile)
    elif qtype == 'number':
        writeToFile(genCounting(imgDatabase, singleWord=True), outputTrainFile, outputTestFile)
    elif qtype == 'all':
        counting = genCounting(imgDatabase, singleWord=False)
        color = genSingleObjColor(imgDatabase, singleWord=False)
        with open('../data/mpi-qa/qa.37.raw.train.txt') as f:
            humanTrain = f.readlines()
        with open('../data/mpi-qa/qa.37.raw.test.txt') as f:
            humanTest = f.readlines()
        humanTrain.extend(counting[0])
        humanTrain.extend(color[0])
        writeToFile((humanTrain, humanTest), outputTrainFile, outputTestFile)

    print 'haha'