import os
import numpy as np

factFolder = '../data/mpi-qa/facts-37-human_seg'
outputTrainFile = '../data/synth-qa/color/color.train.txt'
outputTestFile = '../data/synth-qa/color/color.test.txt'
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

def genSingleObjColor(database):
    qaTrainList = []
    qaTestList = []
    colorTrainStats = {}
    colorTrainQA = {}
    maxColorGet = 200 # At most generate 200 questions for a single color to avoid uneven distribution.
    sum = 0
    for i in range(1, len(database) + 1):
        if database.has_key(i):
            for obj in database[i].iteritems():
                if len(obj[1]) == 1:
                    objname = obj[0]
                    objcolor = obj[1][0]
                    pair = ('what is the color of the %s in the image%d ?\n' % (objname, i), objcolor + '\n', i)
                    if colorTrainStats.has_key(objcolor):
                        colorTrainStats[objcolor] += 1
                        colorTrainQA[objcolor].append(pair)
                    else:
                        colorTrainStats[objcolor] = 1
                        colorTrainQA[objcolor] = [pair]
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
    for item in colorTrainQA.iteritems():
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
            if count > 400: break
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

if __name__ == '__main__':
    split = getTrainTestSplit()
    factFilenames = next(os.walk(factFolder))[2]
    imgDatabase = {}
    for factFilename in factFilenames:
        with open(os.path.join(factFolder,factFilename)) as f:
            lines = f.readlines()
        for line in lines:
            if line[0] != '%':
                imgno, obj, color = parseFact(line)
                if imgDatabase.has_key(imgno):
                    if imgDatabase.has_key(obj):
                        imgDatabase[imgno][obj].append(color)
                    else:
                        imgDatabase[imgno][obj] = [color]
                else:
                    imgDatabase[imgno] = {obj: [color]}
    qaColorTrain, qaColorTest = genSingleObjColor(imgDatabase)
    with open(outputTrainFile, 'w+') as f:
        f.writelines(qaColorTrain)
    with open(outputTestFile, 'w+') as f:
        f.writelines(qaColorTest)

    print 'haha'