import os

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
    for i in range(1, len(database) + 1):
        if database.has_key(i):
            for obj in database[i].iteritems():
                if len(obj[1]) == 1:
                    if split[i] == 1:
                        qaTrainList.append('what is the color of the %s in the image%d ?\n' % (obj[0], i))
                        qaTrainList.append(obj[1][0] + '\n')
                    else:
                        qaTestList.append('what is the color of the %s in the image%d ?\n' % (obj[0], i))
                        qaTestList.append(obj[1][0] + '\n')
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