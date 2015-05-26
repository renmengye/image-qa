import numpy as np
import re
import sys
import os

import prep
import calculate_wups

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

def trainValidSplit(imgids):
    split = {}
    for i in imgids:
        split[i] = 1
    count = 0
    seed = 1
    random = np.random.RandomState(seed)
    print 'Split', split
    for i in split.keys():
        if random.uniform(0, 1, (1)) < 0.1:
            split[i] = 0
        count += 1
    print split
    return split

def dataSplit(data, imgids, split):
    td = []
    vd = []
    for (d, i) in zip(data, imgids):
        if split[i] == 0:
            vd.append(d)
        else:
            td.append(d)
    return (td, vd)

def extractQA(lines):
    questions = []
    answers = []
    imgIds = []
    discard = 0
    preserved = 0
    total = 0
    lineMax = 0
    for i in range(0, len(lines) / 2):
        n = i * 2
        total += 1
        if ',' in lines[n + 1]:
            # No multiple words answer for now.
            discard += 1
            continue
        preserved += 1
        match = re.search('image(\d+)', lines[n])
        number = int((re.search('\d+', match.group())).group())
        line = lines[n]
        line = re.sub(' in the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in this image(\d+)( \?\s)?', '' , line)
        line = re.sub(' on the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' of the image(\d+)( \?\s)?', '' , line)
        line = re.sub(' in image(\d+)( \?\s)?', '' , line)
        line = re.sub(' image(\d+)( \?\s)?', '' , line)
        question = line
        questions.append(question)
        answer = escapeNumber(re.sub('\s$', '', lines[n + 1]))
        answers.append(answer)
        imgIds.append(number)
    print 'Discard', discard
    print 'Preserved', preserved
    print 'Total', total
    return (questions, answers, imgIds)

def getQuestionType(answer):
    if answer == 'one' or answer == 'two' or answer == 'three' or\
        answer == 'four' or answer == 'five' or answer == 'six' or\
        answer == 'seven' or answer == 'eight' or answer == 'nine' or\
        answer == 'ten' or answer == 'eleven' or answer == 'twelve' or\
        answer == 'thirteen' or answer == 'fourteen' or answer == 'fifteen' or\
        answer == 'sixteen' or answer == 'seventeen' or answer == 'eighteen' or\
        answer == 'nineteen' or answer == 'twenty' or answer == 'twenty-one' or\
        answer == 'twenty-two' or answer == 'twenty-three' or answer == 'twenty-four' or\
        answer == 'twenty-five' or answer == 'twenty-six' or answer == 'twenty-seven':
        return 1
    elif answer == 'red' or answer == 'orange' or answer == 'yellow' or\
        answer == 'green' or answer == 'blue' or answer == 'black' or\
        answer == 'white' or answer == 'brown' or answer == 'grey' or\
        answer == 'gray' or answer == 'purple' or answer == 'pink':
        return 2
    else:
        return 0

if __name__ == '__main__':
    """
    Usage: imgword_prep.py -train trainQAFile -test testQAFile -o outputFolder
    """
    trainQAFilename = '../../../data/mpi-qa/qa.37.raw.train.txt'
    testQAFilename = '../../../data/mpi-qa/qa.37.raw.test.txt'
    outputFolder = '../data/daquar-37'
    outputFolderSV = '../data/daquar-37-sv'
    buildObject = True
    buildNumber = True
    buildColor = True

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-train':
            trainQAFilename = sys.argv[i + 1]
        elif sys.argv[i] == '-test':
            testQAFilename = sys.argv[i + 1]
        elif sys.argv[i] == '-o' or sys.argv[i] == '-output':
            outputFolder = sys.argv[i + 1]
            outputFolderSV = outputFolder + '-sv'
        elif sys.argv[i] == '-object':
            buildObject = True
            buildNumber = False
            buildColor = False
        elif sys.argv[i] == '-number':
            buildObject = False
            buildNumber = True
            buildColor = False
        elif sys.argv[i] == '-color':
            buildObject = False
            buildNumber = False
            buildColor = True

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not os.path.exists(outputFolderSV):
        os.makedirs(outputFolderSV)

    # Read train file.
    with open(trainQAFilename) as f:
        lines = f.readlines()

    (questions, answers, imgids) = extractQA(lines)
    maxlen = prep.findMaxlen(questions)

    split = trainValidSplit(imgids)
    trainQuestions, validQuestions = dataSplit(questions, imgids, split)
    trainAnswers, validAnswers = dataSplit(answers, imgids, split)
    trainImgIds, validImgIds = dataSplit(imgids, imgids, split)

    print len(trainQuestions) + len(validQuestions)

    # Read test file.
    with open(testQAFilename) as f:
        lines = f.readlines()

    (testQuestions, testAnswers, testImgIds) = extractQA(lines)
    maxlen2 = prep.findMaxlen(testQuestions)
    maxlen = max(maxlen, maxlen2)
    print 'Maxlen final:', maxlen

    print len(testQuestions)

    # Build a dictionary only for training questions.
    worddict, idict, _ = \
        prep.buildDict(trainQuestions, keystart=1, pr=False)
    ansdict, iansdict, _ = \
        prep.buildDict(trainAnswers, keystart=0, pr=True)
    validAnsDict, validIAnsDict, _ = \
        prep.buildDict(validAnswers, keystart=0, pr=True)
    testAnsDict, testIAnsDict, _ = \
        prep.buildDict(testAnswers, keystart=1, pr=True)

    trainQuestionTypes = np.zeros(len(trainQuestions), dtype=int)
    trainCount = np.zeros(3)
    validQuestionTypes = np.zeros(len(validQuestions), dtype=int)
    validCount = np.zeros(3)
    testQuestionTypes = np.zeros(len(testQuestions), dtype=int)
    testCount = np.zeros(3)
    for i in range(len(trainQuestions)):
        trainQuestionTypes[i] = getQuestionType(trainAnswers[i])
        trainCount[trainQuestionTypes[i]] += 1
    for i in range(len(validQuestions)):
        validQuestionTypes[i] = getQuestionType(validAnswers[i])
        validCount[validQuestionTypes[i]] += 1
    for i in range(len(testQuestions)):
        testQuestionTypes[i] = getQuestionType(testAnswers[i])
        testCount[testQuestionTypes[i]] += 1
    
    print 'Train Questions After Trunk: ', len(trainQuestions)
    print 'Train Question Dist: ', trainCount
    print 'Train Question Dist: ', \
            trainCount / float(len(trainQuestions))

    print 'Valid Questions After Trunk: ', len(validQuestions)
    print 'Valid Question Dist: ', validCount
    print 'Valid Question Dist: ', \
            validCount / float(len(validQuestions))
    trainValidQuestionsLen = len(trainQuestions) + len(validQuestions)
    print 'Train+Valid questions: ', trainValidQuestionsLen
    print 'Train+Valid Dist: ', trainCount + validCount
    print 'Trian+Valid Dist: ', \
            (trainCount + validCount) / float(trainValidQuestionsLen)

    print 'Test Questions After Trunk: ', len(testQuestions)
    print 'Test Question Dist: ', testCount
    print 'Test Question Dist: ', testCount / float(len(testQuestions))

    # Convert to numpy array
    trainQuestions = np.array(trainQuestions, dtype='object')
    trainAnswers = np.array(trainAnswers, dtype='object')
    trainImgIds = np.array(trainImgIds, dtype='object')
    validQuestions = np.array(validQuestions, dtype='object')
    validAnswers = np.array(validAnswers, dtype='object')
    validImgIds = np.array(validImgIds, dtype='object')
    testQuestions = np.array(testQuestions, dtype='object')
    testAnswers = np.array(testAnswers, dtype='object')
    testImgIds = np.array(testImgIds, dtype='object')

    # Filter question types
    if not (buildObject and buildNumber and buildColor):
        # Now only build one type!
        if buildObject:
            typ = 0
        elif buildNumber:
            typ = 1
        elif buildColor:
            typ = 2
        trainQuestions = trainQuestions[trainQuestionTypes == typ]
        trainAnswers = trainAnswers[trainQuestionTypes == typ]
        trainImgIds = trainImgIds[trainQuestionTypes == typ]
        trainQuestionTypes = trainQuestionTypes[trainQuestionTypes == typ]
        validQuestions = validQuestions[validQuestionTypes == typ]
        validAnswers = validAnswers[validQuestionTypes == typ]
        validImgIds = validImgIds[validQuestionTypes == typ]
        validQuestionTypes = validQuestionTypes[validQuestionTypes == typ]
        testQuestions = testQuestions[testQuestionTypes == typ]
        testAnswers = testAnswers[testQuestionTypes == typ]
        testImgIds = testImgIds[testQuestionTypes == typ]
        testQuestionTypes = testQuestionTypes[testQuestionTypes == typ]

    trainInput = prep.combine(\
        prep.lookupQID(trainQuestions, worddict, maxlen), trainImgIds)
    trainInputSV = prep.combineSV(\
        range(len(trainQuestions)), 
        trainImgIds)
    trainTarget = prep.lookupAnsID(trainAnswers, ansdict)
    validInput = prep.combine(\
        prep.lookupQID(validQuestions, worddict, maxlen), validImgIds)
    validInputSV = prep.combineSV(\
        range(len(trainQuestions), len(trainQuestions) + len(validQuestions)),
        validImgIds)
    validTarget = prep.lookupAnsID(validAnswers, ansdict)
    testInput = prep.combine(\
        prep.lookupQID(testQuestions, worddict, maxlen), testImgIds)
    testInputSV = prep.combineSV(\
        range(len(trainQuestions) + len(validQuestions),
            len(trainQuestions) + len(validQuestions) + len(testQuestions)),
        testImgIds)
    testTarget = prep.lookupAnsID(testAnswers, ansdict)

    np.save(\
        os.path.join(outputFolder, 'train.npy'),\
        np.array((trainInput, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'valid.npy'),\
        np.array((validInput, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test.npy'),\
        np.array((testInput, testTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test-qtype.npy'),\
        testQuestionTypes)
    np.save(\
        os.path.join(outputFolder, 'vocab-dict.npy'),\
        np.array((worddict, idict, 
            ansdict, iansdict, 0), dtype=object))

    np.save(\
        os.path.join(outputFolderSV, 'train.npy'),\
        np.array((trainInputSV, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolderSV, 'valid.npy'),\
        np.array((validInputSV, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolderSV, 'test.npy'),\
        np.array((testInputSV, testTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolderSV, 'test-qtype.npy'),\
        testQuestionTypes)
    np.save(\
        os.path.join(outputFolderSV, 'vocab-dict.npy'),\
        np.array((worddict, idict, 
            ansdict, iansdict, 0), dtype=object))

    with open(os.path.join(outputFolder, 'question_vocabs.txt'), 'w+') as f:
        for word in idict:
            f.write(word + '\n')

    with open(os.path.join(outputFolder, 'answer_vocabs.txt'), 'w+') as f:
        for word in iansdict:
            f.write(word + '\n')

    trainImgIds = []
    for i in range(1449):
        if split.has_key(i) and split[i] == 1:
            trainImgIds.append(i)
    with open(os.path.join(outputFolder, 'train_imgids.txt'), 'w+') as f:
        for i in trainImgIds:
            f.write(str(i) + '\n')
    with open(os.path.join(outputFolderSV, 'train_imgids.txt'), 'w+') as f:
        for i in trainImgIds:
            f.write(str(i) + '\n')

    with open(os.path.join(outputFolder, 'questions.txt'), 'w+') as f:
        for q in trainQuestions:
            f.write(q + '\n')
        for q in validQuestions:
            f.write(q + '\n')
        for q in testQuestions:
            f.write(q + '\n')

    # Build baseline solution
    prep.guessBaseline(
                    testQuestions, 
                    testAnswers, 
                    testQuestionTypes, 
                    outputFolder=outputFolder, 
                    calcWups=True)