import numpy as np
import re
import sys
import os
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
        # if count < len(split) / 10:
        #     split[i] = 0
        # else:
        #     break
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
    lineMax = 0
    for i in range(0, len(lines) / 2):
        n = i * 2
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
    return (questions, answers, imgIds)

def buildDict(lines, keystart, pr=False):
    # From word to number.
    word_dict = {}
    # From number to word, numbers need to minus one to convert to list indices.
    word_array = []
    # Word frequency
    word_freq = []
    # if key is 1-based, then 0 is reserved for sentence end.
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
                word_freq[k - keystart] += 1
    word_dict['UNK'] = key
    word_array.append('UNK')

    sorted_x = sorted(range(len(word_freq)), key=lambda k: word_freq[k], reverse=True)
    if pr:
        for x in sorted_x:
            print word_array[x], word_freq[x],
        #print sorted_x
        print 'Dictionary length', len(word_dict)
    return  word_dict, word_array

def lookupAnsID(answers, ansdict):
    ansids = []
    for ans in answers:
        if ansdict.has_key(ans):
            ansids.append(ansdict[ans])
        else:
            ansids.append(ansdict['UNK'])
    return np.array(ansids, dtype=int).reshape(len(ansids), 1)

def lookupQID(questions, worddict):
    wordslist = []
    maxlen = 27
    for q in questions:
        words = q.split(' ')
        wordslist.append(words)
        # if len(words) > maxlen:
        #     maxlen = len(words)
    result = np.zeros((len(questions), maxlen, 1), dtype=int)
    for i,words in enumerate(wordslist):
        for j,w in enumerate(words):
            if worddict.has_key(w):
                result[i, j, 0] = worddict[w]
            else:
                result[i, j, 0] = worddict['UNK']
    return result

def combine(wordids, imgids):
    return np.concatenate(\
        (np.array(imgids).reshape(len(imgids), 1, 1), \
        wordids), axis=1)

def combineAttention(wordids, imgids):
    imgid_t = []
    for n in range(0, wordids.shape[0]):
        for t in range(0, wordids.shape[1]):
            if wordids[n, t, 0] == 0:
                imgid_t.append(0)
            else:
                imgid_t.append(imgids[n])

    return np.concatenate(
            (np.array(imgid_t).reshape(len(imgids), wordids.shape[1], 1),
            wordids), axis=-1)

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
    if len(sys.argv) > 6:
        for i in range(1, len(sys.argv)):
            if sys.argv[i] == '-train':
                trainQAFilename = sys.argv[i + 1]
            elif sys.argv[i] == '-test':
                testQAFilename = sys.argv[i + 1]
            elif sys.argv[i] == '-o':
                outputFolder = sys.argv[i + 1]

    # Read train file.
    with open(trainQAFilename) as f:
        lines = f.readlines()

    (questions, answers, imgids) = extractQA(lines)
    split = trainValidSplit(imgids)
    trainQuestions, validQuestions = dataSplit(questions, imgids, split)
    trainAnswers, validAnswers = dataSplit(answers, imgids, split)
    trainImgIds, validImgIds = dataSplit(imgids, imgids, split)

    print len(trainQuestions) + len(validQuestions)

    # Read test file.
    with open(testQAFilename) as f:
        lines = f.readlines()

    (testQuestions, testAnswers, testImgIds) = extractQA(lines)

    print len(testQuestions)
    # Build a dictionary only for training questions.
    worddict, idict = buildDict(trainQuestions, 1, pr=False)
    ansdict, iansdict = buildDict(trainAnswers, 0, pr=True)
    validAnsDict, validIAnsDict = buildDict(validAnswers, 0, pr=True)
    testAnsDict, testIAnsDict = buildDict(testAnswers, 1, pr=True)

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

    trainInput = combine(\
        lookupQID(trainQuestions, worddict), trainImgIds)
    trainTarget = lookupAnsID(trainAnswers, ansdict)
    validInput = combine(\
        lookupQID(validQuestions, worddict), validImgIds)
    validTarget = lookupAnsID(validAnswers, ansdict)
    testInput = combine(\
        lookupQID(testQuestions, worddict), testImgIds)
    testTarget = lookupAnsID(testAnswers, ansdict)

    worddict_all, idict_all = buildDict(questions, 1)
    ansdict_all, iansdict_all = buildDict(answers, 0, pr=True)

    allInput = combine(\
        lookupQID(questions, worddict), imgids)
    allTarget = lookupAnsID(answers, ansdict)

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
        os.path.join(outputFolder, 'all.npy'),\
        np.array((allInput, allTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'vocab-dict.npy'),\
        np.array((worddict, idict, 
            ansdict, iansdict, 0), dtype=object))
    np.save(\
        os.path.join(outputFolder, 'vocab-dict-all.npy'),\
        np.array((worddict_all, idict_all, 
            ansdict_all, iansdict_all, 0), dtype=object))

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

    # Build baseline solution
    colorAnswer = 'white'
    numberAnswer = 'two'
    objectAnswer = 'table'

    baseline = []
    baselineCorrect = np.zeros(3)
    baselineTotal = np.zeros(3)
    for n in range(0, len(testQuestions)):
        if testQuestionTypes[n] == 0:
            baseline.append(objectAnswer)
            if testAnswers[n] == objectAnswer:
                baselineCorrect[0] += 1
            baselineTotal[0] += 1
        elif testQuestionTypes[n] == 1:
            baseline.append(numberAnswer)
            if testAnswers[n] == numberAnswer:
                baselineCorrect[1] += 1
            baselineTotal[1] += 1
        elif testQuestionTypes[n] == 2:
            baseline.append(colorAnswer)
            if testAnswers[n] == colorAnswer:
                baselineCorrect[2] += 1
            baselineTotal[2] += 1
    baselineRate = baselineCorrect / baselineTotal.astype('float')
    print 'Baseline rate: %.4f' % (np.sum(baselineCorrect) / np.sum(baselineTotal).astype('float'))
    print 'Baseline object: %.4f' % baselineRate[0]
    print 'Baseline number: %.4f' % baselineRate[1]
    print 'Baseline color: %.4f' % baselineRate[2]

    baselineFilename = os.path.join(outputFolder, 'baseline.txt')
    groundTruthFilename = os.path.join(outputFolder, 'ground_truth.txt')
    with open(baselineFilename, 'w+') as f:
        for answer in baseline:
            f.write(answer + '\n')
    with open(groundTruthFilename, 'w+') as f:
        for answer in testAnswers:
            f.write(answer + '\n')

    wups = np.zeros(3)
    for i, thresh in enumerate([-1, 0.9, 0.0]):
        wups[i] = calculate_wups.runAll(groundTruthFilename, baselineFilename, thresh)
    print 'Baseline WUPS -1: %.4f' % wups[0]
    print 'Baseline WUPS 0.9: %.4f' % wups[1]
    print 'Baseline WUPS 0.0: %.4f' % wups[2]
