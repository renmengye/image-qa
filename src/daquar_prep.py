import numpy as np
import re
import sys
import os

import nn
import prep
import word2vec_lookup as word2vec
import word2vec_lookuptxt as word2vec_txt
import word_embedding

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
    if answer == 'one' or answer == 'two' or answer == 'three' or \
        answer == 'four' or answer == 'five' or answer == 'six' or \
        answer == 'seven' or answer == 'eight' or answer == 'nine' or \
        answer == 'ten' or answer == 'eleven' or answer == 'twelve' or \
        answer == 'thirteen' or answer == 'fourteen' or answer == 'fifteen' or \
        answer == 'sixteen' or answer == 'seventeen' or answer == 'eighteen' or \
        answer == 'nineteen' or answer == 'twenty' or answer == 'twenty-one' or \
        answer == 'twenty-two' or answer == 'twenty-three' or answer == 'twenty-four' or \
        answer == 'twenty-five' or answer == 'twenty-six' or answer == 'twenty-seven':
        return 1
    elif answer == 'red' or answer == 'orange' or answer == 'yellow' or \
        answer == 'green' or answer == 'blue' or answer == 'black' or \
        answer == 'white' or answer == 'brown' or answer == 'grey' or \
        answer == 'gray' or answer == 'purple' or answer == 'pink':
        return 2
    else:
        return 0

if __name__ == '__main__':
    """
    Usage:
    python daquar_prep.py \
        -train {train QA raw plain text file} \
        -test {test QA raw plain text file} \
        -o[utput] {output file name, including '.npz' extension} \
        [-word {output word embedding file name, including '.npz' extension, if not provided then no embedding}] \
        [-type {all/object/number/color, all types or type specific dataset}]

    Example:
    python daquar_prep.py \
        -train ../../../data/mpi-qa/qa.37.raw.train.txt \
        -test ../../../data/mpi-qa/qa.37.raw.test.txt \
        -output /ais/gobi3/u/$USER/data/daquar/reduced-37.npz \
        -type all
    """
    trainQAFilename = '/ais/gobi3/u/mren/data/daquar/qa.37.raw.train.txt'
    testQAFilename = '/ais/gobi3/u/mren/data/daquar/qa.37.raw.test.txt'
    outputFilename = '/ais/gobi3/u/mren/data/daquar/reduced-37.npz'
    buildObject = True
    buildNumber = True
    buildColor = True
    lookupWordEmbed = False
    wordEmbedFilename = '/ais/gobi3/u/$USER/data/daquar/reduced-37-word-embed.npz'

    for i, flag in enumerate(sys.argv):
        if flag == '-train':
            trainQAFilename = sys.argv[i + 1]
        elif flag == '-test':
            testQAFilename = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFilename = sys.argv[i + 1]
        elif flag == '-word':
            lookupWordEmbed = True
            wordEmbedFilename = sys.argv[i + 1]
        elif flag == '-type':
            if sys.argv[i + 1] == 'object':
                buildObject = True
                buildNumber = False
                buildColor = False
            elif sys.argv[i + 1] == 'number':
                buildObject = False
                buildNumber = True
                buildColor = False
            elif sys.argv[i + 1] == 'color':
                buildObject = False
                buildNumber = False
                buildColor = True

    print 'Train input:', trainQAFilename
    print 'Test input:', testQAFilename
    outputFolder = os.path.dirname(os.path.abspath(outputFilename))
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

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
        else:
            raise Exception('Unknown separate type')
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

    trainQID = prep.lookupQID(trainQuestions, worddict, maxlen)
    trainInput = prep.combine(np.array(trainQID, dtype=int), trainImgIds)
    trainInputSV = prep.combineSV(\
        range(1, len(trainQuestions) + 1), 
        trainImgIds)
    trainTarget = prep.lookupAnsID(trainAnswers, ansdict)
    
    validQID = prep.lookupQID(validQuestions, worddict, maxlen)
    validInput = prep.combine(np.array(validQID, dtype=int), validImgIds)
    validInputSV = prep.combineSV(\
        range(len(trainQuestions) + 1, len(trainQuestions) + len(validQuestions) + 1),
        validImgIds)
    validTarget = prep.lookupAnsID(validAnswers, ansdict)

    testQID = prep.lookupQID(testQuestions, worddict, maxlen)
    testInput = prep.combine(np.array(testQID, dtype=int), testImgIds)
    testInputSV = prep.combineSV(\
        range(len(trainQuestions) + len(validQuestions) + 1,
            len(trainQuestions) + len(validQuestions) + len(testQuestions) + 1),
        testImgIds)
    testTarget = prep.lookupAnsID(testAnswers, ansdict)

    dataset = nn.Dataset(outputFilename, 'w')
    dataset.setTrainInput(trainInput)
    dataset.setTrainTarget(trainTarget)
    dataset.setValidInput(validInput)
    dataset.setValidTarget(validTarget)
    dataset.setTestInput(testInput)
    dataset.setTestTarget(testTarget)
    dataset.set('testQuestionTypes', testQuestionTypes)
    dataset.set('questionDict', worddict)
    dataset.set('questionIdict', idict)
    dataset.set('ansDict', ansdict)
    dataset.set('ansIdict', iansdict)
    trainImgIds = []
    for i in range(1449):
        if split.has_key(i) and split[i] == 1:
            trainImgIds.append(i)
    dataset.set('trainImageIds', np.array(trainImgIds, dtype='int'))

    print 'Saving dataset...'
    dataset.save()

    # Look up word embedding
    if lookupWordEmbed:
        wordEmbed = nn.Dataset(wordEmbedFilename)
        print 'Building word embeddings...'
        w2v300QuestionEmbedding = word_embedding.getWordEmbedding(word2vec.lookup(idict))
        w2v300AnswerEmbedding = word_embedding.getWordEmbedding(word2vec.lookup(iansdict))
        cw2v300QuestionEmbedding = word_embedding.getWordEmbedding(word2vec_txt.lookup(idict))
        cw2v300AnswerEmbedding = word_embedding.getWordEmbedding(word2vec_txt.lookup(iansdict))
        cw2v500QuestionEmbedding = word_embedding.getWordEmbedding(word2vec_txt.lookup(idict))
        cw2v500AnswerEmbedding = word_embedding.getWordEmbedding(word2vec_txt.lookup(iansdict))
        wordEmbed.set('word2vecGoogleNews300Question', w2v300QuestionEmbedding)
        wordEmbed.set('word2vecGoogleNews300Answer', w2v300AnswerEmbedding)
        wordEmbed.set('word2vecCustom300Question', cw2v300QuestionEmbedding)
        wordEmbed.set('word2vecCustom300Answer', cw2v300AnswerEmbedding)
        wordEmbed.set('word2vecCustom300Question', cw2v500QuestionEmbedding)
        wordEmbed.set('word2vecCustom300Answer', cw2v500AnswerEmbedding)

    # Build baseline solution
    prep.guessBaseline(
                    testQuestions, 
                    testAnswers, 
                    testQuestionTypes, 
                    outputFolder=outputFolder, 
                    calcWups=True)
