import re
import os
import cPickle as pkl
import numpy as np
import operator
import h5py

imgidTrainFilename = '../../../data/mscoco/train/image_list.txt'
imgidValidFilename = '../../../data/mscoco/valid/image_list.txt'
qaTrainFilename = '../../../data/mscoco/train/qa.pkl'
qaValidFilename = '../../../data/mscoco/valid/qa.pkl'
outputFolder = '../data/cocoqa-toy/'
imgHidFeatTrainFilename = '/ais/gobi3/u/mren/data/mscoco/hidden_oxford_train.h5'
imgHidFeatValidFilename = '/ais/gobi3/u/mren/data/mscoco/hidden_oxford_valid.h5'
imgHidFeatOutFilename = '../ais/gobi3/u/mren/data/cocoqa-toy/hidden_oxford.h5'
#imgConvFeatOutFilename = '../data/cocoqa/hidden5_4_conv.txt'

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
    return  word_dict, word_array, word_freq

def removeQuestions(questions, answers, imgids, lowerBound, upperBound=100):
    """
    Removes questions with answer appearing less than N times.
    Probability function to decide whether or not to enroll an answer (remove too frequent answers).
    """
    answerdict, answeridict, answerfreq = buildDict(answers, 0)
    random = np.random.RandomState(2)
    questionsTrunk = []
    answersTrunk = []
    imgidsTrunk = []
    answerfreq2 = []
    for item in answerfreq:
        answerfreq2.append(0)
    for i in range(len(questions)):
        if answerfreq[answerdict[answers[i]]] < lowerBound:
            continue
        else:
            if answerfreq2[answerdict[answers[i]]] <= 100:
                questionsTrunk.append(questions[i])
                answersTrunk.append(answers[i])
                imgidsTrunk.append(imgids[i])
                answerfreq2[answerdict[answers[i]]] += 1
            else:
                # Exponential distribution
                prob = np.exp(-(answerfreq2[answerdict[answers[i]]] - upperBound) / float(2 * upperBound))
                #prob = 1 - (answerfreq2[answerdict[answers[i]]] - 100) / float(1500)
                r = random.uniform(0, 1, [1])
                #print 'Prob', prob, 'freq', answerfreq2[answerdict[answers[i]]], 'random', r
                if r < prob:
                    questionsTrunk.append(questions[i])
                    answersTrunk.append(answers[i])
                    imgidsTrunk.append(imgids[i])
                    answerfreq2[answerdict[answers[i]]] += 1
    return questionsTrunk, answersTrunk, imgidsTrunk


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
    #maxlen = 55
    maxlen = 0
    for q in questions:
        words = q.replace(',', '').split(' ')
        wordslist.append(words)
        if len(words) > maxlen:
            maxlen = len(words)
    print 'Max length', maxlen
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

if __name__ == '__main__':
    # Build image features.
    # numTrain = 3000
    # numValid = 600
    # numTest = 3000
    # imgHidFeat = h5py.File(imgHidFeatTrainFilename)
    # hidFeat = imgHidFeat['hidden7'][0 : numTrain]
    # imgHidFeat = h5py.File(imgHidFeatValidFilename)
    # hidFeat = np.concatenate((hidFeat, imgHidFeat[0 : numValid + numTest]))
    # imgOutFile = h5py.File(imgHidFeatOutFilename, 'w')
    # imgOutFile['hidden7'] = hidFeat

    with open(imgidTrainFilename) as f:
        lines = f.readlines()
    trainLen = 3000
    totalTrainLen = len(lines)
    with open(imgidValidFilename) as f:
        lines.extend(f.readlines())
    validLen = totalTrainLen + 600
    testLen = validLen + 3000

    imgidDict = {} # Mark for train/valid/test.
    imgidDict2 = {} # Reindex the image, 1-based.
    imgidDict3 = [] # Reverse dict for image, 0-based.
    # 3000 images train, 600 images valid, 3000 images test.
    # 0 for train, 1 for valid, 2 for test.

    cocoImgIdRegex = 'COCO_((train)|(val))2014_0*(?P<imgid>[1-9][0-9]*)'

    for i in range(trainLen):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 0
        imgidDict2[imgid] = i + 1
        imgidDict3.append(imgid)

    for i in range(totalTrainLen, validLen):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 1
        imgidDict2[imgid] = i + 1
        imgidDict3.append(imgid)

    for i in range(validLen, testLen):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 2
        imgidDict2[imgid] = i + 1
        imgidDict3.append(imgid)

    with open(qaTrainFilename) as qaf:
        qaAll = pkl.load(qaf)
    with open(qaValidFilename) as qaf:
        qaAll.extend(pkl.load(qaf))

    trainQuestions = []
    trainAnswers = []
    trainImgIds = []
    validQuestions = []
    validAnswers = []
    validImgIds = []
    testQuestions = []
    testAnswers = []
    testImgIds = []

    for item in qaAll:
        imgid = item[2]
        if imgidDict.has_key(imgid):
            if imgidDict[imgid] == 0:
                trainQuestions.append(item[0][:-2])
                trainAnswers.append(item[1])
                trainImgIds.append(imgidDict2[imgid])
            elif imgidDict[imgid] == 1:
                validQuestions.append(item[0][:-2])
                validAnswers.append(item[1])
                validImgIds.append(imgidDict2[imgid])
            elif imgidDict[imgid] == 2:
                testQuestions.append(item[0][:-2])
                testAnswers.append(item[1])
                testImgIds.append(imgidDict2[imgid])

    print 'Train Questions Before Trunk: ', len(trainQuestions)
    print 'Valid Questions Before Trunk: ', len(validQuestions)
    print 'Test Questions Before Trunk: ', len(testQuestions)

    # Truncate rare answers.
    trainQuestions, trainAnswers, trainImgIds = \
        removeQuestions(trainQuestions, trainAnswers, trainImgIds, 5, 100)
    validQuestions, validAnswers, validImgIds = \
        removeQuestions(validQuestions, validAnswers, validImgIds, 2, 20)
    testQuestions, testAnswers, testImgIds = \
        removeQuestions(testQuestions, testAnswers, testImgIds, 5, 100)
    trainCount = [0,0,0]
    validCount = [0,0,0]
    testCount = [0,0,0]
    
    numberAns = {}
    colorAns = {}
    for n in range(0, len(trainQuestions)):
        question = trainQuestions[n]
        if 'how many' in question:
            typ = 1
        elif question.startswith('what is the color'):
            typ = 2
        else:
            typ = 0
        trainCount[typ] += 1
    for n in range(0, len(validQuestions)):
        question = validQuestions[n]
        if 'how many' in question:
            typ = 1
        elif question.startswith('what is the color'):
            typ = 2
        else:
            typ = 0
        validCount[typ] += 1
    for n in range(0, len(testQuestions)):
        question = testQuestions[n]
        if 'how many' in question:
            typ = 1
            numberAns[testAnswers[n]] = 1
        elif question.startswith('what is the color'):
            typ = 2
            colorAns[testAnswers[n]] = 1
        else:
            typ = 0
        testCount[typ] += 1

    print numberAns
    print colorAns
    print 'Train Questions After Trunk: ', len(trainQuestions)
    print 'Train Question Dist: ', trainCount
    print 'Train Question Dist: ', np.array(trainCount) / float(len(trainQuestions))
    print 'Valid Questions After Trunk: ', len(validQuestions)
    print 'Valid Question Dist: ', validCount
    print 'Valid Question Dist: ', np.array(validCount) / float(len(validQuestions))

    trainValidQuestionsLen = len(trainQuestions) + len(validQuestions)
    print 'Train+Valid questions: ', trainValidQuestionsLen
    print 'Train+Valid Dist: ', np.array(trainCount) + np.array(validCount)
    print 'Trian+Valid Dist: ', (np.array(trainCount) + np.array(validCount)) / float(trainValidQuestionsLen)

    print 'Test Questions After Trunk: ', len(testQuestions)
    print 'Test Question Dist: ', testCount
    print 'Test Question Dist: ', np.array(testCount) / float(len(testQuestions))
    
    worddict, idict, _ = buildDict(trainQuestions, 1, pr=False)
    ansdict, iansdict, _ = buildDict(trainAnswers, 0, pr=True)

    print 'Valid answer distribution'
    buildDict(validAnswers, 0, pr=True)
    print 'Test answer distribution'
    buildDict(testAnswers, 0, pr=True)

    trainInput = combine(\
        lookupQID(trainQuestions, worddict), trainImgIds)
    trainTarget = lookupAnsID(trainAnswers, ansdict)
    validInput = combine(\
        lookupQID(validQuestions, worddict), validImgIds)
    validTarget = lookupAnsID(validAnswers, ansdict)
    testInput = combine(\
        lookupQID(testQuestions, worddict), testImgIds)
    testTarget = lookupAnsID(testAnswers, ansdict)

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

    trainInput = combineAttention(\
        lookupQID(trainQuestions, worddict), trainImgIds)
    trainTarget = lookupAnsID(trainAnswers, ansdict)
    validInput = combineAttention(\
        lookupQID(validQuestions, worddict), validImgIds)
    validTarget = lookupAnsID(validAnswers, ansdict)
    testInput = combineAttention(\
        lookupQID(testQuestions, worddict), testImgIds)
    testTarget = lookupAnsID(testAnswers, ansdict)

    np.save(\
        os.path.join(outputFolder, 'train-att.npy'),\
        np.array((trainInput, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'valid-att.npy'),\
        np.array((validInput, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test-toy-att.npy'),\
        np.array((testInput, testTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'vocab-dict.npy'),\
        np.array((worddict, idict, 
            ansdict, iansdict, 0), dtype=object))

    with open(os.path.join(outputFolder, 'question_vocabs.txt'), 'w+') as f:
        for word in idict:
            f.write(word + '\n')

    with open(os.path.join(outputFolder, 'answer_vocabs.txt'), 'w+') as f:
        for word in iansdict:
            f.write(word + '\n')

    with open(os.path.join(outputFolder, 'imgid_dict.pkl'), 'wb') as f:
        pkl.dump(imgidDict3, f)