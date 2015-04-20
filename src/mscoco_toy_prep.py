import re
import os
import cPickle as pkl
import numpy as np
import h5py

imgidTrainFilename = '../../../data/mscoco/train/image_list.txt'
imgidValidFilename = '../../../data/mscoco/valid/image_list.txt'
qaTrainFilename = '../../../data/mscoco/train/qa.pkl'
qaValidFilename = '../../../data/mscoco/valid/qa.pkl'
outputFolder = '../data/cocoqa-toy/'
imgHidFeatTrainFilename = '/ais/gobi3/u/mren/data/mscoco/hidden_oxford_train.h5'
imgHidFeatValidFilename = '/ais/gobi3/u/mren/data/mscoco/hidden_oxford_valid.h5'
imgHidFeatOutFilename = '/ais/gobi3/u/mren/data/cocoqa-toy/hidden_oxford.h5'
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

def removeQuestions(answers, lowerBound, upperBound=100):
    """
    Removes questions with answer appearing less than N times.
    Probability function to decide whether or not to enroll an answer (remove too frequent answers).
    """
    answerdict, answeridict, answerfreq = buildDict(answers, 0)
    random = np.random.RandomState(2)
    answerfreq2 = []
    survivor = []
    for item in answerfreq:
        answerfreq2.append(0)
    for i in range(len(answers)):
        if answerfreq[answerdict[answers[i]]] < lowerBound:
            continue
        else:
            if answerfreq2[answerdict[answers[i]]] <= 100:
                survivor.append(i)
                answerfreq2[answerdict[answers[i]]] += 1
            else:
                # Exponential distribution
                prob = np.exp(-(answerfreq2[answerdict[answers[i]]] - \
                    upperBound) / float(2 * upperBound))
                r = random.uniform(0, 1, [1])
                if r < prob:
                    survivor.append(i)
                    answerfreq2[answerdict[answers[i]]] += 1
    return survivor


def lookupAnsID(answers, ansdict):
    ansids = []
    for ans in answers:
        if ansdict.has_key(ans):
            ansids.append(ansdict[ans])
        else:
            ansids.append(ansdict['UNK'])
    return np.array(ansids, dtype=int).reshape(len(ansids), 1)

def findMaxlen(questions):
    maxlen = 0
    for q in questions:
        words = q.split(' ')
        if len(words) > maxlen:
            maxlen = len(words)
    print 'Maxlen: ', maxlen
    return maxlen

def lookupQID(questions, worddict, maxlen):
    wordslist = []
    for q in questions:
        words = q.split(' ')
        wordslist.append(words)
        if len(words) > maxlen:
            maxlen = len(words)
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

def questionType(q):
    if 'how many' in q:
        typ = 1
    elif q.startswith('what is the color'):
        typ = 2
    elif q.startswith('where'):
        typ = 3
    else:
        typ = 0
    return typ

if __name__ == '__main__':
    # Build image features.
    numTrain = 6000
    numValid = 1200
    numTest = 6000
    imgHidFeatTrain = h5py.File(imgHidFeatTrainFilename)
    imgHidFeatValid = h5py.File(imgHidFeatValidFilename)
    imgOutFile = h5py.File(imgHidFeatOutFilename, 'w')
    
    for name in ['hidden7', 'hidden6', 'hidden5_maxpool']:
        hidFeatTrain = imgHidFeatTrain[name][0 : numTrain]
        hidFeatValid = imgHidFeatValid[name][0 : numValid + numTest]
        hidFeat = np.concatenate((hidFeatTrain, hidFeatValid), axis=0)
        imgOutFile[name] = hidFeat
    
    with open(imgidTrainFilename) as f:
        lines = f.readlines()
    trainLen = numTrain
    totalTrainLen = len(lines)
    
    with open(imgidValidFilename) as f:
        lines.extend(f.readlines())
    validLen = totalTrainLen + numValid
    testLen = validLen + numTest

    imgidDict = {} # Mark for train/valid/test.
    imgidDict2 = {} # Reindex the image, 1-based.
    imgidDict3 = [] # Reverse dict for image, 0-based.
    # 0 for train, 1 for valid, 2 for test.

    cocoImgIdRegex = 'COCO_((train)|(val))2014_0*(?P<imgid>[1-9][0-9]*)'

    for i in range(trainLen):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 0
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)

    for i in range(totalTrainLen, validLen):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 1
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)

    for i in range(validLen, testLen):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 2
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)

    with open(qaTrainFilename) as qaf:
        qaAll = pkl.load(qaf)
    with open(qaValidFilename) as qaf:
        qaAll.extend(pkl.load(qaf))

    trainQuestions = []
    trainAnswers = []
    trainImgIds = []
    trainQuestionTypes = []
    validQuestions = []
    validAnswers = []
    validImgIds = []
    validQuestionTypes = []
    testQuestions = []
    testAnswers = []
    testImgIds = []
    testQuestionTypes = []
    baseline = []
    colorAnswer = 'white'
    numberAnswer = 'two'
    objectAnswer = 'cat'
    locationAnswer = 'room'

    for item in qaAll:
        imgid = item[2]
        if imgidDict.has_key(imgid):
            if imgidDict[imgid] == 0:
                trainQuestions.append(item[0][:-2])
                trainAnswers.append(item[1])
                trainImgIds.append(imgidDict2[imgid])
                trainQuestionTypes.append(item[3])
            elif imgidDict[imgid] == 1:
                validQuestions.append(item[0][:-2])
                validAnswers.append(item[1])
                validImgIds.append(imgidDict2[imgid])
                validQuestionTypes.append(item[3])
            elif imgidDict[imgid] == 2:
                testQuestions.append(item[0][:-2])
                testAnswers.append(item[1])
                testImgIds.append(imgidDict2[imgid])
                testQuestionTypes.append(item[3])

    print 'Train Questions Before Trunk: ', len(trainQuestions)
    print 'Valid Questions Before Trunk: ', len(validQuestions)
    print 'Test Questions Before Trunk: ', len(testQuestions)

    # Shuffle the questions before applying rare-common answer rejection.
    r = np.random.RandomState(1)
    shuffle = r.permutation(len(trainQuestions))
    trainQuestions = np.array(trainQuestions, dtype=object)[shuffle]
    trainAnswers = np.array(trainAnswers, dtype=object)[shuffle]
    trainImgIds = np.array(trainImgIds, dtype=object)[shuffle]
    trainQuestionTypes = np.array(trainQuestionTypes,dtype=object)[shuffle]

    shuffle = r.permutation(len(validQuestions))
    validQuestions = np.array(validQuestions, dtype=object)[shuffle]
    validAnswers = np.array(validAnswers, dtype=object)[shuffle]
    validImgIds = np.array(validImgIds, dtype=object)[shuffle]
    validQuestionTypes = np.array(validQuestionTypes, dtype=object)[shuffle]

    shuffle = r.permutation(len(testQuestions))
    testQuestions = np.array(testQuestions, dtype=object)[shuffle]
    testAnswers = np.array(testAnswers, dtype=object)[shuffle]
    testImgIds = np.array(testImgIds, dtype=object)[shuffle]
    testQuestionTypes = np.array(testQuestionTypes, dtype=object)[shuffle]

    # Truncate rare-common answers.
    survivor = np.array(removeQuestions(trainAnswers, 5, 100))
    trainQuestions = trainQuestions[survivor]
    trainAnswers = trainAnswers[survivor]
    trainImgIds = trainImgIds[survivor]
    trainQuestionTypes = trainQuestionTypes[survivor]

    survivor = np.array(removeQuestions(validAnswers, 2, 20))
    validQuestions = validQuestions[survivor]
    validAnswers = validAnswers[survivor]
    validImgIds = validImgIds[survivor]
    validQuestionTypes = validQuestionTypes[survivor]

    survivor = np.array(removeQuestions(testAnswers, 5, 100))
    testQuestions = testQuestions[survivor]
    testAnswers = testAnswers[survivor]
    testImgIds = testImgIds[survivor]
    testQuestionTypes = testQuestionTypes[survivor]

    trainCount = np.zeros(4, dtype=int)
    validCount = np.zeros(4, dtype=int)
    testCount = np.zeros(4, dtype=int)
    
    for n in range(0, len(trainQuestions)):
        question = trainQuestions[n]
        trainCount[trainQuestionTypes[n]] += 1
    for n in range(0, len(validQuestions)):
        question = validQuestions[n]
        validCount[validQuestionTypes[n]] += 1
    for n in range(0, len(testQuestions)):
        question = testQuestions[n]
        testCount[testQuestionTypes[n]] += 1

    print 'Train Questions After Trunk: ', len(trainQuestions)
    print 'Train Question Dist: ', trainCount
    print 'Train Question Dist: ', trainCount / float(len(trainQuestions))
    print 'Valid Questions After Trunk: ', len(validQuestions)
    print 'Valid Question Dist: ', validCount
    print 'Valid Question Dist: ', validCount / float(len(validQuestions))

    trainValidQuestionsLen = len(trainQuestions) + len(validQuestions)
    print 'Train+Valid questions: ', trainValidQuestionsLen
    print 'Train+Valid Dist: ', trainCount + validCount
    print 'Trian+Valid Dist: ', (trainCount + validCount) / float(trainValidQuestionsLen)

    print 'Test Questions After Trunk: ', len(testQuestions)
    print 'Test Question Dist: ', testCount
    print 'Test Question Dist: ', testCount / float(len(testQuestions))
    
    # Build dictionary based on training questions/answers.
    worddict, idict, _ = buildDict(trainQuestions, 1, pr=False)
    ansdict, iansdict, _ = buildDict(trainAnswers, 0, pr=True)

    print 'Valid answer distribution'
    buildDict(validAnswers, 0, pr=True)
    print 'Test answer distribution'
    buildDict(testAnswers, 0, pr=True)

    # Shuffle the questions again after applying rare-common answer rejection.
    r = np.random.RandomState(2)
    shuffle = r.permutation(len(trainQuestions))
    trainQuestions = np.array(trainQuestions, dtype=object)[shuffle]
    trainAnswers = np.array(trainAnswers, dtype=object)[shuffle]
    trainImgIds = np.array(trainImgIds, dtype=object)[shuffle]
    trainQuestionTypes = np.array(trainQuestionTypes,dtype=object)[shuffle]

    shuffle = r.permutation(len(validQuestions))
    validQuestions = np.array(validQuestions, dtype=object)[shuffle]
    validAnswers = np.array(validAnswers, dtype=object)[shuffle]
    validImgIds = np.array(validImgIds, dtype=object)[shuffle]
    validQuestionTypes = np.array(validQuestionTypes, dtype=object)[shuffle]

    shuffle = r.permutation(len(testQuestions))
    testQuestions = np.array(testQuestions, dtype=object)[shuffle]
    testAnswers = np.array(testAnswers, dtype=object)[shuffle]
    testImgIds = np.array(testImgIds, dtype=object)[shuffle]
    testQuestionTypes = np.array(testQuestionTypes, dtype=object)[shuffle]

    for n in range(0, len(testQuestions)):
        if testQuestionTypes[n] == 0:
            baseline.append(objectAnswer)
        elif testQuestionTypes[n] == 1:
            baseline.append(numberAnswer)
        elif testQuestionTypes[n] == 2:
            baseline.append(colorAnswer)
        elif testQuestionTypes[n] == 3:
            baseline.append(locationAnswer)

    maxlen = findMaxlen(np.concatenate((trainQuestions, validQuestions, testQuestions)))
    # Build output
    trainInput = combine(\
        lookupQID(trainQuestions, worddict, maxlen), trainImgIds)
    trainTarget = lookupAnsID(trainAnswers, ansdict)
    validInput = combine(\
        lookupQID(validQuestions, worddict, maxlen), validImgIds)
    validTarget = lookupAnsID(validAnswers, ansdict)
    testInput = combine(\
        lookupQID(testQuestions, worddict, maxlen), testImgIds)
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

    with open(os.path.join(outputFolder, 'baseline.txt'), 'w+') as f:
        for answer in baseline:
            f.write(answer + '\n')