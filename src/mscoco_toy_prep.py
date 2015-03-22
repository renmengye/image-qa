import re
import os
import cPickle as pkl
import numpy as np
import operator

imgidFilename = '../../../data/mscoco/image_list_train.txt'
qaFilename = '../../../data/mscoco/mscoco_qa_all_train.pkl'
outputFolder = '../data/cocoqa/'
imgHidFeatFilename = '/ais/gobi3/u/rkiros/coco/train_features/hidden7.txt'
imgConvFeatFilename = '/ais/gobi3/u/rkiros/coco/align_train/hidden5_4_conv.txt'
imgHidFeatOutFilename = '../data/cocoqa/hidden7-toy.txt'
imgConvFeatOutFilename = '../data/cocoqa/hidden5_4_conv-toy.txt'

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
          print word_array[x], word_freq[x]
        print sorted_x
        print 'Dictionary length', len(word_dict)
    return  word_dict, word_array, word_freq

def removeQuestions(questions, answers, imgids, lowerBound):
    """
    Removes questions with answer appearing less than N times.
    """
    answerdict, answeridict, answerfreq = buildDict(answers, 0)
    questionsTrunk = []
    answersTrunk = []
    imgidsTrunk = []
    for i in range(len(questions)):
        if answerfreq[answerdict[answers[i]]] < lowerBound:
            continue
        else:
            questionsTrunk.append(questions[i])
            answersTrunk.append(answers[i])
            imgidsTrunk.append(imgids[i])
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
    maxlen = 0
    for q in questions:
        words = q.split(' ')
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
    # hidFeat = []
    # with open(imgHidFeatFilename) as f:
    #     for line in f:
    #         hidFeat.append(line)
    #         if len(hidFeat) == 6600:
    #             break
    # with open(imgHidFeatOutFilename, 'w') as f:
    #     for line in hidFeat:
    #         f.write(line)

    # convFeat = []
    # with open(imgConvFeatFilename) as f:
    #     for line in f:
    #         convFeat.append(line)
    #         if len(convFeat) == 6600:
    #             break
    # with open(imgConvFeatOutFilename, 'w') as f:
    #     for line in convFeat:
    #         f.write(line)

    with open(imgidFilename) as f:
        lines = f.readlines()

    imgidDict = {} # Mark for train/valid/test.
    imgidDict2 = {} # Reindex the image, 1-based.
    # 3000 images train, 600 images valid, 3000 images test.
    # 0 for train, 1 for valid, 2 for test.

    cocoImgIdRegex = 'COCO_train2014_0*(?P<imgid>[1-9][0-9]+)'

    for i in range(3000):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 0
        imgidDict2[imgid] = i + 1

    for i in range(3000, 3600):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 1
        imgidDict2[imgid] = i + 1

    for i in range(3600, 6600):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 2
        imgidDict2[imgid] = i + 1

    with open(qaFilename) as qaf:
        qaAll = pkl.load(qaf)

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
    trainQuestions, trainAnswers, trainImgIds = removeQuestions(trainQuestions, trainAnswers, trainImgIds, 5)
    validQuestions, validAnswers, validImgIds = removeQuestions(validQuestions, validAnswers, validImgIds,  3)
    testQuestions, testAnswers, testImgIds = removeQuestions(testQuestions, testAnswers, testImgIds, 5)
    print 'Train Questions After Trunk: ', len(trainQuestions)
    print 'Valid Questions After Trunk: ', len(validQuestions)
    print 'Test Questions Before Trunk: ', len(testQuestions)
    worddict, idict, _ = buildDict(trainQuestions, 1, pr=False)
    ansdict, iansdict, _ = buildDict(trainAnswers, 0, pr=True)

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
        os.path.join(outputFolder, 'train-toy.npy'),\
        np.array((trainInput, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'valid-toy.npy'),\
        np.array((validInput, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test-toy.npy'),\
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
        os.path.join(outputFolder, 'train-toy-att.npy'),\
        np.array((trainInput, trainTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'valid-toy-att.npy'),\
        np.array((validInput, validTarget, 0),\
            dtype=object))
    np.save(\
        os.path.join(outputFolder, 'test-toy-att.npy'),\
        np.array((testInput, testTarget, 0),\
            dtype=object))

    with open(os.path.join(outputFolder, 'question_vocabs.txt'), 'w+') as f:
        for word in idict:
            f.write(word + '\n')

    with open(os.path.join(outputFolder, 'answer_vocabs.txt'), 'w+') as f:
        for word in iansdict:
            f.write(word + '\n')