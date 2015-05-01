import sys
import os
import nn
import numpy as np
from imageqa_test import *
from imageqa_render import *
from mscoco_prep import *

def combine(wordids, imgids):
    return np.concatenate(\
        (np.array(imgids).reshape(len(imgids), 1, 1), \
        wordids), axis=1)

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

def lookupAnsID(answers, ansdict):
    ansids = []
    for ans in answers:
        if ansdict.has_key(ans):
            ansids.append(ansdict[ans])
        else:
            ansids.append(ansdict['UNK'])
    return np.array(ansids, dtype=int).reshape(len(ansids), 1)

if __name__ == '__main__':
    """
    Usage python imageqa_adhoc.py  -m[odels] {id1},{id2},{id3}...
                                   -n[ames] {name1},{name2}...
                                   -d[ata] {dataFolder}
                                   -i[nput] {listFile}
                                   -k {top K answers}
                                   /-p[icture] {pictureFolder}
                                   /-o[utput] {outputFolder}
                                   /-f[ile] {outputTexFilename}
                                   -daquar/-coco
    Ask adhoc questions on an image
    Input is the following format:
    QID1,Question1,Answer1
    QID2,Question2,Answer2
    """
    dataset = 'coco'
    filename = 'result.tex'
    pictureFolder = 'img'
    K = 1
    modelNames = None
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-m' or sys.argv[i] == '-models':
            modelsStr = sys.argv[i + 1]
            modelIds = modelsStr.split(',')
        elif sys.argv[i] == '-n' or sys.argv[i] == '-names':
            namesStr = sys.argv[i + 1]
            modelNames = namesStr.split(',')
        elif sys.argv[i] == '-d' or sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-i' or sys.argv[i] == '-input':
            inputFile = sys.argv[i + 1]
        elif sys.argv[i] == '-k':
            K = int(sys.argv[i + 1])
        elif sys.argv[i] == '-p' or sys.argv[i] == '-picture':
            pictureFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-o' or sys.argv[i] == '-output':
            outputFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-f' or sys.argv[i] == '-file':
            filename = sys.argv[i + 1]
        elif sys.argv[i] == '-daquar':
            dataset = 'daquar'
        elif sys.argv[i] == '-coco':
            dataset = 'coco'

    resultsFolder = '../results'

    print 'Loading image urls...'
    if dataset == 'coco':
        imgidDictFilename = os.path.join(dataFolder, 'imgid_dict.pkl')
        with open(imgidDictFilename, 'rb') as f:
            imgidDict = pkl.load(f)
        urlDict = readImgDictCoco(imgidDict)
    elif dataset == 'daquar':
        urlDict = readImgDictDaquar()

    print 'Loading test data...'
    vocabDict = np.load(os.path.join(dataFolder, 'vocab-dict.npy'))
    testDataFile = os.path.join(dataFolder, 'test.npy')
    testData = np.load(testDataFile)
    inputTest = testData[0]
    targetTest = testData[1]
    worddict = vocabDict[0]
    ansdict = vocabDict[2]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]
    maxlen = inputTest.shape[1]

    # for i in range(len(questionArray)):
    #     if '_' in questionArray[i]:
    #         questionArray[i] = questionArray[i].replace('_', '\\_')
    # for i in range(len(answerArray)):
    #     if '_' in answerArray[i]:
    #         answerArray[i] = answerArray[i].replace('_', '\\_')

    qids = []
    questions = []
    answers = []
    caption = ''
    with open(inputFile) as f:
        for line in f:
            parts = line.split(',')
            qids.append(int(parts[0]))
            questions.append(parts[1])
            answers.append(parts[2].strip('\n'))
    idx = np.array(qids, dtype='int')
    inputTestSel = inputTest[idx]
    targetTestSel = targetTest[idx]
    imgids = inputTestSel[:, 0, 0]
    adhocInputTestSel = combine(lookupQID(questions, worddict, maxlen), imgids)
    adhocTargetTestSel = lookupAnsID(answers, ansdict)

    # for i in range(len(questions)):
    #     print questions[i], answers[i]
    #     print adhocInputTestSel[i]
    #     print inputTest[qids[i]]
    #     print adhocTargetTestSel

    modelOutputs = []
    for modelName, modelId in zip(modelNames, modelIds):
        print 'Running test data on model %s...' \
                % modelName
        resultFolder = '../results/%s' % modelId
        modelFile = '../results/%s/%s.model.yml' % (modelId, modelId)
        model = nn.load(modelFile)
        model.loadWeights(
            np.load('../results/%s/%s.w.npy' % (modelId, modelId)))
        adhocOutputTest = nn.test(model, adhocInputTestSel)
        #outputTest = nn.test(model, inputTestSel)
        #print adhocOutputTest/outputTest
        modelOutputs.append(adhocOutputTest)

    # Render
    print('Rendering HTML...')
    pages = renderHtml(
                        adhocInputTestSel,
                        adhocTargetTestSel,
                        questionArray,
                        answerArray,
                        urlDict,
                        topK=K,
                        modelOutputs=modelOutputs,
                        modelNames=modelNames,
                        questionIds=qids)
    for i, page in enumerate(pages):
        with open(os.path.join(outputFolder,
            '../%s-%d.html' % (filename, i)), 'w') as f:
            f.write(page)
    print('Rendering LaTeX...')
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    renderLatex(
                adhocInputTestSel,
                adhocTargetTestSel,
                questionArray, 
                answerArray, 
                urlDict, 
                topK=K,
                outputFolder=outputFolder,
                pictureFolder=pictureFolder,
                comments=None,
                caption='',
                modelOutputs=modelOutputs,
                modelNames=modelNames,
                questionIds=idx,
                filename=filename+'.tex')