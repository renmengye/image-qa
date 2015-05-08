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
    Ask adhoc questions with trained models.

    Usage: python imageqa_adhoc.py  
                    -m[odel] {name1:modelId1}
                    -m[odel] {name2:modelId2}
                    -m[odel] {name3:ensembleModelId3,ensembleModelId4,...}
                    ...
                    -d[ata] {dataFolder}
                    -i[nput] {listFile}
                    -o[utput] {outputFolder}
                    [-k {top K answers}]
                    [-p[icture] {pictureFolder}]
                    [-r[esults] {resultsFolder}]
                    [-f[ile] {outputTexFilename}]
                    [-daquar/-coco]
                    [-html/-latex]
    Parameters:
        -m[odel]: Model name and model ID
        -d[ata]: Dataset dataFolder
        -i[nput]: Adhoc question list filename
        -o[utput]: Output folder of the rendered results
        -k: Render top-K answers (default 1)
        -p[icture]: Picture folder, only required in LaTeX mode (default "img")
        -r[esults]: Results folder where trained models are stored (default "../results")
        -f[ile]: Output filename, only required in LaTex mode
        -coco: Use COCO dataset (default)
        -daquar: Use DAQUAR dataset
        -html: Set output format to HTML (default)
        -latex: Set output format to LaTeX.

    Input question list format:
    QID1,Question1,GroundTruthAnswer1
    QID2,Question2,GroundTruthAnswer2
    ...
    """
    usage = '\
    Ask adhoc questions with trained models.\
\
    Usage: python imageqa_adhoc.py  \
                    -m[odel] {name1:modelId1}\
                    -m[odel] {name2:modelId2}\
                    -m[odel] {name3:ensembleModelId3,ensembleModelId4,...}\
                    ...\
                    -d[ata] {dataFolder}\
                    -i[nput] {listFile}\
                    -o[utput] {outputFolder}\
                    [-k {top K answers}]\
                    [-p[icture] {pictureFolder}]\
                    [-r[esults] {resultsFolder}]\
                    [-f[ile] {outputTexFilename}]\
                    [-daquar/-coco]\
                    [-html/-latex]\
    Parameters:\
        -m[odel]: Model name and model ID\
        -d[ata]: Dataset dataFolder\
        -i[nput]: Adhoc question list filename\
        -o[utput]: Output folder of the rendered results\
        -k: Render top-K answers (default 1)\
        -p[icture]: Picture folder, only required in LaTeX mode (default "img")\
        -r[esults]: Results folder where trained models are stored (default "../results")\
        -f[ile]: Output filename, only required in LaTex mode\
        -coco: Use COCO dataset (default)\
        -daquar: Use DAQUAR dataset\
        -html: Set output format to HTML (default)\
        -latex: Set output format to LaTeX\
\
    Input question list format:\
    QID1,Question1,GroundTruthAnswer1\
    QID2,Question2,GroundTruthAnswer2\
    ...'
    dataset = 'coco'
    filename = 'result'
    pictureFolder = 'img'
    K = 1
    resultsFolder = '../results'
    modelNames = []
    modelIds = []
    renderMode = 'html'
    dataFolder = None
    inputFile = None
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            parts = sys.argv[i + 1].split(':')
            modelNames.append(parts[0])
            modelIds.append(parts[1])
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-i' or flag == '-input':
            inputFile = sys.argv[i + 1]
        elif flag == '-k':
            K = int(sys.argv[i + 1])
        elif flag == '-p' or flag == '-picture':
            pictureFolder = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
        elif flag == '-f' or flag == '-file':
            filename = sys.argv[i + 1]
        elif flag == '-daquar':
            dataset = 'daquar'
        elif flag == '-coco':
            dataset = 'coco'
        elif flag == '-latex':
            renderMode = 'latex'
        elif flag == '-html':
            renderMode = 'html'

    if len(modelNames) == 0 or len(modelIds) == 0 or \
        dataFolder is None or inputFile is None:
        print usage
        sys.exit(0)

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
    testQuestionTypeFile = os.path.join(dataFolder, 'test-qtype.npy')
    testQuestionTypes = np.load(testQuestionTypeFile)
    testData = np.load(testDataFile)
    inputTest = testData[0]
    targetTest = testData[1]
    worddict = vocabDict[0]
    ansdict = vocabDict[2]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]
    maxlen = inputTest.shape[1]

    # For LaTeX only, replace underscore in vocabulary.
    questionArrayLatex = []
    answerArrayLatex = []
    for i in range(len(questionArray)):
        questionArrayLatex.append(
                questionArray[i].replace('_', '\\_'))
    for i in range(len(answerArray)):
        answerArrayLatex.append(
                answerArray[i].replace('_', '\\_'))

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
    adhocQuestionTypeTestSel = testQuestionTypes[idx]

    modelOutputs = []
    for modelName, modelId in zip(modelNames, modelIds):
        if ',' in modelId:
            print 'Running test data on ensemble model %s...' \
                    % modelName
            models = loadEnsemble(modelId.split(','), resultsFolder)
            adhocOutputTest = runEnsemble(
                                        adhocInputTestSel, 
                                        models, 
                                        adhocQuestionTypeTestSel)
        else:
            print 'Running test data on model %s...' \
                    % modelName
            model = loadModel(modelId, resultsFolder)
            adhocOutputTest = nn.test(model, adhocInputTestSel)
        modelOutputs.append(adhocOutputTest)

    # Render
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
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
            '%s-%d.html' % (filename, i)), 'w') as f:
            f.write(page)
    print('Rendering LaTeX...')
    renderLatex(
                adhocInputTestSel,
                adhocTargetTestSel,
                questionArrayLatex, 
                answerArrayLatex, 
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