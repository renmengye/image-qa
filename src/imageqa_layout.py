import sys
import os
import nn
import numpy as np
from imageqa_test import *
from imageqa_render import *

if __name__ == '__main__':
    """
    Usage python imageqa_layout.py -m[odels] {id1},{id2},{id3}...
                                   -n[ames] {name1},{name2}...
                                   -d[ata] {dataFolder}
                                   -i[nput] {listFile}
                                   -o[utput] {outputFolder}
                                   -daquar/-coco
    Render a selection of examples into LaTeX.
    Input is the following format:
    QID1[,Comment1]
    QID2[,Comment2]
    ...
    """
    dataset = 'coco'
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
        elif sys.argv[i] == '-o' or sys.argv[i] == '-output':
            outputFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-daquar':
            dataset = 'daquar'
        elif sys.argv[i] == '-coco':
            dataset = 'coco'

    resultsFolder = '../results'
    K = 3 # Top-K answers

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
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]

    selectionIds = []
    selectionComments = []
    with open(inputFile) as f:
        for line in f:
            parts = f.split(',')
            selectionIds.append(int(parts[0]))
            if len(parts) > 1:
                selectionComments.append(parts[1][:-1])
            else:
                selectionComments.append('')
    idx = np.array(selectionIds, dtype='int')
    inputTestSel = inputTest[idx]
    targetTestSel = targetTest[idx]

    modelOutputs = []
    for modelName, modelId in zip(modelNames, modelIds):
        print 'Running test data on model %s...' \
                % modelName
        resultFolder = '../results/%s' % modelId
        modelFile = '../results/%s/%s.model.yml' % (modelId, modelId)
        model = nn.load(modelFile)
        model.loadWeights(
            np.load('../results/%s/%s.w.npy' % (modelId, modelId)))
        outputTest = nn.test(model, inputTestSel)
        modelOutputs.append(outputTest)

    # Render
    print('Rendering LaTeX...')
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    pages = renderLatex(
                inputTestSubset, 
                targetTestSubset, 
                questionArray, 
                answerArray, 
                urlDict, 
                topK=K,
                modelOutputs=modelOutputsSubset,
                modelNames=modelNames,
                questionIds=idx)