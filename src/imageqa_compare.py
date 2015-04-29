import sys
import os
import nn
import numpy as np
from imageqa_test import *
from imageqa_render import *

if __name__ == '__main__':
    """
    Usage python imageqa_compare.py -m[odels] {id1},{id2},{id3}...
                                    -n[ames] {name1},{name2}...
                                    -d[ata] {dataFolder}
                                    -o[utput] {outputFolder}
                                    -daquar/-coco
    """
    dataset = 'coco'
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-m' or sys.argv[i] == '-models':
            modelsStr = sys.argv[i + 1]
            modelIds = modelsStr.split(',')
        elif sys.argv[i] == '-n' or sys.argv[i] == '-names':
            namesStr = sys.argv[i + 1]
            modelNames = namesStr.split(',')
        elif sys.argv[i] == '-d' or sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-o' or sys.argv[i] == '-output':
            outputFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-daquar':
            dataset = 'daquar'
        elif sys.argv[i] == '-coco':
            dataset = 'coco'

    if len(modelNames) != len(modelIds):
        raise Exception('ID list length must be same as name list')

    resultsFolder = '../results'
    K = 3 # Top-K answers
    modelOutputs = []

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
    testQuestionTypeFile = os.path.join(dataFolder, 'test-qtype.npy')
    testQuestionType = np.load(testQuestionTypeFile)

    for modelName in modelNames:
        print 'Running test data on model %s...' \
                % modelName
        outputTest = nn.test(model, inputTest)
        modelOutputs.append(outputTest)

    # Sort questions by question types.
    # categoryInput = np.zeros(4, dtype='object')

    # Render
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    pages = renderHtml(inputTest, modelOutputs, targetTest, 
                questionArray, answerArray, 10, urlDict)
    for i, page in enumerate(pages):
        with open(os.path.join(htmlOutputFolder, 
                htmlHyperLink % i), 'w') as f:
            f.write(page)

    # Sort questions by correctness differences.