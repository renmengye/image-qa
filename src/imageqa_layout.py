import sys
import os
import nn
import numpy as np
from imageqa_test import *
from imageqa_render import *

if __name__ == '__main__':
    """
    Render a selection of examples into LaTeX.
    Usage: python imageqa_layout.py 
                    -m[odel] {name1:modelId1}
                    -m[odel] {name2:modelId2}
                    -m[odel] {name3:ensembleModelId3,ensembleModelId4,...}
                    ...
                    -d[ata] {dataFolder}
                    -i[nput] {listFile}
                    -o[utput] {outputFolder}
                    [-k {top K answers}]
                    [-p[icture] {pictureFolder}]
                    [-f[ile] {outputTexFilename}]
                    [-daquar/-cocoqa]
    Parameters:
        -m[odel]
        -d[ata]
        -i[nput]
        -k
        -p[icture]
        -f[ile]
        -daquar
        -cocoqa
    Input file format:
    QID1[,Comment1]
    QID2[,Comment2]
    ...
    """
    dataset = 'cocoqa'
    filename = 'result'
    pictureFolder = 'img'
    K = 1
    modelNames = []
    modelIds = []
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            parts = sys.argv[i + 1].split(':')
            modelNames.append(parts[0])
            modelIds.append(parts[1])
        elif flag == '-n' or flag == '-names':
            namesStr = sys.argv[i + 1]
            modelNames = namesStr.split(',')
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
        elif flag == '-cocoqa':
            dataset = 'cocoqa'

    resultsFolder = '../results'

    print 'Loading image urls...'
    if dataset == 'cocoqa':
        imgidDictFilename = os.path.join(dataFolder, 'imgid_dict.pkl')
        with open(imgidDictFilename, 'rb') as f:
            imgidDict = pkl.load(f)
        urlDict = readImgDictCocoqa(imgidDict)
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
    caption = ''
    with open(inputFile) as f:
        i = 0
        for line in f:
            if i == 0 and line.startswith('caption:'):
                caption = line[8:-1]
            else:
                parts = line.split(',')
                selectionIds.append(int(parts[0]))
                if len(parts) > 1:
                    selectionComments.append(parts[1][:-1])
                else:
                    selectionComments.append('')
            i += 1
    idx = np.array(selectionIds, dtype='int')
    inputTestSel = inputTest[idx]
    targetTestSel = targetTest[idx]
    
    for i in range(len(questionArray)):
        if '_' in questionArray[i]:
            questionArray[i] = questionArray[i].replace('_', '\\_')
    for i in range(len(answerArray)):
        if '_' in answerArray[i]:
            answerArray[i] = answerArray[i].replace('_', '\\_')

    modelOutputs = []
    for modelName, modelId in zip(modelNames, modelIds):
        if ',' in modelId:
            print 'Running test data on ensemble model %s...' \
                    % modelName
            models = loadEnsemble(modelId.split(','), resultsFolder)
            outputTest = runEnsemble(
                                        inputTest, 
                                        models, 
                                        testQuestionTypes)
        else:
            print 'Running test data on model %s...' \
                    % modelName
            model = loadModel(modelId, resultsFolder)
            outputTest = nn.test(model, inputTest)
        modelOutputs.append(outputTest)

    # Render
    print('Rendering LaTeX...')
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    renderLatex(
                inputTestSel, 
                targetTestSel, 
                questionArray, 
                answerArray, 
                urlDict, 
                topK=K,
                outputFolder=outputFolder,
                pictureFolder=pictureFolder,
                comments=selectionComments,
                caption=caption,
                modelOutputs=modelOutputs,
                modelNames=modelNames,
                questionIds=idx,
                filename=filename + '.tex')