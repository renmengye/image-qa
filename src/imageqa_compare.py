import sys
import os
import nn
import numpy as np
from imageqa_test import *
from imageqa_render import *

nameList = ['object', 'number', 'color', 'location']

def getCatName(i):
    return nameList[i]

def getBinName(n):
    bin = []
    for k in range(numModels):
        bin.append(str(n >> (numModels - k - 1)))
        n = n & (~(1 << (numModels - k - 1)))
    return ''.join(bin)

def getName(catName, binName):
    return catName + '-' + binName


def renderIndex(modelNames, numCategories):
    htmlList = []
    htmlList.append('<html><head><style>%s</style><body>' % \
        'span.good {color:green;} span.bad {color:red;} table{border-spacing:10px;}')
    numModels = len(modelNames)
    numCorrect = 1 << numModels
    htmlList.append('<h1>Models comparisons</h1>')
    htmlList.append('Notes:<br/><span class="good">Green means the model gets correct</span><br/>')
    htmlList.append('<span class="bad">Red means the model gets wrong</span>')
    for i in range(numCategories):
        htmlList.append('<h2>%s</h2>' % getCatName(i))
        htmlList.append('<table>')
        for j in range(numCorrect):
            htmlList.append('<tr>')
            for k, c in enumerate(getBinName(j)):
                htmlList.append('<td>')
                if c == '1':
                    htmlList.append('<span class="good">%s</span>' % modelNames[k])
                elif c == '0':
                    htmlList.append('<span class="bad">%s</span>' % modelNames[k])
                htmlList.append('</td>')
            htmlList.append('<td><a href="%s/0.html">link</a></td>' % \
                getName(getCatName(i), getBinName(j)))
            htmlList.append('</tr>')

        htmlList.append('</table>')
    htmlList.append('</head></body></html>')
    return ''.join(htmlList)

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

    for modelName, modelId in zip(modelNames, modelIds):
        print 'Running test data on model %s...' \
                % modelName
        resultFolder = '../results/%s' % modelId
        modelFile = '../results/%s/%s.model.yml' % (modelId, modelId)
        model = nn.load(modelFile)
        model.loadWeights(
            np.load('../results/%s/%s.w.npy' % (modelId, modelId)))
        outputTest = nn.test(model, inputTest)
        modelOutputs.append(outputTest)

    # Sort questions by question types.
    # Sort questions by correctness differences.
    print('Sorting questions...')
    numCategories = 4
    numModels = len(modelNames)
    numCorrect = 1 << numModels
    numBins = numCategories * numCorrect
    modelAnswers = np.zeros((numModels, inputTest.shape[0]), dtype='int')
    bins = [None] * numBins
    names = []
    for i in range(numCategories):
        catName = getCatName(i)
        for j in range(numCorrect):
            binName = getBinName(j)
            names.append(getName(catName, binName))
    for i in range(numModels):
        modelAnswers[i] = np.argmax(modelOutputs[i], axis=-1)
    for n in range(inputTest.shape[0]):
        correct = targetTest[n, 0]
        bintmp = 0
        for i in range(numModels):
            if modelAnswers[i, n] == correct:
                bintmp += 1 << (numModels - i - 1)
        category = testQuestionType[n]
        binNum = category * numCorrect + bintmp
        if bins[binNum] == None:
            bins[binNum] = [n]
        else:
            bins[binNum].append(n)

    # Render
    print('Rendering webpages...')
    print('Rendering index...')
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    with open(os.path.join(outputFolder, 'index.html'), 'w') as f:
        f.write(renderIndex(modelNames, numCategories))

    for i in range(numBins):
        if bins[i] is not None:
            # Need a better folder name!
            print 'Rendering %s...' % names[i]
            outputSubFolder = os.path.join(outputFolder, names[i])
            idx = np.array(bins[i], dtype='int')
            inputTestSubset = inputTest[idx]
            targetTestSubset = targetTest[idx]
            modelOutputsSubset = []
            for i in range(numModels):
                modelOutputsSubset.append(modelOutputs[i][idx])
            if not os.path.exists(outputSubFolder):
                os.makedirs(outputSubFolder)
            htmlHyperLink = '%d.html'
            pages = renderHtml(inputTestSubset, modelOutputsSubset, 
                        targetTestSubset, 
                        questionArray, answerArray, K, urlDict, modelNames)
            for i, page in enumerate(pages):
                with open(os.path.join(outputSubFolder, 
                        htmlHyperLink % i), 'w') as f:
                    f.write(page)
