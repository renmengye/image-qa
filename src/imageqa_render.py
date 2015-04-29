import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print plt.get_backend()
import matplotlib.cm as cm

import sys
import os
import json
import cPickle as pkl

from nn.func import *
from imageqa_test import *

jsonTrainFilename = '../../../data/mscoco/train/captions.json'
jsonValidFilename = '../../../data/mscoco/valid/captions.json'
htmlHyperLink = '%d.html'
cssHyperLink = 'style.css'
daquarImageFolder = '../../data/nyu-depth-v2/jpg/'

def renderHtml(
                X, 
                Y, 
                T, 
                questionArray, 
                answerArray, 
                topK, 
                urlDict):
    if X.shape[0] < 1000:
        return [renderSinglePage(
            X, Y, T, questionArray, answerArray, 
            topK, urlDict, imgidDict, 0, 1)]
    else:
        result = []
        numPages = X.shape[0] / 2000 + 1
        for i in range(numPages):
            start = 2000 * i
            end = min(X.shape[0], 2000 * (i + 1))
            page = renderSinglePage(
                X[start:end], Y[start:end], T[start:end], 
                questionArray, answerArray,
                topK, urlDict, imgidDict, i, numPages)
            result.append(page)
        return result

def renderMenu(iPage, numPages):
    htmlList = []
    htmlList.append('<div>Navigation: ')
    for n in range(numPages):
        if n != iPage:
            htmlList.append('<a href=%s> %d </a>' % \
                        ((htmlHyperLink % n), n))
        else:
            htmlList.append('<span> %d </span>' % n)

    htmlList.append('</div>')
    return ''.join(htmlList)

def renderCSS():
    cssList = []
    cssList.append('table {width:1250px;border:0;text-align:center;}\n')
    cssList.append('td.item {padding-top:0px;height=550px;}\n')
    cssList.append('div.img {width:310px;height:210px;text-align:top;\
                    margin-top:0px;padding-top:0px;line-height:0px;}\n')
    cssList.append('div.ans {height:300px;text-align:bottom;\
                    overflow:hidden;}')
    cssList.append('img {width:300px; height:200px}\n')
    cssList.append('span.good {color:green;}\n')
    cssList.append('span.bad {color:red;}\n')
    return ''.join(cssList)

def renderSingleItem(
                    imageUrl, 
                    questionIndex, 
                    question, 
                    correctAnswer,
                    topAnswers,
                    topAnswerScores,
                    modelNames=None):
    """
    Render a single item.
    topAnswers: a list of top answer strings
    topAnswerScores: a list of top answer scores
    """
    htmlList = []
    htmlList.append('<td class="item">\
                    <div class="img">\
                    <img src="%s"/></div>\n' % \
                    imageFilename)
    htmlList.append('<div class="ans">Q%d: %s<br/>' % \
                    (questionIndex + 1, question))
    for i, answer in enumerate(topAnswers):
        if answer == correctAnswer:
            colorStr = 'class="good"'
        elif i == 0:
            colorStr = 'class="bad"'
        else:
            colorStr = ''
        htmlList.append('<span %s>%d. %s %.4f</span><br/>' % \
                    (colorStr, i + 1, 
                    answer, topAnswerScores[i]))
    htmlList.append('Correct answer: <span class="good">\
                    %s</span><br/></div></td>' % correctAnswer)

    return ''.join(htmlList)

def renderSinglePage(
                    X, 
                    Y, 
                    T, 
                    questionArray, 
                    answerArray, 
                    topK, 
                    urlDict,
                    iPage, 
                    numPages):
    htmlList = []
    htmlList.append('<html><head>\n')
    htmlList.append('<style>%s</style>' % renderCSS())
    htmlList.append('</head><body>\n')
    htmlList.append('<table>')
    imgPerRow = 4
    htmlList.append(renderMenu(iPage, numPages))
    for n in range(X.shape[0]):
        if np.mod(n, imgPerRow) == 0:
            htmlList.append('<tr>')
        imageId = X[n, 0, 0]
        imageFilename = urlDict[imageId - 1]
        question = decodeQuestion(X[n], questionArray)
        sortIdx = np.argsort(Y[n], axis=0)
        sortIdx = sortIdx[::-1]
        topAnswers = []
        topAnswerScores = []
        for i in range(0, topK):
            topAnswers.append(answerArray[sortIdx[i]])
            topAnswerScores.append(Y[n, sortIdx[i]])
        htmlList.append(renderSingleItem(imageFilename, 
            n, question, answerArray[T[n, 0]], topAnswers, 
            topAnswerScores))
        if np.mod(n, imgPerRow) == imgPerRow - 1:
            htmlList.append('</tr>')
    htmlList.append('</table>')
    htmlList.append(renderMenu(iPage, numPages))
    htmlList.append('</body></html>')
    return ''.join(htmlList)

def readImgDictCoco(imgidDict):
    with open(jsonTrainFilename) as f:
        captiontxt = f.read()
    urlDict = {}
    caption = json.loads(captiontxt)
    for item in caption['images']:
        urlDict[item['id']] = item['url']

    with open(jsonValidFilename) as f:
        captiontxt = f.read()
    caption = json.loads(captiontxt)
    for item in caption['images']:
        urlDict[item['id']] = item['url']
    urlList = [None] * len(imgidDict)
    for i, key in enumerate(imgidDict):
        urlList[i] = urlDict[int(key)]
    return urlList

def readImgDictDaquar():
    urlList = []
    for i in range(1, 1450):
        urlList.append(daquarImageFolder + 'image%d.jpg' % int(i))
    return urlList

if __name__ == '__main__':
    """
    Usage: imageqa_render.py {id} 
                             -d[ata] dataFolder 
                             -o[utput] outputFolder
                             -daquar/-coco
    """
    taskId = sys.argv[1]
    dataset = 'coco'
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '-d' or sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-o' or sys.argv[i] == '-output':
            outputFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-daquar':
            dataset = 'daquar'
        elif sys.argv[i] == '-coco':
            dataset = 'coco'
    print taskId

    vocabDict = np.load(os.path.join(dataFolder, 'vocab-dict.npy'))

    if dataset == 'coco':
        imgidDictFilename = os.path.join(dataFolder, 'imgid_dict.pkl')
        with open(imgidDictFilename, 'rb') as f:
            imgidDict = pkl.load(f)
        urlDict = readImgDictCoco(imgidDict)
    elif dataset == 'daquar':
        urlDict = readImgDictDaquar()

    print 'Loading model...'
    resultFolder = '../results/%s' % taskId
    modelFile = '../results/%s/%s.model.yml' % (taskId, taskId)
    model = nn.load(modelFile)
    model.loadWeights(
        np.load('../results/%s/%s.w.npy' % (taskId, taskId)))

    print 'Loading test data...'
    testDataFile = os.path.join(dataFolder, 'test.npy')
    testData = np.load(testDataFile)
    inputTest = testData[0]
    targetTest = testData[1]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]

    print 'Running model on test data...'
    outputTest = nn.test(model, inputTest)

    # Render
    htmlOutputFolder = os.path.join(resultFolder, outputFolder)
    if not os.path.exists(htmlOutputFolder):
        os.makedirs(htmlOutputFolder)
    pages = renderHtml(inputTest, outputTest, targetTest, 
                questionArray, answerArray, 10, urlDict, imgidDict)
    for i, page in enumerate(pages):
        with open(os.path.join(htmlOutputFolder, 
                htmlHyperLink % i), 'w') as f:
            f.write(page)