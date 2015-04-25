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

def renderHtml(
                X, 
                Y, 
                T, 
                questionArray, 
                answerArray, 
                topK, 
                urlDict, 
                imgidDict):
    if X.shape[0] < 1000:
        return [renderSinglePage(
            X, Y, T, questionArray, answerArray, 
            topK, urlDict, imgidDict, 0, 1)]
    else:
        result = []
        numPages = X.shape[0] / 2000 + 1
        for i in range(numPages):
            page = renderSinglePage(
                X, Y, T, questionArray, answerArray,
                topK, urlDict, imgidDict, i, numPages)
            result.append(page)
        return result

def renderMenu(iPage, numPages):
    htmlList = []
    htmlList.append('<div style="text-align:center">Navigation: ')
    for n in range(numPages):
        if n != iPage:
            htmlList.append('<a href=%s> %d </a>' % \
                        ((htmlHyperLink % n), n))
        else:
            htmlList.append('<span> %d </span>' % n)

    htmlList.append('</div>')
    return ''.join(htmlList)

def renderSinglePage(
                    X, 
                    Y, 
                    T, 
                    questionArray, 
                    answerArray, 
                    topK, 
                    urlDict, 
                    imgidDict, 
                    iPage, 
                    numPages):
    htmlList = []
    htmlList.append('<html><head></head><body>\n')
    htmlList.append('<table style="width:1250px;border=0">')
    imgPerRow = 4
    htmlList.append(renderMenu(iPage, numPages))
    for n in range(X.shape[0]):
        if np.mod(n, imgPerRow) == 0:
            htmlList.append('<tr>')
        imageId = X[n, 0, 0]
        if urlDict.has_key(int(imgidDict[imageId - 1])):
            imageFilename = urlDict[int(imgidDict[imageId - 1])]
        else:
            imageFilename = "unavailable"
        htmlList.append('<td style="padding-top:0px;height=550px">\
                        <div style="width:310px;height:210px;text-align:top;\
                        margin-top:0px;padding-top:0px;line-height:0px">\
                        <img src="%s" width=300 height=200/></div>\n' % \
                        imageFilename)
        sentence = decodeQuestion(X[n], questionArray)
        htmlList.append('<div style="height:300px;text-align:bottom;\
                        overflow:hidden;">Q%d: %s<br/>' % (n + 1, sentence))
        htmlList.append('Top %d answers: (confidence)<br/>' % topK)
        sortIdx = np.argsort(Y[n], axis=0)
        sortIdx = sortIdx[::-1]
        for i in range(0, topK):
            if sortIdx[i] == T[n, 0]:
                colorStr = 'style="color:green"'
            elif i == 0:
                colorStr = 'style="color:red"'
            else:
                colorStr = ''
            htmlList.append('<span %s>%d. %s %.4f</span><br/>' % \
                        (colorStr, i + 1, 
                        answerArray[sortIdx[i]], Y[n, sortIdx[i]]))
        htmlList.append('Correct answer: <span style="color:green">\
                        %s</span><br/></div></td>' % answerArray[T[n, 0]])
        if np.mod(n, imgPerRow) == imgPerRow - 1:
            htmlList.append('</tr>')
    htmlList.append('</table>')
    htmlList.append(renderMenu(iPage, numPages))
    htmlList.append('</body></html>')
    return ''.join(htmlList)

def readImgDict():
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
    return urlDict

if __name__ == '__main__':
    """
    Usage: imageqa_render.py id -data dataFolder
    """
    urlDict = readImgDict()
    taskId = sys.argv[1]
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
    print taskId

    vocabDict = np.load(os.path.join(dataFolder, 'vocab-dict.npy'))
    imgidDictFilename = os.path.join(dataFolder, 'imgid_dict.pkl')

    resultFolder = '../results/%s' % taskId
    modelFile = '../results/%s/%s.model.yml' % (taskId, taskId)
    model = nn.load(modelFile)
    model.loadWeights(
        np.load('../results/%s/%s.w.npy' % (taskId, taskId)))

    testDataFile = os.path.join(dataFolder, 'test.npy')
    testData = np.load(testDataFile)

    inputTest = testData[0]
    outputTest = nn.test(model, inputTest)
    targetTest = testData[1]
    questionArray = vocabDict[1]
    answerArray = vocabDict[3]

    with open(imgidDictFilename, 'rb') as f:
        imgidDict = pkl.load(f)

    # Render
    htmlOutputFolder = os.path.join(resultFolder, 'html')
    if not os.path.exists(htmlOutputFolder):
        os.makedirs(htmlOutputFolder)
    pages = renderHtml(inputTest, outputTest, targetTest, 
                questionArray, answerArray, 10, urlDict, imgidDict)
    for i, page in enumerate(pages):
        with open(os.path.join(htmlOutputFolder, 
                htmlHyperLink % i), 'w') as f:
            f.write(page)
