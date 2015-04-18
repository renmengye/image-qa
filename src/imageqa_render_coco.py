import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print plt.get_backend()
import matplotlib.cm as cm

import sys
import os
import json
import cPickle as pkl

from nn.func import *
#plt.ion()

jsonTrainFilename = '../../../data/mscoco/train/captions.json'
jsonValidFilename = '../../../data/mscoco/valid/captions.json'

def decodeQuestion(X, questionArray):
    sentence = ''
    for t in range(1, X.shape[0]):
        if X[t, 0] == 0:
            break
        sentence += questionArray[X[t, 0]- 1] + ' '
    sentence += '?'
    return sentence

def calcRate(X, Y, T, questionArray):
    correct = np.zeros(4, dtype=int)
    total = np.zeros(4, dtype=int)
    for n in range(0, X.shape[0]):        
        sortIdx = np.argsort(Y[n], axis=0)
        sortIdx = sortIdx[::-1]
        A = sortIdx[0]
        question = decodeQuestion(X[n], questionArray)
        if 'how many' in question:
            typ = 1
        elif question.startswith('what is the color'):
            typ = 2
        elif question.startswith('where'):
            typ = 3
        else:
            typ = 0
        total[typ] += 1
        if A == T[n, 0]:
            correct[typ] += 1
    return correct, total


def renderHtml(X, Y, T, questionArray, answerArray, topK, urlDict, imgidDict):
    htmlList = []
    htmlList.append('<html><head></head><body>\n')
    htmlList.append('<table style="width:1250px;border=0">')
    imgPerRow = 4
    for n in range(0, min(20000, X.shape[0])):
        if np.mod(n, imgPerRow) == 0:
            htmlList.append('<tr>')
        imageId = X[n, 0, 0]
        if urlDict.has_key(int(imgidDict[imageId - 1])):
            imageFilename = urlDict[int(imgidDict[imageId - 1])]
        else:
            imageFilename = "unavailable"
        htmlList.append('<td style="padding-top:0px;height=550px">\
        <div style="width:310px;height:210px;text-align:top;margin-top:0px;\
        padding-top:0px;line-height:0px"><img src="%s" width=300 height=200/></div>\n' % imageFilename)
        sentence = decodeQuestion(X[n], questionArray)
        htmlList.append('<div style="height:300px;text-align:bottom;overflow:hidden;">Q%d: %s<br/>' % (n + 1, sentence))
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
            htmlList.append('<span %s>%d. %s %.4f</span><br/>' % (colorStr, i + 1, answerArray[sortIdx[i]], Y[n, sortIdx[i]]))
        htmlList.append('Correct answer: <span style="color:green">%s</span><br/></div></td>' % answerArray[T[n, 0]])

        if np.mod(n, imgPerRow) == imgPerRow - 1:
            htmlList.append('</tr>')
    htmlList.append('</table></body></html>')
    return ''.join(htmlList)

def scan(X):
    N = X.shape[0]
    numExPerBat = 100
    batchStart = 0
    Y = None
    N = X.shape[0]
    Xend = np.zeros(N, dtype=int) + X.shape[1]
    reachedEnd = np.sum(X, axis=-1) == 0.0

    # Scan for the end of the sequence.
    for n in range(N):
        for t in range(X.shape[1]):
            if reachedEnd[n, t]:
                Xend[n] = t
                break
    return Xend

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
    resultFolder = '../results/%s' % taskId
    print taskId

    # Train
    trainOutputFilename = os.path.join(resultFolder, '%s.train.o.npy' % taskId)
    trainHtmlFilename = os.path.join(resultFolder, '%s.train.o.html' % taskId)
    trainOut = np.load(trainOutputFilename)
    Y = trainOut
    trainData = np.load(os.path.join(dataFolder, 'train.npy'))
    testData = np.load(os.path.join(dataFolder, 'test.npy'))
    vocabDict = np.load(os.path.join(dataFolder, 'vocab-dict.npy'))
    imgidDictFilename = os.path.join(dataFolder, 'imgid_dict.pkl')

    with open(imgidDictFilename, 'rb') as f:
        imgidDict = pkl.load(f)

    X = trainData[0]
    T = trainData[1]
    Xend = scan(X)
    html = renderHtml(X, Y, T, vocabDict[1], vocabDict[3], 10, urlDict, imgidDict)
    with open(trainHtmlFilename, 'w+') as f:
        f.writelines(html)
    correct, total = calcRate(X, Y, T, vocabDict[1])
    print correct, total, np.array(correct, dtype=float) / np.array(total, dtype=float)

    # Test
    testOutputFilename = os.path.join(resultFolder, '%s.test.o.npy' % taskId)
    testHtmlFilename = os.path.join(resultFolder, '%s.test.o.html' % taskId)
    testOut = np.load(testOutputFilename)
    TY = testOut
    TX = testData[0]
    TT = testData[1]
    html = renderHtml(TX, TY, TT, vocabDict[1], vocabDict[3], 10, urlDict, imgidDict)
    with open(testHtmlFilename, 'w+') as f:
        f.writelines(html)
    correct, total = calcRate(TX, TY, TT, vocabDict[1])
    print correct, total, correct / total.astype(float)
