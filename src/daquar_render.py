import sys
import os

from nn.func import *
from imageqa_test import *

imageFolder = '../../data/nyu-depth-v2/jpg/'

def renderHtml(X, Y, T, questionArray, answerArray, topK):
    htmlList = []
    htmlList.append('<html><head></head><body>\n')
    htmlList.append('<table style="width:1250px;border=0">')
    imgPerRow = 4
    for n in range(0, X.shape[0]):
        if np.mod(n, imgPerRow) == 0:
            htmlList.append('<tr>')
        imageId = X[n, 0, 0]
        imageFilename = imageFolder + 'image%d.jpg' % int(imageId)
        htmlList.append('<td style="padding-top:0px;height=550px">\
        <div style="width:310px;height:210px;text-align:top;margin-top:0px;\
        padding-top:0px;line-height:0px"><img src="%s" width=300 height=200/></div>\n' % imageFilename)
        sentence = ''
        for t in range(1, X.shape[1]):
            if X[n, t, 0] == 0:
                break
            sentence += questionArray[X[n, t, 0]- 1] + ' '
        sentence += '?'
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
            if sortIdx[i] >= len(answerArray):
                answer = 'UNK'
            else:
                answer = answerArray[sortIdx[i]]
            htmlList.append('<span %s>%d. %s %.4f</span><br/>' % (colorStr, i + 1, answer, Y[n, sortIdx[i]]))
        htmlList.append('Correct answer: <span style="color:green">%s</span><br/></div></td>' % answerArray[T[n, 0]])

        if np.mod(n, imgPerRow) == imgPerRow - 1:
            htmlList.append('</tr>')
    htmlList.append('</table></body></html>')
    return ''.join(htmlList)

if __name__ == '__main__':
    """
    Usage: imageqa_render.py id -data {dataFolder}
    """
    taskId = sys.argv[1]
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
    resultFolder = '../results/%s' % taskId
    print taskId

    vocabDict = np.load(os.path.join(dataFolder, 'vocab-dict.npy'))

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

    # Render
    testHtmlFilename = os.path.join(resultFolder, '%s.test.o.html' % taskId)
    html = renderHtml(inputTest, outputTest, targetTest, 
        questionArray, answerArray, 10)
    with open(testHtmlFilename, 'w+') as f:
        f.writelines(html)