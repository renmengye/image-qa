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

import requests

jsonTrainFilename = '../../../data/mscoco/train/captions.json'
jsonValidFilename = '../../../data/mscoco/valid/captions.json'
htmlHyperLink = '%d.html'
cssHyperLink = 'style.css'
daquarImageFolder = 'http://www.cs.toronto.edu/~mren/imageqa/data/nyu-depth-v2/jpg/'

def renderLatexAnswerList(
                            correctAnswer, 
                            topAnswers, 
                            topAnswerScores):
    result = []
    for i, answer in enumerate(topAnswers):
        if answer == correctAnswer:
            colorStr = '\\textcolor{green}{%s}'
        elif i == 0:
            colorStr = '\\textcolor{red}{%s}'
        else:
            colorStr = ''
        result.append(colorStr % ('%s (%.4f) ' % \
                    (answer, topAnswerScores[i])))
    return ''.join(result)

def renderLatexSingleItem(
                    questionIndex,
                    question,
                    correctAnswer,
                    comment=None,
                    topAnswers=None,
                    topAnswerScores=None,
                    modelNames=None):
    result = []
    result.append('    \\scalebox{0.3}{\n')
    result.append('        \\includegraphics[width=\\textwidth, height=.7\\textwidth]{img/%d.jpg}}\n' % questionIndex)
    result.append('    \\parbox{5cm}{\n')
    result.append('        \\vskip 0.05in\n')
    result.append('        Q%d: %s\\\\\n' % (questionIndex, question))
    result.append('        Ground truth: %s\\\\\n' % correctAnswer)
    i = 0
    if modelNames is not None and len(modelNames) > 1:
        for modelAnswer, modelAnswerScore, modelName in \
            zip(topAnswers, topAnswerScores, modelNames):
            result.append('%s: ' % modelName)
            result.append(
                renderLatexAnswerList(
                                 correctAnswer,
                                 modelAnswer,
                                 modelAnswerScore))
            if i != len(modelNames) - 1:
                result.append('\\\\')
            i += 1
            result.append('\n')
    elif topAnswers is not None:
        result.append(
            renderLatexAnswerList(
                             correctAnswer,
                             topAnswers, 
                             topAnswerScores))
        result.append('\n')
    if comment is not None:
        result.append('\\\\' + comment)
    result.append('    }\n')
    return ''.join(result)

def renderLatex(
                inputData,
                targetData,
                questionArray,
                answerArray,
                urlDict,
                outputFolder,
                topK=10,
                comments=None,
                modelOutputs=None,
                modelNames=None,
                questionIds=None
                ):
    result = []
    result.append('\\begin{table*}[ht!]\n')
    result.append('\\small\n')
    result.append('\\begin{tabular}{p{5cm} p{5cm} p{5cm}}\n')
    imgPerRow = 3
    imgFolder = os.path.join(outputFolder, 'img')
    for n in range(inputData.shape[0]):
        # Download the images
        imageId = inputData[n, 0, 0]
        imageFilename = urlDict[imageId - 1]
        r = requests.get(imageFilename)
        qid = questionIds[n] if questionIds is not None else n
        if not os.path.exists(imgFolder):
            os.makedirs(imgFolder)
        with open(os.path.join(imgFolder, '%d.jpg' % qid), 'wb') as f:
            f.write(r.content)
        question = decodeQuestion(inputData[n], questionArray)
        answer = answerArray[targetData[n, 0]]
        topAnswers, topAnswerScores = pickTopAnswers(
                                            answerArray,
                                            n,
                                            topK=topK,
                                            modelOutputs=modelOutputs, 
                                            modelNames=modelNames)
        comment = comments[n] \
                if comments is not None else None
        result.append(renderLatexSingleItem(
                                            qid,
                                            question,
                                            answer,
                                            comment=comment,
                                            topAnswers=topAnswers,
                                            topAnswerScores=topAnswerScores,
                                            modelNames=modelNames))
        if np.mod(n, imgPerRow) == imgPerRow - 1:
            result.append('\\\\\n')
            if n != inputData.shape[0] - 1:
                result.append('\\noalign{\\smallskip}\\noalign{\\smallskip}\\noalign{\\smallskip}\n')
        else:
            result.append('&\n')
    result.append('\end{tabular}\n')
    result.append('\caption{}\n')
    result.append('\end{table*}\n')
    latexStr = ''.join(result)
    with open(os.path.join(outputFolder, 'result.tex'), 'w') as f:
        f.write(latexStr)

def renderHtml(
                inputData,
                targetData,
                questionArray,
                answerArray,
                urlDict,
                topK=10,
                modelOutputs=None,
                modelNames=None,
                questionIds=None):
    imgPerPage = 1000
    if inputData.shape[0] < imgPerPage:
        return [renderSinglePage(
                                inputData,
                                targetData,
                                questionArray,
                                answerArray,
                                urlDict,
                                iPage=0,
                                numPages=1,
                                topK=topK,
                                modelOutputs=modelOutputs,
                                modelNames=modelNames, 
                                questionIds=questionIds)]
    else:
        result = []
        numPages = np.ceil(inputData.shape[0] / float(imgPerPage))
        for i in range(numPages):
            start = imgPerPage * i
            end = min(inputData.shape[0], imgPerPage * (i + 1))
            if modelNames is not None:
                modelOutputSlice = []
                for j in range(len(modelNames)):
                    modelOutputSlice.append(modelOutputs[j][start:end])
            elif modelOutputs is not None:
                modelOutputSlice = modelOutputs[start:end]
            else:
                modelOutputSlice = modelOutputs
            page = renderSinglePage(
                                    inputData[start:end],
                                    targetData[start:end],
                                    questionArray,
                                    answerArray,
                                    urlDict, 
                                    iPage=i,
                                    numPages=numPages, 
                                    topK=topK,
                                    modelOutputs=modelOutputSlice,
                                    modelNames=modelNames, 
                                    questionIds=questionIds)
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

def renderCss():
    cssList = []
    cssList.append('table {\
                            width:1200px;\
                            border-spacing:10px;\
                          }\n')
    cssList.append('td.item {\
                             padding:5px;\
                             border:1px solid gray;\
                             vertical-align:top;\
                            }\n')
    cssList.append('div.ans {\
                             margin-top:10px;\
                             width:300px;\
                            }')
    cssList.append('img {width:300px; height:200px;}\n')
    cssList.append('span.good {color:green;}\n')
    cssList.append('span.bad {color:red;}\n')
    return ''.join(cssList)

def renderAnswerList(
                    correctAnswer, 
                    topAnswers, 
                    topAnswerScores):
    htmlList = []
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
    return ''.join(htmlList)

def renderSingleItem(
                    imageFilename,
                    questionIndex,
                    question,
                    correctAnswer,
                    topAnswers=None,
                    topAnswerScores=None,
                    modelNames=None):
    """
    Render a single item.
    topAnswers: a list of top answer strings
    topAnswerScores: a list of top answer scores
    modelNames: if multiple items, then above are list of lists.
    """
    htmlList = []
    htmlList.append('<td class="item">\
                    <div class="img">\
                    <img src="%s"/></div>\n' % \
                    imageFilename)
    htmlList.append('<div class="ans">Q%d: %s<br/>' % \
                    (questionIndex, question))
    htmlList.append('Correct answer: <span class="good">\
                    %s</span><br/>' % correctAnswer)
    if modelNames is not None and len(modelNames) > 1:
        for modelAnswer, modelAnswerScore, modelName in \
            zip(topAnswers, topAnswerScores, modelNames):
            htmlList.append('%s:<br/>' % modelName)
            htmlList.append(
                renderAnswerList(
                                 correctAnswer,
                                 modelAnswer,
                                 modelAnswerScore))
    elif topAnswers is not None:
        htmlList.append(
            renderAnswerList(
                             correctAnswer,
                             topAnswers, 
                             topAnswerScores))
    htmlList.append('</div></td>')
    return ''.join(htmlList)

def pickTopAnswers(
                    answerArray,
                    n,
                    topK=10,
                    modelOutputs=None, 
                    modelNames=None):
    if modelNames is not None and len(modelNames) > 1:
        topAnswers = []
        topAnswerScores = []
        for j, modelOutput in enumerate(modelOutputs):
            sortIdx = np.argsort(modelOutput[n], axis=0)
            sortIdx = sortIdx[::-1]
            topAnswers.append([])
            topAnswerScores.append([])
            for i in range(0, topK):
                topAnswers[-1].append(answerArray[sortIdx[i]])
                topAnswerScores[-1].append(modelOutput[n, sortIdx[i]])
    elif modelOutputs is not None:
        sortIdx = np.argsort(modelOutputs[n], axis=0)
        sortIdx = sortIdx[::-1]
        topAnswers = []
        topAnswerScores = []
        for i in range(0, topK):
            topAnswers.append(answerArray[sortIdx[i]])
            topAnswerScores.append(modelOutputs[n, sortIdx[i]])
        qid = questionIds[n] if questionIds is not None else n
    else:
        topAnswers = None
        topAnswerScores = None
    return topAnswers, topAnswerScores

def renderSinglePage(
                    inputData, 
                    targetData, 
                    questionArray, 
                    answerArray, 
                    urlDict,
                    iPage=0, 
                    numPages=1,
                    topK=10,
                    modelOutputs=None,
                    modelNames=None,
                    questionIds=None):
    htmlList = []
    htmlList.append('<html><head>\n')
    htmlList.append('<style>%s</style>' % renderCss())
    htmlList.append('</head><body>\n')
    htmlList.append('<table>')
    imgPerRow = 4
    htmlList.append(renderMenu(iPage, numPages))
    for n in range(inputData.shape[0]):
        if np.mod(n, imgPerRow) == 0:
            htmlList.append('<tr>')
        imageId = inputData[n, 0, 0]
        imageFilename = urlDict[imageId - 1]
        question = decodeQuestion(inputData[n], questionArray)

        qid = questionIds[n] if questionIds is not None else n
        topAnswers, topAnswerScores = pickTopAnswers(
                                        answerArray, 
                                        n,
                                        topK=topK,
                                        modelOutputs=modelOutputs, 
                                        modelNames=modelNames)
        htmlList.append(renderSingleItem(
                                        imageFilename, 
                                        qid, 
                                        question, 
                                        answerArray[targetData[n, 0]], 
                                        topAnswers=topAnswers, 
                                        topAnswerScores=topAnswerScores, 
                                        modelNames=modelNames))

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
    Usage: imageqa_render.py -id {id} 
                             -d[ata] {dataFolder}
                             -o[utput] {outputFolder}
                             -daquar/-coco
    """
    dataset = 'coco'
    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-d' or sys.argv[i] == '-data':
            dataFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-o' or sys.argv[i] == '-output':
            outputFolder = sys.argv[i + 1]
        elif sys.argv[i] == '-id':
            taskId = sys.argv[i + 1]
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
    pages = renderHtml(
                        inputTest, 
                        targetTest, 
                        questionArray, 
                        answerArray, 
                        urlDict, 
                        topK=10, 
                        modelOutputs=outputTest)
    for i, page in enumerate(pages):
        with open(os.path.join(htmlOutputFolder, 
                htmlHyperLink % i), 'w') as f:
            f.write(page)
