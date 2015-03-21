import matplotlib
#matplotlib.use('TkAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
print plt.get_backend()
import matplotlib.cm as cm

from PIL import Image
import sys
import os
import skimage
import skimage.transform
import skimage.io

from nn.func import *
#plt.ion()

imageFolder = '../../data/nyu-depth-v2/jpg/'

def renderHtml(X, Y, T, questionArray, answerArray, topK, prefix):
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
        for t in range(0, X.shape[1]):
            if X[n, t, 1] == 0:
                break
            sentence += questionArray[X[n, t, 1]- 1] + ' '
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
            htmlList.append('<span %s>%d. %s %.4f</span><br/>' % (colorStr, i + 1, answerArray[sortIdx[i]], Y[n, sortIdx[i]]))
        htmlList.append('Correct answer: <span style="color:green">%s</span><br/></div></td>' % answerArray[T[n, 0]])

        if np.mod(n, imgPerRow) == imgPerRow - 1:
            htmlList.append('</tr>')
    htmlList.append('</table></body></html>')

    # Attention HTML
    #ahtmlList = []
    #for n in range(0, X.shape[0]):

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

def LoadImage(file_name, resize=256, crop=224):
    image = Image.open(file_name)
    width, height = image.size

    if width > height:
        width = (width * resize) / height
        height = resize
    else:
        height = (height * resize) / width
        width = resize
    left = (width - crop) / 2
    top = (height - crop) / 2
    image_resized = image.resize((width, height), Image.BICUBIC).crop((left, top, left + crop, top + crop))
    data = np.array(image_resized.convert('RGB').getdata()).reshape(crop, crop, 3)
    data = data.astype('float32') / 255
    return data

def plotAttention(X, A, Xend, prefix):
    for n in range(10):
        img = LoadImage('../data/nyu-depth-v2/jpg/image%d.jpg' % X[n, 0, 0])
        plt.clf()
        w = np.round(np.sqrt(Xend[n] + 1))
        h = np.ceil((Xend[n] + 1) / float(w))
        plt.subplot(w, h, 1)
        plt.imshow(img)
        plt.axis('off')
        #plt.savefig(os.path.join(resultFolder, '%s-%d-_.png' % (prefix, n)))
        for t in range(Xend[n]):
            word = vocabDict[1][X[n, t, 1] - 1]
            #plt.clf()
            plt.subplot(w, h, t + 2)
            plt.imshow(img)
            alpha = A[n, t].reshape(14, 14)
            alpha_img = skimage.transform.resize(alpha, [img.shape[0], img.shape[1]])
            plt.imshow(alpha_img, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.title(word)
            plt.axis('off')
        plt.savefig(os.path.join(resultFolder, '%s-%d-%d.png' % (prefix, n, t)))
        plt.show()

if __name__ == '__main__':
    """
    Usage: imageqa_render.py id -train trainData.npy -test testData.npy -dict vocabDict.npy
    """
    taskId = sys.argv[1]
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '-train':
            trainDataFile = sys.argv[i + 1]
        elif sys.argv[i] == '-test':
            testDataFile = sys.argv[i + 1]
        elif sys.argv[i] == '-dict':
            dictFile = sys.argv[i + 1]
    resultFolder = '../results/%s' % taskId
    print taskId

    # Train
    trainOutputFilename = os.path.join(resultFolder, '%s.train.o.npy' % taskId)
    trainHtmlFilename = os.path.join(resultFolder, '%s.train.o.html' % taskId)
    trainOut = np.load(trainOutputFilename)
    Y = trainOut[0]
    A = trainOut[1]
    trainData = np.load(trainDataFile)
    testData = np.load(testDataFile)
    vocabDict = np.load(dictFile)
    X = trainData[0]
    T = trainData[1]
    Xend = scan(X)
    prefix = 'train'
    plotAttention(X,A,Xend,prefix)
    html = renderHtml(X, Y, T, vocabDict[1], vocabDict[3], 10, 'train')
    with open(trainHtmlFilename, 'w+') as f:
        f.writelines(html)

    # Test
    testOutputFilename = os.path.join(resultFolder, '%s.test.o.npy' % taskId)
    testHtmlFilename = os.path.join(resultFolder, '%s.test.o.html' % taskId)
    testOut = np.load(testOutputFilename)
    TY = testOut[0]
    TA = testOut[1]
    TX = testData[0]
    TT = testData[1]
    html = renderHtml(TX, TY, TT, vocabDict[1], vocabDict[3], 10, 'test')
    with open(testHtmlFilename, 'w+') as f:
        f.writelines(html)
