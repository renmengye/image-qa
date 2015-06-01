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

import numpy as np
from nn.func import *
import nn
import imageqa_test as it
import imageqa_render as ir

def scan(X):
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

def loadImage(filename, resize=256, crop=224):
    image = Image.open(filename)
    width, height = image.size

    if width > height:
        width = (width * resize) / height
        height = resize
    else:
        height = (height * resize) / width
        width = resize
    left = (width - crop) / 2
    top = (height - crop) / 2
    imageResized = image.resize((width, height), 
        Image.BICUBIC).crop((left, top, left + crop, top + crop))
    data = np.array(
        imageResized.convert('RGB').getdata()).reshape(crop, crop, 3)
    data = data.astype('float32') / 255
    return data

def plotAttention(
                    X, 
                    A, 
                    Xend, 
                    prefix, 
                    resultsFolder,
                    outputFolder,
                    imgPathDict, 
                    questionIdict,
                    Y=None,
                    T=None,
                    ansIdict=None):
    for n in range(X.shape[0]):
        if len(X.shape) == 3:
            img = loadImage(imgPathDict[X[n, 0, 0] - 1])
        elif len(X.shape) == 2:
            img = loadImage(imgPathDict[X[n, 0] - 1])
        plt.clf()

        if mode == 0:
            w = np.round(np.sqrt(Xend[n]))
            h = np.ceil((Xend[n]) / float(w))
        elif mode == 1:
            w = 2
            h = 1
        fig, ax = plt.subplots()
        plt.subplot(w, h, 1)
        plt.imshow(img)
        plt.axis('off')
        words = []

        if mode == 0: timespan = Xend[n]
        elif mode == 1: timespan = 2
        for t in range(1, timespan):
            if mode == 0:
                attention = A[n, t - 1]
            elif mode == 1:
                attention = A[n]
            if len(X.shape) == 3:
                word = questionIdict[X[n, t] - 1]
            elif len(X.shape) == 2 and t == 1:
                word = questionIdict[X[n, 1] - 1]
            else:
                word = ''
            words.append(word)
            plt.subplot(w, h, t + 1)
            plt.imshow(img)
            alpha = attention.reshape(14, 14)
            alphaImage = skimage.transform.resize(
                alpha, [img.shape[0], img.shape[1]])
            plt.imshow(alphaImage, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.axis('off')
        if Y is not None:
            plt.subplot(w, h, 1)
            ans = ansIdict[T[n]]
            outidx = np.argmax(Y[n])
            out = ansIdict[outidx]
            prob = Y[n, outidx]
            plt.title('Q: %s GT: %s A: %s (%.4f)' % (words[0], ans, out, prob))
        else: 
            plt.title(word)
        plt.savefig(
            os.path.join(outputFolder, 
                '%s-%d.png' % (prefix, n)))

if __name__ == '__main__':
    """
    Render visual attention.
    Usage: python visatt_render.py 
                        -m[odel] {model id}
                        -d[ata] {data folder}
                        -o[utput] {output folder}
                        -l[ayer] {attention layer name}
                        [-r[esults] {results folder}]
                        [-dataset {daquar/cocoqa}]
                        [-n[umber] {number of examples}]
                        -mode {0/1}
    Parameters:
        -m[odel]: Model ID
        -d[ata]: Data folder
        -o[utput]: Output folder
        -l[ayer]: Layer name of the attention output, e.g. 'attModel:attOut'
        -r[esults]: Results folder, default "../results"
        -dataset: DAQUAR/COCO-QA dataset, default "cocoqa"
        -n[number]: Render number of examples, default 10
        -mode: Temporary flag
    """
    modelId = sys.argv[1]
    dataFolder = None
    outputFolder = None
    resultsFolder = '../results'
    dataset = 'cocoqa'
    attLayer = 'attModel:attOut'
    N = 10
    mode = 0
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            modelId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
        elif flag == '-l' or flag == '-layer':
            attLayer = sys.argv[i + 1]
        elif flag == '-dataset':
            dataset = sys.argv[i + 1]
        elif flag == '-n' or flag == '-number':
            N = int(sys.argv[i + 1])
        elif flag == '-r' or flag == '-results':
            resultsFolder = sys.argv[i + 1]
        elif flag == '-mode':
            mode = int(sys.argv[i + 1])
    print modelId
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    model = it.loadModel(modelId, resultsFolder)

    #if mode == 1:
    #    W = model.stageDict['controllerHid2'].getWeights()
    #    print W
    #    # Increase the temperature to have a smoother attention output
    #    W /= 5.0
    #    print W
    #    model.stageDict['controllerHid2'].loadWeights(W)
    data = it.loadDataset(dataFolder)
    imgPathDict = ir.loadImgPath(dataset, dataFolder)

    X = data['testData'][0]
    T = data['testData'][1]
    for n in range(N):
        q = it.decodeQuestion(X[n], data['questionIdict'])
        # print q
    Y, layers = nn.test(model, X[0:N], layerNames=[attLayer])
    A = layers[attLayer]

    if mode == 0:
        A = np.concatenate((
            np.zeros((A.shape[0], 1, A.shape[2])) + 1 / float(A.shape[2]),
            A[:, :-1, :]), axis=1)

    print A, A.shape
    np.savetxt(os.path.join(outputFolder, 'attention.txt'), A, delimiter=',')
    Xend = np.zeros(X.shape[0], dtype='int') + A.shape[1] + 1
    plotAttention(
                X=X[0:N, [0, 7], 0],
                A=A[0:N],
                Xend=Xend, 
                prefix='test', 
                resultsFolder=resultsFolder, 
                outputFolder=outputFolder,
                imgPathDict=imgPathDict, 
                questionIdict=data['questionIdict'],
                Y=Y,
                T=T,
                ansIdict=data['ansIdict'])
