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
                    questionIdict):
    for n in range(X.shape[0]):
        img = loadImage(imgPathDict[X[n, 0, 0]])
        plt.clf()
        w = np.round(np.sqrt(Xend[n] + 1))
        h = np.ceil((Xend[n] + 1) / float(w))
        plt.subplot(w, h, 1)
        plt.imshow(img)
        plt.axis('off')
        for t in range(Xend[n]):
            word = questionIdict[X[n, t, 1] - 1]
            plt.subplot(w, h, t + 2)
            plt.imshow(img)
            alpha = A[n, t].reshape(14, 14)
            alphaImage = skimage.transform.resize(
                alpha, [img.shape[0], img.shape[1]])
            plt.imshow(alphaImage, alpha=0.8)
            plt.set_cmap(cm.Greys_r)
            plt.title(word)
            plt.axis('off')
        plt.savefig(
            os.path.join(outputFolder, 
                '%s-%d-%d.png' % (prefix, n, t)))

if __name__ == '__main__':
    """
    Render visual attention.
    Usage: python visatt_render.py 
                        -m[odel] {name:modelId}
                        -d[ata] {dataFolder}
                        -o[utput] {outputFolder}
                        [-r[esults] {resultsFolder}]
                        [-dataset {daquar/cocoqa}]
                        [-n[umber] {number of examples}]
    Parameters:
        -m[odel]: Model ID
        -d[ata]: Data folder
        -o[utput]: Output folder
        -r[esults]: Results folder, default "../results"
        -dataset: DAQUAR/COCO-QA dataset, default "cocoqa"
        -n[number]: Render number of examples, default 10
    """
    modelId = sys.argv[1]
    dataFolder = None
    outputFolder = None
    resultsFolder = '../results'
    dataset = "cocoqa"
    N = 10
    for i, flag in enumerate(sys.argv):
        if flag == '-m' or flag == '-model':
            modelId = sys.argv[i + 1]
        elif flag == '-d' or flag == '-data':
            dataFolder = sys.argv[i + 1]
        elif flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
        elif flag == '-dataset':
            dataset = sys.argv[i + 1]
        elif flag == '-n' or flag == '-number'
            N = int(sys.argv[i + 1])
    print modelId
    if os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    model = it.loadModel(modelId, resultsFolder)
    data = it.loadDataset(dataFolder)
    imgPathDict = ir.loadPathDict(dataset)

    X = data['testData'][0]
    T = data['testData'][1]
    Y, layers = nn.test(model, data['testData'][0], layerNames=['attOut'])
    A = layers['attOut']

    print A, A.shape
    Xend = scan(X[0:N])
    prefix = ''
    plotAttention(
                    X[0:N], 
                    A[0:N], 
                    Xend, 
                    prefix, 
                    resultsFolder, 
                    imgPathDict, 
                    data['questionIdict'])