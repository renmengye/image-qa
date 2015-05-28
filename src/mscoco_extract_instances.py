import json
import os
import cPickle as pkl
import sys
import re

# [u'info', u'licenses', u'images', u'type', u'annotations', u'categories']
import cv2
import numpy as np

imgidTrainFilename = '../../../data/mscoco/train/image_list.txt'
imgidValidFilename = '../../../data/mscoco/valid/image_list.txt'
trainJsonFilename = \
    '/ais/gobi3/datasets/mscoco/annotations/instances_train2014.json'
validtrainJsonFilename = \
    '/ais/gobi3/datasets/mscoco/annotations/instances_val2014.json'

def buildImgIdDict(imgidTrainFilename, imgidValidFilename):
    with open(imgidTrainFilename) as f:
        lines = f.readlines()
    trainStart = 0
    trainEnd = len(lines) * 9 / 10
    validStart = trainEnd
    validEnd = len(lines)
    with open(imgidValidFilename) as f:
        lines.extend(f.readlines())
    testStart = validEnd
    testEnd = len(lines)

    cocoImgIdRegex = 'COCO_((train)|(val))2014_0*(?P<imgid>[1-9][0-9]*)'
    # Mark for train/valid/test.
    imgidDict = {} 
    # Reindex the image, 1-based.
    imgidDict2 = {}
    # Reverse dict for image, 0-based.
    imgidDict3 = []
    # Path for jpg file
    imgPathDict = {}

    # Separate image ids into train-valid-test
    # 0 for train, 1 for valid, 2 for test.
    for i in range(trainStart, trainEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 0
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)
        imgPathDict[imgid] = lines[i][:-1]

    for i in range(validStart, validEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 1
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)
        imgPathDict[imgid] = lines[i][:-1]

    for i in range(testStart, testEnd):
        match = re.search(cocoImgIdRegex, lines[i])
        imgid = match.group('imgid')
        imgidDict[imgid] = 2
        imgidDict2[imgid] = len(imgidDict3) + 1
        imgidDict3.append(imgid)
        imgPathDict[imgid] = lines[i][:-1]

    return imgidDict, imgidDict2, imgidDict3, imgPathDict

def buildImgDict(imgList, imgidDict):
    imgDict = {}
    count = 1
    for img in imgList:
        if not imgDict.has_key(img['id']):
            img['new_id'] = imgidDict[str(img['id'])]
            imgDict[img['id']] = img
            count += 1
    return imgDict

def buildCatDict(catList):
    catDict = {}
    for cat in catList:
        if not catDict.has_key(cat['id']):
            catDict[cat['id']] = cat
    return catDict

def polyFill(img, segmentation):
    polys = []
    for seg in segmentation:
        N = len(seg)
        poly = np.concatenate(
            (np.array(seg[0:N:2]).reshape(N/2, 1), 
             np.array(seg[1:N:2]).reshape(N/2, 1)), axis=-1).astype('int')
        # print poly
        polys.append(poly)
    cv2.fillPoly(img=img, pts=polys, color=(1, 1, 1))

def countPts(img):
    if len(img.shape) == 3:
        count = np.sum(img[:, :, 0])
    elif len(img.shape) == 2:
        count = np.sum(img)
    return count

def distributeAtt(numX, numY, filledPolyImg):
    width = filledPolyImg.shape[1]
    height = filledPolyImg.shape[0]
    w = width / numX
    h = height / numY
    totalCount = 0
    count = np.zeros((numY, numX), dtype='float32')
    for x in range(numX):
        for y in range(numY):
            startX = w * x
            startY = h * y
            if x < numX - 1:
                endX = w * (x + 1)
            else:
                endX = width
            if y < numY - 1:
                endY = h * (y + 1)
            else:
                endY = height
            count[y, x] = \
                countPts(filledPolyImg[startY:endY,startX:endX])
            totalCount += count[y, x]
    count /= totalCount
    return count

def parse(trainJsonFilename, validJsonFilename):
    with open(trainJsonFilename) as f:
        insttxtTrain = f.read()
    instancesTrain = json.loads(insttxtTrain)
    annotations = instancesTrain['annotations']
    images = instancesTrain['images']
    categories = instancesTrain['categories']

    with open(validJsonFilename) as f:
        insttxtVal = f.read()
    instancesVal = json.loads(insttxtVal)
    annotations.extend(instancesVal['annotations'])
    images.extend(instancesVal['images'])
    categories.extend(instancesVal['categories'])
    return annotations, images, categories

def gatherCount(trainJsonFilename, validJsonFilename):
    annotations, images, categories = \
        parse(trainJsonFilename, validtrainJsonFilename)
    splitDict, imgidDict, imgidIdict, imgPathDict = \
        buildImgIdDict(imgidTrainFilename, imgidValidFilename)
    catDict = buildCatDict(categories)
    imgDict = buildImgDict(images, imgidDict)

    inputData = []
    targetData = []
    countDict = {}

    # Each annotation is only responsible for one object.
    # Multiple object of same category => multiple annotations.
    L = len(annotations)
    print 'Total instances:', L
    for i in range(L):
        if i % 1000 == 0: print i
        img = imgDict[ann['image_id']]
        catId = ann['category_id']
        if countDict.has_key(img):
            if countDict[img].has_key(catId):
                countDict[img][catId] += 1
            else:
                countDict[img][catId] = 1
        else:
            countDict[img] = {catId: 1}
    count = [0] * 10
    for i in countDict.iterkeys():
        for j in countDict[i].iterkeys():
            if countDict[i][j] < 10:
                count[countDict[i][j]] += 1
    print count
    return countDict


def gatherAttention(trainJsonFilename, validJsonFilename):
    annotations, images, categories = \
        parse(trainJsonFilename, validtrainJsonFilename)
    splitDict, imgidDict, imgidIdict, imgPathDict = \
        buildImgIdDict(imgidTrainFilename, imgidValidFilename)
    catDict = buildCatDict(categories)
    imgDict = buildImgDict(images, imgidDict)

    trainInput = []
    trainTarget = []
    validInput = []
    validTarget = []
    testInput = []
    testTarget = []

    L = len(annotations)
    print 'Total instances:', L
    invalidCount = 0

    for i in range(L):
        if i % 1000 == 0: print 'Num', i, 'Invalid', invalidCount
        ann = annotations[i]
        seg = ann['segmentation']
        catId = ann['category_id']
        img = imgDict[ann['image_id']]
        imgid = str(img['id'])
        width = int(img['width'])
        height = int(img['height'])

        if type(seg) is not list:
            invalidCount += 1
            continue

        zeroMat = np.zeros((height, width, 3), dtype='uint8')
        polyFill(zeroMat, seg)
        att = distributeAtt(14, 14, zeroMat)

        # Assemble input data
        # imgId, catId
        inputData = [img['new_id'], catId]

        # Assemble target data
        # attention, flattened
        att = att.reshape(196)
        targetData = att

        s = splitDict[imgid]
        if s == 0:
            trainInput.append(inputData)
            trainTarget.append(targetData)
        elif s == 1:
            validInput.append(inputData)
            validTarget.append(targetData)
        elif s == 2:
            testInput.append(inputData)
            testTarget.append(targetData)

    trainInput = np.array(trainInput, dtype='int')
    trainTarget = np.array(trainTarget, dtype='float32')
    validInput = np.array(validInput, dtype='int')
    validTarget = np.array(validTarget, dtype='float32')
    testInput = np.array(testInput, dtype='int')
    testTarget = np.array(testTarget, dtype='float32')
    
    trainData = np.array((trainInput, trainTarget, 0), dtype='object')
    validData = np.array((validInput, validTarget, 0), dtype='object')
    testData = np.array((testInput, testTarget, 0), dtype='object')

    return trainData, validData, testData, catDict

if __name__ == '__main__':
    """
    Usage:
    python mscoco_extract_instances.py 
                        -t[ype] {attention/count}
                        [-o[utput] {outputFolder}]
    """
    outputFolder = None
    for i, flag in enumerate(sys.argv):
        if flag == '-o' or flag == '-output':
            outputFolder = sys.argv[i + 1]
        elif flag == '-t' or flag == '-type':
            outputType = sys.argv[i + 1]

    if outputType == 'attention':
        if outputFolder is None: outputFolder = '../data/coco-att'
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        trainData, validData, testData, catDict = \
            gatherAttention(trainJsonFilename, validtrainJsonFilename)
        np.save(os.path.join(outputFolder, 'train.npy'), trainData)
        np.save(os.path.join(outputFolder, 'valid.npy'), validData)
        np.save(os.path.join(outputFolder, 'test.npy'), testData)

        with open(os.path.join(outputFolder, 'question_vocabs.txt'), 'w') as f:
            for k in catDict.iterkeys():
                f.write('%s\n' % catDict[k]['name'].replace(' ', '_'))

        with open(os.path.join(outputFolder, 'answer_vocabs.txt'), 'w') as f:
            for k in catDict.iterkeys():
                f.write('%s\n' % catDict[k]['name'].replace(' ', '_'))
    elif outputType == 'count':
        if outputFolder is None: outputFolder = '../data/coco-count'    
        f not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
        countDict = gatherAttention(trainJsonFilename, validtrainJsonFilename)
        with open(os.path.join(outputFolder, 'count.pkl'), 'wb') as f:
            pkl.dump(countDict, f)